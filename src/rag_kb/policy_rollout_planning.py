"""受限的策略落地规划：仅动态检索授权证据，生成可引用的 Todo 草案。"""

from __future__ import annotations

import importlib
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Callable, Protocol

from pydantic import BaseModel, Field, ValidationError

from rag_kb.models import (
    PlanStatus, PolicyRolloutClarification, PolicyRolloutPlanVersion,
    PolicyRolloutTaskResponse, PolicyRolloutTodo, Role, TodoCitation,
)
from rag_kb.observability import RunObserver
from rag_kb.policy_rollout_store import PolicyRolloutTaskStore
from rag_kb.retriever import RetrievalCandidate, Retriever

logger = logging.getLogger(__name__)
REFUSAL_MESSAGE = "当前无法完成策略落地规划"
ALLOWED_TOOL_NAMES = frozenset({"search_authorized_evidence"})


class PlanningError(RuntimeError):
    pass


class DraftTodo(BaseModel):
    todo_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=500)
    detail: str = Field(min_length=1, max_length=4000)
    depends_on: list[str] = Field(default_factory=list, max_length=50)
    chunk_ids: list[str] = Field(default_factory=list, min_length=1, max_length=20)


class PlanningDraft(BaseModel):
    status: PlanStatus
    todos: list[DraftTodo] = Field(default_factory=list, max_length=50)
    clarification_questions: list[str] = Field(default_factory=list, max_length=10)
    used_chunk_ids: list[str] = Field(default_factory=list, max_length=128)


class ToolBindable(Protocol):
    def bind_tools(self, tools: list, **kwargs): ...


class BoundPlanningTools:
    """唯一工具绑定角色和 Retriever，模型不能选择 ACL 或底层数据源。"""
    def __init__(self, retriever: Retriever, role: Role, max_tool_calls: int) -> None:
        self._retriever, self._role, self._max_tool_calls = retriever, role, max_tool_calls
        self._tool_calls = 0
        self.candidates: dict[str, RetrievalCandidate] = {}

    def search_authorized_evidence(self, query: str) -> str:
        self._tool_calls += 1
        if self._tool_calls > self._max_tool_calls or not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise PlanningError("invalid_tool_call")
        try:
            results = self._retriever.search(query, self._role)
        except Exception as exc:
            raise PlanningError("retrieval_failed") from exc
        for item in results.candidates:
            self.candidates[item.chunk_id] = item
        return json.dumps({"evidence": [{"chunk_id": item.chunk_id, "source_locator": item.source_locator, "text": item.text} for item in results.candidates]}, ensure_ascii=False)

    def langchain_tools(self) -> list:
        StructuredTool = importlib.import_module("langchain_core.tools").StructuredTool
        return [StructuredTool.from_function(self.search_authorized_evidence, name="search_authorized_evidence", description="在服务端绑定角色的授权有效文档中检索证据。只传检索查询。")]


class PolicyRolloutPlanningService:
    # ponytail: v1 is request-synchronous/read-only; durable execution is deferred.
    def __init__(
        self, store: PolicyRolloutTaskStore, retriever: Retriever, model: ToolBindable | None,
        *, citation_checker: Callable[[list[TodoCitation], Role], bool] | None = None,
        max_steps: int = 4, max_tool_calls: int = 3, claim_ttl_seconds: int = 300,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.store, self.retriever, self.model = store, retriever, model
        self.citation_checker = citation_checker or (lambda citations, _role: not citations)
        self.max_steps, self.max_tool_calls = max_steps, max_tool_calls
        self.claim_ttl_seconds, self.now = claim_ttl_seconds, now or (lambda: datetime.now(UTC))

    def create(self, goal: str, role: Role) -> PolicyRolloutTaskResponse:
        task_id = str(uuid.uuid4())
        claim_token = str(uuid.uuid4())
        self.store.create(task_id, role, goal, PlanStatus.PLANNING, None, claim_token=claim_token)
        return self._plan(task_id, role, goal, [], None, claim_token)

    def continue_task(self, task_id: str, role: Role, answers: list[str]) -> PolicyRolloutTaskResponse | None:
        previous = self.get(task_id, role)
        if previous is None or previous.status != PlanStatus.CLARIFICATION:
            return None
        questions = previous.clarification.questions if previous.clarification else []
        clarification = PolicyRolloutClarification(questions=questions, answers=answers)
        claim_token = str(uuid.uuid4())
        if not self.store.claim_clarification(task_id, role, clarification, claim_token=claim_token):
            return None
        return self._plan(task_id, role, previous.goal, answers, clarification, claim_token)

    def get(self, task_id: str, role: Role) -> PolicyRolloutTaskResponse | None:
        self.store.reclaim_stale_planning(task_id, role, self.claim_ttl_seconds, now=self.now())
        result = self.store.get(task_id, role, str(uuid.uuid4()))
        if result is None or not self._citations_currently_authorized(result, role):
            return None
        return result

    def _plan(
        self, task_id: str, role: Role, goal: str, answers: list[str],
        claimed_clarification: PolicyRolloutClarification | None, claim_token: str,
    ) -> PolicyRolloutTaskResponse:
        observer, started = RunObserver(str(role), task_id), time.perf_counter()
        tools = BoundPlanningTools(self.retriever, role, self.max_tool_calls)
        current = self.store.get(task_id, role, observer.run_id)
        if current is None:
            raise PlanningError("task_persistence_failed")
        try:
            draft = self._run_agent(tools, goal, answers, observer)
            if draft.status == PlanStatus.CLARIFICATION:
                if draft.todos or not draft.clarification_questions:
                    raise PlanningError("invalid_model_output")
                clarification = PolicyRolloutClarification(questions=draft.clarification_questions, answers=answers)
                if not self.store.update_state(task_id, role, PlanStatus.CLARIFICATION, clarification, claim_token=claim_token):
                    raise PlanningError("planning_claim_lost")
            elif draft.status == PlanStatus.PLANNED:
                plan = self._validated_plan(draft, tools)
                if self.store.complete_claim_and_append_plan(task_id, role, plan, current.clarification, claim_token) is None:
                    raise PlanningError("planning_claim_lost")
            else:
                raise PlanningError("invalid_model_output")
        except Exception as exc:
            logger.error("策略落地规划失败(error_code=policy_rollout_planning_failed)", exc_info=True)
            if claimed_clarification is None:
                self.store.update_state(task_id, role, PlanStatus.FAILED, None, claim_token=claim_token)
            else:
                self.store.release_claim(task_id, role, claimed_clarification, claim_token)
            observer.emit("policy_rollout_planning", "completed", status="refusal", error_code="planning_refused", started_at=started)
            return self._stored(task_id, role, observer.run_id)
        observer.emit("policy_rollout_planning", "completed", status="success", counts={"candidates": len(tools.candidates)}, started_at=started)
        return self._stored(task_id, role, observer.run_id)

    def _stored(self, task_id: str, role: Role, run_id: str) -> PolicyRolloutTaskResponse:
        result = self.store.get(task_id, role, run_id)
        if result is None or not self._citations_currently_authorized(result, role):
            raise PlanningError("task_persistence_failed")
        return result

    def _citations_currently_authorized(self, task: PolicyRolloutTaskResponse, role: Role) -> bool:
        citations = [citation for plan in task.plan_versions for todo in plan.todos for citation in todo.citations]
        try:
            return self.citation_checker(citations, role)
        except Exception:
            return False

    def _run_agent(self, tools: BoundPlanningTools, goal: str, answers: list[str], observer: RunObserver) -> PlanningDraft:
        if self.model is None:
            raise PlanningError("agent_runtime_unavailable")
        langchain_tools = tools.langchain_tools()
        if {tool.name for tool in langchain_tools} != ALLOWED_TOOL_NAMES:
            raise PlanningError("tool_allowlist_invalid")
        create_agent = importlib.import_module("langchain.agents").create_agent
        HumanMessage = importlib.import_module("langchain_core.messages").HumanMessage
        agent = create_agent(model=self.model, tools=langchain_tools, name="bounded_policy_rollout_planner", system_prompt=("你只能调用提供的只读检索工具，不能选择角色、ACL、集合或执行 Todo。必须先检索，再只返回 JSON：status 为 planned 或 clarification；planned 含 todos(todo_id,title,detail,depends_on,chunk_ids) 与 used_chunk_ids；clarification 含 clarification_questions。"))
        try:
            result = agent.invoke({"messages": [HumanMessage(content=json.dumps({"goal": goal, "clarification_answers": answers}, ensure_ascii=False))]}, config={"recursion_limit": max(3, self.max_steps * 2 + 1)})
            messages = result.get("messages") if isinstance(result, dict) else None
            if not messages:
                raise PlanningError("invalid_model_output")
            content = getattr(messages[-1], "content", messages[-1])
            return PlanningDraft.model_validate(content) if isinstance(content, dict) else PlanningDraft.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            raise PlanningError("invalid_model_output") from exc
        except PlanningError:
            raise
        except Exception as exc:
            raise PlanningError("agent_execution_failed") from exc

    @staticmethod
    def _validated_plan(draft: PlanningDraft, tools: BoundPlanningTools) -> PolicyRolloutPlanVersion:
        if (
            not draft.todos
            or len(draft.used_chunk_ids) != len(set(draft.used_chunk_ids))
            or not set(draft.used_chunk_ids).issubset(tools.candidates)
        ):
            raise PlanningError("invalid_citations")
        ids = [item.todo_id for item in draft.todos]
        if len(ids) != len(set(ids)):
            raise PlanningError("duplicate_todo_id")
        known = set(ids)
        graph = {item.todo_id: set(item.depends_on) for item in draft.todos}
        if any(not deps.issubset(known) or todo_id in deps for todo_id, deps in graph.items()):
            raise PlanningError("invalid_dependencies")
        visiting, complete = set(), set()
        def visit(node: str) -> None:
            if node in visiting: raise PlanningError("cyclic_dependencies")
            if node not in complete:
                visiting.add(node)
                for dependency in graph[node]: visit(dependency)
                visiting.remove(node); complete.add(node)
        for todo_id in graph: visit(todo_id)
        todos = []
        for item in draft.todos:
            if (
                len(item.chunk_ids) != len(set(item.chunk_ids))
                or not set(item.chunk_ids).issubset(draft.used_chunk_ids)
                or not set(item.chunk_ids).issubset(tools.candidates)
            ):
                raise PlanningError("invalid_citations")
            citations = [TodoCitation(document_id=c.document_id, document_version=c.document_version, source_file_name=c.source_file_name, source_locator=c.source_locator, chunk_id=c.chunk_id) for chunk_id in item.chunk_ids for c in [tools.candidates[chunk_id]]]
            todos.append(PolicyRolloutTodo(todo_id=item.todo_id, title=item.title, detail=item.detail, depends_on=item.depends_on, citations=citations))
        return PolicyRolloutPlanVersion(version=0, todos=todos, created_at=datetime.now(UTC))
