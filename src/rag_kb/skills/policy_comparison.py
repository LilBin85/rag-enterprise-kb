"""受限策略版本比较技能。

技能只负责调用两个服务端绑定的只读工具并给出简短摘要；版本读取、确定性
diff、引用和失败关闭均由应用层完成。此模块不在导入时依赖顶层 ``langchain``。
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field, ValidationError

from rag_kb.models import (
    Citation,
    PolicyChange,
    PolicyComparison,
    PolicyComparisonResponse,
    Refusal,
    Role,
)
from rag_kb.observability import RunObserver
from rag_kb.store import ChunkStore

# 保留历史 logger 名称，避免迁移改变现有日志路由。
logger = logging.getLogger("rag_kb.policy_compare")

REFUSAL_MESSAGE = "当前无法完成策略版本比较"
ALLOWED_TOOL_NAMES = frozenset({"fetch_authorized_version", "compare_authorized_versions"})


class PolicyAgentVerdict(BaseModel):
    """模型唯一允许返回的结构；模型不能生成 diff 或引用元数据。"""

    summary: str = Field(min_length=1, max_length=2000)
    used_chunk_ids: list[str] = Field(default_factory=list, max_length=128)


class PolicyToolError(RuntimeError):
    """工具边界、参数或存储失败。"""


class BoundToolModel(Protocol):
    def invoke(self, messages: list): ...


class ToolBindable(Protocol):
    def bind_tools(self, tools: list, **kwargs) -> BoundToolModel: ...


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    document_id: str
    document_version: int
    source_file_name: str
    source_locator: str
    text: str
    content_hash: str


def _evidence(payload: dict) -> Evidence:
    """仅从已通过 Qdrant ACL 的 payload 构造证据对象。"""
    return Evidence(
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        document_version=int(payload["document_version"]),
        source_file_name=str(payload["source_file_name"]),
        source_locator=str(payload["source_locator"]),
        text=str(payload["text"]),
        content_hash=str(payload["content_hash"]),
    )


def _section_map(evidence: list[Evidence]) -> dict[str, str]:
    """按稳定章节定位符聚合块，避免让模型决定变化边界。"""
    grouped: dict[str, list[Evidence]] = {}
    for item in sorted(evidence, key=lambda value: value.chunk_id):
        grouped.setdefault(item.source_locator, []).append(item)
    return {section: "\n".join(item.text for item in items) for section, items in grouped.items()}


def deterministic_diff(from_evidence: list[Evidence], to_evidence: list[Evidence]) -> list[PolicyChange]:
    """基于章节正文生成可重复的 added/modified/removed 结果。"""
    before = _section_map(from_evidence)
    after = _section_map(to_evidence)
    changes: list[PolicyChange] = []
    for section in sorted(set(before) | set(after)):
        if section not in before:
            changes.append(PolicyChange(section=section, change="added", to_text=after[section]))
        elif section not in after:
            changes.append(PolicyChange(section=section, change="removed", from_text=before[section]))
        elif before[section] != after[section]:
            changes.append(
                PolicyChange(
                    section=section,
                    change="modified",
                    from_text=before[section],
                    to_text=after[section],
                )
            )
    return changes


class BoundPolicyTools:
    """服务端闭包工具：文档、角色、版本和 ACL 永远不由模型传入。"""

    def __init__(
        self,
        reader: Callable[[str, int, Role], list[dict]],
        document_id: str,
        from_version: int,
        to_version: int,
        role: Role,
        max_tool_calls: int = 3,
    ) -> None:
        self._reader = reader
        self._document_id = document_id
        self._from_version = from_version
        self._to_version = to_version
        self._role = role
        self._max_tool_calls = max_tool_calls
        self._tool_calls = 0
        self._evidence: dict[int, list[Evidence]] = {}
        self.changes: list[PolicyChange] | None = None

    def _claim_tool_call(self) -> None:
        """在工具边界限制调用次数，避免 Agent 无限循环。"""
        self._tool_calls += 1
        if self._tool_calls > self._max_tool_calls:
            raise PolicyToolError("max_tool_calls_exceeded")

    def fetch_authorized_version(self, version: int) -> str:
        """读取一个已绑定版本；返回给 Agent 的字段仍来自授权结果。"""
        self._claim_tool_call()
        if version not in {self._from_version, self._to_version}:
            raise PolicyToolError("version_not_bound")
        if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
            raise PolicyToolError("invalid_version")
        try:
            raw_payloads = self._reader(self._document_id, version, self._role)
            evidence = [_evidence(item) for item in raw_payloads]
        except Exception as exc:
            raise PolicyToolError("evidence_read_failed") from exc
        if not evidence or any(
            item.document_id != self._document_id or item.document_version != version
            for item in evidence
        ):
            raise PolicyToolError("evidence_unavailable")
        raw_statuses = {str(item.get("status")) for item in raw_payloads}
        if raw_statuses - {"active", "superseded"}:
            raise PolicyToolError("lifecycle_not_comparable")
        if any(str(self._role) not in {str(role) for role in item.get("allowed_roles", [])} for item in raw_payloads):
            raise PolicyToolError("acl_mismatch")
        self._evidence[version] = evidence
        # 工具返回最小必要上下文，不把 ACL/status 等控制字段交给模型。
        return json.dumps(
            {
                "version": version,
                "evidence": [
                    {"chunk_id": item.chunk_id, "source_locator": item.source_locator, "text": item.text}
                    for item in evidence
                ],
            },
            ensure_ascii=False,
        )

    def compare_authorized_versions(self) -> str:
        """只对已读取的两个授权版本执行确定性比较。"""
        self._claim_tool_call()
        if set(self._evidence) != {self._from_version, self._to_version}:
            raise PolicyToolError("evidence_not_loaded")
        self.changes = deterministic_diff(
            self._evidence[self._from_version], self._evidence[self._to_version]
        )
        return json.dumps(
            {
                "from_version": self._from_version,
                "to_version": self._to_version,
                "changes": [item.model_dump() for item in self.changes],
            },
            ensure_ascii=False,
        )

    def langchain_tools(self) -> list:
        """运行时才创建 LangChain 工具，保持无顶层 langchain 环境可导入。"""
        StructuredTool = importlib.import_module("langchain_core.tools").StructuredTool

        return [
            StructuredTool.from_function(
                self.fetch_authorized_version,
                name="fetch_authorized_version",
                description="读取服务端绑定的授权策略版本证据。只能传请求中的 from_version 或 to_version。",
            ),
            StructuredTool.from_function(
                self.compare_authorized_versions,
                name="compare_authorized_versions",
                description="比较已经读取的两个授权版本；无参数。",
            ),
        ]


class DeepSeekPolicyAgentModel:
    """ChatDeepSeek 的可选运行时适配；构造失败不会影响项目其它导入。"""

    def __init__(
        self,
        settings,
        *,
        max_retries: int | None = None,
        tool_choice: str = "any",
        single_tool_call: bool = False,
    ) -> None:
        ChatDeepSeek = importlib.import_module("langchain_deepseek").ChatDeepSeek
        options = {
            "model": settings.deepseek_model,
            "api_key": settings.deepseek_api_key,
            "base_url": settings.deepseek_base_url,
            "temperature": 0,
            "timeout": getattr(settings, "agent_timeout_seconds", 20),
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if max_retries is not None:
            options["max_retries"] = max_retries
        self._llm = ChatDeepSeek(**options)
        self._tool_choice = tool_choice
        self._single_tool_call = single_tool_call

    def bind_tools(self, tools: list, **kwargs) -> Any:
        """兼容 create_agent 的 tool_choice，同时禁止并行工具调用。"""
        kwargs["parallel_tool_calls"] = False
        kwargs["tool_choice"] = self._tool_choice
        model = self._llm.bind_tools(tools, **kwargs)
        return _SingleToolCallBoundModel(model, {tool.name for tool in tools}) if getattr(self, "_single_tool_call", False) else model


class _SingleToolCallBoundModel:
    """Provider 忽略并行禁令时，只保留该轮第一个已解析工具调用。"""

    def __init__(self, model, tool_names: set[str]) -> None:
        self._model, self._tool_names = model, tool_names

    def invoke(self, *args, **kwargs):
        result = self._model.invoke(*args, **kwargs)
        tool_calls = getattr(result, "tool_calls", None)
        if isinstance(tool_calls, list) and len(tool_calls) > 1:
            valid_calls = [call for call in tool_calls if self._valid_call(call)]
            valid = valid_calls[0] if valid_calls else None
            return result.model_copy(update={"tool_calls": [valid] if valid else []})
        return result

    def _valid_call(self, call) -> bool:
        return bool(
            isinstance(call, dict)
            and call.get("name") in self._tool_names
            and isinstance(call.get("args"), dict)
            and isinstance(call.get("id"), str)
            and call["id"]
        )

    def __getattr__(self, name):
        return getattr(self._model, name)


class PolicyComparisonService:
    """版本比较服务：验证 → ACL 工具 → 有界 Agent → 确定性 diff → 应用层引用。"""

    def __init__(
        self,
        store: ChunkStore,
        model=None,
        *,
        max_steps: int = 4,
        max_tool_calls: int = 3,
    ) -> None:
        self.store = store
        self.model: ToolBindable | None = model
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls

    def compare(self, document_id: str, from_version: int, to_version: int, role: Role) -> PolicyComparisonResponse:
        observer = RunObserver(str(role), document_id)
        started = time.perf_counter()
        observer.emit("policy_comparison", "validation", status="started", started_at=started)
        if not self._valid_request(document_id, from_version, to_version):
            return self._refusal(observer, role, "invalid_request", started)
        tools = BoundPolicyTools(
            self.store.scroll_authorized_version,
            document_id,
            from_version,
            to_version,
            role,
            max_tool_calls=self.max_tool_calls,
        )
        try:
            if self.model is None:
                raise PolicyToolError("agent_runtime_unavailable")
            verdict = self._run_agent(tools, from_version, to_version, observer)
            changes = tools.changes
            if changes is None:
                return self._refusal(observer, role, "diff_not_computed", started)
            citations = self._citations(tools, changes)
            if not changes or not citations:
                return self._refusal(observer, role, "insufficient_diff_evidence", started)
            result = PolicyComparison(
                document_id=document_id,
                from_version=from_version,
                to_version=to_version,
                changes=changes,
                summary=verdict.summary,
                citations=citations,
            )
        except PolicyToolError as exc:
            return self._refusal(observer, role, str(exc), started)
        except Exception:
            logger.error("策略比较失败(error_code=policy_comparison_failed)", exc_info=True)
            return self._refusal(observer, role, "policy_comparison_failed", started)
        observer.emit("policy_comparison", "completed", status="success", counts={"changes": len(changes)}, started_at=started)
        return PolicyComparisonResponse(
            verdict="comparison", comparison=result, role=role, run_id=observer.run_id
        )

    def _run_agent(
        self, tools: BoundPolicyTools, from_version: int, to_version: int, observer: RunObserver
    ) -> PolicyAgentVerdict:
        """用 LangChain create_agent 驱动工具，但把安全边界留在本服务。"""
        create_agent = importlib.import_module("langchain.agents").create_agent
        langchain_tools = tools.langchain_tools()
        by_name = {tool.name: tool for tool in langchain_tools}
        if set(by_name) != ALLOWED_TOOL_NAMES:
            raise PolicyToolError("tool_allowlist_invalid")
        model = self.model
        if model is None:
            raise PolicyToolError("agent_runtime_unavailable")
        messages_module = importlib.import_module("langchain_core.messages")
        HumanMessage = messages_module.HumanMessage
        agent = create_agent(
            model=model,
            tools=langchain_tools,
            system_prompt=(
                "你只能调用提供的两个只读工具。不得选择文档、角色、集合、状态或 ACL。"
                "先读取请求指定的两个版本，再调用比较工具，最后只返回 JSON: "
                '{"summary":"...","used_chunk_ids":["..."]}。'
            ),
            name="bounded_policy_comparison",
        )
        started = time.perf_counter()
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=f"比较请求中的版本 {from_version} 与 {to_version}。")]},
                config={"recursion_limit": max(3, self.max_steps * 2 + 1)},
            )
        except Exception as exc:
            observer.emit(
                "policy_comparison", "agent", status="error", error_code="agent_execution_failed",
                started_at=started,
            )
            raise PolicyToolError("agent_execution_failed") from exc
        messages = result.get("messages") if isinstance(result, dict) else None
        if not messages:
            raise PolicyToolError("invalid_model_output")
        observer.emit(
            "policy_comparison", "agent", status="success",
            counts={"tool_calls": tools._tool_calls}, started_at=started,
        )
        return self._parse_verdict(getattr(messages[-1], "content", messages[-1]))

    @staticmethod
    def _parse_verdict(content) -> PolicyAgentVerdict:
        if isinstance(content, dict):
            return PolicyAgentVerdict.model_validate(content)
        if isinstance(content, str):
            try:
                return PolicyAgentVerdict.model_validate_json(content)
            except (ValidationError, ValueError) as exc:
                raise PolicyToolError("invalid_model_output") from exc
        raise PolicyToolError("invalid_model_output")

    @staticmethod
    def _valid_request(document_id: str, from_version: int, to_version: int) -> bool:
        return bool(
            isinstance(document_id, str)
            and document_id
            and len(document_id) <= 256
            and isinstance(from_version, int)
            and not isinstance(from_version, bool)
            and isinstance(to_version, int)
            and not isinstance(to_version, bool)
            and 0 < from_version < to_version
        )

    @staticmethod
    def _citations(tools: BoundPolicyTools, changes: list[PolicyChange]) -> list[Citation]:
        sections = {item.section for item in changes}
        result: list[Citation] = []
        for version in (tools._from_version, tools._to_version):
            for item in tools._evidence[version]:
                if item.source_locator in sections:
                    result.append(
                        Citation(
                            document_id=item.document_id,
                            document_version=item.document_version,
                            source_file_name=item.source_file_name,
                            source_locator=item.source_locator,
                            chunk_id=item.chunk_id,
                        )
                    )
        return result

    @staticmethod
    def _refusal(
        observer: RunObserver, role: Role, error_code: str, started_at: float
    ) -> PolicyComparisonResponse:
        observer.emit(
            "policy_comparison", "completed", status="refusal", error_code=error_code,
            started_at=started_at,
        )
        return PolicyComparisonResponse(
            verdict="refusal",
            refusal=Refusal(kind="comparison_refused", message=REFUSAL_MESSAGE),
            role=role,
            run_id=observer.run_id,
        )


__all__ = [
    "ALLOWED_TOOL_NAMES",
    "REFUSAL_MESSAGE",
    "BoundPolicyTools",
    "BoundToolModel",
    "DeepSeekPolicyAgentModel",
    "Evidence",
    "PolicyAgentVerdict",
    "PolicyComparisonService",
    "PolicyToolError",
    "ToolBindable",
    "_SingleToolCallBoundModel",
    "_evidence",
    "_section_map",
    "deterministic_diff",
]
