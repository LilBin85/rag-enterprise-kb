"""受限的策略落地规划：仅动态检索授权证据，生成可引用的 Todo 草案。"""

from __future__ import annotations

import importlib
import hashlib
import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag_kb.models import (
    PlanStatus, PolicyRolloutClarification, PolicyRolloutPlanVersion,
    PolicyRolloutTaskResponse, PolicyRolloutTodo, Role, TodoCitation,
)
from rag_kb.observability import RunObserver, classify_model_failure
from rag_kb.policy_rollout_store import PolicyRolloutTaskStore
from rag_kb.retriever import RetrievalCandidate, Retriever
from rag_kb.retrieval_policy import QueryIntent, QueryPolicy, hash_identifier

logger = logging.getLogger(__name__)
REFUSAL_MESSAGE = "当前无法完成策略落地规划"
STRUCTURAL_CODES = frozenset({
    "invalid_model_output", "invalid_citations", "duplicate_todo_id",
    "invalid_dependencies", "cyclic_dependencies", "unsupported_todo_content",
    "incomplete_evidence_actions",
})
MODEL_FAILURE_CODES = frozenset({
    "json_truncated", "schema_validation_failed", "provider_timeout",
    "abnormal_finish_reason", "provider_empty_response", "adapter_protocol_error",
    "model_output_parse_failed", "provider_request_failed",
})
REPAIRABLE_MODEL_CODES = frozenset({
    "invalid_model_output", "json_truncated", "schema_validation_failed",
    "provider_empty_response", "model_output_parse_failed",
})
SAFE_GROUNDED_FALLBACK_CODES = REPAIRABLE_MODEL_CODES | {
    "unsupported_todo_content", "incomplete_evidence_actions",
}
GROUNDED_COMPOSER_CODE = "grounded_composer_after_validation"
GROUNDED_COMPOSER_OUTCOME = "constrained_composer"


def _provider_failure_code(error: BaseException) -> str:
    """只细分可可靠识别的传输失败；未知异常不能伪装成可修复草稿。"""
    classified = classify_model_failure(error)
    if classified in {"provider_timeout", "provider_request_failed"}:
        return classified
    return "provider_request_failed"

_EXPLICIT_MISSING_MARKERS = (
    "尚未确定", "仍未确定", "还未确定", "暂未确定", "有待确认", "待确认", "未指定", "未提供",
)
_GENERIC_SUPPORT_TERMS = frozenset({
    "制度", "员工", "计划", "执行", "工作", "事项", "相关", "根据", "依据", "完成", "制定",
    "进行", "安排", "要求", "需要", "负责", "部门", "公司", "流程", "政策", "规则",
})
_ACTION_SUPPORT_GROUPS = (
    frozenset({"核对", "确认", "复核", "检查", "校验", "盘点"}),
    frozenset({"配置", "设置", "设定"}),
    frozenset({"发布", "通知", "公告", "告知"}),
    frozenset({"保留", "记录", "留存", "归档"}),
    frozenset({"迁移", "导入", "转移"}),
    frozenset({"查阅", "访问", "获取", "读取"}),
    frozenset({"申请", "审批", "批准"}),
    frozenset({"订购", "采购", "购买", "支付"}),
    frozenset({"删除", "销毁", "清除"}),
    frozenset({"发送", "分发", "推送"}),
)
# 自动组合器只拥有“核对 / 配置 / 发布通知”三类制度落地动作。其余动词即使
# 出现在证据中也绝不从模型自由文本带入最终 Todo，需由专用执行能力另行建模。
_COMPOSABLE_ACTION_GROUPS = frozenset({0, 1, 2})
_ACTION_SENTENCE_SPLIT = re.compile(r"[。！？；;\n]+")
_ACTION_CLAUSE_SPLIT = re.compile(r"[，,]+")
_DIRECT_DENIAL_MARKER = re.compile(
    r"(?:不得|不应|不可|不能|不必|不需|不需要|不允许|禁止|严禁|无需|无须|避免|停止|不要|切勿|"
    r"无权|没有权限|无权限|未获(?:得)?权限|未授权|未获授权|未批准|未审批|未许可|权限不足|"
    r"仅限|只限|暂不|不建议|不予|暂停|暂缓|取消|拒绝)"
)
_RESTRICTED_ACTION_SENTENCE = re.compile(
    r"(?:不得|不应|不可|不能|不必|不需|不需要|不允许|禁止|严禁|无需|无须|避免|停止|不要|切勿|"
    r"无权|没有权限|无权限|未获(?:得)?权限|未授权|未获授权|未批准|未审批|未许可|权限不足|"
    r"权限(?:未知|不明|未确认|待确认)|授权状态(?:未知|不明|未确认|待确认)|"
    r"仅限|只限|(?:^|[：:,，])限[^。！？；;\n]{0,40}(?:访问|读取|查阅|获取|发送|分发|推送)|"
    r"暂不|不建议|不予|暂停|暂缓|取消|拒绝|待定|待确认|"
    r"未经|除非|否则|仅当|如果|若(?!干)|一旦|假如|倘若|"
    r"只有[^。！？；;\n]{0,60}才|才可|方可|方能|才能|即可|"
    r"(?:经|获得|取得|获)[^。！？；;\n]{0,40}(?:批准|审批|授权|同意|许可|通过)|"
    # 仅识别紧邻的审批条件，不能把“审批流程……随后发布”误判成受限句。
    r"(?:批准|审批|授权|同意|许可)(?:通过)?(?:后|前|才|方可|方能|即可|才能)|"
    r"通过(?:后|前|才|方可|方能|即可|才能)|"
    r"(?:条件|情况|前提)[^。！？；;\n]{0,6}(?:下|时|后|前)|前提是)"
)
_NON_BUSINESS_VARIABLE_TERMS = frozenset({
    "证据", "授权证据", "权限", "访问权限", "授权", "制度", "政策", "规则", "条款",
    "文档", "资料", "来源", "检索结果", "搜索结果",
})


def _required_evidence_action_terms(evidence: str) -> tuple[str, ...]:
    """只提取证据正文中同时明确出现的关键动作，避免凭目标臆造要求。"""
    implementation = ("配置", "审批流程", "通知")
    review = ("核对", "适用资格", "年假余额")
    terms: list[str] = []
    for sentence in _positive_action_sentences(evidence):
        if all(term in sentence for term in implementation):
            terms.extend(implementation)
        if all(term in sentence for term in review):
            terms.extend(review)
    return tuple(dict.fromkeys(terms))


def _explicit_missing_business_variables(goal: str) -> list[str]:
    """只提取目标中明确声明未定的变量，避免把一般不确定性升级为澄清。"""
    if not isinstance(goal, str):
        return []
    markers = "|".join(map(re.escape, _EXPLICIT_MISSING_MARKERS))
    variables: list[str] = []
    for clause in re.split(r"[。！？；;\n]+", goal):
        for match in re.finditer(rf"(?P<prefix>.{{1,80}}?)(?:{markers})", clause):
            prefix = re.split(r"[，,:：]", match.group("prefix"))[-1].strip()
            prefix = re.sub(r"^(?:但|且|并且|然而|不过|目前|当前|其中|关于|对于|仍需)+", "", prefix).strip()
            label_match = re.search(r"[A-Za-z0-9_\-\u4e00-\u9fff]{2,24}$", prefix)
            if not label_match:
                continue
            label = label_match.group(0)
            if any(term in label for term in _NON_BUSINESS_VARIABLE_TERMS):
                continue
            if label not in variables:
                variables.append(label)
    return variables


def _business_input_terms(variables: list[str], answers: list[str]) -> set[str]:
    terms = {value.strip().casefold() for value in variables if len(value.strip()) >= 2}
    for answer in answers:
        normalized = re.sub(r"\s+", "", answer).casefold()
        if len(normalized) >= 2:
            terms.add(normalized)
        for value in re.split(r"[，,、；;：:=为是]", normalized):
            if len(value) >= 2:
                terms.add(value)
    return terms


def _support_terms(value: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|\d+(?:\.\d+)?", value.casefold()):
        terms.add(token)
    for segment in re.findall(r"[\u4e00-\u9fff]+", value):
        for size in (2, 3, 4):
            terms.update(segment[index:index + size] for index in range(max(0, len(segment) - size + 1)))
    return {term for term in terms if term not in _GENERIC_SUPPORT_TERMS}


def _todo_instruction_text(title: str, detail: str) -> str:
    """证据摘录只用于审计展示，不能反向证明 Todo 自己声明了必要动作。"""
    instruction_detail = detail
    for marker in ("授权制度依据：", "授权制度依据摘要：", "证据原文：", "证据摘要："):
        instruction_detail = instruction_detail.split(marker, 1)[0]
    return f"{title} {instruction_detail}".strip()


def _todo_is_supported(
    title: str, detail: str, evidence_texts: list[str], authority_texts: list[str] | None = None,
) -> bool:
    todo_text = _todo_instruction_text(title, detail)
    evidence_text = " ".join(evidence_texts)
    authority_text = " ".join(authority_texts or [])
    todo_terms = _support_terms(todo_text)
    evidence_terms = _support_terms(evidence_text)
    if not todo_terms & evidence_terms:
        return False
    # 最终 Todo 由组合器产生，只允许组合器拥有的三类动作；“审批流程”里的
    # 名词“审批”不能被误当成另一条需要单独执行的审批动作。
    todo_actions = set(_positive_composable_action_terms(todo_text))
    if not todo_actions:
        return False
    supported_actions = {
        group
        for source in [*evidence_texts, *(authority_texts or [])]
        for group in _positive_action_terms(source)
    }
    restricted_actions = {
        group
        for source in [*evidence_texts, *(authority_texts or [])]
        for group in _restricted_action_group_indexes(source)
    }
    if todo_actions & restricted_actions or not todo_actions.issubset(supported_actions):
        return False
    todo_numbers = set(re.findall(r"\d+(?:\.\d+)?", todo_text))
    supported_numbers = set(re.findall(r"\d+(?:\.\d+)?", f"{evidence_text} {authority_text}"))
    return todo_numbers.issubset(supported_numbers)


def _action_group_indexes(value: str) -> set[int]:
    return {
        index for index, group in enumerate(_ACTION_SUPPORT_GROUPS)
        if any(action in value for action in group)
    }


def _action_sentences(value: str) -> list[str]:
    return [sentence.strip() for sentence in _ACTION_SENTENCE_SPLIT.split(value) if sentence.strip()]


def _positive_action_sentences(value: str) -> list[str]:
    """风险标记作用于整个中文句/分句；不尝试把条件中的动作改写成无条件动作。"""
    return [
        sentence for sentence in _action_sentences(value)
        if not _RESTRICTED_ACTION_SENTENCE.search(sentence)
    ]


def _positive_action_terms(
    value: str, allowed_groups: frozenset[int] | None = None,
) -> dict[int, str]:
    terms: dict[int, str] = {}
    for sentence in _positive_action_sentences(value):
        for index, group in enumerate(_ACTION_SUPPORT_GROUPS):
            if allowed_groups is not None and index not in allowed_groups:
                continue
            matches = [
                (match.start(), term)
                for term in group
                for match in re.finditer(re.escape(term), sentence)
            ]
            if matches and index not in terms:
                terms[index] = min(matches)[1]
    return terms


def _positive_composable_action_terms(value: str) -> dict[int, str]:
    return _positive_action_terms(value, _COMPOSABLE_ACTION_GROUPS)


def _restricted_action_group_indexes(value: str) -> set[int]:
    """仅把含限制条件的分句（及其紧随的受条件动作）标为受限。

    不能让“先核对；未完成核对前不得迁移”中的迁移限制反向否定前一个
    独立的核对动作；但“经批准后，发送资料”仍需把后续发送视为条件动作。
    """
    restricted: set[int] = set()
    for sentence in _action_sentences(value):
        clauses = [clause.strip() for clause in _ACTION_CLAUSE_SPLIT.split(sentence) if clause.strip()]
        for position, clause in enumerate(clauses):
            if not _RESTRICTED_ACTION_SENTENCE.search(clause):
                continue
            direct_denial = _DIRECT_DENIAL_MARKER.search(clause)
            # “未完成核对前不得迁移”只限制迁移；不能把条件中的“核对”错判为
            # 被禁止的核对动作。没有直接否定词时保留整个条件分句以失败关闭。
            action_clause = (
                clause[direct_denial.start():] if direct_denial else clause
            ).replace("待确认", "").replace("未确认", "")
            # 条件写在前一分句时，保守地同时限制紧随其后的动作；否定词写在
            # 当前分句时不扩散到无关的前序动作。
            if re.search(r"(?:未经|除非|仅当|只有|经|获得|取得|获|批准|审批|授权|同意|许可|通过|条件|前提)", clause):
                if position + 1 < len(clauses):
                    action_clause = f"{action_clause} {clauses[position + 1]}"
            restricted.update(
                index
                for index, group in enumerate(_ACTION_SUPPORT_GROUPS)
                if any(action in action_clause for action in group)
            )
    return restricted


def _high_level_action_is_grounded(value: str, evidence_texts: list[str]) -> bool:
    """组合器只接受由所选证据直接支持的高层动作，不采纳任意模型文本。"""
    evidence = " ".join(evidence_texts)
    action_groups = set(_positive_composable_action_terms(value))
    positive_evidence_groups = {
        group for source in evidence_texts for group in _positive_composable_action_terms(source)
    }
    restricted_evidence_groups = {
        group for source in evidence_texts for group in _restricted_action_group_indexes(source)
    }
    if (
        not action_groups
        or not action_groups.issubset(positive_evidence_groups)
        or bool(action_groups & restricted_evidence_groups)
    ):
        return False
    if not _support_terms(value) & _support_terms(evidence):
        return False
    proposed_numbers = set(re.findall(r"\d+(?:\.\d+)?", value))
    evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence))
    return proposed_numbers.issubset(evidence_numbers)


_USAGE_METADATA_TOKEN_KEYS = frozenset({
    "input_tokens", "output_tokens", "total_tokens",
})
_PROVIDER_TOKEN_USAGE_KEYS = frozenset({
    "prompt_tokens", "completion_tokens", "total_tokens",
    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
    "cached_tokens", "reasoning_tokens",
})


def _safe_token_counts(value: Any, allowed_keys: frozenset[str]) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        key: count
        for key, count in value.items()
        if key in allowed_keys and type(count) is int and count >= 0
    }


class PlanningError(RuntimeError):
    """稳定错误码；可选载荷只允许安全的 Token 整数计数。"""

    def __init__(
        self, code: str, *, usage_metadata: Any = None, response_metadata: Any = None,
    ) -> None:
        super().__init__(code)
        safe_usage = _safe_token_counts(usage_metadata, _USAGE_METADATA_TOKEN_KEYS)
        if safe_usage:
            self.usage_metadata = safe_usage
        token_usage = (
            response_metadata.get("token_usage")
            if isinstance(response_metadata, dict) else None
        )
        safe_provider_usage = _safe_token_counts(
            token_usage, _PROVIDER_TOKEN_USAGE_KEYS,
        )
        if safe_provider_usage:
            self.response_metadata = {"token_usage": safe_provider_usage}


class DraftTodo(BaseModel):
    todo_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=500)
    detail: str = Field(min_length=1, max_length=4000)
    depends_on: list[str] = Field(default_factory=list, max_length=50)
    chunk_ids: list[str] = Field(default_factory=list, min_length=1, max_length=20)


class DraftActionIntent(BaseModel):
    """模型选择的高层动作；应用层负责把它编排成可执行、可引用的 Todo。"""

    # ID 只是模型意图的稳定标签，不是授权依据；缺失时由组合器按顺序补齐。
    intent_id: str | None = Field(default=None, min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1000)
    chunk_ids: list[str] = Field(default_factory=list, min_length=1, max_length=20)


class PlanningDraft(BaseModel):
    action: Literal["propose", "clarify", "refuse"] | None = None
    status: str | None = None
    todos: list[DraftTodo] = Field(default_factory=list, max_length=50)
    action_intents: list[DraftActionIntent] = Field(default_factory=list, max_length=50)
    clarification_questions: list[str] = Field(default_factory=list, max_length=10)
    used_chunk_ids: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def terminal_action_is_valid(self) -> "PlanningDraft":
        legacy: dict[str, Literal["propose", "clarify", "refuse"]] = {
            "draft": "propose", "planned": "propose", "clarification": "clarify",
            "refused": "refuse", "refuse": "refuse",
        }
        if self.action is None:
            resolved = legacy.get(self.status or "")
            if resolved is None:
                raise ValueError("invalid_terminal_action")
            self.action = resolved
            self.status = None
        elif self.status is not None:
            legacy_action = legacy.get(self.status)
            if legacy_action is not None and legacy_action != self.action:
                raise ValueError("invalid_terminal_action")
            # 未知/非枚举状态只是旧协议噪声，action 已由外层或模型显式决定，
            # 直接丢弃，避免自由文本被后续链路读取。
            self.status = None
        if self.action == "propose" and not self.used_chunk_ids and self.action_intents:
            # used_chunk_ids 是 intents.chunk_ids 的投影，不扩大模型选择的证据集合。
            self.used_chunk_ids = sorted({
                chunk_id
                for intent in self.action_intents
                for chunk_id in intent.chunk_ids
            })
        return self


class PlannerDecision(BaseModel):
    """Planner 每轮唯一允许输出的应用控制决策。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    action: Literal["search", "clarify", "propose", "refuse"]
    search_query: str | None = Field(default=None, min_length=1, max_length=4000)
    clarification_questions: list[str] | None = Field(default=None, min_length=1, max_length=10)
    proposal: PlanningDraft | None = None
    refusal_reason: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_proposal_action(cls, data: Any) -> Any:
        """仅回填语义已由外层 action 决定的安全草案动作，不读取模型自由文本。"""
        if not isinstance(data, dict) or data.get("action") != "propose":
            return data
        proposal = data.get("proposal")
        if isinstance(proposal, dict):
            effective_action = proposal.get("action")
            effective_status = proposal.get("status")
            legacy_status_actions = {
                "draft", "planned", "clarification", "refused", "refuse",
            }
            missing_action = effective_action in (None, "")
            missing_or_unknown_status = (
                effective_status in (None, "")
                or effective_status not in legacy_status_actions
            )
            if missing_action and missing_or_unknown_status:
                data = dict(data)
                data["proposal"] = {**proposal, "action": "propose"}
                return data
        if (
            isinstance(proposal, PlanningDraft)
            and proposal.action in (None, "")
            and proposal.status in (None, "")
        ):
            data = dict(data)
            data["proposal"] = proposal.model_copy(update={"action": "propose"})
            return data
        return data

    @model_validator(mode="after")
    def action_has_exactly_one_payload(self) -> "PlannerDecision":
        payloads = {
            "search": self.search_query,
            "clarify": self.clarification_questions,
            "propose": self.proposal,
            "refuse": self.refusal_reason,
        }
        if payloads[self.action] is None or sum(value is not None for value in payloads.values()) != 1:
            raise ValueError("action_requires_exactly_one_payload")
        if self.action == "propose" and (self.proposal is None or self.proposal.action != "propose"):
            raise ValueError("proposal_requires_propose_action")
        return self


class PlannerDecisionModel(Protocol):
    def invoke(self, messages: list) -> Any: ...


class DeepSeekPlannerDecisionModel:
    """Planner 专用的严格结构化 DeepSeek 调用，不绑定工具。"""

    def __init__(self, settings, *, max_retries: int | None = None) -> None:
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
        self._model = ChatDeepSeek(**options).with_structured_output(PlannerDecision, include_raw=True)

    def invoke(self, messages: list) -> PlannerDecision:
        try:
            result = self._model.invoke(messages)
        except Exception as exc:
            # 非本适配器抛出的异常没有可判定的结构化输出。仅保留可可靠
            # 识别的传输错误码，未知异常不得误当可修复草稿而重复计费。
            raise PlanningError(
                _provider_failure_code(exc),
                usage_metadata=getattr(exc, "usage_metadata", None),
                response_metadata=getattr(exc, "response_metadata", None),
            ) from exc
        if not isinstance(result, dict):
            raise PlanningError("adapter_protocol_error")
        raw = result.get("raw")
        parsing_error = result.get("parsing_error")
        if parsing_error is not None or result.get("parsed") is None:
            raise PlanningError(
                classify_model_failure(parsing_error, raw_response=raw),
                usage_metadata=getattr(raw, "usage_metadata", None),
                response_metadata=getattr(raw, "response_metadata", None),
            )
        validation_failure: PlanningError | None = None
        try:
            decision = PlannerDecision.model_validate(result["parsed"])
        except (ValidationError, ValueError) as exc:
            validation_failure = PlanningError(
                classify_model_failure(exc, raw_response=raw),
                usage_metadata=getattr(raw, "usage_metadata", None),
                response_metadata=getattr(raw, "response_metadata", None),
            )
        if validation_failure is not None:
            # 离开 except 后再抛出，避免把含模型字段的 ValidationError 挂到异常上下文。
            raise validation_failure
        usage = getattr(raw, "usage_metadata", None)
        if isinstance(usage, dict):
            object.__setattr__(decision, "usage_metadata", dict(usage))
        metadata = getattr(raw, "response_metadata", None)
        token_usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
        if isinstance(token_usage, dict):
            object.__setattr__(decision, "response_metadata", {"token_usage": dict(token_usage)})
        return decision


class BoundPlanningTools:
    """唯一工具绑定角色和 Retriever，模型不能选择 ACL 或底层数据源。"""
    def __init__(
        self, retriever: Retriever, role: Role, max_tool_calls: int,
        event: Callable[[str, str, int], None] | None = None, *, initial_tool_calls: int = 0,
    ) -> None:
        self._retriever, self._role, self._max_tool_calls = retriever, role, max_tool_calls
        self._tool_calls = initial_tool_calls
        self.candidates: dict[str, RetrievalCandidate] = {}
        self._event = event
        self.last_result_count = 0
        self.last_result_ids: set[str] = set()

    def search_authorized_evidence(self, query: str) -> str:
        if (
            self._tool_calls >= self._max_tool_calls
            or not isinstance(query, str) or not query.strip() or len(query) > 4000
        ):
            raise PlanningError("invalid_tool_call")
        self._tool_calls += 1
        try:
            results = self._retriever.search(query, self._role)
        except Exception as exc:
            raise PlanningError("retrieval_failed") from exc
        self.last_result_count = len(results.candidates)
        self.last_result_ids = {item.chunk_id for item in results.candidates}
        for item in results.candidates:
            self.candidates[item.chunk_id] = item
        if self._event:
            self._event("search", "success", self._tool_calls)
        return json.dumps({"evidence": [{"chunk_id": item.chunk_id, "source_locator": item.source_locator, "text": item.text} for item in results.candidates]}, ensure_ascii=False)


class GroundedPlanComposer:
    """把受证据约束的模型意图编排成 Todo；不修复伪造引用或无依据动作。"""

    def __init__(self, query_policy: QueryPolicy) -> None:
        self.query_policy = query_policy

    @staticmethod
    def _validate_legacy_dependencies(todos: list[DraftTodo]) -> None:
        ids = [item.todo_id for item in todos]
        if len(ids) != len(set(ids)):
            raise PlanningError("duplicate_todo_id")
        known = set(ids)
        graph = {item.todo_id: set(item.depends_on) for item in todos}
        if any(not deps.issubset(known) or todo_id in deps for todo_id, deps in graph.items()):
            raise PlanningError("invalid_dependencies")
        visiting: set[str] = set()
        complete: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise PlanningError("cyclic_dependencies")
            if node in complete:
                return
            visiting.add(node)
            for dependency in graph[node]:
                visit(dependency)
            visiting.remove(node)
            complete.add(node)

        for todo_id in graph:
            visit(todo_id)

    @staticmethod
    def _detail(goal: str, answers: list[str], evidence: str) -> str:
        parts: list[str] = []
        if answers:
            answer_text = "；".join(answers)
            parts.append(f"用户已确认的业务约束（非制度证据）：{answer_text[:800]}。")
        parts.append(f"任务范围（非制度证据）：{goal[:1000]}。")
        prefix = "".join(parts)
        evidence_budget = max(200, 4000 - len(prefix) - len("授权制度依据："))
        return f"{prefix}授权制度依据：{evidence[:evidence_budget]}"[:4000]

    def _build_draft(
        self, selected_chunk_ids: list[str], tools: BoundPlanningTools,
        goal: str, answers: list[str],
    ) -> PlanningDraft:
        topics = self.query_policy.router.topics(goal)
        topic = (topics[0] if topics else "授权制度")[:80]
        todos: list[DraftTodo] = []
        review_ids: list[str] = []
        implementation_ids: list[str] = []
        for index, chunk_id in enumerate(sorted(selected_chunk_ids), start=1):
            candidate = tools.candidates[chunk_id]
            evidence = candidate.text
            positive_actions = _positive_composable_action_terms(evidence)
            action_groups = set(positive_actions)
            if not action_groups:
                raise PlanningError("unsupported_todo_content")
            required_terms = set(_required_evidence_action_terms(evidence))
            supports_review = bool(action_groups & {0})
            supports_implementation = bool(action_groups & {1, 2})
            detail = self._detail(goal, answers, evidence)
            if supports_review:
                todo_id = f"grounded-review-{index}"
                title = (
                    "核对员工适用资格与年假余额"
                    if {"核对", "适用资格", "年假余额"}.issubset(required_terms)
                    else f"核对{topic}授权要求"
                )
                review_ids.append(todo_id)
                todos.append(DraftTodo(
                    todo_id=todo_id, title=title, detail=detail, chunk_ids=[chunk_id],
                ))
            if supports_implementation:
                todo_id = f"grounded-implement-{index}"
                title = (
                    "配置审批流程并通知员工"
                    if (
                        {1, 2}.issubset(action_groups)
                        and "审批流程" in evidence
                        and "员工" in evidence
                    )
                    else f"{'并'.join(positive_actions[group] for group in sorted(action_groups - {0}))}{topic}要求"
                )
                implementation_ids.append(todo_id)
                todos.append(DraftTodo(
                    todo_id=todo_id, title=title, detail=detail, chunk_ids=[chunk_id],
                ))
            generic_groups = (
                set()
                if supports_implementation
                else action_groups - ({0} if supports_review else set())
            )
            if generic_groups:
                action_terms = [positive_actions[group] for group in sorted(generic_groups)]
                todos.append(DraftTodo(
                    todo_id=f"grounded-action-{index}",
                    title=f"{'并'.join(action_terms)}{topic}要求",
                    detail=detail,
                    chunk_ids=[chunk_id],
                ))
        if review_ids and implementation_ids:
            todos = [
                item.model_copy(update={"depends_on": list(review_ids)})
                if item.todo_id in implementation_ids else item
                for item in todos
            ]
        return PlanningDraft(
            action="propose", todos=todos, used_chunk_ids=sorted(selected_chunk_ids),
        )

    def compose(
        self, draft: PlanningDraft, tools: BoundPlanningTools,
        *, goal: str, answers: list[str],
    ) -> PlanningDraft:
        """只在模型选择真实授权证据且高层动作受其支持时进行确定性编排。"""
        if _explicit_missing_business_variables(goal) and not answers:
            raise PlanningError("invalid_model_output")
        selected = draft.used_chunk_ids
        if (
            not tools.candidates
            or not selected
            or len(selected) != len(set(selected))
            or not set(selected).issubset(tools.candidates)
        ):
            raise PlanningError("invalid_citations")
        if draft.action_intents and draft.todos:
            raise PlanningError("invalid_model_output")
        if draft.action_intents:
            positive_intents = [
                item for item in draft.action_intents
                if _positive_action_terms(item.description)
            ]
            has_any_action = any(
                _action_group_indexes(item.description)
                for item in draft.action_intents
            )
            if positive_intents:
                actionable_intents = positive_intents
            elif has_any_action:
                raise PlanningError("unsupported_todo_content")
            else:
                return self._build_draft(selected, tools, goal, answers)
            intent_ids = [
                item.intent_id or f"model-intent-{index}"
                for index, item in enumerate(actionable_intents, start=1)
            ]
            if len(intent_ids) != len(set(intent_ids)):
                raise PlanningError("duplicate_todo_id")
            intents = [
                (item.description, item.chunk_ids) for item in actionable_intents
            ]
        elif draft.todos:
            self._validate_legacy_dependencies(draft.todos)
            intents = [
                (_todo_instruction_text(item.title, item.detail), item.chunk_ids)
                for item in draft.todos
            ]
        else:
            raise PlanningError("invalid_citations")
        referenced_chunks = {
            chunk_id
            for item in draft.action_intents
            for chunk_id in item.chunk_ids
        } if draft.action_intents else {
            chunk_id
            for _description, chunk_ids in intents
            for chunk_id in chunk_ids
        }
        covered_actions: dict[str, set[int]] = {
            chunk_id: set() for chunk_id in selected
        }
        for description, chunk_ids in intents:
            if (
                len(chunk_ids) != len(set(chunk_ids))
                or not set(chunk_ids).issubset(selected)
                or not set(chunk_ids).issubset(tools.candidates)
            ):
                raise PlanningError("invalid_citations")
            evidence_texts = [tools.candidates[chunk_id].text for chunk_id in chunk_ids]
            if not _high_level_action_is_grounded(description, evidence_texts):
                raise PlanningError("unsupported_todo_content")
            intent_actions = set(_positive_composable_action_terms(description))
            mapped_actions: set[int] = set()
            for chunk_id in chunk_ids:
                chunk_actions = set(_positive_composable_action_terms(
                    tools.candidates[chunk_id].text,
                ))
                local_actions = intent_actions & chunk_actions
                if not local_actions:
                    raise PlanningError("unsupported_todo_content")
                covered_actions[chunk_id].update(local_actions)
                mapped_actions.update(local_actions)
            if mapped_actions != intent_actions:
                raise PlanningError("unsupported_todo_content")
        excluded_actions = {
            group
            for source in [goal, *answers]
            for group in _restricted_action_group_indexes(source)
        }
        selected_actions = {
            group
            for chunk_id in selected
            for group in _positive_composable_action_terms(tools.candidates[chunk_id].text)
        }
        if excluded_actions & selected_actions:
            raise PlanningError("unsupported_todo_content")
        for chunk_id in selected:
            required_actions = set(_positive_composable_action_terms(
                tools.candidates[chunk_id].text,
            ))
            if not required_actions:
                raise PlanningError("unsupported_todo_content")
            if not covered_actions[chunk_id] and chunk_id in referenced_chunks:
                continue
            if not covered_actions[chunk_id]:
                raise PlanningError("incomplete_evidence_actions")
        return self._build_draft(selected, tools, goal, answers)


class PolicyRolloutPlanningService:
    # ponytail: v1 is request-synchronous/read-only; durable execution is deferred.
    def __init__(
        self, store: PolicyRolloutTaskStore, retriever: Retriever, model: PlannerDecisionModel | None,
        *, citation_checker: Callable[[list[TodoCitation], Role], bool] | None = None,
        max_tool_calls: int = 3, max_model_decisions: int = 6,
        max_repairs: int = 2, claim_ttl_seconds: int = 300,
        now: Callable[[], datetime] | None = None,
        query_policy: QueryPolicy | None = None,
        preflight_named_policy_search: bool = False,
    ) -> None:
        self.store, self.retriever, self.model = store, retriever, model
        self.citation_checker = citation_checker or (lambda citations, _role: not citations)
        self.max_tool_calls = max_tool_calls
        self.max_model_decisions, self.max_repairs = max_model_decisions, max_repairs
        self.claim_ttl_seconds, self.now = claim_ttl_seconds, now or (lambda: datetime.now(UTC))
        self.query_policy = query_policy or QueryPolicy()
        self.plan_composer = GroundedPlanComposer(self.query_policy)
        self.preflight_named_policy_search = preflight_named_policy_search

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
        current = self.store.get(task_id, role, observer.run_id)
        if current is None:
            raise PlanningError("task_persistence_failed")
        progress = self.store.planning_progress(task_id, role, claim_token)
        if progress is None:
            raise PlanningError("planning_claim_lost")
        decisions = int(progress["model_decisions"])
        repairs = int(progress["repairs"])
        attempted_searches = int(progress["attempted_searches"])
        executed_searches = int(progress["executed_searches"])
        deduplicated_searches = int(progress["deduplicated_searches"])
        rejected_searches = int(progress["rejected_searches"])
        seen_query_hashes = set(progress["seen_query_hashes"])
        seen_query_structures = set(progress["seen_query_structures"])
        seen_chunk_hashes = set(progress["seen_chunk_hashes"])
        query_rewrites = int(progress["query_rewrites"])
        tools = BoundPlanningTools(
            self.retriever, role, self.max_tool_calls, initial_tool_calls=executed_searches,
        )

        def save_progress() -> None:
            if not self.store.save_planning_progress(
                task_id, role, claim_token, model_decisions=decisions, repairs=repairs,
                attempted_searches=attempted_searches, executed_searches=executed_searches,
                deduplicated_searches=deduplicated_searches,
                rejected_searches=rejected_searches, seen_query_hashes=seen_query_hashes,
                seen_query_structures=seen_query_structures,
                seen_chunk_hashes=seen_chunk_hashes, query_rewrites=query_rewrites,
            ):
                raise PlanningError("planning_claim_lost")

        def trace(action: str, outcome: str, *, code: str | None = None) -> None:
            if not self.store.append_trace_event(
                task_id, role, claim_token, action=action, outcome=outcome,
                model_decisions=decisions, search_calls=executed_searches, repairs=repairs,
                constraint_code=code, duration_ms=round((time.perf_counter() - started) * 1000, 3),
                attempted_searches=attempted_searches, executed_searches=executed_searches,
                deduplicated_searches=deduplicated_searches, rejected_searches=rejected_searches,
            ):
                raise PlanningError("planning_claim_lost")

        def terminal_event(action: str, outcome: str, code: str | None) -> dict:
            return {
                "action": action, "outcome": outcome, "model_decisions": decisions,
                "search_calls": executed_searches, "attempted_searches": attempted_searches,
                "executed_searches": executed_searches,
                "deduplicated_searches": deduplicated_searches,
                "rejected_searches": rejected_searches, "repairs": repairs,
                "constraint_code": code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            }

        try:
            constraint_code = None
            repair_feedback = None
            repair_constraint_history: set[str] = set()
            phase_queries: set[str] = set()
            phase_executed_searches = 0
            phase_business_query_rejections = 0
            phase_out_of_scope_rejections = 0
            phase_unjustified_clarifications = 0
            phase_terminal_refusal_reconsiderations = 0
            terminal_only = False
            search_feedback: str | None = None
            explicit_missing_variables = _explicit_missing_business_variables(goal)
            business_terms = _business_input_terms(explicit_missing_variables, answers)
            if (
                explicit_missing_variables
                and not answers
                and claimed_clarification is None
                and decisions == 0
                and executed_searches == 0
            ):
                constraint_code = "explicit_missing_business_variable"
                clarification = PolicyRolloutClarification(
                    questions=[f"请确认{variable}。" for variable in explicit_missing_variables],
                    answers=[],
                )
                event = terminal_event("clarify", "clarification", constraint_code)
                if not self.store.update_state(
                    task_id, role, PlanStatus.CLARIFICATION, clarification,
                    claim_token=claim_token, trace_event=event,
                ):
                    raise PlanningError("planning_claim_lost")
                observer.emit(
                    "policy_rollout_planning", "completed", status="success",
                    counts={"candidates": 0}, started_at=started,
                )
                return self._stored(task_id, role, observer.run_id)
            # 对明确点名的制度主题，先由服务端完成一次受权检索。这样 ACL 空集会
            # 直接拒绝，不会先把权限或证据问题错误升级为要求用户补充信息。
            preflight_queries = self.query_policy.preflight_queries(goal)
            if (
                self.preflight_named_policy_search
                and preflight_queries
                and not answers
                and claimed_clarification is None
                and decisions == 0
                and executed_searches == 0
            ):
                preflight = self.query_policy.classify(
                    preflight_queries[0], goal,
                    business_terms=business_terms,
                    missing_variables=explicit_missing_variables,
                    answers=answers,
                )
                if preflight.intent != QueryIntent.EVIDENCE_QUERY or preflight.query is None:
                    raise PlanningError("invalid_tool_call")
                assert preflight.structure_fingerprint is not None
                query_key = preflight.query.casefold()
                seen_query_hashes.add(hashlib.sha256(query_key.encode("utf-8")).hexdigest())
                seen_query_structures.add(preflight.structure_fingerprint)
                phase_queries.add(query_key)
                attempted_searches += 1
                save_progress()
                try:
                    tools.search_authorized_evidence(preflight.query)
                finally:
                    executed_searches = tools._tool_calls
                    save_progress()
                phase_executed_searches += 1
                returned_hashes = {
                    hash_identifier(chunk_id) for chunk_id in tools.last_result_ids
                }
                if tools.last_result_count == 0:
                    trace("search", "empty")
                    constraint_code = "authorized_evidence_empty"
                    event = terminal_event("refuse", "refused", constraint_code)
                    if not self.store.update_state(
                        task_id, role, PlanStatus.REFUSED, current.clarification,
                        claim_token=claim_token, trace_event=event,
                    ):
                        raise PlanningError("planning_claim_lost")
                    observer.emit(
                        "policy_rollout_planning", "completed", status="refusal",
                        error_code="planning_refused", started_at=started,
                    )
                    return self._stored(task_id, role, observer.run_id)
                seen_chunk_hashes.update(returned_hashes)
                save_progress()
                trace("search", "success")
                search_feedback = "preflight_authorized_evidence"
                # 对目标已明确要求的实施维度，最多补一次服务端受控检索。它只
                # 依赖目标中的显式 facet，不使用模型分类，不改变 ACL 或预算。
                for query_seed in preflight_queries[1:]:
                    if executed_searches >= self.max_tool_calls:
                        break
                    facet_query = self.query_policy.classify(
                        query_seed, goal, business_terms=business_terms,
                        missing_variables=explicit_missing_variables, answers=answers,
                    )
                    if facet_query.intent != QueryIntent.EVIDENCE_QUERY or facet_query.query is None:
                        continue
                    assert facet_query.structure_fingerprint is not None
                    query_key = facet_query.query.casefold()
                    query_hash = hashlib.sha256(query_key.encode("utf-8")).hexdigest()
                    if query_hash in seen_query_hashes:
                        continue
                    seen_query_hashes.add(query_hash)
                    seen_query_structures.add(facet_query.structure_fingerprint)
                    phase_queries.add(query_key)
                    attempted_searches += 1
                    save_progress()
                    try:
                        tools.search_authorized_evidence(facet_query.query)
                    finally:
                        executed_searches = tools._tool_calls
                        save_progress()
                    phase_executed_searches += 1
                    returned_hashes = {
                        hash_identifier(chunk_id) for chunk_id in tools.last_result_ids
                    }
                    new_chunk_hashes = returned_hashes - seen_chunk_hashes
                    if not tools.last_result_count:
                        # 首次授权检索为空才是拒绝条件；补检索为空不推翻已有证据。
                        trace("search", "empty")
                    elif new_chunk_hashes:
                        seen_chunk_hashes.update(new_chunk_hashes)
                        save_progress()
                        trace("search", "success")
                    else:
                        trace("search", "no_new_evidence", code="no_new_evidence")
            if preflight_queries and tools.candidates:
                # 服务端已完成受权预检索。此时已有证据集合，模型应直接 propose 或
                # refuse，不应再消耗预算做无新证据搜索。
                terminal_only = True
                search_feedback = "preflight_authorized_evidence_terminal_only"
            while True:
                emergency_fallback_decision = False
                if decisions >= self.max_model_decisions:
                    raise PlanningError("max_model_decisions_exceeded")
                decisions += 1
                save_progress()
                try:
                    decision = self._run_decision(
                        tools, goal, answers,
                        claimed_clarification.questions if claimed_clarification else [],
                        repair_feedback, sorted(phase_queries), search_feedback,
                    )
                except PlanningError as exc:
                    if str(exc) not in REPAIRABLE_MODEL_CODES or repairs >= self.max_repairs:
                        if (
                            tools.candidates
                            and repairs > 0
                            and repairs >= self.max_repairs
                            and repair_constraint_history
                            and repair_constraint_history.issubset(SAFE_GROUNDED_FALLBACK_CODES)
                        ):
                            constraint_code = "grounded_fallback_after_model_failure"
                            trace("propose", "grounded_fallback", code=constraint_code)
                            emergency_fallback_decision = True
                            decision = PlannerDecision(
                                action="propose",
                                proposal=self._grounded_fallback_draft(tools, goal, answers),
                            )
                        else:
                            raise
                    else:
                        repairs += 1
                        save_progress()
                        constraint_code = str(exc)
                        repair_constraint_history.add(constraint_code)
                        trace("repair", "requested", code=constraint_code)
                        repair_feedback = {"prior_draft": None, "constraint_codes": [constraint_code]}
                        continue
                if decision.action == "search":
                    assert decision.search_query is not None
                    attempted_searches += 1
                    policy_decision = self.query_policy.classify(
                        decision.search_query, goal, business_terms=business_terms,
                        missing_variables=explicit_missing_variables, answers=answers,
                    )
                    if policy_decision.intent == QueryIntent.INVALID_QUERY:
                        rejected_searches += 1
                        save_progress()
                        trace("search", "rejected", code="invalid_tool_call")
                        raise PlanningError("invalid_tool_call")
                    if policy_decision.intent == QueryIntent.MISSING_VARIABLE:
                        constraint_code = "explicit_missing_business_variable"
                        decision = PlannerDecision(
                            action="clarify",
                            clarification_questions=[f"请确认{value}。" for value in explicit_missing_variables],
                        )
                    elif policy_decision.intent == QueryIntent.BUSINESS_INPUT:
                        rejected_searches += 1
                        phase_business_query_rejections += 1
                        save_progress()
                        constraint_code = "business_input_is_not_evidence_query"
                        trace("search", "rejected", code=constraint_code)
                        if phase_business_query_rejections >= 2 or terminal_only:
                            decision = PlannerDecision(action="refuse", refusal_reason=constraint_code)
                        else:
                            search_feedback = constraint_code
                            continue
                    elif policy_decision.intent == QueryIntent.OUT_OF_SCOPE:
                        rejected_searches += 1
                        phase_out_of_scope_rejections += 1
                        save_progress()
                        constraint_code = "query_outside_goal_scope"
                        trace("search", "rejected", code=constraint_code)
                        if tools.candidates:
                            search_feedback = "query_outside_goal_scope_terminal_only"
                            terminal_only = True
                            continue
                        if phase_out_of_scope_rejections >= 2:
                            decision = PlannerDecision(action="refuse", refusal_reason=constraint_code)
                        else:
                            search_feedback = constraint_code
                            continue
                    else:
                        assert policy_decision.query is not None
                        assert policy_decision.structure_fingerprint is not None
                        query = policy_decision.query
                        query_key = query.casefold()
                        query_hash = hashlib.sha256(query_key.encode("utf-8")).hexdigest()
                        query_structure = policy_decision.structure_fingerprint
                        was_seen = query_hash in seen_query_hashes
                        structure_seen = query_structure in seen_query_structures
                        is_rehydration = bool(
                            answers and phase_executed_searches == 0
                            and (was_seen or structure_seen)
                        )
                        if terminal_only:
                            rejected_searches += 1
                            save_progress()
                            constraint_code = "search_after_terminal_evidence_state"
                            trace("search", "rejected", code=constraint_code)
                            decision = PlannerDecision(action="refuse", refusal_reason=constraint_code)
                        elif answers and phase_executed_searches >= 1:
                            rejected_searches += 1
                            save_progress()
                            constraint_code = "search_after_clarification_answers"
                            trace("search", "rejected", code=constraint_code)
                            search_feedback = "search_after_clarification_answers_terminal_only"
                            terminal_only = True
                            continue
                        elif was_seen and not is_rehydration:
                            deduplicated_searches += 1
                            save_progress()
                            trace("search", "deduplicated", code="duplicate_search_query")
                            search_feedback = (
                                "duplicate_search_query"
                                if repair_feedback else "duplicate_search_query_terminal_only"
                            )
                            terminal_only = repair_feedback is None
                            continue
                        elif structure_seen and not is_rehydration:
                            deduplicated_searches += 1
                            save_progress()
                            trace("search", "deduplicated", code="semantically_duplicate_query")
                            search_feedback = (
                                "semantically_duplicate_query"
                                if repair_feedback else "semantically_duplicate_query_terminal_only"
                            )
                            terminal_only = repair_feedback is None
                            continue
                        elif seen_query_structures and query_rewrites >= 1 and not is_rehydration:
                            rejected_searches += 1
                            save_progress()
                            constraint_code = "query_rewrite_limit_exceeded"
                            trace("search", "rejected", code=constraint_code)
                            decision = PlannerDecision(action="refuse", refusal_reason=constraint_code)
                        elif executed_searches >= self.max_tool_calls:
                            rejected_searches += 1
                            save_progress()
                            trace("search", "rejected", code="max_searches_exceeded")
                            raise PlanningError("max_searches_exceeded")
                        else:
                            if seen_query_structures and not is_rehydration:
                                query_rewrites += 1
                            seen_query_hashes.add(query_hash)
                            seen_query_structures.add(query_structure)
                            phase_queries.add(query_key)
                            save_progress()
                            search_feedback = None
                            try:
                                tools.search_authorized_evidence(query)
                            finally:
                                executed_searches = tools._tool_calls
                                save_progress()
                            phase_executed_searches += 1
                            returned_hashes = {
                                hash_identifier(chunk_id) for chunk_id in tools.last_result_ids
                            }
                            new_chunk_hashes = returned_hashes - seen_chunk_hashes
                            if tools.last_result_count == 0:
                                trace("search", "empty")
                                constraint_code = "authorized_evidence_empty"
                                decision = PlannerDecision(action="refuse", refusal_reason=constraint_code)
                            elif is_rehydration:
                                trace("search", "rehydrated")
                                continue
                            elif not new_chunk_hashes:
                                constraint_code = "no_new_evidence"
                                trace("search", "no_new_evidence", code=constraint_code)
                                search_feedback = "no_new_evidence_terminal_only"
                                terminal_only = True
                                continue
                            else:
                                seen_chunk_hashes.update(new_chunk_hashes)
                                save_progress()
                                trace("search", "success")
                                if query_rewrites >= 1:
                                    search_feedback = "query_rewrite_limit_reached_terminal_only"
                                    terminal_only = True
                                continue
                draft = self._terminal_draft(decision)
                if draft.action == "propose":
                    try:
                        if emergency_fallback_decision:
                            plan = self._validated_plan(
                                draft, tools, goal=goal, answers=answers,
                            )
                        else:
                            draft = self.plan_composer.compose(
                                draft, tools, goal=goal, answers=answers,
                            )
                            plan = self._validated_plan(
                                draft, tools, goal=goal, answers=answers,
                            )
                            constraint_code = GROUNDED_COMPOSER_CODE
                            trace(
                                "propose", GROUNDED_COMPOSER_OUTCOME,
                                code=constraint_code,
                            )
                    except PlanningError as exc:
                        trace("propose", str(exc), code=str(exc))
                        if str(exc) not in STRUCTURAL_CODES or repairs >= self.max_repairs:
                            if (
                                tools.candidates
                                and repairs > 0
                                and repairs >= self.max_repairs
                                and str(exc) in {
                                    "unsupported_todo_content", "incomplete_evidence_actions",
                                }
                            ):
                                constraint_code = "grounded_fallback_after_unsupported_todo"
                                draft = self._grounded_fallback_draft(tools, goal, answers)
                                trace("propose", "grounded_fallback", code=constraint_code)
                                plan = self._validated_plan(draft, tools, goal=goal, answers=answers)
                            else:
                                raise
                        else:
                            repairs += 1
                            save_progress()
                            constraint_code = str(exc)
                            repair_constraint_history.add(constraint_code)
                            trace("repair", "requested", code=constraint_code)
                            repair_feedback = {"prior_draft": draft.model_dump(mode="json"), "constraint_codes": [constraint_code]}
                            continue
                    event = terminal_event("propose", "planned", constraint_code)
                    if self.store.complete_claim_and_append_plan(task_id, role, plan, current.clarification, claim_token, trace_event=event) is None:
                        raise PlanningError("planning_claim_lost")
                    action, outcome = "propose", "planned"
                    break
                if draft.action == "clarify":
                    if draft.todos or not draft.clarification_questions:
                        raise PlanningError("invalid_model_output")
                    if answers:
                        constraint_code = "clarification_after_answers"
                        trace("clarify", "rejected", code=constraint_code)
                        event = terminal_event("refuse", "refused", constraint_code)
                        if not self.store.update_state(
                            task_id, role, PlanStatus.REFUSED, claimed_clarification,
                            claim_token=claim_token, trace_event=event,
                        ):
                            raise PlanningError("planning_claim_lost")
                        action, outcome = "refuse", "refused"
                        break
                    if not explicit_missing_variables:
                        phase_unjustified_clarifications += 1
                        constraint_code = "clarification_not_justified"
                        trace("clarify", "rejected", code=constraint_code)
                        if phase_unjustified_clarifications >= 2:
                            decision = PlannerDecision(action="refuse", refusal_reason=constraint_code)
                            draft = self._terminal_draft(decision)
                        else:
                            search_feedback = "clarification_not_justified"
                            continue
                    if draft.action == "refuse":
                        event = terminal_event("refuse", "refused", constraint_code)
                        if not self.store.update_state(
                            task_id, role, PlanStatus.REFUSED, current.clarification,
                            claim_token=claim_token, trace_event=event,
                        ):
                            raise PlanningError("planning_claim_lost")
                        action, outcome = "refuse", "refused"
                        break
                    clarification = PolicyRolloutClarification(questions=draft.clarification_questions, answers=answers)
                    event = terminal_event("clarify", "clarification", constraint_code)
                    if not self.store.update_state(task_id, role, PlanStatus.CLARIFICATION, clarification, claim_token=claim_token, trace_event=event):
                        raise PlanningError("planning_claim_lost")
                    action, outcome = "clarify", "clarification"
                    break
                if draft.action == "refuse":
                    if draft.todos or draft.clarification_questions:
                        raise PlanningError("invalid_model_output")
                    if (
                        tools.candidates
                        and search_feedback in {
                            "no_new_evidence_terminal_only",
                            "search_after_clarification_answers_terminal_only",
                            "query_outside_goal_scope_terminal_only",
                        }
                        and phase_terminal_refusal_reconsiderations == 0
                    ):
                        phase_terminal_refusal_reconsiderations += 1
                        constraint_code = "refusal_reconsideration_required"
                        trace("refuse", "rejected", code=constraint_code)
                        search_feedback = "refusal_reconsideration_required_terminal_only"
                        terminal_only = True
                        continue
                    event = terminal_event("refuse", "refused", constraint_code)
                    if not self.store.update_state(task_id, role, PlanStatus.REFUSED, current.clarification, claim_token=claim_token, trace_event=event):
                        raise PlanningError("planning_claim_lost")
                    action, outcome = "refuse", "refused"
                    break
                raise PlanningError("invalid_model_output")
        except Exception as exc:
            logger.error("策略落地规划失败(error_code=policy_rollout_planning_failed)")
            if claimed_clarification is None:
                code = str(exc) if str(exc) in STRUCTURAL_CODES | MODEL_FAILURE_CODES | {"invalid_tool_call", "max_searches_exceeded", "max_model_decisions_exceeded", "planning_claim_lost", "agent_runtime_unavailable", "retrieval_failed"} else "planning_failed"
                self.store.update_state(
                    task_id, role, PlanStatus.FAILED, None, claim_token=claim_token,
                    trace_event=terminal_event("failed", "failed", code),
                )
            else:
                try:
                    trace("failed", "continuation_failed", code=str(exc))
                except PlanningError:
                    pass
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

    def _run_decision(
        self, tools: BoundPlanningTools, goal: str, answers: list[str], questions: list[str],
        repair: dict | None, prior_queries: list[str], search_feedback: str | None,
    ) -> PlannerDecision:
        if self.model is None:
            raise PlanningError("agent_runtime_unavailable")
        messages_module = importlib.import_module("langchain_core.messages")
        HumanMessage, SystemMessage = messages_module.HumanMessage, messages_module.SystemMessage
        try:
            request = {
                "goal": goal,
                "clarification_answers": answers,
                "clarification_context": [
                    {"question": question, "answer": answers[index] if index < len(answers) else ""}
                    for index, question in enumerate(questions)
                ],
                "evidence": [
                    {"chunk_id": item.chunk_id, "source_locator": item.source_locator, "text": item.text}
                    for item in tools.candidates.values()
                ],
                "prior_search_queries": prior_queries,
            }
            if repair:
                request["repair"] = repair
            if search_feedback:
                request["search_feedback"] = search_feedback
            result = self.model.invoke([
                SystemMessage(content=(
                    "你是受限的策略落地规划决策器。严格按优先级选择："
                    "搜索后若明确无授权证据或明确无权访问，必须 refuse；"
                    "只有缺失一个会使所有安全可执行计划都无法成立的关键业务变量时才能 clarify；"
                    "不得因一般性不确定、偏好信息、证据不足或权限不足而 clarify。"
                    "若目标已明确写明某个关键业务变量尚未确定并要求先确认，必须在搜索前 clarify，"
                    "不得把该缺失变量本身改写成制度检索查询。"
                    "clarification_answers 非空时答案是权威约束，必须用于 Todo 的内容、范围、依赖或最终行动，"
                    "不得再次 clarify；至多允许一次用于恢复证据的授权检索，之后必须 propose 或 refuse。"
                    "当前证据足够形成计划时直接 propose；无授权证据或无权访问时 refuse；否则 search 新查询。"
                    "充分、无授权或无权证据出现后停止搜索。不得重复 prior_search_queries。"
                    "查询改写最多一次，且必须改变版本、日期、条款、证据类型等结构化约束；纯近义改写禁止。"
                    "search_feedback 包含 terminal_only 时，下一步只能 propose 或 refuse，不得 search 或 clarify。"
                    "search_feedback 为 preflight_authorized_evidence_terminal_only 时，表示服务端已完成受权检索；"
                    "若现有证据足以支持 goal 的交付动作，必须直接 propose；确实不足时才 refuse。"
                    "若 search_feedback 为 refusal_reconsideration_required_terminal_only，表示已有授权证据且首次拒绝缺少充分依据；"
                    "必须重新核对 goal、clarification_answers 与 evidence，证据足够时 propose，确实不足时才 refuse。"
                    "search_feedback 为 clarification_not_justified 时，不得再次 clarify。"
                    "search_feedback 为 business_input_is_not_evidence_query 时，说明查询错误地包含了业务输入或澄清答案；"
                    "必须改查制度依据，不得查询用户刚提供的部门、人员、范围或其他业务值。"
                    "search_feedback 为 query_outside_goal_scope 时，说明查询违反了目标中明确排除的范围；"
                    "不得再查询该范围。goal 明确要求的交付动作可作为 Todo 行动依据，但制度事实仍须引用证据支持。"
                    "每次只能返回一个结构化决策：search 仅填写新的检索查询；"
                    "clarify 仅填写澄清问题列表；propose 仅填写可引用的完整 PlanningDraft；"
                    "refuse 仅填写拒绝原因。不得选择角色、ACL、数据源或执行 Todo；"
                    "propose 时优先填写 used_chunk_ids 与 action_intents，并将 todos 留空："
                    "每个 action_intent 只写一个由其 chunk_ids 直接支持的高层动作，不得加入证据中没有的动作或数字；"
                    "应用层会据此生成合法 Todo、依赖与引用。兼容旧格式时也可直接填写 todos，但不得同时填写两种格式。"
                    "goal 可支持交付动作与任务范围，clarification_answers 可支持用户确认的业务约束，"
                    "但二者都不能替代制度证据；每个 Todo 的制度事实及制度数字必须由该 Todo 自己引用的证据直接支持。"
                    "已有证据与修复反馈均由应用提供。"
                )),
                HumanMessage(content=json.dumps(request, ensure_ascii=False)),
            ])
            if isinstance(result, PlannerDecision):
                return result
            return PlannerDecision.model_validate(result) if isinstance(result, dict) else PlannerDecision.model_validate_json(result)
        except (ValidationError, ValueError) as exc:
            raise PlanningError(classify_model_failure(exc)) from exc
        except PlanningError:
            raise
        except Exception as exc:
            raise PlanningError(
                _provider_failure_code(exc),
                usage_metadata=getattr(exc, "usage_metadata", None),
                response_metadata=getattr(exc, "response_metadata", None),
            ) from exc

    @staticmethod
    def _terminal_draft(decision: PlannerDecision) -> PlanningDraft:
        if decision.action == "propose":
            assert decision.proposal is not None
            return decision.proposal
        if decision.action == "clarify":
            assert decision.clarification_questions is not None
            return PlanningDraft(action="clarify", clarification_questions=decision.clarification_questions)
        if decision.action == "refuse":
            return PlanningDraft(action="refuse")
        raise PlanningError("invalid_model_output")

    @staticmethod
    def _validated_plan(
        draft: PlanningDraft, tools: BoundPlanningTools, *, goal: str, answers: list[str],
    ) -> PolicyRolloutPlanVersion:
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
            if not _todo_is_supported(
                item.title, item.detail,
                [tools.candidates[chunk_id].text for chunk_id in item.chunk_ids],
                [goal, *answers],
            ):
                raise PlanningError("unsupported_todo_content")
            citations = [TodoCitation(document_id=c.document_id, document_version=c.document_version, source_file_name=c.source_file_name, source_locator=c.source_locator, chunk_id=c.chunk_id) for chunk_id in item.chunk_ids for c in [tools.candidates[chunk_id]]]
            todos.append(PolicyRolloutTodo(todo_id=item.todo_id, title=item.title, detail=item.detail, depends_on=item.depends_on, citations=citations))
        for chunk_id in draft.used_chunk_ids:
            candidate = tools.candidates[chunk_id]
            required_terms = _required_evidence_action_terms(candidate.text)
            if not required_terms:
                continue
            cited_content = " ".join(
                _todo_instruction_text(item.title, item.detail)
                for item in draft.todos if chunk_id in item.chunk_ids
            )
            normalized_content = re.sub(r"\s+", "", cited_content).casefold()
            if not all(
                re.sub(r"\s+", "", term).casefold() in normalized_content
                for term in required_terms
            ):
                raise PlanningError("incomplete_evidence_actions")
        return PolicyRolloutPlanVersion(version=0, todos=todos, created_at=datetime.now(UTC))

    def _grounded_fallback_draft(
        self, tools: BoundPlanningTools, goal: str, answers: list[str],
    ) -> PlanningDraft:
        """紧急计划也复用句级正向动作提取；条件或否定证据直接 fail-closed。"""
        return self.plan_composer._build_draft(
            sorted(tools.candidates), tools, goal, answers,
        )
