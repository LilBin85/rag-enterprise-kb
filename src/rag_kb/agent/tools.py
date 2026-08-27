"""LangChain 只读工具薄适配器；全部委托 ToolExecutionService，不直接接触数据层。"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime

from rag_kb.agent.context import AgentRunContext
from rag_kb.application.tool_execution_service import (
    _FATAL_TOOL_CODES,
    ToolExecutionError,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolExecutionService,
)
from rag_kb.domain.fingerprints import canonical_fingerprint

RESEARCH_TOOL_NAMES = (
    "search_knowledge",
    "fetch_evidence",
    "list_document_versions",
    "compare_document_versions",
)

# M2 批次 1 精确开放的两个 WRITE 工具；HITL interrupt_on 必须与之逐字一致。
CONTROLLED_WRITE_TOOL_NAMES = (
    "create_report_artifact",
    "create_work_item_draft",
)

_MAX_ITEMS = 20
_MAX_LIST_ITEMS = 50
_MAX_STRING_CHARS = 2_000
_CONTENT_FIELDS = {
    "search_knowledge": frozenset(
        {
            "document_id",
            "version_id",
            "document_version",
            "chunk_id",
            "content_hash",
            "source_locator",
            "source_file_name",
            "fused_score",
            "rerank_score",
            "text",
        }
    ),
    "fetch_evidence": frozenset(
        {
            "document_id",
            "document_version",
            "source_file_name",
            "source_locator",
            "chunk_id",
        }
    ),
    "list_document_versions": frozenset(
        {
            "version_id",
            "version",
            "lifecycle_status",
            "content_hash",
            "source_file_name",
        }
    ),
    "compare_document_versions": frozenset(
        {"document_id", "left_version", "right_version", "changes"}
    ),
}
_CHUNK_FIELDS = frozenset(
    {
        "version",
        "chunk_id",
        "content_hash",
        "source_locator",
        "source_file_name",
        "text",
        "evidence_id",
    }
)
_CHANGE_FIELDS = frozenset({"kind", "left", "right"})

# 瞬时类错误码白名单：只有表内错误允许模型重试；权限/参数/版本类永久失败不可重试。
_RETRYABLE_CODES = frozenset(
    {
        "tool_timeout",
        "tool_execution_unavailable",
        "previous_execution_in_progress",
        "tool_result_readback_failed",
        "execution_deadline_expired",
    }
)


def _trim_value(value: object) -> object:
    if isinstance(value, str):
        return value[:_MAX_STRING_CHARS]
    if isinstance(value, list):
        return [_trim_value(item) for item in value[:_MAX_LIST_ITEMS]]
    if isinstance(value, dict):
        return {key: _trim_value(item) for key, item in value.items()}
    return value


def trim_tool_content(tool_name: str, items: object) -> list[object]:
    """按工具 allowlist 裁剪业务内容；保持结构合法且大小有界。"""
    if not isinstance(items, list):
        return []
    fields = _CONTENT_FIELDS[tool_name]
    trimmed: list[object] = []
    for item in items[:_MAX_ITEMS]:
        if not isinstance(item, dict):
            continue
        safe = {key: _trim_value(item[key]) for key in fields if key in item}
        if tool_name == "compare_document_versions" and isinstance(safe.get("changes"), list):
            safe["changes"] = [
                {
                    key: (
                        [
                            {
                                field: _trim_value(chunk[field])
                                for field in _CHUNK_FIELDS
                                if field in chunk
                            }
                            for chunk in change[key][:_MAX_LIST_ITEMS]
                            if isinstance(chunk, dict)
                        ]
                        if key in {"left", "right"} and isinstance(change.get(key), list)
                        else _trim_value(change[key])
                    )
                    for key in _CHANGE_FIELDS
                    if key in change
                }
                for change in safe["changes"][:_MAX_LIST_ITEMS]
                if isinstance(change, dict)
            ]
        if safe:
            trimmed.append(safe)
    return trimmed


def render_tool_message(
    tool_name: str,
    result: ToolExecutionResult | None,
    *,
    error_code: str | None = None,
) -> str:
    """构造进入 ToolMessage 的受控 JSON；失败显式可见，不伪装成空成功。"""
    status = result.status if result is not None else "failed"
    code = error_code if result is None else result.error_code
    payload: dict[str, object] = {
        "tool_call_id": result.tool_call_id if result is not None else None,
        "tool_name": tool_name,
        "status": status,
        "error_code": code,
        "retryable": status == "failed" and code in _RETRYABLE_CODES,
    }
    if result is not None and result.result is not None:
        body = result.result
        payload["evidence_ids"] = list(body.get("evidence_ids") or [])
        payload["content"] = trim_tool_content(tool_name, body.get("items"))
        # 模型只看服务端调用 ID，不看对象存储路径。
        payload["result_ref"] = result.tool_call_id
        payload["result_hash"] = result.result_hash
    elif result is not None and result.result_hash:
        payload["result_ref"] = result.tool_call_id
        payload["result_hash"] = result.result_hash
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _execute_read_tool(
    tool_name: str,
    arguments: dict[str, object],
    runtime: ToolRuntime,
    executor: ToolExecutionService,
) -> str:
    context = runtime.context
    if not isinstance(context, AgentRunContext):
        return render_tool_message(
            tool_name, None, error_code="agent_runtime_context_invalid"
        )
    if not runtime.tool_call_id:
        return render_tool_message(tool_name, None, error_code="agent_tool_call_id_missing")
    request = ToolExecutionRequest(
        auth=context.auth,
        task_id=context.task_id,
        tool_name=tool_name,
        arguments=arguments,
        worker_id=context.worker_id,
        lease_generation=context.lease_generation,
        idempotency_key=canonical_fingerprint(
            "agent_tool_call",
            {
                "task_id": context.task_id,
                "tool_name": tool_name,
                "tool_call_id": runtime.tool_call_id,
            },
        ),
    )
    try:
        result = executor.execute(request)
    except ToolExecutionError as exc:
        if exc.code in _FATAL_TOOL_CODES:
            # 失租/身份失效：穿透 middleware 使 invoke 失败，禁止渲染 ToolMessage 续跑。
            raise
        return render_tool_message(tool_name, None, error_code=exc.code)
    except Exception as exc:
        code = getattr(exc, "code", None)
        return render_tool_message(
            tool_name,
            None,
            error_code=code if isinstance(code, str) and code else "tool_execution_unavailable",
        )
    return render_tool_message(tool_name, result)


def _execute_write_tool(
    tool_name: str,
    arguments: dict[str, object],
    runtime: ToolRuntime,
    executor: ToolExecutionService,
) -> str:
    context = runtime.context
    if not isinstance(context, AgentRunContext):
        return render_tool_message(
            tool_name, None, error_code="agent_runtime_context_invalid"
        )
    if not runtime.tool_call_id:
        return render_tool_message(tool_name, None, error_code="agent_tool_call_id_missing")
    # 控制字段只取服务端 AgentRunContext；批次 1 恒无审批事实，
    # 请求在授权门（approval_required / tool_plan_binding_required）fail-closed。
    request = ToolExecutionRequest(
        auth=context.auth,
        task_id=context.task_id,
        tool_name=tool_name,
        arguments=arguments,
        worker_id=context.worker_id,
        lease_generation=context.lease_generation,
        idempotency_key=context.idempotency_key,
        approval_id=context.approval_id,
        plan_revision=context.plan_revision,
        plan_digest=context.plan_digest,
        evidence_digest=context.evidence_digest,
    )
    try:
        result = executor.execute(request)
    except ToolExecutionError as exc:
        if exc.code in _FATAL_TOOL_CODES:
            # 失租/身份失效：穿透 middleware 使 invoke 失败，禁止渲染 ToolMessage 续跑。
            raise
        return render_tool_message(tool_name, None, error_code=exc.code)
    except Exception as exc:
        code = getattr(exc, "code", None)
        return render_tool_message(
            tool_name,
            None,
            error_code=code if isinstance(code, str) and code else "tool_execution_unavailable",
        )
    return render_tool_message(tool_name, result)


def build_research_tools(executor: ToolExecutionService) -> list[BaseTool]:
    """只暴露 4 个只读工具；写工具暴露数为 0。"""

    def _description(tool_name: str) -> str:
        from rag_kb.application.tool_registry import builtin_tool_definitions

        return next(
            definition.description
            for definition in builtin_tool_definitions()
            if definition.name == tool_name
        )

    @tool("search_knowledge", description=_description("search_knowledge"))
    def search_knowledge(
        query: str, runtime: ToolRuntime[AgentRunContext], limit: int = 5
    ) -> str:
        return _execute_read_tool("search_knowledge", {"query": query, "limit": limit}, runtime, executor)

    @tool("fetch_evidence", description=_description("fetch_evidence"))
    def fetch_evidence(
        evidence_ids: list[str], runtime: ToolRuntime[AgentRunContext]
    ) -> str:
        return _execute_read_tool("fetch_evidence", {"evidence_ids": evidence_ids}, runtime, executor)

    @tool("list_document_versions", description=_description("list_document_versions"))
    def list_document_versions(
        document_id: str, runtime: ToolRuntime[AgentRunContext]
    ) -> str:
        return _execute_read_tool("list_document_versions", {"document_id": document_id}, runtime, executor)

    @tool("compare_document_versions", description=_description("compare_document_versions"))
    def compare_document_versions(
        document_id: str,
        left_version: int,
        right_version: int,
        runtime: ToolRuntime[AgentRunContext],
    ) -> str:
        return _execute_read_tool(
            "compare_document_versions",
            {"document_id": document_id, "left_version": left_version, "right_version": right_version},
            runtime,
            executor,
        )

    return [
        search_knowledge,
        fetch_evidence,
        list_document_versions,
        compare_document_versions,
    ]


def build_controlled_action_tools(executor: ToolExecutionService) -> list[BaseTool]:
    """暴露 4 READ + 精确 2 WRITE；WRITE 由 HITL 暂停并要求服务端审批事实。"""

    def _description(tool_name: str) -> str:
        from rag_kb.application.tool_registry import builtin_tool_definitions

        return next(
            definition.description
            for definition in builtin_tool_definitions()
            if definition.name == tool_name
        )

    @tool("create_report_artifact", description=_description("create_report_artifact"))
    def create_report_artifact(
        title: str,
        content: str,
        format: str,
        runtime: ToolRuntime[AgentRunContext],
    ) -> str:
        return _execute_write_tool(
            "create_report_artifact",
            {"title": title, "content": content, "format": format},
            runtime,
            executor,
        )

    @tool("create_work_item_draft", description=_description("create_work_item_draft"))
    def create_work_item_draft(
        title: str,
        description: str,
        runtime: ToolRuntime[AgentRunContext],
        owner_group: str | None = None,
        due_date: str | None = None,
    ) -> str:
        arguments: dict[str, object] = {"title": title, "description": description}
        if owner_group is not None:
            arguments["owner_group"] = owner_group
        if due_date is not None:
            arguments["due_date"] = due_date
        return _execute_write_tool(
            "create_work_item_draft", arguments, runtime, executor
        )

    return [
        *build_research_tools(executor),
        create_report_artifact,
        create_work_item_draft,
    ]
