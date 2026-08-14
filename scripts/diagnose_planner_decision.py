"""Phase 1: 无工具 PlannerDecision 结构化输出诊断。"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rag_kb.config import Settings, get_settings
from rag_kb.observability import LocalDiagnosticTrace, classify_model_failure, finish_reason
from rag_kb.policy_rollout_planning import PlannerDecision, PlanningDraft


SYSTEM_PROMPT = (
    "你是无工具规划决策器。严格按优先级判断：无授权证据或明确拒绝访问时选 refuse；"
    "目标缺少必要信息时选 clarify；已有充分证据且请求要求形成计划时选 propose；"
    "否则选 search。search 只填写 search_query，clarify 只填写 clarification_questions，"
    "propose 只填写可引用的完整 PlanningDraft，refuse 只填写 refusal_reason；其他字段必须为 null。"
)

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "search", "goal": "确认研发发布流程的回滚时限",
        "evidence": "已有文档索引，但未提供回滚时限正文。", "access": "engineer 可访问研发流程文档。",
        "expected": "search",
    },
    {
        "name": "clarify", "goal": "帮我处理这个", "evidence": "没有可识别的业务目标。",
        "access": "用户身份已通过认证。", "expected": "clarify",
    },
    {
        "name": "propose", "goal": "制定研发发布流程落地计划",
        "evidence": "chunk_id=release:v1:chunk-01；已授权证据：发布窗口为周二、周四 14:00-17:00；P0 故障须 30 分钟内回滚。",
        "access": "engineer 可访问该研发流程文档。", "expected": "propose",
        "proposal": {"action": "propose", "used_chunk_ids": ["release:v1:chunk-01"], "todos": [{"todo_id": "rollback", "title": "核对回滚时限", "detail": "按证据核对发布回滚要求。", "chunk_ids": ["release:v1:chunk-01"]}]},
    },
    {
        "name": "refuse", "goal": "获取其他部门受限薪资文档",
        "evidence": "检索结果为空，且没有授权证据。", "access": "employee 无权访问薪资文档。",
        "expected": "refuse",
    },
)


def build_structured_model(settings: Settings):
    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return llm.with_structured_output(PlannerDecision, include_raw=True)


def _model_dump(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else {"object_type": type(value).__name__}


def _raw_response_details(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    return {
        "object_type": type(raw).__name__,
        "content": getattr(raw, "content", None),
        "tool_calls": getattr(raw, "tool_calls", None),
        "invalid_tool_calls": getattr(raw, "invalid_tool_calls", None),
        "response_metadata": getattr(raw, "response_metadata", None),
        "usage_metadata": getattr(raw, "usage_metadata", None),
    }


def run_diagnostic(model, cases=CASES, *, trace: LocalDiagnosticTrace | None = None) -> bool:
    from langchain_core.messages import HumanMessage, SystemMessage

    passed = True
    for case in cases:
        input_payload = {key: case[key] for key in ("goal", "evidence", "access")}
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(input_payload, ensure_ascii=False, sort_keys=True)),
        ]
        raw = None
        parsed = None
        parsing_error = None
        try:
            response = model.invoke(messages)
            if isinstance(response, dict) and {"raw", "parsed", "parsing_error"} <= response.keys():
                raw = response["raw"]
                parsed = response["parsed"]
                parsing_error = response["parsing_error"]
                if parsing_error is not None:
                    raise parsing_error
                if parsed is None:
                    raise ValueError("unexpected response shape")
                decision = parsed
            else:
                decision = response
                parsed = response
            action = decision.get("action") if isinstance(decision, dict) else getattr(decision, "action", None)
            ok = action == case["expected"]
            error_type = "none" if ok else "ActionMismatch"
            error_code = "none" if ok else "action_mismatch"
            error_message = None
        except Exception as exc:
            action = "error"
            ok = False
            error_type = type(exc).__name__
            error_code = classify_model_failure(exc, raw_response=raw)
            error_message = str(exc)
        safe_action = action if isinstance(action, str) and action in {"search", "clarify", "propose", "refuse"} else "error"
        print(
            f"case={case['name']} action={safe_action} pass={str(ok).lower()} "
            f"error-type={error_type} error-code={error_code}"
        )
        if trace is not None:
            trace.write_event(
                {
                    "case": case["name"],
                    "expected_action": case["expected"],
                    "observed_action": safe_action,
                    "passed": ok,
                    "prompt": {"system": SYSTEM_PROMPT, "input": input_payload},
                    "raw_response": _raw_response_details(raw),
                    "parsed_response": _model_dump(parsed),
                    "parsing_error": (
                        None
                        if parsing_error is None and error_message is None
                        else {
                            "type": type(parsing_error).__name__ if parsing_error is not None else error_type,
                            "message": str(parsing_error) if parsing_error is not None else error_message,
                        }
                    ),
                    "finish_reason": finish_reason(raw),
                    "error_code": error_code,
                }
            )
        passed = passed and ok
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-file",
        help="显式写入当前工作目录内的新 JSONL 文件；内容可能敏感，诊断后应删除。",
    )
    parser.add_argument("--trace-ttl-minutes", type=int, default=60)
    parser.add_argument("--case", action="append", dest="case_names")
    args = parser.parse_args([] if argv is None else argv)

    trace = None
    try:
        if args.trace_file:
            trace = LocalDiagnosticTrace(
                args.trace_file,
                root=Path.cwd(),
                ttl_seconds=args.trace_ttl_minutes * 60,
            )
        model = build_structured_model(get_settings())
    except Exception as exc:
        print(
            f"case=setup action=error pass=false error-type={type(exc).__name__} "
            f"error-code={classify_model_failure(exc)}"
        )
        if trace is not None:
            trace.close()
        return 1
    try:
        selected = CASES
        if args.case_names:
            known = {case["name"]: case for case in CASES}
            missing = set(args.case_names) - set(known)
            if missing:
                parser.error(f"unknown diagnostic case: {', '.join(sorted(missing))}")
            selected = tuple(known[name] for name in args.case_names)
        return 0 if run_diagnostic(model, cases=selected, trace=trace) else 1
    finally:
        if trace is not None:
            trace.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
