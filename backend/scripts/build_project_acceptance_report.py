"""从已有机器报告生成项目验收汇总；缺失证据永远标记 unverified。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# 验收链路锚定仓库根：Git 工作目录与未跟踪文件定位覆盖整个仓库；
# 后端契约 schema 位于后端工程根。
BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
SCHEMA = BACKEND_ROOT / "schemas" / "project-acceptance-report.schema.json"
_M1_RESEARCH_TOOL_TRACE = [
    "list_document_versions",
    "compare_document_versions",
    "search_knowledge",
    "fetch_evidence",
]
# 与 agent.checkpointer 的 _TASK_CHECKPOINT_MSGPACK_TYPES 派生结果一致；
# 旧 checkpoint 仍引用 rag_kb.agent.result 路径，新写入使用 application 契约路径。
_STRICT_MSGPACK_ALLOWLIST = sorted(
    f"{module}.{name}"
    for module, name in (
        ("rag_kb.agent.result", "TaskResultCandidate"),
        ("rag_kb.application.agent_contract", "TaskResultCandidate"),
    )
)
_M2_READ_TOOLS = _M1_RESEARCH_TOOL_TRACE
_M2_WRITE_TOOLS = ["create_report_artifact", "create_work_item_draft"]
_M2_CRASH_RECOVERY_STEPS = (
    "before_execution",
    "after_execution_before_checkpoint",
    "after_checkpoint_before_sync",
)
_M2_PRIVACY_COUNTS = (
    "storage_key",
    "prompt",
    "content",
    "credentials",
    "resume_payload",
)

_SKIP_REASON_DESCRIPTIONS = {
    "postgres_test_url_unconfigured": "未配置 PostgreSQL 测试地址",
    "qdrant_unavailable": "Qdrant 不可达",
}


def sha256_file(path: Path) -> str | None:
    if path is None or not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def status_object(status: str, values: dict[str, Any] | None = None, source: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "values": values or {}}
    if source is not None:
        result["source"] = source
    return result


def read_json(path: Path | None) -> tuple[str, dict[str, Any] | None]:
    if path is None or not path.is_file():
        return "unverified", None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "failed", None
    if not isinstance(value, dict):
        return "failed", None
    return "passed", value


def observed_report_status(parse_status: str, value: dict[str, Any] | None) -> str:
    if parse_status != "passed" or value is None:
        return parse_status
    status = value.get("status")
    if status in {"completed", "passed"}:
        return "passed"
    if status in {"failed", "run_failed"}:
        return "failed"
    return "unverified"


def source_binding_status(value: dict[str, Any], revision: str | None, worktree_fingerprint: str | None) -> str:
    binding = value.get("source_binding")
    if not isinstance(binding, dict) or not isinstance(binding.get("git_revision"), str):
        return "failed"
    if revision is None:
        return "unverified"
    if binding["git_revision"] != revision:
        return "failed"
    source_worktree = binding.get("worktree_sha256")
    if source_worktree is not None and (
        not isinstance(source_worktree, str)
        or len(source_worktree) != 64
        or any(character not in "0123456789abcdef" for character in source_worktree)
        or (worktree_fingerprint is not None and source_worktree != worktree_fingerprint)
    ):
        return "failed"
    return "passed"


def strict_msgpack_valid(value: dict[str, Any]) -> bool:
    return (
        value.get("runtime_flag") is True
        and value.get("serializer") == "JsonPlusSerializer"
        and value.get("pickle_fallback") is False
        and value.get("allowed_msgpack_modules") == _STRICT_MSGPACK_ALLOWLIST
    )


def m2_terminal_audit_valid(
    ledger: Any, coverage: Any, primary_task_id: Any
) -> bool:
    if (
        not isinstance(ledger, list)
        or not isinstance(coverage, dict)
        or type(coverage.get("task_count")) is not int
        or len(ledger) != coverage["task_count"]
        or len(ledger) < 30
    ):
        return False
    task_ids: set[str] = set()
    for entry in ledger:
        if not isinstance(entry, dict):
            return False
        task_id = entry.get("task_id")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in task_ids
            or entry.get("status") not in {"succeeded", "refused"}
            or not isinstance(entry.get("stop_reason"), str)
            or not entry["stop_reason"]
            or entry.get("error_code") is not None
            and (not isinstance(entry["error_code"], str) or not entry["error_code"])
            or not isinstance(entry.get("correlation_id"), str)
            or not entry["correlation_id"]
            or type(entry.get("model_calls")) is not int
            or entry["model_calls"] < 0
            or type(entry.get("tool_calls")) is not int
            or entry["tool_calls"] < 0
        ):
            return False
        task_ids.add(task_id)
    return primary_task_id in task_ids


def m2_terminal_binding_valid(binding: Any, terminal_binding: Any) -> bool:
    if not isinstance(binding, dict) or not isinstance(terminal_binding, dict):
        return False
    approval = terminal_binding.get("approval")
    tool_call = terminal_binding.get("tool_call")
    artifact = terminal_binding.get("artifact")
    result = terminal_binding.get("task_result")
    if not all(isinstance(value, dict) for value in (approval, tool_call, artifact, result)):
        return False
    assert isinstance(approval, dict)
    assert isinstance(tool_call, dict)
    assert isinstance(artifact, dict)
    assert isinstance(result, dict)
    return (
        approval.get("approval_id") == binding.get("approval_id")
        and isinstance(approval.get("checkpoint_action_id"), str)
        and bool(approval["checkpoint_action_id"])
        and approval.get("status") == "executed"
        and tool_call.get("tool_call_id") == binding.get("tool_call_id")
        and tool_call.get("approval_id") == approval.get("approval_id")
        and tool_call.get("status") == "succeeded"
        and artifact.get("artifact_id") == binding.get("artifact_id")
        and artifact.get("tool_call_id") == tool_call.get("tool_call_id")
        and artifact.get("content_hash") == binding.get("artifact_hash")
        and artifact.get("status") == "ready"
        and result.get("task_id") == binding.get("task_id")
        and result.get("result_kind") == "controlled_action"
        and result.get("status") == "succeeded"
        and isinstance(result.get("stop_reason"), str)
        and bool(result["stop_reason"])
        and result.get("artifact_id") == artifact.get("artifact_id")
    )


def m1_e2e_status(
    parse_status: str,
    value: dict[str, Any] | None,
    revision: str | None,
    worktree_fingerprint: str | None,
) -> str:
    status = observed_report_status(parse_status, value)
    if status != "passed" or value is None:
        return status
    strict = value.get("strict_msgpack")
    if (
        value.get("execution_mode") != "live_postgres_msgpack"
        or not isinstance(strict, dict)
        or not strict_msgpack_valid(strict)
        or not isinstance(value.get("document_version_endpoint"), str)
        or value.get("document_versions") != 2
        or value.get("tool_calls") != 4
        or value.get("tool_trace") != _M1_RESEARCH_TOOL_TRACE
        or not isinstance(value.get("checkpoint_encodings"), list)
        or "pickle" in value["checkpoint_encodings"]
        or value.get("result_readable") is not True
        or not isinstance(value.get("citation_count"), int)
        or value["citation_count"] < 1
    ):
        return "failed"
    return source_binding_status(value, revision, worktree_fingerprint)


def m2_e2e_status(
    parse_status: str,
    value: dict[str, Any] | None,
    revision: str | None,
    worktree_fingerprint: str | None,
) -> str:
    status = observed_report_status(parse_status, value)
    if status != "passed" or value is None:
        return status
    strict = value.get("strict_msgpack")
    binding = value.get("binding")
    artifact = value.get("artifact")
    tools = value.get("tool_exposure")
    trace = value.get("tool_trace")
    hitl = value.get("hitl")
    coverage = value.get("coverage")
    privacy = value.get("privacy_scan")
    access = value.get("access_checks")
    side_effects = value.get("side_effects")
    ledger = value.get("task_ledger")
    terminal_binding = value.get("terminal_binding")
    if (
        value.get("execution_mode") != "live_postgres_msgpack"
        or value.get("capability") != "task-agent"
        or value.get("protocol") != "m2-fenced-v1"
        or not isinstance(strict, dict)
        or not strict_msgpack_valid(strict)
        or not isinstance(value.get("checkpoint_encodings"), list)
        or "pickle" in value["checkpoint_encodings"]
        or tools != {"read": _M2_READ_TOOLS, "write": _M2_WRITE_TOOLS}
        or not isinstance(trace, list)
        or not trace
        or any(tool not in _M2_READ_TOOLS + _M2_WRITE_TOOLS for tool in trace)
        or not isinstance(binding, dict)
        or not all(isinstance(binding.get(name), str) and binding[name] for name in (
            "task_id", "thread_id", "checkpoint_id", "interrupt_id", "approval_id", "tool_call_id", "artifact_id", "artifact_hash", "result_artifact_id"
        ))
        or binding["task_id"] != binding["thread_id"]
        or binding["artifact_id"] != binding["result_artifact_id"]
        or not isinstance(artifact, dict)
        or artifact.get("artifact_id") != binding["artifact_id"]
        or artifact.get("content_hash") != binding["artifact_hash"]
        or artifact.get("status") != "ready"
        or artifact.get("content_readable") is not True
        or artifact.get("hash_verified") is not True
        or not isinstance(hitl, dict)
        or hitl.get("adapter_calls_before_approval") != 0
        or hitl.get("adapter_calls_after_approval") != 1
        or not isinstance(side_effects, dict)
        or any(side_effects.get(name) != 0 for name in ("unapproved", "duplicate", "unknown_replay"))
        or not isinstance(access, dict)
        or access.get("cross_tenant_status") != 404
        or access.get("nonowner_status") != 404
        or not isinstance(privacy, dict)
        or any(privacy.get(name) != 0 for name in _M2_PRIVACY_COUNTS)
        or not isinstance(value.get("cleanup"), dict)
        or value["cleanup"].get("completed") is not True
        or not isinstance(coverage, dict)
        or type(coverage.get("task_count")) is not int
        or coverage["task_count"] < 30
        or not m2_terminal_audit_valid(ledger, coverage, binding.get("task_id"))
        or not m2_terminal_binding_valid(binding, terminal_binding)
        or type(coverage.get("scenario_count")) is not int
        or coverage["scenario_count"] < 20
        or not isinstance(coverage.get("crash_recovery"), dict)
        or any(coverage["crash_recovery"].get(name) is not True for name in _M2_CRASH_RECOVERY_STEPS)
    ):
        return "failed"
    return source_binding_status(value, revision, worktree_fingerprint)


def pytest_values(path: Path | None) -> tuple[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return "unverified", {}
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return "failed", {}
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        return "failed", {}
    values = {
        "total": int(suite.attrib.get("tests", 0)),
        "failures": int(suite.attrib.get("failures", 0)),
        "errors": int(suite.attrib.get("errors", 0)),
        "skipped": int(suite.attrib.get("skipped", 0)),
        "duration_seconds": float(suite.attrib.get("time", 0.0)),
    }
    values["skipped_details"] = [
        {
            "test": "::".join(
                part
                for part in (case.attrib.get("classname"), case.attrib.get("name"))
                if part
            )
            or "unknown",
            "reason": _SKIP_REASON_DESCRIPTIONS.get(
                (skipped.attrib.get("message") or (skipped.text or "").strip()),
                "跳过详情未写入报告",
            ),
        }
        for case in root.iter("testcase")
        for skipped in (case.find("skipped"),)
        if skipped is not None
    ]
    values["passed"] = max(0, values["total"] - values["failures"] - values["errors"] - values["skipped"])
    return ("failed" if values["failures"] or values["errors"] or values["skipped"] else "passed"), values


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_worktree_fingerprint() -> str | None:
    """哈希当前工作树；排除输出报告，避免报告把自身变更纳入身份。"""
    try:
        diff = subprocess.check_output(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--binary",
                "HEAD",
                "--",
                ".",
                ":(exclude)docs/acceptance/latest.json",
                ":(exclude)docs/acceptance/m1-research-e2e.json",
                ":(exclude)docs/acceptance/m2-controlled-write-e2e.json",
            ],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        untracked = subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                ".",
                ":(exclude)docs/acceptance/latest.json",
                ":(exclude)docs/acceptance/m1-research-e2e.json",
                ":(exclude)docs/acceptance/m2-controlled-write-e2e.json",
            ],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    digest = hashlib.sha256(diff)
    root = REPO_ROOT.resolve()
    for raw_path in sorted(path for path in untracked.split(b"\0") if path):
        digest.update(b"\0" + raw_path)
        try:
            path = (REPO_ROOT / raw_path.decode("utf-8", errors="surrogateescape")).resolve()
            path.relative_to(root)
            digest.update(path.read_bytes() if path.is_file() else b"<not-a-file>")
        except (OSError, ValueError):
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def artifact(name: str, path: Path | None) -> dict[str, Any]:
    digest = sha256_file(path) if path else None
    return {"name": name, "path": str(path) if path else "", "sha256": digest, "status": "available" if digest else "missing"}


def build_report(*, pytest_xml: Path | None, retrieval_report: Path | None, task_report: Path | None, dataset: Path | None, config: Path | None, model_report: Path | None = None, m1_e2e_report: Path | None = None, m2_e2e_report: Path | None = None) -> dict[str, Any]:
    revision = git_revision()
    worktree_fingerprint = git_worktree_fingerprint()
    pytest_status, pytest_data = pytest_values(pytest_xml)
    retrieval_status, retrieval_data = read_json(retrieval_report)
    task_status, task_data = read_json(task_report)
    _, model_data = read_json(model_report)
    m1_e2e_parse_status, m1_e2e_data = read_json(m1_e2e_report)
    m2_e2e_parse_status, m2_e2e_data = read_json(m2_e2e_report)
    retrieval_status = observed_report_status(retrieval_status, retrieval_data)
    task_status = observed_report_status(task_status, task_data)
    m1_e2e_report_status = m1_e2e_status(m1_e2e_parse_status, m1_e2e_data, revision, worktree_fingerprint)
    m2_e2e_report_status = m2_e2e_status(m2_e2e_parse_status, m2_e2e_data, revision, worktree_fingerprint)
    dataset_status = "available" if dataset and dataset.is_file() else "missing"
    config_status = "passed" if config and config.is_file() else "unverified"
    artifacts = [artifact("pytest_junit", pytest_xml), artifact("retrieval_report", retrieval_report), artifact("legacy_task_report", task_report), artifact("legacy_live_model_report", model_report), artifact("m1_research_e2e", m1_e2e_report), artifact("m2_controlled_write_e2e", m2_e2e_report), artifact("dataset", dataset), artifact("config", config)]
    required_artifact_names = {"pytest_junit", "retrieval_report", "m1_research_e2e", "m2_controlled_write_e2e", "dataset"}
    code_status = "passed" if revision and worktree_fingerprint else "unverified"
    all_inputs = [code_status, pytest_status, retrieval_status, m1_e2e_report_status, m2_e2e_report_status, "passed" if dataset_status == "available" else "unverified"]
    retrieval_metrics = (
        retrieval_data.get("test", {}).get("metrics", {})
        if retrieval_status == "passed" and retrieval_data
        else {}
    )
    quality_values = {
        key: retrieval_metrics.get(key)
        for key in ("recall_at_5", "mrr", "no_answer_accuracy")
        if retrieval_metrics.get(key) is not None
    }
    quality_passed = bool(quality_values) and (
        quality_values.get("recall_at_5", 0) >= 0.90
        and quality_values.get("mrr", 0) >= 0.85
        and quality_values.get("no_answer_accuracy", 0) >= 0.90
    )
    acl_value = retrieval_metrics.get("acl_leak_rate")
    security_values = {"acl_leak_rate": acl_value} if acl_value is not None else {}
    security_status = (
        "passed" if acl_value == 0 else "failed" if acl_value is not None else "unverified"
    )
    quality_status = (
        "passed" if quality_passed else "failed" if quality_values else "unverified"
    )
    if task_report is None:
        legacy_task_contract_status = "unverified"
    elif task_status != "passed":
        legacy_task_contract_status = task_status
    elif task_data and task_data.get("execution_mode") == "deterministic_contract_simulation":
        legacy_task_contract_status = "passed"
    else:
        legacy_task_contract_status = "failed"
    all_inputs.extend((quality_status, security_status))
    if config is not None:
        all_inputs.append(config_status)
    if task_report is not None:
        all_inputs.append(legacy_task_contract_status)
    has_failure = "failed" in all_inputs
    has_unverified = "unverified" in all_inputs or any(
        item["name"] in required_artifact_names and item["status"] == "missing"
        for item in artifacts
    )
    verdict_status = "failed" if has_failure else "unverified" if has_unverified else "passed"
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_on": datetime.now(UTC).isoformat(),
        "code": status_object(
            code_status,
            {
                "git_revision": revision,
                "worktree_sha256": worktree_fingerprint,
            },
        ),
        "dataset": status_object("passed" if dataset_status == "available" else "unverified", {"sha256": sha256_file(dataset), "path": str(dataset) if dataset else None}),
        "samples": status_object("passed" if dataset_status == "available" else "unverified", {"dataset": "observed_only"}),
        "outcomes": status_object("failed" if has_failure else "passed" if not has_unverified else "unverified", {"pytest": pytest_data, "retrieval": retrieval_data or {}, "legacy_task": task_data or {}, "legacy_model": model_data or {}, "m1_research_e2e": m1_e2e_data or {}, "m2_controlled_write_e2e": m2_e2e_data or {}}),
        "model": status_object(m1_e2e_report_status, {key: m1_e2e_data.get(key) for key in ("execution_mode", "citation_count", "strict_msgpack", "source_binding") if m1_e2e_data and m1_e2e_data.get(key) is not None}),
        "tool": status_object(m1_e2e_report_status, {"tool_trace": m1_e2e_data.get("tool_trace"), "tool_calls": m1_e2e_data.get("tool_calls")} if m1_e2e_data else {}),
        "m2": status_object(m2_e2e_report_status, {key: m2_e2e_data.get(key) for key in ("capability", "protocol", "tool_exposure", "coverage", "task_ledger", "terminal_binding", "source_binding") if m2_e2e_data and m2_e2e_data.get(key) is not None}),
        "security": status_object(security_status, security_values), "quality": status_object(quality_status, quality_values),
        "gates": [
            {"name": "code_binding", "status": code_status, "evidence": [revision or "unverified"]},
            {"name": "dataset", "status": "passed" if dataset_status == "available" else "unverified", "evidence": [str(dataset) if dataset else "missing"]},
            {"name": "pytest", "status": pytest_status if pytest_status in {"passed", "failed"} else "unverified", "evidence": [str(pytest_xml) if pytest_xml else "missing"]},
            {"name": "retrieval_quality", "status": quality_status, "evidence": [str(retrieval_report) if retrieval_report else "missing"]},
            {"name": "acl_leak_rate", "status": security_status, "evidence": [str(retrieval_report) if retrieval_report else "missing"]},
            {"name": "m1_research_e2e", "status": m1_e2e_report_status, "evidence": [str(m1_e2e_report) if m1_e2e_report else "missing"]},
            {"name": "m2_controlled_write_e2e", "status": m2_e2e_report_status, "evidence": [str(m2_e2e_report) if m2_e2e_report else "missing"]},
        ],
        "failures": [] if not has_failure else [{"source": "input_reports", "status": "failed"}],
        "evidence_artifacts": artifacts,
        "report_hash": "0" * 64,
        "verdict": {"status": verdict_status, "reason": "observed inputs only; missing metrics are not inferred"},
    }
    if config is not None:
        report["config"] = status_object(config_status, {"path": str(config)})
        report["gates"].append({"name": "config", "status": config_status, "evidence": [str(config)]})
    if task_report is not None:
        report["gates"].append({"name": "legacy_task_contract", "status": legacy_task_contract_status, "evidence": [str(task_report)]})
    canonical = json.dumps({k: v for k, v in report.items() if k != "report_hash"}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    report["report_hash"] = hashlib.sha256(canonical).hexdigest()
    return report


def validate_report(report: dict[str, Any]) -> None:
    import jsonschema
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytest-xml", type=Path)
    parser.add_argument("--retrieval-report", type=Path)
    parser.add_argument("--task-report", type=Path)
    parser.add_argument("--model-report", type=Path)
    parser.add_argument("--m1-e2e-report", type=Path)
    parser.add_argument("--m2-e2e-report", type=Path, help="最终 passed verdict 必需的 M2 严格真实 E2E 报告")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(pytest_xml=args.pytest_xml, retrieval_report=args.retrieval_report, task_report=args.task_report, dataset=args.dataset, config=args.config, model_report=args.model_report, m1_e2e_report=args.m1_e2e_report, m2_e2e_report=args.m2_e2e_report)
    validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
