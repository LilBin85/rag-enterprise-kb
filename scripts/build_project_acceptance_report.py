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


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "project-acceptance-report.schema.json"
_M1_RESEARCH_TOOL_TRACE = [
    "list_document_versions",
    "compare_document_versions",
    "search_knowledge",
    "fetch_evidence",
]

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


def m1_e2e_status(parse_status: str, value: dict[str, Any] | None) -> str:
    status = observed_report_status(parse_status, value)
    if status != "passed" or value is None:
        return status
    strict = value.get("strict_msgpack")
    if (
        value.get("execution_mode") != "live_postgres_msgpack"
        or not isinstance(strict, dict)
        or strict.get("runtime_flag") is not True
        or strict.get("pickle_fallback") is not False
        or not isinstance(value.get("document_version_endpoint"), str)
        or value.get("document_versions") != 2
        or value.get("tool_calls") != 4
        or value.get("tool_trace") != _M1_RESEARCH_TOOL_TRACE
        or "pickle" in (value.get("checkpoint_encodings") or [])
        or value.get("result_readable") is not True
        or not isinstance(value.get("citation_count"), int)
        or value["citation_count"] < 1
    ):
        return "failed"
    return "passed"


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
    return ("failed" if values["failures"] or values["errors"] else "passed"), values


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
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
            ],
            cwd=ROOT,
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
            ],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    digest = hashlib.sha256(diff)
    root = ROOT.resolve()
    for raw_path in sorted(path for path in untracked.split(b"\0") if path):
        digest.update(b"\0" + raw_path)
        try:
            path = (ROOT / raw_path.decode("utf-8", errors="surrogateescape")).resolve()
            path.relative_to(root)
            digest.update(path.read_bytes() if path.is_file() else b"<not-a-file>")
        except (OSError, ValueError):
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def artifact(name: str, path: Path | None) -> dict[str, Any]:
    digest = sha256_file(path) if path else None
    return {"name": name, "path": str(path) if path else "", "sha256": digest, "status": "available" if digest else "missing"}


def build_report(*, pytest_xml: Path | None, retrieval_report: Path | None, task_report: Path | None, dataset: Path | None, config: Path | None, model_report: Path | None = None, m1_e2e_report: Path | None = None) -> dict[str, Any]:
    pytest_status, pytest_data = pytest_values(pytest_xml)
    retrieval_status, retrieval_data = read_json(retrieval_report)
    task_status, task_data = read_json(task_report)
    model_report_status, model_data = read_json(model_report)
    m1_e2e_parse_status, m1_e2e_data = read_json(m1_e2e_report)
    retrieval_status = observed_report_status(retrieval_status, retrieval_data)
    task_status = observed_report_status(task_status, task_data)
    model_report_status = observed_report_status(model_report_status, model_data)
    m1_e2e_report_status = m1_e2e_status(m1_e2e_parse_status, m1_e2e_data)
    dataset_status = "available" if dataset and dataset.is_file() else "missing"
    config_status = "passed" if config and config.is_file() else "unverified"
    artifacts = [artifact("pytest_junit", pytest_xml), artifact("retrieval_report", retrieval_report), artifact("legacy_task_report", task_report), artifact("legacy_live_model_report", model_report), artifact("m1_research_e2e", m1_e2e_report), artifact("dataset", dataset), artifact("config", config)]
    required_artifact_names = {"pytest_junit", "retrieval_report", "m1_research_e2e", "dataset"}
    all_inputs = [pytest_status, retrieval_status, m1_e2e_report_status, "passed" if dataset_status == "available" else "unverified"]
    has_failure = "failed" in all_inputs
    has_unverified = "unverified" in all_inputs or any(item["name"] in required_artifact_names and item["status"] == "missing" for item in artifacts)
    verdict_status = "failed" if has_failure else "unverified" if has_unverified else "passed"
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
    live_model_observed = bool(model_data and model_data.get("live") is True)
    model_status = model_report_status if live_model_observed else "unverified"
    legacy_task_contract_values = (
        {
            key: task_data.get(key)
            for key in ("execution_mode", "summary", "dataset")
            if task_data.get(key) is not None
        }
        if task_status == "passed" and task_data
        else {}
    )
    legacy_task_contract_status = (
        task_status
        if task_data and task_data.get("execution_mode") == "deterministic_contract_simulation"
        else "unverified"
    )
    latencies = [
        case.get("latency_ms")
        for case in (model_data or {}).get("cases", [])
        if isinstance(case.get("latency_ms"), (int, float))
    ]
    latency_values = (
        {
            "sample_count": len(latencies),
            "average_ms": round(sum(latencies) / len(latencies), 3),
            "maximum_ms": max(latencies),
        }
        if latencies
        else {}
    )
    latency_status = model_status if latencies else "unverified"
    if "failed" in {quality_status, security_status}:
        verdict_status = "failed"
    elif "unverified" in {quality_status, security_status} and verdict_status == "passed":
        verdict_status = "unverified"
    revision = git_revision()
    worktree_fingerprint = git_worktree_fingerprint()
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_on": datetime.now(UTC).isoformat(),
        "code": status_object(
            "passed" if revision and worktree_fingerprint else "unverified",
            {
                "git_revision": revision,
                "worktree_sha256": worktree_fingerprint,
            },
        ),
        "dataset": status_object("passed" if dataset_status == "available" else "unverified", {"sha256": sha256_file(dataset), "path": str(dataset) if dataset else None}),
        "config": status_object(config_status, {"path": str(config) if config else None}),
        "samples": status_object("passed" if dataset_status == "available" else "unverified", {"dataset": "observed_only"}),
        "outcomes": status_object("failed" if has_failure else "passed" if not has_unverified else "unverified", {"pytest": pytest_data, "retrieval": retrieval_data or {}, "legacy_task": task_data or {}, "legacy_model": model_data or {}, "m1_research_e2e": m1_e2e_data or {}}),
        "model": status_object(m1_e2e_report_status, {key: m1_e2e_data.get(key) for key in ("execution_mode", "citation_count", "strict_msgpack") if m1_e2e_data and m1_e2e_data.get(key) is not None}), "tool": status_object(m1_e2e_report_status, {"tool_trace": m1_e2e_data.get("tool_trace"), "tool_calls": m1_e2e_data.get("tool_calls")} if m1_e2e_data else {}), "latency": status_object(latency_status, latency_values),
        "security": status_object(security_status, security_values), "quality": status_object(quality_status, quality_values), "cost": status_object("unverified", {}),
        "gates": [
            {"name": "pytest", "status": pytest_status if pytest_status in {"passed", "failed"} else "unverified", "evidence": [str(pytest_xml) if pytest_xml else "missing"]},
            {"name": "retrieval_quality", "status": quality_status, "evidence": [str(retrieval_report) if retrieval_report else "missing"]},
            {"name": "acl_leak_rate", "status": security_status, "evidence": [str(retrieval_report) if retrieval_report else "missing"]},
            {"name": "m1_research_e2e", "status": m1_e2e_report_status, "evidence": [str(m1_e2e_report) if m1_e2e_report else "missing"]},
            {"name": "legacy_task_contract", "status": legacy_task_contract_status, "evidence": [str(task_report) if task_report else "missing"]},
        ],
        "failures": [] if not has_failure else [{"source": "input_reports", "status": "failed"}],
        "evidence_artifacts": artifacts,
        "report_hash": "0" * 64,
        "verdict": {"status": verdict_status, "reason": "observed inputs only; missing metrics are not inferred"},
    }
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
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(pytest_xml=args.pytest_xml, retrieval_report=args.retrieval_report, task_report=args.task_report, dataset=args.dataset, config=args.config, model_report=args.model_report, m1_e2e_report=args.m1_e2e_report)
    validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
