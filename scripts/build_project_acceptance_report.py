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


def sha256_file(path: Path) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    values["passed"] = max(0, values["total"] - values["failures"] - values["errors"] - values["skipped"])
    return ("failed" if values["failures"] or values["errors"] else "passed"), values


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def artifact(name: str, path: Path | None) -> dict[str, Any]:
    digest = sha256_file(path) if path else None
    return {"name": name, "path": str(path) if path else "", "sha256": digest, "status": "available" if digest else "missing"}


def build_report(*, pytest_xml: Path | None, retrieval_report: Path | None, task_report: Path | None, dataset: Path | None, config: Path | None, model_report: Path | None = None) -> dict[str, Any]:
    pytest_status, pytest_data = pytest_values(pytest_xml)
    retrieval_status, retrieval_data = read_json(retrieval_report)
    task_status, task_data = read_json(task_report)
    model_report_status, model_data = read_json(model_report)
    retrieval_status = observed_report_status(retrieval_status, retrieval_data)
    task_status = observed_report_status(task_status, task_data)
    model_report_status = observed_report_status(model_report_status, model_data)
    dataset_status = "available" if dataset and dataset.is_file() else "missing"
    config_status = "passed" if config and config.is_file() else "unverified"
    artifacts = [artifact("pytest_junit", pytest_xml), artifact("retrieval_report", retrieval_report), artifact("task_report", task_report), artifact("live_model_report", model_report), artifact("dataset", dataset), artifact("config", config)]
    all_inputs = [pytest_status, retrieval_status, task_status, model_report_status, "passed" if dataset_status == "available" else "unverified"]
    has_failure = "failed" in all_inputs
    has_unverified = "unverified" in all_inputs or any(item["status"] == "missing" for item in artifacts)
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
    model_values = (
        {
            key: model_data.get(key)
            for key in ("provider", "model", "request_count", "passed", "failed")
            if model_data.get(key) is not None
        }
        if live_model_observed
        else {}
    )
    model_status = model_report_status if live_model_observed else "unverified"
    task_contract_values = (
        {
            key: task_data.get(key)
            for key in ("execution_mode", "summary", "dataset")
            if task_data.get(key) is not None
        }
        if task_status == "passed" and task_data
        else {}
    )
    task_contract_status = (
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
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_on": datetime.now(UTC).isoformat(),
        "code": status_object("passed" if revision else "unverified", {"git_revision": revision}),
        "dataset": status_object("passed" if dataset_status == "available" else "unverified", {"sha256": sha256_file(dataset), "path": str(dataset) if dataset else None}),
        "config": status_object(config_status, {"path": str(config) if config else None}),
        "samples": status_object("passed" if dataset_status == "available" else "unverified", {"dataset": "observed_only"}),
        "outcomes": status_object("failed" if has_failure else "passed" if not has_unverified else "unverified", {"pytest": pytest_data, "retrieval": retrieval_data or {}, "task": task_data or {}, "model": model_data or {}}),
        "model": status_object(model_status, model_values), "tool": status_object(task_contract_status, task_contract_values), "latency": status_object(latency_status, latency_values),
        "security": status_object(security_status, security_values), "quality": status_object(quality_status, quality_values), "cost": status_object("unverified", {}),
        "gates": [
            {"name": "pytest", "status": pytest_status if pytest_status in {"passed", "failed"} else "unverified", "evidence": [str(pytest_xml) if pytest_xml else "missing"]},
            {"name": "retrieval_quality", "status": quality_status, "evidence": [str(retrieval_report) if retrieval_report else "missing"]},
            {"name": "acl_leak_rate", "status": security_status, "evidence": [str(retrieval_report) if retrieval_report else "missing"]},
            {"name": "live_model", "status": model_status, "evidence": [str(model_report) if model_report else "missing"]},
            {"name": "task_contract", "status": task_contract_status, "evidence": [str(task_report) if task_report else "missing"]},
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
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(pytest_xml=args.pytest_xml, retrieval_report=args.retrieval_report, task_report=args.task_report, dataset=args.dataset, config=args.config, model_report=args.model_report)
    validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
