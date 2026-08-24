"""运行 manifest 驱动的策略比较基准,只把实际运行结果写入 data/local/benchmarks。"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage
from jsonschema import validate

from rag_kb.ingestion import chunk_items, parse_document
from rag_kb.models import ChunkStatus, DocumentSpec, Role
from rag_kb.skills.policy_comparison import PolicyComparisonService


class ManifestEvidenceStore:
    """基准专用只读 fixture；复用 manifest 和固定语料，不写 Qdrant。"""

    def __init__(self, manifest_path: Path, manifest: dict) -> None:
        self._data: dict[tuple[str, int], list[dict]] = {}
        entries = manifest["documents"]
        max_versions: dict[str, int] = {}
        for entry in entries:
            max_versions[entry["document_id"]] = max(max_versions.get(entry["document_id"], 0), entry["version"])
        for entry in entries:
            spec = DocumentSpec(**entry)
            source = manifest_path.parent / entry["file"]
            if source.suffix.lower() == ".pdf":
                markdown_source = manifest_path.parent / "sources" / f"{source.stem}.md"
                if markdown_source.is_file():
                    source = markdown_source
            chunks = chunk_items(
                parse_document(source), spec, max_chars=800,
                now=datetime(2026, 1, 1, tzinfo=UTC),
            )
            status = (
                ChunkStatus.SUPERSEDED
                if entry["version"] < max_versions[entry["document_id"]]
                else ChunkStatus.ACTIVE
            )
            self._data[(entry["document_id"], entry["version"])] = [
                {**chunk.to_payload(), "status": str(status)} for chunk in chunks
            ]

    def scroll_authorized_version(self, document_id: str, version: int, role: Role) -> list[dict]:
        payloads = self._data.get((document_id, version), [])
        if not payloads or any(str(payload["status"]) not in {"active", "superseded"} for payload in payloads):
            return []
        if any(str(role) not in payload["allowed_roles"] for payload in payloads):
            return []
        return payloads


class BenchmarkAgentModel:
    """确定性工具调用模型;用于测量安全工作流而非评估真实 LLM 质量。"""

    def __init__(self, from_version: int, to_version: int) -> None:
        self.versions = (from_version, to_version)
        self.turn = 0

    def bind_tools(self, _tools, **_kwargs):
        return self

    def invoke(self, _messages):
        self.turn += 1
        if self.turn <= 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fetch_authorized_version",
                        "args": {"version": self.versions[self.turn - 1]},
                        "id": f"benchmark-{self.turn}",
                    }
                ]
            )
        if self.turn == 3:
            return AIMessage(
                content="",
                tool_calls=[{"name": "compare_authorized_versions", "args": {}, "id": "benchmark-3"}]
            )
        return AIMessage(
            content=json.dumps({"summary": "确定性基准摘要", "used_chunk_ids": []}),
        )


def run_benchmark(service: PolicyComparisonService, manifest: dict) -> dict:
    """逐个执行清单用例,报告只保留计数和耗时,不保存正文/模型输出。"""
    case_reports: list[dict] = []
    totals = {
        "case_pass": 0,
        "acl_leakage_count": 0,
        "version_mismatch": 0,
        "citation_integrity": 0,
        "guardrail_violations": 0,
        "refusal_incorrect_count": 0,
        "timing_ms": 0.0,
        "timing_sample_size": 0,
    }
    for case in manifest.get("policy_comparisons", []):
        started = time.perf_counter()
        from_version = int(case["from_version"])
        to_version = int(case["to_version"])
        role = Role(case["role"])
        service.model = BenchmarkAgentModel(from_version, to_version)
        response = service.compare(case["document_id"], from_version, to_version, role)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        body = response.model_dump_json()
        forbidden = case.get("must_not_mention", [])
        acl_leakage = sum(1 for value in forbidden if value in body)
        version_mismatch = 0
        citation_integrity = 0
        if response.verdict == "comparison" and response.comparison:
            version_mismatch = int(
                response.comparison.document_id != case["document_id"]
                or
                response.comparison.from_version != from_version
                or response.comparison.to_version != to_version
                or any(c.document_id != case["document_id"] for c in response.comparison.citations)
            )
            authorized_ids = {
                item["chunk_id"]
                for version in (from_version, to_version)
                for item in service.store.scroll_authorized_version(case["document_id"], version, role)
            }
            citation_integrity = sum(c.chunk_id not in authorized_ids for c in response.comparison.citations)
            expected_changes = set(case.get("expect_changes", []))
            actual_changes = {change.change for change in response.comparison.changes}
            version_mismatch += int(bool(expected_changes) and expected_changes != actual_changes)
        expected_refusal = case["expect"] == "refusal"
        actual_refusal = response.verdict == "refusal" and response.refusal is not None
        refusal_correctness = int(expected_refusal != actual_refusal)
        guardrail_violations = int((case["expect"] == "comparison") != (response.verdict == "comparison"))
        metrics = {
            "acl_leakage_count": acl_leakage,
            "version_mismatch": version_mismatch,
            "citation_integrity": citation_integrity,
            "guardrail_violations": guardrail_violations,
            "refusal_incorrect_count": refusal_correctness,
        }
        passed = not any(metrics.values())
        case_reports.append({"name": case["name"], "status": "pass" if passed else "fail", "duration_ms": duration_ms, "metrics": metrics})
        totals["case_pass"] += int(passed)
        for key, value in metrics.items():
            totals[key] += value
        totals["timing_ms"] += duration_ms
        totals["timing_sample_size"] += 1
    return {
        "schema_version": 1,
        "manifest": "data/corpus/manifest.yaml",
        "sample_size": len(case_reports),
        "metrics": totals,
        "cases": case_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus/manifest.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/local/benchmarks/policy-comparison-report.json"))
    parser.add_argument(
        "--schema", type=Path, default=Path("schemas/policy-comparison-report.schema.json")
    )
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    store = ManifestEvidenceStore(args.manifest, manifest)
    service = PolicyComparisonService(store, model=None)
    report = run_benchmark(service, manifest)
    validate(report, json.loads(args.schema.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
