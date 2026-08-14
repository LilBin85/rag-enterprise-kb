"""运行冻结的真实 BGE-M3 + Qdrant 检索验收。

本脚本刻意不用 fixture alias、LLM、BM25 或 reranker。每次运行会创建一个随机
Qdrant collection，真实摄取 ``data/corpus/manifest.yaml``，使休假制度经历 v1→v2
生命周期，再经过生产 ``Retriever`` 的 dense embedding、candidate_k、active/ACL 过滤。
candidate_k/context_k/阈值只在 calibration split 上联合选择，test split 仅作一次冻结
配置的留出评估；无论成功还是失败都会删除本次创建的 collection。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from rag_kb.config import get_settings
from rag_kb.embedder import BgeM3Embedder
from rag_kb.ingestion import load_manifest
from rag_kb.lifecycle import DocumentLifecycle
from rag_kb.retrieval_acceptance import (
    calibrate_retrieval_configuration,
    evaluate_ranked_results,
    load_acceptance_dataset,
    make_report_header,
    quality_gates,
    serialize_evaluation,
    summarize_retrieval_costs,
    validate_gold_labels_against_ingested_corpus,
    write_report,
)
from rag_kb.retriever import Retriever
from rag_kb.store import ChunkStore


def _ingest_manifest(lifecycle: DocumentLifecycle, store: ChunkStore, corpus_dir: Path) -> list[dict[str, Any]]:
    """按 manifest 顺序摄取；同一 document_id 的后续条目走版本更新路径。"""
    records: list[dict[str, Any]] = []
    known_documents: set[str] = set()
    for spec in load_manifest(corpus_dir):
        source = corpus_dir / spec.file
        if spec.document_id in known_documents:
            result = lifecycle.update_document(spec.document_id, spec, source)
        else:
            result = lifecycle.add_document(spec, source)
            known_documents.add(spec.document_id)
        records.append({
            "document_id": result["document_id"],
            "version": result["version"],
            "chunk_count": result["chunk_count"],
            "previous_version": result.get("previous_version"),
        })
    return records


def _run_all_queries(
    retriever: Retriever, cases
) -> tuple[dict[str, Any], dict[str, float]]:
    """逐条调用生产 Retriever，确保每个样本经过同一个 ACL/Qdrant 边界。"""
    results: dict[str, Any] = {}
    latency_ms: dict[str, float] = {}
    for case in cases:
        started = time.perf_counter()
        results[case.case_id] = retriever.search(case.question, case.role)
        latency_ms[case.case_id] = (time.perf_counter() - started) * 1000
    return results, latency_ms


def _metrics_payload(metrics) -> dict[str, Any]:
    return asdict(metrics)


def _acceptance_outcome(
    *, calibration_passed: bool, test_passed: bool, replay_passed: bool = True
) -> tuple[str, int]:
    """校准集与冻结测试集都必须通过，避免用 test 偶然成绩掩盖选参失败。"""
    if not calibration_passed:
        return "calibration_quality_gates_failed", 1
    if not test_passed:
        return "test_quality_gates_failed", 1
    if not replay_passed:
        return "selected_configuration_replay_mismatch", 1
    return "passed", 0


def _record_collection_cleanup(store, collection: str, report: dict[str, Any]) -> bool:
    """记录隔离 collection 清理结果；只暴露稳定异常类型。"""
    report["collection_cleanup"]["attempted"] = True
    try:
        store.client.delete_collection(collection)
        report["collection_cleanup"]["succeeded"] = True
        return True
    except Exception as exc:
        report["collection_cleanup"]["succeeded"] = False
        report["collection_cleanup"]["error_code"] = type(exc).__name__
        return False


def _parse_k_values(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("K 列表必须是逗号分隔的正整数") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("K 列表必须至少包含一个正整数")
    return values


def run_acceptance(
    *,
    dataset_path: Path,
    schema_path: Path,
    output_path: Path,
    report_schema_path: Path = Path("schemas/retrieval-acceptance-report.schema.json"),
    collection_prefix: str = "retrieval_acceptance",
    candidate_k_values: Sequence[int] = (5, 10),
    context_k_values: Sequence[int] = (2, 3, 5),
) -> tuple[dict[str, Any], int]:
    settings = get_settings()
    dataset = load_acceptance_dataset(dataset_path, schema_path)
    candidate_k_values = tuple(sorted(set(candidate_k_values)))
    context_k_values = tuple(sorted(set(context_k_values)))
    if (
        not candidate_k_values
        or not context_k_values
        or any(value <= 0 for value in (*candidate_k_values, *context_k_values))
    ):
        raise ValueError("retrieval_k_calibration_grid_requires_positive_integers")
    if dataset.required_top_k not in context_k_values:
        raise ValueError("retrieval_context_grid_must_include_frozen_top_k_baseline")
    if not any(
        candidate_k >= context_k
        for candidate_k in candidate_k_values
        for context_k in context_k_values
    ):
        raise ValueError("retrieval_k_calibration_grid_has_no_valid_pair")
    if not any(candidate_k >= dataset.required_top_k for candidate_k in candidate_k_values):
        raise ValueError("retrieval_candidate_grid_cannot_run_frozen_top_k_baseline")
    raw_candidate_k = max(candidate_k_values)

    collection = f"{collection_prefix}_{uuid.uuid4().hex[:12]}"
    store = ChunkStore(settings.qdrant_url, collection)
    report = make_report_header(
        dataset,
        collection=collection,
        qdrant_url=settings.qdrant_url,
        model_name="BAAI/bge-m3",
        batch_size=0,
    )
    report["backend"]["embedding_batch_size"] = "per_document_chunk_batch"
    report["backend"]["query_execution"] = "sequential_production_retriever_search"
    report["backend"]["raw_candidate_k"] = raw_candidate_k
    report["parameter_search"] = {
        "selection_split": "calibration",
        "test_labels_used_for_selection": False,
        "candidate_k_values": list(candidate_k_values),
        "context_k_values": list(context_k_values),
        "raw_candidate_pool_executions_per_case": 1,
        "selected_configuration_replay_executions_per_case": 1,
        "production_configuration_changed": False,
    }
    report["collection_cleanup"] = {"attempted": False, "succeeded": False}
    write_report(output_path, report, schema_path=report_schema_path)

    collection_created = False
    try:
        settings.prepare_runtime_env()
        embedder = BgeM3Embedder()
        store.ensure_collection(settings.embedding_dim)
        collection_created = True
        with tempfile.TemporaryDirectory(prefix="rag-kb-retrieval-acceptance-") as temp_root:
            lifecycle = DocumentLifecycle(
                store=store,
                embedder=embedder,
                upload_dir=Path(temp_root) / "uploads",
                max_chars=settings.chunk_max_chars,
            )
            report["ingestion"] = {
                "documents": _ingest_manifest(lifecycle, store, settings.corpus_dir),
            }

        labels = validate_gold_labels_against_ingested_corpus(dataset, store)
        report["gold_label_validation"] = labels
        if not labels["passed"]:
            report["status"] = "label_validation_failed"
            return report, 2

        # 一次取回最大的 ACL 授权候选池；校准器仅在 calibration 上做前缀重放。
        # test 结果虽同时执行以节省 embedding 启动成本，但其标签不进入选参函数。
        retriever = Retriever(
            store,
            embedder,
            top_k=raw_candidate_k,
            candidate_k=raw_candidate_k,
            context_k=raw_candidate_k,
            score_threshold=None,
        )
        raw_results, query_latency_ms = _run_all_queries(retriever, dataset.cases)
        calibration_cases = tuple(case for case in dataset.cases if case.split == "calibration")
        test_cases = tuple(case for case in dataset.cases if case.split == "test")
        calibration = calibrate_retrieval_configuration(
            calibration_cases,
            raw_results,
            candidate_k_values=candidate_k_values,
            context_k_values=context_k_values,
        )
        calibration_evaluation = evaluate_ranked_results(
            calibration_cases,
            raw_results,
            candidate_k=calibration.selected_candidate_k,
            context_k=calibration.selected_context_k,
            score_threshold=calibration.selected_threshold,
        )
        test_evaluation = evaluate_ranked_results(
            test_cases,
            raw_results,
            candidate_k=calibration.selected_candidate_k,
            context_k=calibration.selected_context_k,
            score_threshold=calibration.selected_threshold,
        )
        combined_evaluation = evaluate_ranked_results(
            dataset.cases,
            raw_results,
            candidate_k=calibration.selected_candidate_k,
            context_k=calibration.selected_context_k,
            score_threshold=calibration.selected_threshold,
        )
        # 用冻结配置真实再跑一次，校验离线前缀重放与生产 Retriever 一致，并单列
        # selected candidate_k 的端到端延迟。该次执行发生在选参之后，不反馈标签。
        selected_retriever = Retriever(
            store,
            embedder,
            candidate_k=calibration.selected_candidate_k,
            context_k=calibration.selected_context_k,
            score_threshold=calibration.selected_threshold,
        )
        selected_results, selected_query_latency_ms = _run_all_queries(
            selected_retriever, dataset.cases
        )
        expected_chunk_ids = {
            case.case_id: case.returned_chunk_ids for case in combined_evaluation.cases
        }
        replay_mismatches = sorted(
            case.case_id
            for case in dataset.cases
            if tuple(
                candidate.chunk_id
                for candidate in selected_results[case.case_id].candidates
            )
            != expected_chunk_ids[case.case_id]
        )
        replay_passed = not replay_mismatches
        test_gates = quality_gates(
            test_evaluation.metrics, k=calibration.selected_context_k
        )
        calibration_gates = quality_gates(
            calibration_evaluation.metrics, k=calibration.selected_context_k
        )
        calibration_passed = (
            calibration.all_quality_gates_passed_on_calibration
            and calibration_gates["passed"]
        )
        status, exit_code = _acceptance_outcome(
            calibration_passed=calibration_passed,
            test_passed=test_gates["passed"],
            replay_passed=replay_passed,
        )
        report.update({
            "status": status,
            "threshold_calibration": {
                "split": "calibration",
                "selected_candidate_k": calibration.selected_candidate_k,
                "selected_context_k": calibration.selected_context_k,
                "selected_score_threshold": calibration.selected_threshold,
                "k_pair_count": calibration.k_pair_count,
                "score_threshold_candidate_count": (
                    calibration.score_threshold_candidate_count
                ),
                "evaluated_configuration_count": (
                    calibration.evaluated_configuration_count
                ),
                "quality_gates": calibration.selected_gates,
                "all_quality_gates_passed": calibration.all_quality_gates_passed_on_calibration,
                "metrics": _metrics_payload(calibration.selected_metrics),
            },
            "calibration": {
                **serialize_evaluation(calibration_evaluation),
                "quality_gates": calibration_gates,
            },
            "test": {
                **serialize_evaluation(test_evaluation),
                "quality_gates": test_gates,
            },
            "all_cases": serialize_evaluation(combined_evaluation),
            "selected_configuration_replay": {
                "passed": replay_passed,
                "mismatched_case_ids": replay_mismatches,
                "labels_used_for_selection": False,
            },
            "costs": summarize_retrieval_costs(
                raw_results,
                combined_evaluation,
                raw_candidate_k=raw_candidate_k,
                selected_candidate_k=calibration.selected_candidate_k,
                selected_context_k=calibration.selected_context_k,
                query_latency_ms=query_latency_ms,
                selected_query_latency_ms=selected_query_latency_ms,
            ),
        })
        return report, exit_code
    except Exception as exc:
        report.update({
            "status": "run_failed",
            "error_code": type(exc).__name__,
        })
        return report, 3
    finally:
        cleanup_failed = False
        if collection_created:
            cleanup_failed = not _record_collection_cleanup(store, collection, report)
            if cleanup_failed:
                # 清理失败不能保留 passed；先落盘一个 Schema 合法的失败报告，
                # 再以非零异常终止，避免调用方误判验收成功。
                report["status"] = "run_failed"
                report["error_code"] = "collection_cleanup_failed"
        write_report(output_path, report, schema_path=report_schema_path)
        if cleanup_failed:
            raise RuntimeError("retrieval_acceptance_collection_cleanup_failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/benchmarks/retrieval_acceptance.yaml")
    )
    parser.add_argument(
        "--schema", type=Path,
        default=Path("schemas/retrieval-acceptance-dataset.schema.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/local/benchmarks/retrieval-acceptance-report.json"),
    )
    parser.add_argument(
        "--report-schema", type=Path,
        default=Path("schemas/retrieval-acceptance-report.schema.json"),
    )
    parser.add_argument("--collection-prefix", default="retrieval_acceptance")
    parser.add_argument(
        "--candidate-k-values",
        type=_parse_k_values,
        default=(5, 10),
        help="只在 calibration 扫描的候选池 K，逗号分隔（默认 5,10）",
    )
    parser.add_argument(
        "--context-k-values",
        type=_parse_k_values,
        default=(2, 3, 5),
        help="只在 calibration 扫描的上下文 K，逗号分隔（默认 2,3,5）",
    )
    args = parser.parse_args(argv)
    report, exit_code = run_acceptance(
        dataset_path=args.dataset,
        schema_path=args.schema,
        output_path=args.output,
        report_schema_path=args.report_schema,
        collection_prefix=args.collection_prefix,
        candidate_k_values=args.candidate_k_values,
        context_k_values=args.context_k_values,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
