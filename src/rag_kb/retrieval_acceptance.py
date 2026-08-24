"""真实 BGE-M3 + Qdrant 检索验收集、指标与可复现报告。

该模块只描述评测契约与计算指标；生产运行仍复用 ``Retriever`` 的服务端
``status=active AND allowed_roles contains role`` 过滤条件。它刻意不包含关键词
alias 逻辑、LLM 或 BM25/reranker，防止离线 fixture 成绩被误写成真实 dense 检索成绩。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from rag_kb.models import ChunkStatus, Role
from rag_kb.retriever import RetrievalCandidate, RetrievalResult

_REQUIRED_CATEGORY_MINIMUMS = {
    "semantic": 25,
    "exact_term": 10,
    "acl": 10,
    "no_answer": 10,
    "multi_chunk": 5,
}
_ACCEPTANCE_THRESHOLDS = {
    "recall_at_k": 0.90,
    "mrr": 0.80,
    "ndcg_at_k": 0.80,
    "correct_version_recall_at_k": 0.95,
    "exact_term_recall_at_k": 0.90,
    "acl_filtered_recall_at_k": 0.90,
    "no_answer_accuracy": 0.95,
    "acl_leak_rate": 0.0,
}


@dataclass(frozen=True)
class AcceptanceCase:
    """一个冻结、可追溯的真实检索评测样本。"""

    case_id: str
    split: str
    category: str
    question: str
    role: Role
    answerable: bool
    gold_chunk_ids: frozenset[str]
    gold_versions: frozenset[tuple[str, int]]
    exact_terms: tuple[str, ...]
    acl_filtered: bool
    forbidden_document_ids: frozenset[str]
    no_answer_kind: str | None = None


@dataclass(frozen=True)
class AcceptanceDataset:
    """验收数据集元数据与解析后的样本。"""

    path: Path
    version: int
    name: str
    description: str
    corpus_manifest: str
    required_top_k: int
    cases: tuple[AcceptanceCase, ...]

    @property
    def category_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(case.category for case in self.cases).items()))

    @property
    def split_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(case.split for case in self.cases).items()))


@dataclass(frozen=True)
class RetrievalAcceptanceMetrics:
    """冻结验收指标；None 表示数据集缺少该类样本，不伪造为零。"""

    case_count: int
    positive_case_count: int
    no_answer_case_count: int
    exact_term_case_count: int
    acl_positive_case_count: int
    acl_denied_case_count: int
    recall_at_k: float | None
    mrr: float | None
    ndcg_at_k: float | None
    correct_version_recall_at_k: float | None
    exact_term_recall_at_k: float | None
    acl_filtered_recall_at_k: float | None
    no_answer_accuracy: float | None
    acl_leak_rate: float | None
    acl_leak_count: int
    acl_leak_case_count: int


@dataclass(frozen=True)
class CaseEvaluation:
    """报告中的单 case 诊断数据；永远不包含证据正文。"""

    case_id: str
    split: str
    category: str
    answerable: bool
    returned_chunk_ids: tuple[str, ...]
    returned_scores: tuple[float, ...]
    first_gold_rank: int | None
    gold_hit_count: int
    no_answer_correct: bool | None
    acl_leak_document_ids: tuple[str, ...]
    candidate_count: int
    threshold_suppressed_count: int
    context_truncated_count: int


@dataclass(frozen=True)
class EvaluationResult:
    metrics: RetrievalAcceptanceMetrics
    cases: tuple[CaseEvaluation, ...]


@dataclass(frozen=True)
class ThresholdCalibration:
    """只使用 calibration split 选择相似度阈值的可审计结果。"""

    selected_threshold: float | None
    selected_metrics: RetrievalAcceptanceMetrics
    selected_gates: dict[str, bool]
    candidate_count: int
    all_quality_gates_passed_on_calibration: bool


@dataclass(frozen=True)
class RetrievalConfigurationCalibration:
    """仅由 calibration split 共同选择 candidate/context K 与阈值。"""

    selected_candidate_k: int
    selected_context_k: int
    selected_threshold: float | None
    selected_metrics: RetrievalAcceptanceMetrics
    selected_gates: dict[str, bool]
    k_pair_count: int
    score_threshold_candidate_count: int
    evaluated_configuration_count: int
    all_quality_gates_passed_on_calibration: bool


def load_acceptance_dataset(dataset_path: Path, schema_path: Path) -> AcceptanceDataset:
    """读取并验证 YAML 与 JSON Schema，再校验跨字段的评测语义。"""
    raw = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(raw)

    cases = tuple(_parse_case(item) for item in raw["cases"])
    dataset = AcceptanceDataset(
        path=dataset_path,
        version=raw["version"],
        name=raw["name"],
        description=raw["description"],
        corpus_manifest=raw["corpus_manifest"],
        required_top_k=raw["required_top_k"],
        cases=cases,
    )
    validate_dataset_semantics(dataset)
    return dataset


def _parse_case(raw: Mapping[str, Any]) -> AcceptanceCase:
    return AcceptanceCase(
        case_id=raw["case_id"],
        split=raw["split"],
        category=raw["category"],
        question=raw["query"],
        role=Role(raw["role"]),
        answerable=raw["answerable"],
        gold_chunk_ids=frozenset(raw["gold_chunk_ids"]),
        gold_versions=frozenset(
            (item["document_id"], item["version"]) for item in raw["gold_versions"]
        ),
        exact_terms=tuple(raw["exact_terms"]),
        acl_filtered=raw["acl_filtered"],
        forbidden_document_ids=frozenset(raw["forbidden_document_ids"]),
        no_answer_kind=raw.get("no_answer_kind"),
    )


def validate_dataset_semantics(dataset: AcceptanceDataset) -> None:
    """补足 JSON Schema 难以表达的分层、唯一性与 Gold 一致性约束。"""
    minimum_cases = 60 if dataset.version == 1 else 20
    if len(dataset.cases) < minimum_cases:
        raise ValueError(
            f"retrieval_acceptance_dataset_requires_at_least_{minimum_cases}_cases"
        )
    case_ids = [case.case_id for case in dataset.cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("retrieval_acceptance_dataset_case_ids_must_be_unique")
    category_minimums = (
        _REQUIRED_CATEGORY_MINIMUMS
        if dataset.version == 1
        else {category: 2 for category in _REQUIRED_CATEGORY_MINIMUMS}
    )
    for category, minimum in category_minimums.items():
        actual = dataset.category_counts.get(category, 0)
        if actual < minimum:
            raise ValueError(
                f"retrieval_acceptance_dataset_category_too_small:{category}:{actual}<{minimum}"
            )
    if not dataset.split_counts.get("calibration") or not dataset.split_counts.get("test"):
        raise ValueError("retrieval_acceptance_dataset_requires_calibration_and_test_splits")
    if dataset.version >= 2 and not dataset.split_counts.get("development"):
        raise ValueError("retrieval_acceptance_dataset_requires_development_split")

    for case in dataset.cases:
        if case.answerable:
            if not case.gold_chunk_ids or not case.gold_versions:
                raise ValueError(f"answerable_case_requires_gold:{case.case_id}")
            parsed_versions = {
                _chunk_document_version(chunk_id) for chunk_id in case.gold_chunk_ids
            }
            if not parsed_versions.issubset(case.gold_versions):
                raise ValueError(f"gold_chunk_version_not_declared:{case.case_id}")
        elif case.gold_chunk_ids or case.gold_versions:
            raise ValueError(f"no_answer_case_cannot_have_gold:{case.case_id}")
        if case.category == "exact_term" and not case.exact_terms:
            raise ValueError(f"exact_term_case_requires_terms:{case.case_id}")
        if case.category == "multi_chunk" and len(case.gold_chunk_ids) < 2:
            raise ValueError(f"multi_chunk_case_requires_multiple_gold_chunks:{case.case_id}")
        if case.category == "acl" and not case.acl_filtered:
            raise ValueError(f"acl_category_requires_acl_filter_marker:{case.case_id}")
        if case.no_answer_kind == "acl_denied" and not case.forbidden_document_ids:
            raise ValueError(f"acl_denied_case_requires_forbidden_documents:{case.case_id}")


def validate_gold_labels_against_ingested_corpus(
    dataset: AcceptanceDataset,
    store,
) -> dict[str, Any]:
    """在真实摄取后校验 Gold 块、版本、角色和精确术语均有可追溯依据。

    这一步只读取本次隔离 collection 的 payload，不会把正文写入长期报告。
    """
    by_document: dict[str, list[dict[str, Any]]] = {}
    document_ids = {
        document_id
        for case in dataset.cases
        for document_id, _version in case.gold_versions
    } | {
        document_id for case in dataset.cases for document_id in case.forbidden_document_ids
    }
    for document_id in document_ids:
        by_document[document_id] = store.scroll_document(document_id)

    active_by_chunk: dict[str, dict[str, Any]] = {}
    all_by_chunk: dict[str, dict[str, Any]] = {}
    for payloads in by_document.values():
        for payload in payloads:
            chunk_id = str(payload.get("chunk_id", ""))
            all_by_chunk[chunk_id] = payload
            if payload.get("status") == str(ChunkStatus.ACTIVE):
                active_by_chunk[chunk_id] = payload

    errors: list[str] = []
    for case in dataset.cases:
        gold_payloads: list[dict[str, Any]] = []
        for chunk_id in case.gold_chunk_ids:
            payload = active_by_chunk.get(chunk_id)
            if payload is None:
                errors.append(f"gold_chunk_not_active:{case.case_id}:{chunk_id}")
                continue
            if str(case.role) not in {str(role) for role in payload.get("allowed_roles", [])}:
                errors.append(f"gold_chunk_not_authorized_for_role:{case.case_id}:{chunk_id}")
            gold_payloads.append(payload)
        if case.exact_terms and gold_payloads:
            normalized_text = _normalize_text_for_term_match(
                "\n".join(str(payload.get("text", "")) for payload in gold_payloads)
            )
            for term in case.exact_terms:
                if _normalize_text_for_term_match(term) not in normalized_text:
                    errors.append(f"exact_term_not_grounded:{case.case_id}:{term}")
        if case.no_answer_kind == "acl_denied":
            for forbidden_document_id in case.forbidden_document_ids:
                payloads = by_document.get(forbidden_document_id, [])
                if not payloads:
                    errors.append(
                        f"forbidden_document_missing_from_corpus:{case.case_id}:{forbidden_document_id}"
                    )
                    continue
                if any(
                    payload.get("status") == str(ChunkStatus.ACTIVE)
                    and str(case.role) in {str(role) for role in payload.get("allowed_roles", [])}
                    for payload in payloads
                ):
                    errors.append(
                        f"acl_denied_case_has_authorized_forbidden_document:{case.case_id}:{forbidden_document_id}"
                    )
    return {
        "passed": not errors,
        "active_chunk_count": len(active_by_chunk),
        "ingested_chunk_count": len(all_by_chunk),
        "error_count": len(errors),
        "errors": errors,
    }


def evaluate_ranked_results(
    cases: Iterable[AcceptanceCase],
    results_by_case_id: Mapping[str, RetrievalResult],
    *,
    top_k: int | None = None,
    candidate_k: int | None = None,
    context_k: int | None = None,
    score_threshold: float | None,
) -> EvaluationResult:
    """从真实候选池计算指标，不执行任何 fixture alias 匹配。

    兼容调用方可继续只传 ``top_k``。新调用按 candidate_k 截取一次原始池，
    再应用 score threshold，最后按 context_k 截断；这与生产 Retriever 顺序一致。
    """
    candidate_k, context_k = _resolve_evaluation_limits(
        top_k=top_k, candidate_k=candidate_k, context_k=context_k
    )
    if score_threshold is not None and (
        isinstance(score_threshold, bool)
        or not isinstance(score_threshold, (int, float))
        or not math.isfinite(float(score_threshold))
    ):
        raise ValueError("retrieval_acceptance_threshold_must_be_finite")
    selected_cases = tuple(cases)
    if not selected_cases:
        raise ValueError("retrieval_acceptance_cases_required")

    positive_case_count = 0
    no_answer_case_count = 0
    exact_term_case_count = 0
    acl_positive_case_count = 0
    acl_denied_case_count = 0
    recall_hits = 0
    recall_total = 0
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    version_hits = 0
    version_total = 0
    exact_term_hits = 0
    acl_hits = 0
    acl_total = 0
    no_answer_hits = 0
    acl_leak_count = 0
    acl_leak_case_count = 0
    case_evaluations: list[CaseEvaluation] = []

    for case in selected_cases:
        try:
            raw_result = results_by_case_id[case.case_id]
        except KeyError as exc:
            raise ValueError(f"missing_retrieval_result:{case.case_id}") from exc
        candidate_pool, threshold_kept, ranked = _select_candidates(
            raw_result.candidates,
            candidate_k=candidate_k,
            context_k=context_k,
            score_threshold=score_threshold,
        )
        ranked_ids = [candidate.chunk_id for candidate in ranked]
        returned_versions = {
            (candidate.document_id, candidate.document_version) for candidate in ranked
        }
        first_gold_rank: int | None = None
        gold_hit_count = 0
        if case.answerable:
            positive_case_count += 1
            gold_hit_count = len(set(ranked_ids) & case.gold_chunk_ids)
            recall_hits += gold_hit_count
            recall_total += len(case.gold_chunk_ids)
            first_gold_rank = next(
                (
                    rank
                    for rank, chunk_id in enumerate(ranked_ids, start=1)
                    if chunk_id in case.gold_chunk_ids
                ),
                None,
            )
            reciprocal_ranks.append(0.0 if first_gold_rank is None else 1.0 / first_gold_rank)
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, chunk_id in enumerate(ranked_ids, start=1)
                if chunk_id in case.gold_chunk_ids
            )
            ideal_dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, min(context_k, len(case.gold_chunk_ids)) + 1)
            )
            ndcg_scores.append(dcg / ideal_dcg)
            version_hits += len(returned_versions & case.gold_versions)
            version_total += len(case.gold_versions)
            if case.category == "exact_term":
                exact_term_case_count += 1
                exact_term_hits += int(gold_hit_count > 0)
            if case.acl_filtered:
                acl_positive_case_count += 1
                acl_hits += gold_hit_count
                acl_total += len(case.gold_chunk_ids)
        else:
            # 分层考核口径：拒答准确率只统计 out_of_scope 负样本——它们的确定性
            # 拦截（数量/锚点）仍在检索层生效。acl_denied 负样本考核的是"不许把
            # 无权限文档检索出来"，由 acl_leak_rate 门承担；其最终拒答由回答
            # 模型 verdict 层完成，不属于检索层评分范围。
            if case.no_answer_kind != "acl_denied":
                no_answer_case_count += 1
                no_answer_hits += int(not ranked)

        leaked_document_ids = tuple(
            sorted(
                {
                    candidate.document_id
                    for candidate in ranked
                    if candidate.document_id in case.forbidden_document_ids
                }
            )
        )
        if case.no_answer_kind == "acl_denied":
            acl_denied_case_count += 1
            if leaked_document_ids:
                acl_leak_case_count += 1
                acl_leak_count += len(leaked_document_ids)

        case_evaluations.append(
            CaseEvaluation(
                case_id=case.case_id,
                split=case.split,
                category=case.category,
                answerable=case.answerable,
                returned_chunk_ids=tuple(ranked_ids),
                returned_scores=tuple(round(candidate.score, 8) for candidate in ranked),
                first_gold_rank=first_gold_rank,
                gold_hit_count=gold_hit_count,
                no_answer_correct=(
                    # acl_denied 样本不在检索层考核拒答正确性（见计数注释），置 None
                    (not ranked)
                    if not case.answerable and case.no_answer_kind != "acl_denied"
                    else None
                ),
                acl_leak_document_ids=leaked_document_ids,
                candidate_count=len(candidate_pool),
                threshold_suppressed_count=len(candidate_pool) - len(threshold_kept),
                context_truncated_count=len(threshold_kept) - len(ranked),
            )
        )

    metrics = RetrievalAcceptanceMetrics(
        case_count=len(selected_cases),
        positive_case_count=positive_case_count,
        no_answer_case_count=no_answer_case_count,
        exact_term_case_count=exact_term_case_count,
        acl_positive_case_count=acl_positive_case_count,
        acl_denied_case_count=acl_denied_case_count,
        recall_at_k=_ratio(recall_hits, recall_total),
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
        ndcg_at_k=sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else None,
        correct_version_recall_at_k=_ratio(version_hits, version_total),
        exact_term_recall_at_k=_ratio(exact_term_hits, exact_term_case_count),
        acl_filtered_recall_at_k=_ratio(acl_hits, acl_total),
        no_answer_accuracy=_ratio(no_answer_hits, no_answer_case_count),
        acl_leak_rate=_ratio(acl_leak_case_count, acl_denied_case_count),
        acl_leak_count=acl_leak_count,
        acl_leak_case_count=acl_leak_case_count,
    )
    return EvaluationResult(metrics=metrics, cases=tuple(case_evaluations))


def quality_gates(
    metrics: RetrievalAcceptanceMetrics, *, k: int = 5
) -> dict[str, bool]:
    """按应届生项目验收线生成明确的检索质量/安全门。"""
    if k <= 0:
        raise ValueError("retrieval_acceptance_gate_k_must_be_positive")
    gates = {
        f"recall_at_{k}": _meets(metrics.recall_at_k, _ACCEPTANCE_THRESHOLDS["recall_at_k"]),
        "mrr": _meets(metrics.mrr, _ACCEPTANCE_THRESHOLDS["mrr"]),
        f"ndcg_at_{k}": _meets(metrics.ndcg_at_k, _ACCEPTANCE_THRESHOLDS["ndcg_at_k"]),
        "correct_version_recall": _meets(
            metrics.correct_version_recall_at_k,
            _ACCEPTANCE_THRESHOLDS["correct_version_recall_at_k"],
        ),
        "exact_term_recall": _meets(
            metrics.exact_term_recall_at_k,
            _ACCEPTANCE_THRESHOLDS["exact_term_recall_at_k"],
        ),
        "acl_filtered_recall": _meets(
            metrics.acl_filtered_recall_at_k,
            _ACCEPTANCE_THRESHOLDS["acl_filtered_recall_at_k"],
        ),
        "no_answer_accuracy": _meets(
            metrics.no_answer_accuracy,
            _ACCEPTANCE_THRESHOLDS["no_answer_accuracy"],
        ),
        "acl_leak_rate": metrics.acl_leak_rate == _ACCEPTANCE_THRESHOLDS["acl_leak_rate"],
    }
    return {**gates, "passed": all(gates.values())}


def calibrate_score_threshold(
    cases: Iterable[AcceptanceCase],
    results_by_case_id: Mapping[str, RetrievalResult],
    *,
    top_k: int,
) -> ThresholdCalibration:
    """兼容入口：在 candidate_k=context_k=top_k 时仅选择阈值。

    新代码应优先调用 :func:`calibrate_retrieval_configuration` 联合选择参数。
    """
    selected = calibrate_retrieval_configuration(
        cases,
        results_by_case_id,
        candidate_k_values=(top_k,),
        context_k_values=(top_k,),
    )
    return ThresholdCalibration(
        selected_threshold=selected.selected_threshold,
        selected_metrics=selected.selected_metrics,
        selected_gates=selected.selected_gates,
        candidate_count=selected.score_threshold_candidate_count,
        all_quality_gates_passed_on_calibration=(
            selected.all_quality_gates_passed_on_calibration
        ),
    )


def calibrate_retrieval_configuration(
    cases: Iterable[AcceptanceCase],
    results_by_case_id: Mapping[str, RetrievalResult],
    *,
    candidate_k_values: Sequence[int],
    context_k_values: Sequence[int],
) -> RetrievalConfigurationCalibration:
    """只在 calibration 标签上联合扫描 candidate_k/context_k/threshold。

    调用方应先以 ``max(candidate_k_values)`` 真实检索一次原始候选池。本函数只做
    前缀重放，不重复 embedding/Qdrant，也不读取 test case 标签。通过门和质量指标
    优先；完全同质时才选择更小 context_k、candidate_k 与更保守的低阈值。
    """
    calibration_cases = tuple(cases)
    if not calibration_cases:
        raise ValueError("retrieval_acceptance_calibration_cases_required")
    if any(case.split != "calibration" for case in calibration_cases):
        raise ValueError("retrieval_calibration_must_only_receive_calibration_split")
    candidate_values = _positive_unique_ints(
        candidate_k_values, error_code="retrieval_candidate_k_values_required"
    )
    context_values = _positive_unique_ints(
        context_k_values, error_code="retrieval_context_k_values_required"
    )
    pairs = tuple(
        (candidate_k, context_k)
        for candidate_k in candidate_values
        for context_k in context_values
        if context_k <= candidate_k
    )
    if not pairs:
        raise ValueError("retrieval_calibration_requires_context_not_above_candidate")

    scored: list[
        tuple[
            tuple[float, ...],
            int,
            int,
            float | None,
            EvaluationResult,
            dict[str, bool],
            int,
        ]
    ] = []
    evaluated_configuration_count = 0
    for candidate_k, context_k in pairs:
        thresholds = _score_threshold_candidates(
            calibration_cases, results_by_case_id, candidate_k=candidate_k
        )
        for threshold in thresholds:
            result = evaluate_ranked_results(
                calibration_cases,
                results_by_case_id,
                candidate_k=candidate_k,
                context_k=context_k,
                score_threshold=threshold,
            )
            gates = quality_gates(result.metrics, k=context_k)
            evaluated_configuration_count += 1
            scored.append(
                (
                    _configuration_objective(
                        result.metrics,
                        gates,
                        candidate_k=candidate_k,
                        context_k=context_k,
                        threshold=threshold,
                    ),
                    candidate_k,
                    context_k,
                    threshold,
                    result,
                    gates,
                    len(thresholds),
                )
            )
    (
        _objective,
        candidate_k,
        context_k,
        threshold,
        result,
        gates,
        threshold_candidate_count,
    ) = max(scored, key=lambda item: item[0])
    return RetrievalConfigurationCalibration(
        selected_candidate_k=candidate_k,
        selected_context_k=context_k,
        selected_threshold=threshold,
        selected_metrics=result.metrics,
        selected_gates=gates,
        k_pair_count=len(pairs),
        score_threshold_candidate_count=threshold_candidate_count,
        evaluated_configuration_count=evaluated_configuration_count,
        all_quality_gates_passed_on_calibration=gates["passed"],
    )


def serialize_evaluation(result: EvaluationResult) -> dict[str, Any]:
    """转换成稳定 JSON 数据；每 case 只保存 ID、排名和分数。"""
    cases: list[dict[str, Any]] = []
    for case in result.cases:
        payload = asdict(case)
        # dataclasses.asdict 会保留 tuple；报告契约是 JSON array，必须在
        # Schema 校验前显式转换，避免真实 runner 到最终落盘才失败。
        for name in ("returned_chunk_ids", "returned_scores", "acl_leak_document_ids"):
            payload[name] = list(payload[name])
        cases.append(payload)
    return {
        "metrics": asdict(result.metrics),
        "cases": cases,
    }


def make_report_header(
    dataset: AcceptanceDataset,
    *,
    collection: str,
    qdrant_url: str,
    model_name: str,
    batch_size: int,
) -> dict[str, Any]:
    """创建报告固定顶层字段，供 CLI 在清理后补写结果。"""
    return {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "backend": {
            "mode": "real_bge_m3_qdrant",
            "model": model_name,
            "qdrant_url": qdrant_url,
            "isolated_collection": collection,
            "embedding_batch_size": batch_size,
            "acl_filter": "status=active AND allowed_roles contains role",
            "retrieval_order": (
                "qdrant_server_acl -> candidate_k -> score_threshold -> context_k"
            ),
        },
        "dataset": {
            "path": str(dataset.path),
            "name": dataset.name,
            "total_cases": len(dataset.cases),
            "split_counts": dataset.split_counts,
            "category_counts": dataset.category_counts,
            "required_top_k": dataset.required_top_k,
            "frozen_baseline_context_k": dataset.required_top_k,
        },
        "thresholds": _ACCEPTANCE_THRESHOLDS,
    }


def summarize_retrieval_costs(
    results_by_case_id: Mapping[str, RetrievalResult],
    evaluation: EvaluationResult,
    *,
    raw_candidate_k: int,
    selected_candidate_k: int,
    selected_context_k: int,
    query_latency_ms: Mapping[str, float],
    selected_query_latency_ms: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """汇总候选/上下文与端到端检索延迟；不保存查询或证据正文。"""
    if raw_candidate_k < selected_candidate_k or selected_context_k > selected_candidate_k:
        raise ValueError("retrieval_cost_summary_invalid_k_relationship")
    case_ids = [case.case_id for case in evaluation.cases]
    if set(case_ids) - set(results_by_case_id):
        raise ValueError("retrieval_cost_summary_missing_results")
    if set(case_ids) - set(query_latency_ms):
        raise ValueError("retrieval_cost_summary_missing_latency")
    if selected_query_latency_ms is not None and set(case_ids) - set(
        selected_query_latency_ms
    ):
        raise ValueError("retrieval_cost_summary_missing_selected_latency")

    raw_counts = [
        result.candidate_count
        if result.candidate_count is not None
        else len(result.candidates)
        for case_id in case_ids
        for result in (results_by_case_id[case_id],)
    ]
    selected_candidate_counts = [case.candidate_count for case in evaluation.cases]
    context_counts = [len(case.returned_chunk_ids) for case in evaluation.cases]
    threshold_suppressed = [case.threshold_suppressed_count for case in evaluation.cases]
    context_truncated = [case.context_truncated_count for case in evaluation.cases]
    latencies = [float(query_latency_ms[case_id]) for case_id in case_ids]
    candidate_text_by_case = {
        case_id: {
            candidate.chunk_id: candidate.text
            for candidate in results_by_case_id[case_id].candidates
        }
        for case_id in case_ids
    }
    context_characters = [
        sum(
            len(candidate_text_by_case[case.case_id].get(chunk_id, ""))
            for chunk_id in case.returned_chunk_ids
        )
        for case in evaluation.cases
    ]
    payload: dict[str, Any] = {
        "scope": "raw_calibration_pool_plus_optional_selected_configuration_replay",
        "raw_candidate_pool": {
            "configured_k": raw_candidate_k,
            **_count_summary(raw_counts),
        },
        "selected_candidate_pool": {
            "configured_k": selected_candidate_k,
            **_count_summary(selected_candidate_counts),
        },
        "selected_context": {
            "configured_k": selected_context_k,
            **_count_summary(context_counts),
            "character_count": _count_summary(context_characters),
            "threshold_suppressed_total": sum(threshold_suppressed),
            "context_truncated_total": sum(context_truncated),
        },
        "query_latency_ms": {
            "scope": "raw_candidate_k_embedding_plus_acl_filtered_qdrant_search",
            "configured_candidate_k": raw_candidate_k,
            **_latency_summary(latencies),
        },
    }
    if selected_query_latency_ms is not None:
        payload["selected_configuration_query_latency_ms"] = {
            "scope": "selected_candidate_k_embedding_plus_acl_filtered_qdrant_search",
            "configured_candidate_k": selected_candidate_k,
            **_latency_summary(
                [float(selected_query_latency_ms[case_id]) for case_id in case_ids]
            ),
        }
    return payload


def validate_retrieval_acceptance_report(
    report: Mapping[str, Any], schema_path: Path
) -> None:
    """在落盘前校验机器报告；Schema 不允许查询或证据正文字段。"""
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)


def write_report(
    output_path: Path,
    report: Mapping[str, Any],
    *,
    schema_path: Path | None = None,
) -> None:
    if schema_path is not None:
        validate_retrieval_acceptance_report(report, schema_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _chunk_document_version(chunk_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"(?P<document_id>[a-z0-9-]+):v(?P<version>[1-9][0-9]*):chunk-[0-9]{2,}", chunk_id)
    if match is None:
        raise ValueError(f"invalid_gold_chunk_id:{chunk_id}")
    return match.group("document_id"), int(match.group("version"))


def _normalize_text_for_term_match(text: str) -> str:
    return re.sub(r"[\s（）()]", "", text).lower()


def _resolve_evaluation_limits(
    *, top_k: int | None, candidate_k: int | None, context_k: int | None
) -> tuple[int, int]:
    if top_k is None and candidate_k is None and context_k is None:
        raise ValueError("retrieval_acceptance_k_required")
    resolved_candidate_k = (
        candidate_k
        if candidate_k is not None
        else top_k if top_k is not None else context_k
    )
    resolved_context_k = (
        context_k
        if context_k is not None
        else top_k if top_k is not None else candidate_k
    )
    assert resolved_candidate_k is not None and resolved_context_k is not None
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (resolved_candidate_k, resolved_context_k)
    ):
        raise ValueError("retrieval_acceptance_k_must_be_integers")
    if resolved_candidate_k <= 0 or resolved_context_k <= 0:
        raise ValueError("retrieval_acceptance_k_must_be_positive")
    if resolved_context_k > resolved_candidate_k:
        raise ValueError("retrieval_acceptance_context_k_cannot_exceed_candidate_k")
    return resolved_candidate_k, resolved_context_k


def _select_candidates(
    candidates: Sequence[RetrievalCandidate],
    *,
    candidate_k: int,
    context_k: int,
    score_threshold: float | None,
) -> tuple[list[RetrievalCandidate], list[RetrievalCandidate], list[RetrievalCandidate]]:
    """按生产顺序返回候选池、阈值通过集和最终上下文。"""
    candidate_pool: list[RetrievalCandidate] = []
    seen_chunk_ids: set[str] = set()
    # candidate_k 是 Qdrant 原始排名前缀；不能因异常重复 ID 向更深排名补位。
    for candidate in candidates[:candidate_k]:
        if candidate.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(candidate.chunk_id)
        candidate_pool.append(candidate)
    threshold_kept = [
        candidate
        for candidate in candidate_pool
        if score_threshold is None or candidate.score >= score_threshold
    ]
    return candidate_pool, threshold_kept, threshold_kept[:context_k]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _meets(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _score_threshold_candidates(
    cases: Sequence[AcceptanceCase],
    results_by_case_id: Mapping[str, RetrievalResult],
    *,
    candidate_k: int,
) -> tuple[float | None, ...]:
    """返回能改变 ``score >= threshold`` 过滤结果的确定性边界。"""
    scores = {
        candidate.score
        for case in cases
        for candidate in results_by_case_id[case.case_id].candidates[:candidate_k]
        if math.isfinite(candidate.score)
    }
    # None 表示生产 Retriever 不做阈值裁切；nextafter(score, +∞) 则排除该分数。
    boundaries = {None}
    boundaries.update(math.nextafter(score, math.inf) for score in scores)
    return tuple(sorted((value for value in boundaries if value is not None), reverse=True)) + (
        (None,) if None in boundaries else ()
    )


def _configuration_objective(
    metrics: RetrievalAcceptanceMetrics,
    gates: Mapping[str, bool],
    *,
    candidate_k: int,
    context_k: int,
    threshold: float | None,
) -> tuple[float, ...]:
    """先守 ACL 和全部质量门；过门后优先选择成本更低的配置。

    质量门是可接受性的硬边界，而不是用更多上下文换取极小离线分数增益的
    排名特征。只有在上下文/候选规模相同的情况下，质量余量才参与破同分。
    没有配置通过全部门时仍返回最安全、诊断价值最高的一项，但调用方必须把
    整次验收判为失败。
    """
    acl_safe = float(bool(gates.get("acl_leak_rate", False)))
    quality = _quality_objective(metrics, gates)
    if gates.get("passed", False):
        return (
            acl_safe,
            1.0,
            -float(context_k),
            -float(candidate_k),
            *quality,
            _threshold_preference(threshold),
        )
    return (
        acl_safe,
        0.0,
        *quality,
        -float(context_k),
        -float(candidate_k),
        _threshold_preference(threshold),
    )


def _quality_objective(
    metrics: RetrievalAcceptanceMetrics, gates: Mapping[str, bool]
) -> tuple[float, ...]:
    gate_count = sum(value for name, value in gates.items() if name != "passed")
    return (
        float(gate_count),
        float(metrics.recall_at_k or 0.0),
        float(metrics.mrr or 0.0),
        float(metrics.ndcg_at_k or 0.0),
        float(metrics.no_answer_accuracy or 0.0),
        float(metrics.correct_version_recall_at_k or 0.0),
        float(metrics.exact_term_recall_at_k or 0.0),
        float(metrics.acl_filtered_recall_at_k or 0.0),
    )


def _threshold_preference(threshold: float | None) -> float:
    # None 视为最低阈值；同分时保留更多授权候选，减少无依据拒答。
    return -(threshold if threshold is not None else float("-inf"))


def _positive_unique_ints(values: Sequence[int], *, error_code: str) -> tuple[int, ...]:
    normalized = tuple(sorted(set(values)))
    if not normalized or any(isinstance(value, bool) or value <= 0 for value in normalized):
        raise ValueError(error_code)
    return normalized


def _count_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    return {
        "sample_count": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _latency_summary(values: Sequence[float]) -> dict[str, int | float | None]:
    return {
        "sample_count": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _percentile(values: Sequence[int | float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
