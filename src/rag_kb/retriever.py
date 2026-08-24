"""授权检索(蓝图 7.1 检索基线 / 6.2 ACL 规则)。

ACL 是检索约束,不是提示词约束:每次 Qdrant 查询都强制
``status = active`` 且 ``allowed_roles`` 包含当前用户角色;不满足的块不进入
Top-K 候选,也不得作为上下文、引用或错误信息的一部分(蓝图 6.2)。
Qdrant 对数组字段的 match 语义为"任一元素相等即匹配",与 allowed_roles 设计对齐。
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client.http import models

from rag_kb.models import ChunkStatus, Role
from rag_kb.store import ChunkStore


@dataclass(frozen=True)
class RetrievalCandidate:
    """一个授权候选块:阶段 4 组装引用所需的全部来源字段(蓝图 7.3)。"""

    chunk_id: str
    document_id: str
    document_version: int
    title: str
    source_file_name: str
    source_locator: str
    score: float
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    """检索结果:最终上下文 + 候选池裁切统计。

    suppressed_count 用于区分蓝图 7.4 的两类拒答:
    - candidates 空且 suppressed_count 为 0 → 无任何授权候选(含无权限);
    - candidates 空但 suppressed_count > 0 → 有授权候选但相关性不足。

    candidate_count 是 Qdrant 在服务端 ACL 后实际返回的候选数；
    context_truncated_count 是通过阈值但因 context_k 未进入上下文的数量。
    新字段均有默认值，旧适配器的 ``RetrievalResult(candidates, count)`` 保持兼容。
    """

    candidates: list[RetrievalCandidate]
    suppressed_count: int
    candidate_count: int | None = None
    context_truncated_count: int = 0


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """真实 Retriever 可直接执行的冻结检索评测样本。

    relevant_chunk_ids 只列当前角色应当召回的 Gold 块；无答案或 ACL 拒答样本
    将其留空并设置 expected_no_answer。is_acl_filtered 用于单独统计经过 ACL
    过滤后仍应可见的正样本召回率。expected_document_versions 用于校验没有
    召回旧版本，元素格式为 ``(document_id, version)``。
    """

    case_id: str
    question: str
    role: Role
    relevant_chunk_ids: frozenset[str] = frozenset()
    expected_no_answer: bool = False
    is_acl_filtered: bool = False
    expected_document_versions: frozenset[tuple[str, int]] = frozenset()

    def __post_init__(self) -> None:
        if not self.case_id or not self.question:
            raise ValueError("evaluation_case_id_and_question_required")
        if self.expected_no_answer and self.relevant_chunk_ids:
            raise ValueError("no_answer_case_cannot_have_relevant_chunks")


@dataclass(frozen=True)
class RetrievalEvaluationMetrics:
    """检索评测聚合结果；无对应样本的指标为 None，而不是伪造 0。"""

    k: int
    case_count: int
    positive_case_count: int
    no_answer_case_count: int
    recall_at_k: float | None
    mean_reciprocal_rank: float | None
    ndcg_at_k: float | None
    acl_filtered_recall_at_k: float | None
    no_answer_accuracy: float | None
    correct_version_recall_at_k: float | None


class Retriever:
    """单一 dense 检索基线:候选池 + 阈值 + 上下文截断。

    执行顺序固定为 Qdrant 服务端 ACL → candidate_k 候选池 → score threshold
    → context_k 截断。``top_k`` 是兼容参数；不提供两个新参数时行为与旧版一致。
    """

    def __init__(
        self,
        store: ChunkStore,
        embedder,
        *,
        top_k: int = 5,
        candidate_k: int | None = None,
        context_k: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k if candidate_k is not None else top_k
        self.context_k = context_k if context_k is not None else top_k
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.candidate_k, self.context_k)
        ):
            raise ValueError("retrieval_limits_must_be_integers")
        if self.candidate_k <= 0 or self.context_k <= 0:
            raise ValueError("retrieval_limits_must_be_positive")
        if self.context_k > self.candidate_k:
            raise ValueError("retrieval_context_k_cannot_exceed_candidate_k")
        # 兼容少量读取 retriever.top_k 的旧调用方；它表示最终返回上限。
        self.top_k = self.context_k
        if score_threshold is not None and (
            isinstance(score_threshold, bool)
            or not isinstance(score_threshold, (int, float))
            or not math.isfinite(float(score_threshold))
        ):
            raise ValueError("retrieval_score_threshold_must_be_finite")
        self.score_threshold = (
            float(score_threshold) if score_threshold is not None else None
        )

    def search(self, question: str, role: Role) -> RetrievalResult:
        vector = self.embedder.embed([question])[0]
        return self._search_vector(vector, role)

    @staticmethod
    def _acl_filter(role: Role) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="status", match=models.MatchValue(value=str(ChunkStatus.ACTIVE))
                ),
                models.FieldCondition(
                    key="allowed_roles", match=models.MatchValue(value=str(role))
                ),
            ]
        )

    def _search_vector(self, vector: list[float], role: Role) -> RetrievalResult:
        # ACL 必须在 Qdrant 查询入口执行；不得先宽检索再由 Python/模型过滤。
        response = self.store.client.query_points(
            collection_name=self.store.collection,
            query=vector,
            query_filter=self._acl_filter(role),
            limit=self.candidate_k,
            with_payload=True,
            with_vectors=False,
        )
        candidate_pool = [self._to_candidate(point) for point in response.points]
        threshold_kept = candidate_pool
        if self.score_threshold is not None:
            threshold_kept = [
                candidate
                for candidate in candidate_pool
                if candidate.score >= self.score_threshold
            ]
        context = threshold_kept[: self.context_k]
        return RetrievalResult(
            candidates=context,
            suppressed_count=len(candidate_pool) - len(threshold_kept),
            candidate_count=len(candidate_pool),
            context_truncated_count=len(threshold_kept) - len(context),
        )

    @staticmethod
    def _to_candidate(point: models.ScoredPoint) -> RetrievalCandidate:
        payload = point.payload or {}
        return RetrievalCandidate(
            chunk_id=payload["chunk_id"],
            document_id=payload["document_id"],
            document_version=payload["document_version"],
            title=payload["title"],
            source_file_name=payload["source_file_name"],
            source_locator=payload["source_locator"],
            score=point.score,
            text=payload["text"],
        )


def evaluate_retriever(
    retriever: Retriever,
    cases: Sequence[RetrievalEvaluationCase],
    *,
    k: int,
) -> RetrievalEvaluationMetrics:
    """通过生产 Retriever 边界计算 Recall/MRR/nDCG/ACL/拒答/版本指标。

    本函数不替换真实 BGE-M3 或 Qdrant；调用方给它生产 Retriever 时，评测会
    完整经过 embedding、ACL filter、Top-K、阈值与 Qdrant 排序。测试中也可用
    冻结假 Retriever 验证指标公式，避免把 fixture alias 匹配误当成真实召回。
    """
    if k <= 0:
        raise ValueError("retrieval_evaluation_k_must_be_positive")
    if not cases:
        raise ValueError("retrieval_evaluation_cases_required")

    positive_count = 0
    no_answer_count = 0
    recall_hits = 0
    recall_total = 0
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    acl_hits = 0
    acl_total = 0
    no_answer_hits = 0
    version_hits = 0
    version_total = 0

    for case in cases:
        result = retriever.search(case.question, case.role)
        # Qdrant point ID 唯一，但评测层仍按 chunk_id 保留首个候选，避免异常适配器
        # 的重复项虚增 nDCG 或版本命中。
        ranked: list[RetrievalCandidate] = []
        seen_chunk_ids: set[str] = set()
        for candidate in result.candidates:
            if candidate.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(candidate.chunk_id)
            ranked.append(candidate)
            if len(ranked) == k:
                break

        if case.expected_no_answer:
            no_answer_count += 1
            no_answer_hits += int(not ranked)

        relevant = case.relevant_chunk_ids
        if relevant:
            positive_count += 1
            ranked_ids = [candidate.chunk_id for candidate in ranked]
            hit_count = len(set(ranked_ids) & relevant)
            recall_hits += hit_count
            recall_total += len(relevant)
            first_rank = next(
                (index for index, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in relevant),
                None,
            )
            reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
            dcg = sum(
                1.0 / math.log2(index + 1)
                for index, chunk_id in enumerate(ranked_ids, start=1)
                if chunk_id in relevant
            )
            ideal_hits = min(len(relevant), k)
            ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
            ndcg_scores.append(dcg / ideal_dcg)
            if case.is_acl_filtered:
                acl_hits += hit_count
                acl_total += len(relevant)

        expected_versions = case.expected_document_versions
        if expected_versions:
            returned_versions = {
                (candidate.document_id, candidate.document_version) for candidate in ranked
            }
            version_hits += len(returned_versions & expected_versions)
            version_total += len(expected_versions)

    return RetrievalEvaluationMetrics(
        k=k,
        case_count=len(cases),
        positive_case_count=positive_count,
        no_answer_case_count=no_answer_count,
        recall_at_k=recall_hits / recall_total if recall_total else None,
        mean_reciprocal_rank=sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks
        else None,
        ndcg_at_k=sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else None,
        acl_filtered_recall_at_k=acl_hits / acl_total if acl_total else None,
        no_answer_accuracy=no_answer_hits / no_answer_count if no_answer_count else None,
        correct_version_recall_at_k=version_hits / version_total if version_total else None,
    )
