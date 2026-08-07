"""授权检索(蓝图 7.1 检索基线 / 6.2 ACL 规则)。

ACL 是检索约束,不是提示词约束:每次 Qdrant 查询都强制
``status = active`` 且 ``allowed_roles`` 包含当前用户角色;不满足的块不进入
Top-K 候选,也不得作为上下文、引用或错误信息的一部分(蓝图 6.2)。
Qdrant 对数组字段的 match 语义为"任一元素相等即匹配",与 allowed_roles 设计对齐。
"""

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
    """检索结果:候选 + 被阈值裁掉的授权候选数。

    suppressed_count 用于区分蓝图 7.4 的两类拒答:
    - candidates 空且 suppressed_count 为 0 → 无任何授权候选(含无权限);
    - candidates 空但 suppressed_count > 0 → 有授权候选但相关性不足。
    """

    candidates: list[RetrievalCandidate]
    suppressed_count: int


class Retriever:
    """单一 dense 检索基线:Top-K + 可选相似度阈值(蓝图 7.1,阈值待冻结评测校准)。"""

    def __init__(
        self,
        store: ChunkStore,
        embedder,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold

    def search(self, question: str, role: Role) -> RetrievalResult:
        vector = self.embedder.embed([question])[0]
        acl_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="status", match=models.MatchValue(value=str(ChunkStatus.ACTIVE))
                ),
                models.FieldCondition(
                    key="allowed_roles", match=models.MatchValue(value=str(role))
                ),
            ]
        )
        # 先按 Top-K 取回(不带阈值),再在内存中按阈值裁切,以统计被裁数量。
        response = self.store.client.query_points(
            collection_name=self.store.collection,
            query=vector,
            query_filter=acl_filter,
            limit=self.top_k,
            with_payload=True,
            with_vectors=False,
        )
        candidates = [self._to_candidate(point) for point in response.points]
        if self.score_threshold is not None:
            kept = [c for c in candidates if c.score >= self.score_threshold]
            return RetrievalResult(candidates=kept, suppressed_count=len(candidates) - len(kept))
        return RetrievalResult(candidates=candidates, suppressed_count=0)

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
