"""Qdrant 存储封装(蓝图 5.1:写入能力只由摄取路径使用)。

设计决策(蓝图 3.3 取舍原则):不设独立文档注册表——版本与生命周期状态只存于
块 payload,文档级查询通过 scroll 聚合。单一真相源,无双写不一致;规模增长后
再按第 15 节触发器拆分注册表。
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_kb.models import ChunkRecord, ChunkStatus, Role, TodoCitation

_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-enterprise-kb")
_PAYLOAD_INDEXES = (
    ("document_id", models.PayloadSchemaType.KEYWORD),
    ("status", models.PayloadSchemaType.KEYWORD),
    ("allowed_roles", models.PayloadSchemaType.KEYWORD),
    ("document_version", models.PayloadSchemaType.INTEGER),
)


class ChunkStore:
    def __init__(self, url: str, collection: str) -> None:
        self.client = QdrantClient(url=url, timeout=10)
        self.collection = collection

    @staticmethod
    def point_id(chunk_id: str) -> str:
        """chunk_id → 确定性 UUID5:重复摄取幂等(同一块覆盖同一向量点)。"""
        return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))

    def ensure_collection(self, dim: int) -> None:
        """创建 collection 并为 ACL/生命周期过滤字段建 payload 索引(免全表扫描)。"""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
        payload_schema = self.client.get_collection(self.collection).payload_schema or {}
        for field, schema in _PAYLOAD_INDEXES:
            current = payload_schema.get(field)
            if current is None or current.data_type != schema:
                self.client.create_payload_index(self.collection, field, schema)

    def upsert_chunks(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(id=self.point_id(chunk.chunk_id), vector=vector, payload=chunk.to_payload())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def scroll_document(self, document_id: str) -> list[dict]:
        """取一个文档全部版本的块 payload(MVP 单页假设:单文档块数远小于 limit)。"""
        results, _offset = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            ),
            with_payload=True,
            with_vectors=False,
            limit=256,
        )
        return [r.payload for r in results]

    def scroll_authorized_version(self, document_id: str, version: int, role: Role) -> list[dict]:
        """只读取指定文档版本中当前角色有权看到的 active/superseded 块。

        这是比较工作流的存储边界:工具闭包只能调用本方法，任何未授权正文都不会
        进入 Agent 上下文。若版本状态不一致或不是允许比较的状态，整体返回空集。
        """
        all_payloads = self._scroll_all(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    ),
                    models.FieldCondition(
                        key="document_version", match=models.MatchValue(value=version)
                    ),
                ]
            )
        )
        statuses = {p.get("status") for p in all_payloads}
        if not all_payloads or len(statuses) != 1 or statuses - {
            str(ChunkStatus.ACTIVE), str(ChunkStatus.SUPERSEDED)
        }:
            return []
        payloads = [
            payload
            for payload in all_payloads
            if str(role) in {str(value) for value in payload.get("allowed_roles", [])}
        ]
        return (
            payloads
            if len(payloads) == len(all_payloads)
            and {p.get("chunk_id") for p in payloads} == {p.get("chunk_id") for p in all_payloads}
            else []
        )

    def citations_currently_authorized(self, citations: list[TodoCitation], role: Role) -> bool:
        """仅以当前 payload 校验已持久化引用，不读取正文或向模型暴露任何数据。"""
        unique = {citation.chunk_id: citation for citation in citations}
        if not unique:
            return True
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=[self.point_id(chunk_id) for chunk_id in unique],
            with_payload=[
                "chunk_id", "document_id", "document_version", "source_file_name",
                "source_locator", "status", "allowed_roles",
            ],
            with_vectors=False,
        )
        if len(points) != len(unique):
            return False
        payloads = {str((point.payload or {}).get("chunk_id")): point.payload or {} for point in points}
        for chunk_id, citation in unique.items():
            payload = payloads.get(chunk_id)
            if payload is None or payload.get("status") != str(ChunkStatus.ACTIVE):
                return False
            if str(role) not in {str(value) for value in payload.get("allowed_roles", [])}:
                return False
            if (
                payload.get("document_id") != citation.document_id
                or payload.get("document_version") != citation.document_version
                or payload.get("source_file_name") != citation.source_file_name
                or payload.get("source_locator") != citation.source_locator
            ):
                return False
        return True

    def _scroll_all(self, scroll_filter: models.Filter) -> list[dict]:
        """完整消费 Qdrant 游标;分页异常时抛错,调用方统一失败关闭。"""
        payloads: list[dict] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            payloads.extend(point.payload or {} for point in points)
            if offset is None:
                return payloads

    def set_status(
        self,
        document_id: str,
        status: ChunkStatus,
        *,
        version: int | None = None,
        only_active: bool = False,
    ) -> None:
        """按文档(可限定版本/仅 active)批量改写生命周期状态。"""
        must = [models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        if version is not None:
            must.append(models.FieldCondition(key="document_version", match=models.MatchValue(value=version)))
        if only_active:
            must.append(
                models.FieldCondition(key="status", match=models.MatchValue(value=str(ChunkStatus.ACTIVE)))
            )
        self.client.set_payload(
            collection_name=self.collection,
            payload={"status": str(status)},
            points=models.Filter(must=must),
            wait=True,
        )
