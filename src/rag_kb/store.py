"""Qdrant 存储封装(蓝图 5.1:写入能力只由摄取路径使用)。

设计决策(蓝图 3.3 取舍原则):不设独立文档注册表——版本与生命周期状态只存于
块 payload,文档级查询通过 scroll 聚合。单一真相源,无双写不一致;规模增长后
再按第 15 节触发器拆分注册表。
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_kb.models import ChunkRecord, ChunkStatus

_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-enterprise-kb")


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
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        for field, schema in [
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("status", models.PayloadSchemaType.KEYWORD),
            ("allowed_roles", models.PayloadSchemaType.KEYWORD),
            ("document_version", models.PayloadSchemaType.INTEGER),
        ]:
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
