"""Qdrant 文档 collection 的共享、失败关闭初始化逻辑。"""

from __future__ import annotations

from qdrant_client.http import models


_PAYLOAD_INDEXES = (
    ("tenant_id", models.PayloadSchemaType.KEYWORD),
    ("version_id", models.PayloadSchemaType.KEYWORD),
    ("document_id", models.PayloadSchemaType.KEYWORD),
    ("projection_generation", models.PayloadSchemaType.INTEGER),
    ("projection_key", models.PayloadSchemaType.KEYWORD),
    ("lifecycle_status", models.PayloadSchemaType.KEYWORD),
    ("current", models.PayloadSchemaType.BOOL),
    ("allowed_principals", models.PayloadSchemaType.KEYWORD),
    ("index_version", models.PayloadSchemaType.KEYWORD),
)


class DocumentIndexError(RuntimeError):
    """仅传播固定安全错误码，绝不暴露底层 Qdrant 异常。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _get_collection_after_create(client, collection: str) -> object:
    """创建请求不确定后读取集合信息；给服务端异步落地一个短等待窗口。

    创建请求可能客户端超时但服务端仍在异步执行（例如高负载下 >2s），
    此时紧随其后的 get 会短暂 404。最多额外等待约 1 秒并复查一次。
    """
    import time

    attempts = 3
    for attempt in range(attempts):
        try:
            return client.get_collection(collection)
        except DocumentIndexError:
            raise
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError("unreachable")


def ensure_document_collection(client, *, collection: str, dimension: int) -> None:
    """创建或验证文档 collection；绝不删除或重建不兼容集合。"""
    if not collection or dimension <= 0:
        raise DocumentIndexError("document_index_collection_schema_invalid")

    try:
        exists = client.collection_exists(collection)
        if not exists:
            try:
                client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=dimension, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "keyword": models.SparseVectorParams(modifier=models.Modifier.IDF)
                    },
                )
            except Exception:
                # 另一实例可能已在我们之前完成创建。只有完全兼容才接受该竞态。
                pass

        info = _get_collection_after_create(client, collection)
        _validate_vector_schema(info, dimension)
        payload_schema = getattr(info, "payload_schema", None) or {}
        for field, schema_type in _PAYLOAD_INDEXES:
            current = payload_schema.get(field)
            if current is None:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=schema_type,
                    wait=True,
                )
            elif getattr(current, "data_type", None) != schema_type:
                raise DocumentIndexError("document_index_collection_schema_invalid")

        # create_payload_index(wait=True) 仍需以服务端状态为准，防止静默失败或竞态。
        validate_document_collection(client.get_collection(collection), dimension=dimension)
    except DocumentIndexError:
        raise
    except Exception as exc:
        raise DocumentIndexError("document_index_collection_unavailable") from exc


def validate_document_collection(info, *, dimension: int) -> None:
    """只读验证完整 collection schema，供 readiness 与写入路径复用。"""
    if dimension <= 0:
        raise DocumentIndexError("document_index_collection_schema_invalid")
    try:
        _validate_vector_schema(info, dimension)
        payload_schema = getattr(info, "payload_schema", None) or {}
        for field, schema_type in _PAYLOAD_INDEXES:
            current = payload_schema.get(field)
            if current is None or getattr(current, "data_type", None) != schema_type:
                raise DocumentIndexError("document_index_collection_schema_invalid")
    except DocumentIndexError:
        raise
    except Exception as exc:
        raise DocumentIndexError("document_index_collection_schema_invalid") from exc


def _validate_vector_schema(info, dimension: int) -> None:
    try:
        params = info.config.params
        vectors = params.vectors
        sparse_vectors = params.sparse_vectors
        dense = vectors.get("dense") if isinstance(vectors, dict) else None
        keyword = sparse_vectors.get("keyword") if isinstance(sparse_vectors, dict) else None
        if (
            dense is None
            or dense.size != dimension
            or dense.distance != models.Distance.COSINE
            or keyword is None
            or keyword.modifier != models.Modifier.IDF
        ):
            raise DocumentIndexError("document_index_collection_schema_invalid")
    except DocumentIndexError:
        raise
    except Exception as exc:
        raise DocumentIndexError("document_index_collection_schema_invalid") from exc
