"""在业务进程启动前幂等初始化企业 Qdrant collection。"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from rag_kb.application.ports.indexing import DocumentIndexError
from rag_kb.config import get_settings
from rag_kb.infrastructure.vectorstore.qdrant_collection import ensure_document_collection


logger = logging.getLogger(__name__)


def run() -> int:
    settings = get_settings()
    try:
        ensure_document_collection(
            QdrantClient(url=settings.qdrant_url),
            collection=settings.qdrant_v2_collection,
            dimension=settings.embedding_dim,
        )
    except DocumentIndexError as exc:
        logger.error("Qdrant collection 初始化失败 error_code=%s", exc.code)
        return 1
    logger.info("Qdrant collection 初始化完成")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
