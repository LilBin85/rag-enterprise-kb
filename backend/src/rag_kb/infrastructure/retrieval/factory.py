"""共享 Retrieval Core 的生产装配。"""

from typing import Literal

from qdrant_client import QdrantClient

from rag_kb.application.enterprise_retriever import EnterpriseRetriever
from rag_kb.config import Settings, resolve_retrieval_limits
from rag_kb.embedder import BgeM3Embedder
from rag_kb.infrastructure.database import Database
from rag_kb.infrastructure.database.acl_repository import DatabaseAuthorizationPort
from rag_kb.infrastructure.retrieval.reranker import BgeReranker
from rag_kb.infrastructure.vectorstore.qdrant_authorized_search import QdrantAuthorizedSearch


def build_enterprise_retriever(
    settings: Settings, database: Database, *, consumer: Literal["fast", "durable"]
) -> EnterpriseRetriever:
    candidate_k, context_k = resolve_retrieval_limits(settings)
    embedder = BgeM3Embedder()
    vectors = QdrantAuthorizedSearch(
        QdrantClient(url=settings.qdrant_url),
        embedder,
        collection=settings.qdrant_v2_collection,
    )
    return EnterpriseRetriever(
        DatabaseAuthorizationPort(database),
        vectors,
        candidate_k=candidate_k,
        context_k=context_k,
        context_token_budget=settings.retrieval_context_token_budget,
        rrf_k=settings.retrieval_rrf_k,
        hybrid_enabled=settings.enable_hybrid_retrieval,
        dense_score_threshold=settings.retrieval_score_threshold,
        consumer=consumer,
        reranker=BgeReranker(settings.reranker_model) if settings.enable_reranker else None,
    )
