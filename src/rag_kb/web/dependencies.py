"""HTTP 依赖装配；不得在构造函数中执行迁移或下载模型。"""

from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import Depends, Header, HTTPException, Request
from qdrant_client import QdrantClient
from sqlalchemy import text

from rag_kb.config import get_settings, task_budgets_from_settings
from rag_kb.application.approval_service import ApprovalService
from rag_kb.application.artifact_content_service import ArtifactContentService
from rag_kb.application.query_service import KnowledgeQueryService
from rag_kb.application.task_query_service import TaskQueryService
from rag_kb.application.task_result_service import TaskResultService
from rag_kb.agent.result import AgentResultStore
from rag_kb.application.document_service import DocumentService
from rag_kb.domain.identity import AuthContext
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider, IdentityProvider
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.document_lifecycle import DatabaseDocumentLifecycle
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
)
from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
from rag_kb.infrastructure.database.task_read import (
    DatabaseArtifactContentReader,
    DatabaseTaskQueryReader,
    DatabaseTaskResultReader,
)
from rag_kb.infrastructure.database.approval_lifecycle import (
    DatabaseApprovalLifecycle,
)
from rag_kb.infrastructure.object_store import LocalObjectStore
from rag_kb.infrastructure.vectorstore.qdrant_collection import (
    validate_document_collection,
)
from rag_kb.infrastructure.vectorstore.qdrant_document_index import QdrantDocumentDeletion
from rag_kb.observability import normalize_request_id


EXPECTED_DATABASE_REVISION = "20260831_0015"


class ReadinessFailure(RuntimeError):
    def __init__(self, component: str) -> None:
        super().__init__(component)
        self.component = component


class ReadinessProbe(Protocol):
    def check(self) -> dict[str, str]: ...


class SystemReadinessProbe:
    def __init__(
        self,
        qdrant_url: str,
        database_url: str,
        *,
        collection: str,
        embedding_dim: int,
    ) -> None:
        self._qdrant_url = qdrant_url
        self._database = build_database(database_url)
        self._collection = collection
        self._embedding_dim = embedding_dim

    def check(self) -> dict[str, str]:
        try:
            info = QdrantClient(url=self._qdrant_url, timeout=3).get_collection(
                self._collection
            )
        except Exception as exc:
            raise ReadinessFailure("qdrant") from exc
        try:
            validate_document_collection(info, dimension=self._embedding_dim)
        except Exception as exc:
            raise ReadinessFailure("qdrant_schema") from exc
        try:
            with self._database.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                revision = connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar_one_or_none()
                unconverged = connection.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM document_versions
                        JOIN documents
                          ON documents.tenant_id = document_versions.tenant_id
                         AND documents.document_id = document_versions.document_id
                        WHERE document_versions.lifecycle_status = 'active'
                          AND (
                              document_versions.applied_projection_generation IS NULL
                              OR documents.applied_projection_version_id IS NULL
                              OR documents.applied_projection_generation IS NULL
                              OR documents.applied_projection_version_id
                                 != document_versions.version_id
                              OR documents.applied_projection_generation
                                 != document_versions.applied_projection_generation
                          )
                        """
                    )
                ).scalar_one()
        except Exception as exc:
            raise ReadinessFailure("postgresql") from exc
        if revision != EXPECTED_DATABASE_REVISION:
            raise ReadinessFailure("database_schema")
        if unconverged:
            raise ReadinessFailure("projection_convergence")
        return {
            "qdrant": "ok",
            "qdrant_schema": "current",
            "postgresql": "ok",
            "database_schema": "current",
        }


@lru_cache
def build_identity_provider() -> IdentityProvider:
    settings = get_settings()
    return DemoTokenIdentityProvider.from_role_tokens(settings.demo_role_tokens)


@lru_cache
def build_readiness_probe() -> ReadinessProbe:
    settings = get_settings()
    return SystemReadinessProbe(
        settings.qdrant_url,
        settings.database_url,
        collection=settings.qdrant_v2_collection,
        embedding_dim=settings.embedding_dim,
    )


@lru_cache
def build_knowledge_query_service() -> KnowledgeQueryService:
    # 新问答链只读取现有 schema/collection；请求路径不执行 DDL 或模型下载探测。
    from rag_kb.answerer import DeepSeekAnswerModel
    from rag_kb.application.enterprise_retriever import EnterpriseRetriever
    from rag_kb.config import resolve_retrieval_limits
    from rag_kb.infrastructure.database.knowledge_query_lifecycle import (
        DatabaseKnowledgeQueryLifecycle,
    )
    from rag_kb.embedder import BgeM3Embedder
    from rag_kb.infrastructure.database.acl_repository import DatabaseAuthorizationPort
    from rag_kb.infrastructure.vectorstore.qdrant_authorized_search import QdrantAuthorizedSearch

    settings = get_settings()
    candidate_k, context_k = resolve_retrieval_limits(settings)
    database = build_database(settings.database_url)
    embedder = BgeM3Embedder()
    vectors = QdrantAuthorizedSearch(
        QdrantClient(url=settings.qdrant_url),
        embedder,
        collection=settings.qdrant_v2_collection,
    )
    retriever = EnterpriseRetriever(
        DatabaseAuthorizationPort(database),
        vectors,
        candidate_k=candidate_k,
        context_k=context_k,
        context_token_budget=settings.retrieval_context_token_budget,
        rrf_k=settings.retrieval_rrf_k,
        hybrid_enabled=settings.enable_hybrid_retrieval,
        dense_score_threshold=settings.retrieval_score_threshold,
    )
    return KnowledgeQueryService(
        DatabaseKnowledgeQueryLifecycle(database),
        retriever,
        DeepSeekAnswerModel(settings),
    )


@lru_cache
def build_document_service() -> DocumentService:
    settings = get_settings()
    database = build_database(settings.database_url)
    return DocumentService(
        DatabaseDocumentLifecycle(database),
        LocalObjectStore(settings.object_store_root),
        vector_deleter=QdrantDocumentDeletion(
            QdrantClient(url=settings.qdrant_url),
            collection=settings.qdrant_v2_collection,
        ),
    )


@lru_cache
def build_task_repository() -> DatabaseTaskRepository:
    settings = get_settings()
    return DatabaseTaskRepository(
        build_database(settings.database_url),
        default_budgets=task_budgets_from_settings(settings),
        task_capability_name=settings.task_agent_capability_name,
        task_capability_protocol_version=settings.task_agent_protocol_version,
    )


@lru_cache
def build_task_query_service() -> TaskQueryService:
    return TaskQueryService(DatabaseTaskQueryReader(build_database(get_settings().database_url)))


@lru_cache
def build_artifact_content_service() -> ArtifactContentService:
    settings = get_settings()
    return ArtifactContentService(DatabaseArtifactContentReader(
        build_database(settings.database_url),
        LocalObjectStore(settings.object_store_root),
    ))


def build_task_result_service() -> TaskResultService:
    settings = get_settings()
    return TaskResultService(DatabaseTaskResultReader(
        build_database(settings.database_url),
        AgentResultStore(LocalObjectStore(settings.object_store_root)),
        EvidenceLedgerRepository(),
    ))


@lru_cache
def build_approval_service() -> ApprovalService:
    database = build_database(get_settings().database_url)
    return ApprovalService(
        DatabaseApprovalLifecycle(
            database,
            resume_repository=build_task_repository(),
        )
    )


def resolve_auth_context(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    demo_token: Annotated[str | None, Header(alias="X-Demo-Role-Token")] = None,
    provider: Annotated[IdentityProvider, Depends(build_identity_provider)] = None,
) -> AuthContext:
    """从服务端身份提供器解析统一上下文；错误响应不泄露令牌或角色存在性。"""
    bearer = None
    if authorization:
        scheme, separator, value = authorization.partition(" ")
        if separator and scheme.lower() == "bearer":
            bearer = value.strip()
    token = bearer or demo_token
    if not token:
        raise HTTPException(status_code=403, detail="身份验证失败")
    if provider is None:
        raise HTTPException(status_code=503, detail="身份服务不可用")
    request_id = getattr(request.state, "request_id", None)
    if not isinstance(request_id, str):
        request_id = normalize_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
    context = provider.authenticate(
        token,
        request_id=request_id,
    )
    if context is None:
        raise HTTPException(status_code=403, detail="身份验证失败")
    return context
