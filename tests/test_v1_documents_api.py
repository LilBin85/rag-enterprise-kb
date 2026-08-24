"""阶段2文档 HTTP 路由回归：只使用内存假服务和演示身份，不连接外部组件。"""

import json
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_kb.application.document_service import (
    DocumentAccessDenied,
    DocumentDeletionResult,
    DocumentResult,
    DocumentUploadResult,
    DocumentVersionResult,
    IngestionJobResult,
)
from rag_kb.domain.identity import AuthContext, Permission, Principal
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.infrastructure.database.document_repository import (
    DocumentRegistryConflict,
    RegisteredUpload,
)
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import build_document_service, build_identity_provider, resolve_auth_context


class FakeDocumentService:
    def __init__(self, tenant_id: str = "local-demo") -> None:
        self.tenant_id = tenant_id
        self.uploads: list[dict[str, object]] = []

    def _check(self, auth: AuthContext) -> None:
        if auth.tenant_id != self.tenant_id:
            raise DocumentAccessDenied("cross_tenant_access_denied")
        if not auth.has_permission(Permission.DOCUMENT_READ):
            raise DocumentAccessDenied("document_read_denied")

    def register_upload(self, **kwargs: object) -> DocumentUploadResult:
        auth = kwargs["auth"]
        assert isinstance(auth, AuthContext)
        if auth.tenant_id != self.tenant_id or not auth.has_permission(Permission.DOCUMENT_WRITE):
            raise DocumentAccessDenied("document_write_denied")
        document_id = str(kwargs["document_id"])
        if "/" in document_id or "\\" in document_id or document_id in {".", ".."}:
            raise ValueError("invalid_document_id")
        if document_id == "existing-doc":
            raise DocumentRegistryConflict(
                "document_already_exists",
                recovery=RegisteredUpload(
                    tenant_id=self.tenant_id,
                    document_id=document_id,
                    version_id="version-existing",
                    version=1,
                    job_id="job-existing",
                    source_object_key="hidden",
                    lifecycle_status="active",
                    ingestion_status="succeeded",
                ),
            )
        self.uploads.append(kwargs)
        return DocumentUploadResult(
            document_id=document_id,
            version_id="version-1",
            version=1,
            ingestion_job_id="job-1",
            lifecycle_status="pending",
            ingestion_status="pending",
        )

    def get_document(self, document_id: str, auth: AuthContext) -> DocumentResult | None:
        self._check(auth)
        if document_id != "doc-1":
            return None
        return DocumentResult(
            document_id="doc-1",
            title="测试制度",
            department="hr",
            status="pending",
            versions=(DocumentVersionResult(version_id="version-1", version=1, lifecycle_status="pending"),),
        )

    def get_ingestion_job(self, job_id: str, auth: AuthContext) -> IngestionJobResult | None:
        self._check(auth)
        if job_id != "job-1":
            return None
        return IngestionJobResult(
            job_id="job-1", version_id="version-1", status="pending", attempts=0
        )

    def list_ingestion_jobs(
        self,
        document_id: str,
        auth: AuthContext,
        *,
        limit: int = 100,
    ) -> tuple[IngestionJobResult, ...] | None:
        self._check(auth)
        if document_id != "doc-1":
            return None
        return (IngestionJobResult(job_id="job-1", version_id="version-1", status="pending", attempts=0),)

    def delete_document(self, document_id: str, auth: AuthContext) -> DocumentDeletionResult | None:
        if auth.tenant_id != self.tenant_id or not auth.has_permission(Permission.DOCUMENT_WRITE):
            raise DocumentAccessDenied("document_write_denied")
        if document_id != "doc-1":
            return None
        return DocumentDeletionResult(
            document_id=document_id,
            deleted_versions=1,
            deleted_ingestion_jobs=1,
        )


@pytest.fixture()
def setup_app() -> Iterator[tuple[TestClient, FakeDocumentService]]:
    app = create_app()
    service = FakeDocumentService()
    provider = DemoTokenIdentityProvider(
        {
            "write-token": Principal(
                tenant_id="local-demo",
                user_id="writer",
                roles=frozenset({"admin"}),
            ),
            "read-only-token": Principal(
                tenant_id="local-demo",
                user_id="reader",
                roles=frozenset({"employee"}),
                attributes={"permissions": ["document:read"]},
            ),
        }
    )
    app.dependency_overrides[build_document_service] = lambda: service
    app.dependency_overrides[build_identity_provider] = lambda: provider
    with TestClient(app) as client:
        yield client, service
    app.dependency_overrides.clear()


def _metadata(document_id: str = "doc-1") -> str:
    return json.dumps(
        {"document_id": document_id, "title": "测试制度", "department": "hr"},
        ensure_ascii=False,
    )


def _upload(client: TestClient, *, token: str = "write-token", document_id: str = "doc-1", key: str | None = "idem-1"):
    headers = {"X-Demo-Role-Token": token}
    if key is not None:
        headers["Idempotency-Key"] = key
    return client.post(
        "/v1/documents",
        headers=headers,
        files={"file": ("policy.md", b"safe fixture", "text/markdown")},
        data={"metadata": _metadata(document_id)},
    )


def test_multipart_upload_requires_identity_and_idempotency_key(setup_app) -> None:
    client, service = setup_app
    missing_identity = client.post(
        "/v1/documents",
        files={"file": ("policy.md", b"safe fixture", "text/markdown")},
        data={"metadata": _metadata()},
    )
    assert missing_identity.status_code == 403
    missing_key = _upload(client, key=None)
    assert missing_key.status_code == 422
    assert service.uploads == []


def test_multipart_upload_returns_202_and_document_job_queries_work(setup_app) -> None:
    client, service = setup_app
    response = _upload(client)
    assert response.status_code == 202
    assert response.json()["ingestion_job_id"] == "job-1"
    assert service.uploads[0]["content"] == b"safe fixture"

    document = client.get("/v1/documents/doc-1", headers={"X-Demo-Role-Token": "read-only-token"})
    job = client.get("/v1/ingestion-jobs/job-1", headers={"X-Demo-Role-Token": "read-only-token"})
    assert document.status_code == 200
    assert document.json()["versions"][0]["version"] == 1
    assert job.status_code == 200
    assert job.json()["status"] == "pending"

    jobs = client.get(
        "/v1/documents/doc-1/ingestion-jobs",
        headers={"X-Demo-Role-Token": "read-only-token"},
    )
    assert jobs.status_code == 200
    assert jobs.json()[0]["job_id"] == "job-1"


def test_existing_document_conflict_returns_safe_recovery_reference(setup_app) -> None:
    client, _ = setup_app

    response = _upload(client, document_id="existing-doc", key="new-key")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "document_already_exists"
    assert detail["recovery"]["document_id"] == "existing-doc"
    assert detail["recovery"]["ingestion_job_id"] == "job-existing"


def test_delete_document_requires_writer_and_returns_deletion_receipt(setup_app) -> None:
    client, _ = setup_app

    denied = client.delete("/v1/documents/doc-1", headers={"X-Demo-Role-Token": "read-only-token"})
    deleted = client.delete("/v1/documents/doc-1", headers={"X-Demo-Role-Token": "write-token"})

    assert denied.status_code == 403
    assert deleted.status_code == 200
    assert deleted.json()["deleted_ingestion_jobs"] == 1


def test_unauthorized_and_cross_tenant_requests_are_rejected(setup_app) -> None:
    client, _ = setup_app
    unauthorized = client.get("/v1/documents/doc-1", headers={"X-Demo-Role-Token": "read-only-token"})
    assert unauthorized.status_code == 200  # 读权限允许查询
    denied_upload = _upload(client, token="read-only-token", key="read-upload")
    assert denied_upload.status_code == 403

    app = client.app
    app.dependency_overrides[resolve_auth_context] = lambda: AuthContext(
        tenant_id="other-tenant", user_id="other-user", request_id="request-1", roles=frozenset({"admin"})
    )
    cross_tenant = client.get("/v1/documents/doc-1")
    assert cross_tenant.status_code == 403
    app.dependency_overrides.pop(resolve_auth_context)


def test_path_escape_is_rejected_before_upload_is_accepted(setup_app) -> None:
    client, service = setup_app
    response = _upload(client, document_id="../escape", key="escape-key")
    assert response.status_code == 422
    assert service.uploads == []


def test_upload_body_is_bounded_before_service_dispatch(setup_app, monkeypatch) -> None:
    client, service = setup_app
    monkeypatch.setattr(
        "rag_kb.web.routers.documents.get_settings",
        lambda: SimpleNamespace(max_upload_bytes=4),
    )

    response = client.post(
        "/v1/documents",
        headers={"X-Demo-Role-Token": "write-token", "Idempotency-Key": "oversized"},
        files={"file": ("policy.md", b"12345", "text/markdown")},
        data={"metadata": _metadata()},
    )

    assert response.status_code == 413
    assert service.uploads == []
