"""M2 批次 3 产物正文读取：授权、生命周期和 hash 必须全部 fail-closed。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_kb.application.artifact_content_service import (
    ArtifactContentError,
    ArtifactContentService,
)
from rag_kb.domain.identity import AuthContext, Principal
from rag_kb.domain.tasks import TaskMode
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import (
    ArtifactRow,
    PrincipalRow,
    TenantRow,
    ToolCallRow,
)
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
from rag_kb.infrastructure.database.task_read import DatabaseArtifactContentReader
from rag_kb.infrastructure.object_store.local import LocalObjectStore
from rag_kb.web.app import create_app
from rag_kb.web.dependencies import (
    build_artifact_content_service,
    build_identity_provider,
)


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


@pytest.fixture()
def stage(tmp_path: Path):
    database = build_database(f"sqlite+pysqlite:///{(tmp_path / 'artifacts.sqlite3').as_posix()}")
    Base.metadata.create_all(database.engine)
    now = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
    owner = AuthContext(
        tenant_id="tenant-a",
        user_id="owner-a",
        attributes={"permissions": ["task:create", "task:read"]},
        request_id="owner-request",
    )
    non_owner = owner.model_copy(update={"user_id": "reader-a", "request_id": "reader-request"})
    outsider = owner.model_copy(update={"tenant_id": "tenant-b", "user_id": "owner-b", "request_id": "outside-request"})
    with database.sessions.begin() as session:
        session.add_all(
            [
                TenantRow(tenant_id="tenant-a", name="tenant-a"),
                TenantRow(tenant_id="tenant-b", name="tenant-b"),
            ]
        )
        session.flush()
        session.add_all(
            [
                PrincipalRow(tenant_id="tenant-a", user_id="owner-a", roles=[], attributes=owner.attributes),
                PrincipalRow(tenant_id="tenant-a", user_id="reader-a", roles=[], attributes=non_owner.attributes),
                PrincipalRow(tenant_id="tenant-b", user_id="owner-b", roles=[], attributes=outsider.attributes),
            ]
        )
    repository = DatabaseTaskRepository(database, now=lambda: now)
    repository.create_task(auth=owner, goal="生成报告", task_mode=TaskMode.RESEARCH, task_id="task-a")
    objects = LocalObjectStore(tmp_path / "objects")
    content = b"approved artifact body"
    storage_key = "artifacts/tenant/task/report/report.md"
    objects.put_exclusive(storage_key, content)
    with database.sessions.begin() as session:
        session.add(
            ToolCallRow(
                tool_call_id="call-a",
                task_id="task-a",
                tenant_id="tenant-a",
                approval_id=None,
                plan_revision=None,
                plan_digest=None,
                evidence_digest=None,
                tool_name="search_knowledge",
                tool_version="1.0.0",
                arguments_hash="sha256:" + "a" * 64,
                principal_hash="sha256:" + "p" * 64,
                mode="read",
                idempotency_key=None,
                risk_level="low",
                status="running",
                execution_deadline=now + timedelta(minutes=1),
            )
        )
        session.flush()
        session.add(
            ArtifactRow(
                artifact_id="artifact-a",
                task_id="task-a",
                tenant_id="tenant-a",
                tool_call_id="call-a",
                artifact_type="report",
                storage_key=storage_key,
                content_hash=_sha256(content),
                status="ready",
                content_expires_at=now + timedelta(days=1),
            )
        )
    service = ArtifactContentService(
        DatabaseArtifactContentReader(database, objects, now=lambda: now)
    )
    return SimpleNamespace(
        database=database,
        now=now,
        owner=owner,
        non_owner=non_owner,
        outsider=outsider,
        objects=objects,
        storage_key=storage_key,
        service=service,
    )


def _set_artifact(stage, **values: object) -> None:
    with stage.database.sessions.begin() as session:
        row = session.get(ArtifactRow, "artifact-a")
        assert row is not None
        for key, value in values.items():
            setattr(row, key, value)


def test_owner_with_task_read_can_read_ready_artifact(stage) -> None:
    assert stage.service.read(auth=stage.owner, task_id="task-a", artifact_id="artifact-a") == b"approved artifact body"


@pytest.mark.parametrize("status", ["creating", "failed", "unknown"])
def test_not_ready_artifact_is_conflict(stage, status: str) -> None:
    _set_artifact(stage, status=status)

    with pytest.raises(ArtifactContentError, match="artifact_content_unavailable"):
        stage.service.read(auth=stage.owner, task_id="task-a", artifact_id="artifact-a")


def test_expired_and_purged_artifact_are_gone(stage) -> None:
    _set_artifact(stage, content_expires_at=stage.now - timedelta(seconds=1))
    with pytest.raises(ArtifactContentError, match="artifact_content_expired"):
        stage.service.read(auth=stage.owner, task_id="task-a", artifact_id="artifact-a")

    _set_artifact(
        stage,
        content_expires_at=stage.now + timedelta(days=1),
        storage_key=None,
        content_purged_at=stage.now,
    )
    with pytest.raises(ArtifactContentError, match="artifact_content_expired"):
        stage.service.read(auth=stage.owner, task_id="task-a", artifact_id="artifact-a")


@pytest.mark.parametrize("tampered", [False, True])
def test_missing_or_tampered_object_is_internal_failure_without_storage_key(stage, tampered: bool) -> None:
    if tampered:
        stage.objects.delete_exact(stage.storage_key)
        stage.objects.put_exclusive(stage.storage_key, b"tampered")
    else:
        stage.objects.delete_exact(stage.storage_key)

    with pytest.raises(ArtifactContentError) as captured:
        stage.service.read(auth=stage.owner, task_id="task-a", artifact_id="artifact-a")

    assert captured.value.code == "artifact_content_integrity_failed"
    assert stage.storage_key not in str(captured.value)


def test_cross_tenant_and_non_owner_are_indistinguishable_not_found(stage) -> None:
    for auth in (stage.non_owner, stage.outsider):
        with pytest.raises(ArtifactContentError, match="artifact_not_found"):
            stage.service.read(auth=auth, task_id="task-a", artifact_id="artifact-a")


def test_http_content_contract_and_web_static_storage_key_gate(stage) -> None:
    app = create_app()
    provider = DemoTokenIdentityProvider(
        {
            "owner-token": Principal(
                tenant_id="tenant-a",
                user_id="owner-a",
                attributes={"permissions": ["task:read"]},
            ),
            "reader-token": Principal(
                tenant_id="tenant-a",
                user_id="reader-a",
                attributes={"permissions": ["task:read"]},
            ),
        }
    )
    app.dependency_overrides[build_identity_provider] = lambda: provider
    app.dependency_overrides[build_artifact_content_service] = lambda: stage.service
    with TestClient(app) as client:
        response = client.get(
            "/v1/tasks/task-a/artifacts/artifact-a/content",
            headers={"Authorization": "Bearer owner-token"},
        )
        denied = client.get(
            "/v1/tasks/task-a/artifacts/artifact-a/content",
            headers={"Authorization": "Bearer reader-token"},
        )

    assert response.status_code == 200
    assert response.content == b"approved artifact body"
    assert "content-type" not in response.headers
    assert stage.storage_key not in response.text
    assert denied.status_code == 404
    source_root = Path(__file__).parents[1] / "src" / "rag_kb" / "web"
    assert all("storage_key" not in path.read_text(encoding="utf-8") for path in source_root.rglob("*.py"))
