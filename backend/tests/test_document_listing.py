from datetime import UTC, datetime, timedelta

import pytest

from rag_kb.domain.identity import AuthContext
from rag_kb.infrastructure.database.base import Base
from rag_kb.infrastructure.database.document_lifecycle import DatabaseDocumentLifecycle
from rag_kb.infrastructure.database.models import ACLGrantRow, DocumentRow, PrincipalRow, TenantRow
from rag_kb.infrastructure.database.session import build_database


def _auth(*, admin: bool = False) -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id="alice",
        roles=frozenset({"admin"}) if admin else frozenset(),
        attributes={"permissions": ["document:read"]},
        request_id="document-listing",
    )


@pytest.fixture()
def lifecycle(tmp_path) -> DatabaseDocumentLifecycle:
    database = build_database(f"sqlite:///{tmp_path / 'documents.db'}")
    Base.metadata.create_all(database.engine)
    base = datetime(2026, 9, 9, tzinfo=UTC)
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="租户A", active=True))
        session.flush()
        session.add_all(
            (
                PrincipalRow(
                    tenant_id="tenant-a",
                    user_id="alice",
                    roles=[],
                    attributes={},
                    active=True,
                ),
                PrincipalRow(
                    tenant_id="tenant-a",
                    user_id="bob",
                    roles=[],
                    attributes={},
                    active=True,
                ),
            )
        )
        session.flush()
        for document_id, title, owner, created_at in (
            ("new", "最新制度", "alice", base + timedelta(minutes=2)),
            ("shared", "共享制度", "alice", base + timedelta(minutes=1)),
            ("hidden", "无权制度", "bob", base),
        ):
            session.add(
                DocumentRow(
                    tenant_id="tenant-a",
                    document_id=document_id,
                    title=title,
                    department="安全",
                    status="active",
                    created_by=owner,
                    created_at=created_at,
                    updated_at=created_at,
                )
            )
        session.add_all(
            (
                ACLGrantRow(
                    grant_id="grant-new",
                    tenant_id="tenant-a",
                    resource_type="document",
                    resource_id="new",
                    subject_type="user",
                    subject_id="alice",
                    permissions=["document:read"],
                ),
                ACLGrantRow(
                    grant_id="grant-shared",
                    tenant_id="tenant-a",
                    resource_type="document",
                    resource_id="shared",
                    subject_type="user",
                    subject_id="alice",
                    permissions=["document:read"],
                ),
                ACLGrantRow(
                    grant_id="grant-hidden",
                    tenant_id="tenant-a",
                    resource_type="document",
                    resource_id="hidden",
                    subject_type="user",
                    subject_id="bob",
                    permissions=["document:read"],
                ),
            )
        )
    return DatabaseDocumentLifecycle(database)


def test_document_listing_is_acl_scoped_and_cursor_stable(
    lifecycle: DatabaseDocumentLifecycle,
) -> None:
    first = lifecycle.list_documents(auth=_auth(), limit=1, cursor=None)
    second = lifecycle.list_documents(auth=_auth(), limit=1, cursor=first.next_cursor)

    assert [item.document_id for item in first.items] == ["new"]
    assert first.next_cursor is not None
    assert [item.document_id for item in second.items] == ["shared"]
    assert second.next_cursor is None


def test_document_listing_admin_and_invalid_cursor(lifecycle: DatabaseDocumentLifecycle) -> None:
    all_documents = lifecycle.list_documents(auth=_auth(admin=True), limit=10, cursor=None)

    assert [item.document_id for item in all_documents.items] == ["new", "shared", "hidden"]
    with pytest.raises(ValueError, match="document_cursor_invalid"):
        lifecycle.list_documents(auth=_auth(), limit=20, cursor="not-a-cursor")
