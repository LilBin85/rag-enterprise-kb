"""M2 批次 3 审批 HTTP 契约：客户端只能提交业务 decision。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from inspect import getsource
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_kb.application.approval_service import ApprovalServiceError
from rag_kb.domain.approvals import ApprovalBinding
from rag_kb.domain.identity import Principal
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.infrastructure.database.task_repository import TaskRepositoryError
from rag_kb.web.app import create_app
from rag_kb.web.routers import approvals as approvals_router
from rag_kb.web.dependencies import (
    build_approval_service,
    build_identity_provider,
)


def _binding(*, status: str = "pending") -> ApprovalBinding:
    now = datetime.now(UTC)
    return ApprovalBinding(
        approval_id="approval-1",
        task_id="task-1",
        tenant_id="tenant-a",
        plan_revision=1,
        plan_digest="sha256:" + "p" * 64,
        tool_name="create_report_artifact",
        tool_version="1.0.0",
        canonical_arguments_hash="sha256:" + "a" * 64,
        principal_hash="sha256:" + "u" * 64,
        evidence_hash="sha256:" + "e" * 64,
        expires_at=now + timedelta(minutes=10),
        status=status,
        checkpoint_binding_hash="sha256:" + "c" * 64,
        thread_id="task-1",
        checkpoint_ns="",
        checkpoint_id="checkpoint-1",
        interrupt_id="interrupt-1",
        tool_call_id="call-1",
        approved_by="approver-a" if status in {"approved", "executed"} else None,
        approved_at=now if status in {"approved", "executed"} else None,
        rejected_by="approver-a" if status == "rejected" else None,
        rejected_at=now if status == "rejected" else None,
        rejection_reason="not_ready" if status == "rejected" else None,
    )


class FakeApprovalService:
    def __init__(self) -> None:
        self.binding = _binding()
        self.summary = "等待人工审批：报告产物「安全标题」"
        self.task_state_version = 4
        self.calls: list[dict[str, object]] = []
        self.next_error: Exception | None = None

    def list_details(self, *, tenant_id: str, status: str | None, task_id: str | None):
        if tenant_id != "tenant-a" or (
            status is not None and self.binding.status != status
        ) or (task_id is not None and self.binding.task_id != task_id):
            return ()
        return (
            SimpleNamespace(
                binding=self.binding,
                summary=self.summary,
                task_state_version=self.task_state_version,
            ),
        )

    def get_detail(self, *, tenant_id: str, approval_id: str):
        if tenant_id != "tenant-a" or approval_id != self.binding.approval_id:
            raise ApprovalServiceError("approval_not_found")
        return SimpleNamespace(
            binding=self.binding,
            summary=self.summary,
            task_state_version=self.task_state_version,
        )

    def decide_and_enqueue(self, **kwargs: object) -> object:
        if self.next_error is not None:
            error = self.next_error
            self.next_error = None
            raise error
        decision = kwargs["decision"]
        binding = self.binding
        if decision == "approve":
            if binding.status == "rejected":
                raise ApprovalServiceError("approval_decision_conflict")
            self.binding = binding.model_copy(
                update={
                    "status": "approved",
                    "approved_by": "approver-a",
                    "approved_at": datetime.now(UTC),
                }
            )
        else:
            if binding.status in {"approved", "executed"}:
                raise ApprovalServiceError("approval_decision_conflict")
            self.binding = binding.model_copy(
                update={
                    "status": "rejected",
                    "rejected_by": "approver-a",
                    "rejected_at": datetime.now(UTC),
                    "rejection_reason": str(kwargs.get("reason") or "rejected_by_approver"),
                }
            )
        self.calls.append(kwargs)
        return SimpleNamespace(command_id="command-1")


@pytest.fixture()
def setup_app():
    app = create_app()
    service = FakeApprovalService()
    provider = DemoTokenIdentityProvider(
        {
            "approver-token": Principal(
                tenant_id="tenant-a",
                user_id="approver-a",
                attributes={"permissions": ["task:approve"]},
            ),
            "owner-token": Principal(
                tenant_id="tenant-a",
                user_id="owner-a",
                attributes={"permissions": ["task:create", "task:read"]},
            ),
            "self-token": Principal(
                tenant_id="tenant-a",
                user_id="owner-a",
                attributes={"permissions": ["task:approve"]},
            ),
            "cross-token": Principal(
                tenant_id="tenant-b",
                user_id="approver-b",
                attributes={"permissions": ["task:approve"]},
            ),
        }
    )
    app.dependency_overrides[build_identity_provider] = lambda: provider
    app.dependency_overrides[build_approval_service] = lambda: service
    with TestClient(app) as client:
        yield client, service
    app.dependency_overrides.clear()


def _headers(token: str = "approver-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "X-Request-ID": "approval-request"}


def test_approver_can_list_and_read_safe_detail(setup_app) -> None:
    client, _service = setup_app

    listed = client.get("/v1/approvals?status=pending&task_id=task-1", headers=_headers())
    detail = client.get("/v1/approvals/approval-1", headers=_headers())

    assert listed.status_code == 200
    assert [item["approval_id"] for item in listed.json()] == ["approval-1"]
    assert detail.status_code == 200
    assert detail.json()["summary"] == "等待人工审批：报告产物「安全标题」"
    assert detail.json()["task_state_version"] == 4
    assert detail.json()["checkpoint_binding_hash"].startswith("sha256:")
    assert {"thread_id", "checkpoint_id", "interrupt_id", "tool_call_id"}.isdisjoint(
        detail.json()
    )


def test_approvals_require_permission_and_hide_cross_tenant_existence(setup_app) -> None:
    client, _service = setup_app

    denied = client.get("/v1/approvals", headers=_headers("owner-token"))
    cross = client.get("/v1/approvals/approval-1", headers=_headers("cross-token"))

    assert denied.status_code == 403
    assert cross.status_code == 404


def test_decision_only_enqueues_server_approval_binding_and_is_idempotent(setup_app) -> None:
    client, service = setup_app
    body = {"decision": "approve", "expected_task_state_version": 4}

    first = client.post("/v1/approvals/approval-1/decision", headers=_headers(), json=body)
    repeated = client.post("/v1/approvals/approval-1/decision", headers=_headers(), json=body)
    opposite = client.post(
        "/v1/approvals/approval-1/decision",
        headers=_headers(),
        json={"decision": "reject", "expected_task_state_version": 4},
    )

    assert first.status_code == 202
    assert repeated.status_code == 202
    assert opposite.status_code == 409
    assert service.binding.status == "approved"
    assert first.json()["resume_command"] == {
        "command_id": "command-1",
        "status": "accepted",
    }
    assert repeated.json()["resume_command"] == first.json()["resume_command"]
    assert all(call["approval_id"] == "approval-1" for call in service.calls)
    assert all(call["expected_task_state_version"] == 4 for call in service.calls)


def test_rejection_reason_is_not_returned_by_approval_api(setup_app) -> None:
    client, _service = setup_app
    reason = "仅限审批审计的私密原因"

    decided = client.post(
        "/v1/approvals/approval-1/decision",
        headers=_headers(),
        json={"decision": "reject", "reason": reason, "expected_task_state_version": 4},
    )
    detail = client.get("/v1/approvals/approval-1", headers=_headers())

    assert decided.status_code == 202
    assert detail.status_code == 200
    assert reason not in decided.text
    assert reason not in detail.text
    assert "rejection_reason" not in decided.json()
    assert "rejection_reason" not in detail.json()


def test_decision_rejects_server_field_injection_and_preserves_binding(setup_app) -> None:
    client, service = setup_app

    response = client.post(
        "/v1/approvals/approval-1/decision",
        headers=_headers(),
        json={
            "decision": "reject",
            "reason": "无需创建",
            "expected_task_state_version": 4,
            "thread_id": "client-thread",
            "checkpoint_id": "client-checkpoint",
            "resume": {"decisions": [{"type": "approve"}]},
            "tool_name": "create_work_item_draft",
            "tool_version": "999",
            "approval_id": "other",
            "plan_digest": "sha256:fake",
            "evidence_digest": "sha256:fake",
            "storage_key": "artifacts/private/sentinel",
        },
    )

    assert response.status_code == 422
    assert "artifacts/private/sentinel" not in response.text
    assert service.binding.status == "pending"
    assert service.calls == []


def test_decision_routes_wrong_interrupt_and_self_approval_to_stable_errors(setup_app) -> None:
    client, service = setup_app
    service.next_error = TaskRepositoryError("task_not_waiting_for_approval")

    clarification = client.post(
        "/v1/approvals/approval-1/decision",
        headers=_headers(),
        json={"decision": "approve", "expected_task_state_version": 4},
    )
    service.next_error = ApprovalServiceError("approval_self_rejected")
    self_approval = client.post(
        "/v1/approvals/approval-1/decision",
        headers=_headers("self-token"),
        json={"decision": "approve", "expected_task_state_version": 4},
    )
    service.next_error = ApprovalServiceError("approval_resume_state_conflict")
    checkpoint_mismatch = client.post(
        "/v1/approvals/approval-1/decision",
        headers=_headers(),
        json={"decision": "approve", "expected_task_state_version": 4},
    )

    assert clarification.status_code == 409
    assert self_approval.status_code == 403
    assert self_approval.json()["detail"]["code"] == "approval_self_rejected"
    assert checkpoint_mismatch.status_code == 409


def test_decision_router_has_no_agent_execution_dependency() -> None:
    source = getsource(approvals_router)

    assert "build_task_resume_worker" not in source
    assert "TaskWorker" not in source
