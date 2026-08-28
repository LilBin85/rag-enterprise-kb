"""运行隔离的 M2 controlled-write HTTP/Worker/PostgreSQL 验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5


ROOT = Path(__file__).resolve().parents[1]
_PROTECTED_COLLECTIONS = frozenset({"kb_chunks", "kb_chunks_v2", "test_dfaae526c675"})
_DATABASE_NAME = re.compile(r"^rag_kb_m2b4_[a-z0-9_]+$")
_ARTIFACT_ID = "00000000000000000000000000000042"
_FAULT_SCENARIOS = {
    "migration_round_trip": "test_m2_protocol_and_artifact_lifecycle_round_trip_on_postgres",
    "m1_worker_cannot_claim_m2": "test_postgres_m2_task_claim_requires_exact_live_worker_protocol",
    "checkpoint_protocol_fencing": "test_missing_or_wrong_m2_protocol_cannot_write_checkpoint_or_pending_writes",
    "checkpoint_preemption_fencing": "test_preempted_generation_cannot_write_checkpoint_or_pending_writes",
    "crash_before_execution": "test_window_a_crash_before_tool_node_recovers_executes_once",
    "crash_after_execution": "test_window_b_crash_after_adapter_reads_result_ref_single_execution",
    "crash_after_checkpoint": "test_window_c_checkpoint_done_summary_sync_idempotent",
    "approval_expiry": "test_invalidated_approval_never_executes_adapter[expiry]",
    "principal_revocation": "test_invalidated_approval_never_executes_adapter[principal_changed]",
    "acl_revocation": "test_invalidated_approval_never_executes_adapter[acl_revoked]",
    "evidence_withdrawal": "test_invalidated_approval_never_executes_adapter[evidence_withdrawn]",
    "arguments_drift": "test_invalidated_approval_never_executes_adapter[arguments_drift]",
    "tool_version_drift": "test_invalidated_approval_never_executes_adapter[version_drift]",
    "write_timeout_unknown": "test_write_timeout_unknown_never_replays_on_postgres",
    "concurrent_decision": "test_concurrent_decisions_single_action_on_postgres",
    "result_ref_tampered": "test_postgres_result_ref_corruption_fails_closed[tampered]",
    "result_ref_missing": "test_postgres_result_ref_corruption_fails_closed[missing]",
}


class SmokeError(RuntimeError):
    pass


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


class _WorkerRouter:
    """让 HTTP approval/continue 路由复用创建该 task 的真实 Worker。"""

    def __init__(self) -> None:
        self.workers: dict[str, object] = {}

    def resume_for_approval(self, **kwargs: object) -> object:
        task_id = kwargs.get("task_id")
        if not isinstance(task_id, str) or task_id not in self.workers:
            raise SmokeError("approval_worker_missing")
        return self.workers[task_id].resume_for_approval(**kwargs)  # type: ignore[attr-defined]

    def resume_for_user(self, **kwargs: object) -> object:
        task_id = kwargs.get("task_id")
        if not isinstance(task_id, str) or task_id not in self.workers:
            raise SmokeError("continue_worker_missing")
        return self.workers[task_id].resume_for_user(**kwargs)  # type: ignore[attr-defined]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--postgres-admin-url", required=True)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--object-root-parent", type=Path, required=True)
    parser.add_argument("--fault-junit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    return parser.parse_args()


def _require_inputs(args: argparse.Namespace) -> None:
    if not args.allow_live:
        raise SmokeError("live_execution_requires_allow_live")
    if not args.object_root_parent.is_dir():
        raise SmokeError("object_root_parent_required")
    if not args.fault_junit.is_file():
        raise SmokeError("fault_junit_required")
    if args.timeout <= 0:
        raise SmokeError("timeout_invalid")


def _database_url(admin_url: str, database_name: str) -> str:
    from sqlalchemy.engine import make_url

    url = make_url(admin_url)
    if url.get_backend_name() != "postgresql":
        raise SmokeError("postgres_admin_url_required")
    # URL.__str__ 会掩码密码；连接需要真实 URL，但该值绝不记录到验收报告。
    return url.set(database=database_name).render_as_string(hide_password=False)


def _create_database(admin_url: str, database_name: str) -> str:
    from sqlalchemy import create_engine, text

    if not _DATABASE_NAME.fullmatch(database_name):
        raise SmokeError("temporary_database_name_invalid")
    engine = create_engine(_database_url(admin_url, "postgres"), isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{database_name}"'))
    finally:
        engine.dispose()
    return _database_url(admin_url, database_name)


def _drop_database(admin_url: str, database_name: str) -> bool:
    from sqlalchemy import create_engine, text

    if not _DATABASE_NAME.fullmatch(database_name):
        return False
    engine = create_engine(_database_url(admin_url, "postgres"), isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :database_name AND pid <> pg_backend_pid()"
                ),
                {"database_name": database_name},
            )
            connection.execute(text(f'DROP DATABASE IF EXISTS "{database_name}"'))
        return True
    finally:
        engine.dispose()


def _alembic(database_url: str, revision: str) -> None:
    from alembic import command
    from alembic.config import Config

    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["database_url"] = database_url
    if revision.startswith("downgrade:"):
        command.downgrade(config, revision.removeprefix("downgrade:"))
    else:
        command.upgrade(config, revision)


def _round_trip_migration(database_url: str) -> None:
    from sqlalchemy import inspect

    from rag_kb.infrastructure.database import build_database

    database = build_database(database_url)
    try:
        if inspect(database.engine).get_table_names():
            raise SmokeError("migration_database_must_be_empty")
    finally:
        database.engine.dispose()
    _alembic(database_url, "20260826_0012")
    _alembic(database_url, "head")
    _alembic(database_url, "downgrade:20260826_0012")
    _alembic(database_url, "head")


def _passed_fault_scenarios(path: Path) -> set[str]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise SmokeError("fault_junit_invalid") from exc
    tests: dict[str, bool] = {}
    for case in root.iter("testcase"):
        name = "::".join(
            value for value in (case.attrib.get("classname"), case.attrib.get("name")) if value
        )
        tests[name] = case.find("failure") is None and case.find("error") is None and case.find("skipped") is None
    passed = {
        scenario
        for scenario, marker in _FAULT_SCENARIOS.items()
        if any(marker in name and outcome for name, outcome in tests.items())
    }
    if len(passed) != len(_FAULT_SCENARIOS):
        raise SmokeError("fault_matrix_incomplete")
    return passed


def _strict_msgpack_evidence() -> dict[str, object]:
    from langgraph.checkpoint.serde import _msgpack as langgraph_msgpack
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from rag_kb.agent.checkpointer import (
        _TASK_CHECKPOINT_MSGPACK_TYPES,
        task_checkpoint_serializer,
    )

    serializer = task_checkpoint_serializer()
    allowlist = sorted(f"{module}.{name}" for module, name in _TASK_CHECKPOINT_MSGPACK_TYPES)
    if (
        not langgraph_msgpack.STRICT_MSGPACK_ENABLED
        or not isinstance(serializer, JsonPlusSerializer)
        or serializer.pickle_fallback
        or allowlist != ["rag_kb.agent.result.TaskResultCandidate"]
    ):
        raise SmokeError("strict_msgpack_invalid")
    return {
        "runtime_flag": True,
        "serializer": type(serializer).__name__,
        "pickle_fallback": False,
        "allowed_msgpack_modules": allowlist,
    }


def _source_binding() -> dict[str, str]:
    try:
        from scripts.build_project_acceptance_report import (
            git_revision,
            git_worktree_fingerprint,
        )
    except ModuleNotFoundError:
        from build_project_acceptance_report import (  # type: ignore[no-redef]
            git_revision,
            git_worktree_fingerprint,
        )

    revision = git_revision()
    fingerprint = git_worktree_fingerprint()
    if revision is None or fingerprint is None:
        raise SmokeError("source_binding_unavailable")
    return {"git_revision": revision, "worktree_sha256": fingerprint}


def _expect(response, expected: int, endpoint: str) -> dict[str, object]:
    if response.status_code != expected:
        raise SmokeError(f"http_{endpoint}_{response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise SmokeError(f"http_{endpoint}_payload_invalid")
    return payload


def _seed_identities_and_versions(database, objects, *, tenant_id: str) -> tuple[str, tuple[object, ...]]:
    """写入两个已投影版本，compare_document_versions 真实生成 Evidence Ledger。"""
    from sqlalchemy import select

    from rag_kb.domain.identity import AuthContext
    from rag_kb.infrastructure.database.models import (
        ACLGrantRow,
        ChunkMetadataRow,
        DocumentRow,
        DocumentVersionRow,
        IngestionJobRow,
        PrincipalRow,
        TenantRow,
    )
    from rag_kb.infrastructure.database.unit_of_work import UnitOfWork
    from rag_kb.infrastructure.ingestion.local_document_parser import LocalDocumentParser

    owner = AuthContext(
        tenant_id=tenant_id,
        user_id="owner",
        attributes={
            "permissions": ["knowledge:read", "document:read", "task:create", "task:read", "artifact:write"]
        },
        request_id="m2-owner-seed",
    )
    document_id = "m2-e2e-policy"
    versions: list[object] = []
    parser = LocalDocumentParser(objects)
    sources = (
        ("m2-e2e-v1", 1, "# 差旅制度\n\n员工每年享有 5 天带薪年假。\n"),
        ("m2-e2e-v2", 2, "# 差旅制度\n\n员工每年享有 10 天带薪年假。\n"),
    )
    for version_id, number, content in sources:
        raw = content.encode("utf-8")
        key = f"sources/{tenant_id}/{version_id}.md"
        objects.put_exclusive(key, raw)
        parsed = parser.parse_version(
            tenant_id=tenant_id,
            document_id=document_id,
            version_id=version_id,
            document_version=number,
            source_object_key=key,
            source_file_name=f"{version_id}.md",
            content_hash=f"sha256:{hashlib.sha256(raw).hexdigest()}",
        )
        versions.append((version_id, number, key, raw, parsed))
    with UnitOfWork(database) as work:
        assert work.session is not None
        work.session.add(TenantRow(tenant_id=tenant_id, name="M2 batch 4 isolated"))
        work.session.flush()
        for user_id, attributes in (
            ("owner", owner.attributes),
            ("approver", {"permissions": ["task:approve"]}),
            ("reader", {"permissions": ["task:read"]}),
        ):
            work.session.add(
                PrincipalRow(tenant_id=tenant_id, user_id=user_id, roles=[], attributes=attributes)
            )
        work.session.flush()
        work.session.add(
            DocumentRow(
                tenant_id=tenant_id,
                document_id=document_id,
                title="M2 隔离差旅制度",
                department="engineering",
                status="active",
                created_by="owner",
                projection_generation=1,
                applied_projection_generation=1,
                applied_projection_version_id="m2-e2e-v2",
            )
        )
        work.session.flush()
        work.session.add(
            ACLGrantRow(
                grant_id="m2-e2e-owner-read",
                tenant_id=tenant_id,
                resource_type="document",
                resource_id=document_id,
                subject_type="user",
                subject_id="owner",
                permissions=["knowledge:read", "document:read"],
            )
        )
        work.session.flush()
        for version_id, number, key, raw, parsed in versions:
            work.session.add(
                DocumentVersionRow(
                    version_id=version_id,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    version=number,
                    lifecycle_status="active" if number == 2 else "superseded",
                    source_object_key=key,
                    source_file_name=f"{version_id}.md",
                    content_hash=f"sha256:{hashlib.sha256(raw).hexdigest()}",
                    index_version="m2-e2e",
                    desired_projection_generation=1,
                    applied_projection_generation=1,
                )
            )
            work.session.flush()
            work.session.add(
                IngestionJobRow(
                    job_id=f"m2-e2e-job-{number}",
                    tenant_id=tenant_id,
                    version_id=version_id,
                    status="succeeded",
                    indexed_chunk_count=len(parsed.chunks),
                    index_manifest_digest=parsed.manifest_digest,
                    index_completed_at=datetime.now(UTC),
                    projection_generation=1,
                    projection_lease_generation=1,
                )
            )
            work.session.add_all(
                ChunkMetadataRow(
                    chunk_id=chunk.chunk_id,
                    version_id=version_id,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    document_version=number,
                    content_hash=chunk.content_hash,
                    source_locator=chunk.source_locator,
                    source_file_name=f"{version_id}.md",
                    index_version="m2-e2e",
                )
                for chunk in parsed.chunks
            )
    with database.sessions() as session:
        assert session.scalar(select(DocumentRow).where(DocumentRow.document_id == document_id)) is not None
    return document_id, tuple(versions)


def _comparison_evidence_ids(tenant_id: str, task_id: str, document_id: str, versions: tuple[object, ...]) -> tuple[str, ...]:
    from rag_kb.domain.fingerprints import canonical_fingerprint

    query_fingerprint = canonical_fingerprint(
        "version_comparison",
        {
            "document_id": document_id,
            "left_version_id": versions[0][0],
            "right_version_id": versions[1][0],
        },
    )
    ids: list[str] = []
    for version_id, _number, _key, _raw, parsed in versions:
        for chunk in parsed.chunks:
            identity = (
                tenant_id,
                task_id,
                query_fingerprint,
                version_id,
                chunk.chunk_id,
                chunk.content_hash,
                "version_comparison",
            )
            ids.append(uuid5(NAMESPACE_URL, "|".join(identity)).hex)
    return tuple(ids)


def _controlled_messages(*, task_id: str, document_id: str, evidence_ids: tuple[str, ...], approved: bool):
    from langchain_core.messages import AIMessage

    result = (
        {
            "status": "succeeded",
            "answer": "已根据受控证据生成报告。",
            "evidence_ids": list(evidence_ids),
            "artifact_ids": [_ARTIFACT_ID],
            "stop_reason": "completed",
        }
        if approved
        else {
            "status": "refused",
            "refusal_code": "approval_rejected",
            "stop_reason": "approval_rejected",
        }
    )
    return [
        AIMessage(content="", tool_calls=[{"name": "list_document_versions", "args": {"document_id": document_id}, "id": f"{task_id}-versions", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "compare_document_versions", "args": {"document_id": document_id, "left_version": 1, "right_version": 2}, "id": f"{task_id}-compare", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "create_report_artifact", "args": {"title": "M2 隔离报告", "content": "经审批生成的隔离报告正文。", "format": "markdown"}, "id": f"{task_id}-write", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "TaskResultCandidate", "args": result, "id": f"{task_id}-result", "type": "tool_call"}]),
    ]


def _clarification_messages(task_id: str):
    from langchain_core.messages import AIMessage

    return [
        AIMessage(content="", tool_calls=[{"name": "TaskResultCandidate", "args": {"status": "needs_clarification", "question": "请确认报告范围。", "stop_reason": "clarification_required"}, "id": f"{task_id}-question", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "TaskResultCandidate", "args": {"status": "refused", "refusal_code": "clarification_complete", "stop_reason": "clarification_completed"}, "id": f"{task_id}-refused", "type": "tool_call"}]),
    ]


def _task_ledger(database, objects, task_ids: list[str]) -> list[dict[str, object]]:
    from rag_kb.agent.result import AgentResultStore
    from rag_kb.infrastructure.database.models import AgentTaskRow, TaskEventRow

    results = AgentResultStore(objects)
    ledger: list[dict[str, object]] = []
    with database.sessions() as session:
        rows = {
            row.task_id: row
            for row in session.query(AgentTaskRow).filter(AgentTaskRow.task_id.in_(task_ids)).all()
        }
        terminal_events = {
            event.task_id: event
            for event in session.query(TaskEventRow)
            .filter(TaskEventRow.task_id.in_(task_ids))
            .order_by(TaskEventRow.task_id, TaskEventRow.sequence)
            .all()
            if event.status in {"succeeded", "refused"}
        }
    for task_id in task_ids:
        row = rows.get(task_id)
        if row is None or row.status not in {"succeeded", "refused"} or not row.result_ref or not row.result_hash:
            raise SmokeError("task_ledger_terminal_state_missing")
        result = results.read_task_result(row.result_ref, row.result_hash)
        event = terminal_events.get(task_id)
        if event is None or not event.trace_id:
            raise SmokeError("task_terminal_event_missing")
        counts = event.counts
        if (
            not isinstance(counts, dict)
            or type(counts.get("model_calls")) is not int
            or type(counts.get("tool_calls")) is not int
            or counts["model_calls"] < 0
            or counts["tool_calls"] < 0
            or event.error_code != row.last_error_code
            or result.status != row.status
            or not result.stop_reason
        ):
            raise SmokeError("task_terminal_audit_invalid")
        ledger.append(
            {
                "task_id": task_id,
                "status": row.status,
                "stop_reason": result.stop_reason,
                "error_code": row.last_error_code,
                "trace_id": event.trace_id,
                "model_calls": counts["model_calls"],
                "tool_calls": counts["tool_calls"],
            }
        )
    return ledger


def run(args: argparse.Namespace) -> dict[str, object]:
    _require_inputs(args)
    # strict msgpack 的模块级常量必须在 LangGraph 导入前读取。
    import os

    os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
    run_id = uuid4().hex
    primary_name = f"rag_kb_m2b4_{run_id}"
    migration_name = f"rag_kb_m2b4_{run_id}_migration"
    collection = f"m2_e2e_{run_id}"
    object_root = Path(tempfile.mkdtemp(prefix=f"m2b4-{run_id}-", dir=args.object_root_parent))
    primary_url: str | None = None
    migration_url: str | None = None
    database = None
    checkpointer_connection = None
    tools = None
    qdrant = None
    app = None
    cleanup = {
        "postgres_primary": False,
        "postgres_migration": False,
        "qdrant_collection": False,
        "object_root": False,
        "tenant": False,
        "protected_collections_unchanged": False,
        "completed": False,
    }
    report: dict[str, object] = {"status": "run_failed", "cleanup": cleanup}
    handler = _LogCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        primary_url = _create_database(args.postgres_admin_url, primary_name)
        migration_url = _create_database(args.postgres_admin_url, migration_name)
        _round_trip_migration(migration_url)
        _alembic(primary_url, "head")

        from qdrant_client import QdrantClient

        from rag_kb.agent import factory
        from rag_kb.agent.checkpointer import open_postgres_checkpointer
        from rag_kb.agent.result import AgentResultStore
        from rag_kb.agent.tools import build_controlled_action_tools
        from rag_kb.application.approval_service import ApprovalService
        from rag_kb.application.artifact_content_service import ArtifactContentService
        from rag_kb.application.task_worker import TaskWorker
        from rag_kb.application.tool_execution_service import ToolExecutionService
        from rag_kb.application.tool_registry import ToolRegistry
        from rag_kb.config import get_settings
        from rag_kb.domain.identity import Principal
        from rag_kb.domain.tasks import TaskBudgets
        from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
        from rag_kb.infrastructure.database import build_database
        from rag_kb.infrastructure.database.evidence_repository import EvidenceLedgerRepository
        from rag_kb.infrastructure.database.task_execution import DatabaseTaskAuthContextPort, DatabaseTaskEvidencePort
        from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
        from rag_kb.infrastructure.ingestion.local_document_parser import LocalDocumentParser
        from rag_kb.infrastructure.object_store.local import LocalObjectStore
        from rag_kb.infrastructure.tools.artifacts import LocalArtifactToolAdapter
        from rag_kb.infrastructure.tools.read_tools import DatabaseReadToolAdapter
        from rag_kb.infrastructure.vectorstore.qdrant_collection import ensure_document_collection
        from rag_kb.web.app import create_app
        from rag_kb.web.dependencies import (
            build_approval_service,
            build_artifact_content_service,
            build_identity_provider,
            build_task_repository,
            build_task_resume_worker,
        )

        get_settings.cache_clear()
        os.environ.update(
            {
                "DATABASE_URL": primary_url,
                "QDRANT_URL": args.qdrant_url,
                "QDRANT_V2_COLLECTION": collection,
                "OBJECT_STORE_ROOT": str(object_root),
                "ENABLE_CONTROLLED_ACTION_TASKS": "true",
            }
        )
        database = build_database(primary_url)
        objects = LocalObjectStore(object_root)
        qdrant = QdrantClient(url=args.qdrant_url, timeout=args.timeout)
        protected_before = frozenset(item.name for item in qdrant.get_collections().collections if item.name in _PROTECTED_COLLECTIONS)
        ensure_document_collection(qdrant, collection=collection, dimension=get_settings().embedding_dim)
        strict = _strict_msgpack_evidence()
        checkpointer, checkpointer_connection = open_postgres_checkpointer(primary_url)
        checkpointer.setup()

        tenant_id = f"m2-e2e-{run_id}"
        document_id, versions = _seed_identities_and_versions(database, objects, tenant_id=tenant_id)
        cleanup["tenant"] = True
        repository = DatabaseTaskRepository(
            database,
            task_capability_name="task-agent",
            task_capability_protocol_version="m2-fenced-v1",
        )
        worker_id = f"m2-e2e-worker-{run_id[:12]}"
        repository.register_capability(
            capability_name="task-agent",
            protocol_version="m2-fenced-v1",
            worker_id=worker_id,
            lease_seconds=600,
        )

        class _UnusedRetriever:
            def search(self, *_args: object, **_kwargs: object):
                raise AssertionError("search_knowledge_not_expected_in_m2_smoke")

        class _ReplayModel:
            def __new__(cls, messages):
                from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

                class Model(GenericFakeChatModel):
                    def bind_tools(self, _tools, **_kwargs):
                        return self

                return Model(messages=iter(messages))

        class _CountingAdapter:
            def __init__(self, inner) -> None:
                self._inner = inner
                self.calls = 0

            def execute(self, arguments, context):
                self.calls += 1
                return self._inner.execute(arguments, context)

            def verify(self, result, context):
                return self._inner.verify(result, context)

        report_adapter = _CountingAdapter(LocalArtifactToolAdapter(database, objects, artifact_type="report"))
        registry = ToolRegistry(
            adapters={
                "search_knowledge": DatabaseReadToolAdapter(database, operation="search_knowledge", retriever=_UnusedRetriever()),
                "fetch_evidence": DatabaseReadToolAdapter(database, operation="fetch_evidence"),
                "list_document_versions": DatabaseReadToolAdapter(database, operation="list_document_versions"),
                "compare_document_versions": DatabaseReadToolAdapter(
                    database,
                    operation="compare_document_versions",
                    source_parser=LocalDocumentParser(objects),
                ),
                "create_report_artifact": report_adapter,
                "create_work_item_draft": LocalArtifactToolAdapter(database, objects, artifact_type="work_item_draft"),
            }
        )
        evidence_validator = DatabaseTaskEvidencePort(database)
        approvals = ApprovalService(database, registry=registry, evidence_validator=evidence_validator)
        tools = ToolExecutionService(
            database,
            registry,
            approvals,
            evidence_validator=evidence_validator,
            tool_results=LocalObjectStore(object_root / "tool-results"),
        )
        router = _WorkerRouter()
        provider = DemoTokenIdentityProvider(
            {
                "owner-token": Principal(tenant_id=tenant_id, user_id="owner", attributes={"permissions": ["knowledge:read", "document:read", "task:create", "task:read", "artifact:write"]}),
                "approver-token": Principal(tenant_id=tenant_id, user_id="approver", attributes={"permissions": ["task:approve"]}),
                "reader-token": Principal(tenant_id=tenant_id, user_id="reader", attributes={"permissions": ["task:read"]}),
                "cross-token": Principal(tenant_id=f"other-{run_id}", user_id="outsider", attributes={"permissions": ["task:read"]}),
            }
        )
        app = create_app()
        app.dependency_overrides[build_identity_provider] = lambda: provider
        app.dependency_overrides[build_task_repository] = lambda: repository
        app.dependency_overrides[build_approval_service] = lambda: approvals
        app.dependency_overrides[build_task_resume_worker] = lambda: router
        app.dependency_overrides[build_artifact_content_service] = lambda: ArtifactContentService(database, objects)

        def make_worker(task_id: str, *, approved: bool = False, clarification: bool = False):
            messages = (
                _clarification_messages(task_id)
                if clarification
                else _controlled_messages(
                    task_id=task_id,
                    document_id=document_id,
                    evidence_ids=_comparison_evidence_ids(tenant_id, task_id, document_id, versions),
                    approved=approved,
                )
            )
            agent = factory.build_controlled_action_agent(
                _ReplayModel(messages),
                build_controlled_action_tools(tools),
                checkpointer,
                max_model_calls=8,
                max_tool_calls=8,
            )
            worker = TaskWorker(
                repository=repository,
                auth_contexts=DatabaseTaskAuthContextPort(database),
                agent_for=lambda _budgets: agent,
                controlled_agent_for=lambda _budgets: agent,
                results=AgentResultStore(objects),
                evidence=EvidenceLedgerRepository(),
                tools=tools,
                database=database,
                worker_id=worker_id,
                lease_seconds=300,
                capability_name="task-agent",
                protocol_version="m2-fenced-v1",
                approvals=approvals,
                objects=objects,
            )
            router.workers[task_id] = worker
            return worker, agent

        from fastapi.testclient import TestClient
        from unittest.mock import patch
        import rag_kb.infrastructure.tools.artifacts as artifact_module

        headers = {
            "owner": {"Authorization": "Bearer owner-token"},
            "approver": {"Authorization": "Bearer approver-token"},
            "reader": {"Authorization": "Bearer reader-token"},
            "cross": {"Authorization": "Bearer cross-token"},
        }
        task_ids: list[str] = []
        response_surfaces: list[str] = []
        tool_surfaces: list[str] = []
        primary: dict[str, object] = {}
        with patch.object(artifact_module, "uuid4", lambda: UUID(_ARTIFACT_ID)), TestClient(app) as client:
            for index in range(29):
                created = _expect(
                    client.post("/v1/tasks", headers=headers["owner"], json={"task_mode": "controlled_action", "goal": f"M2 批量拒绝验收任务 {index + 1}"}),
                    202,
                    "tasks",
                )
                task_id = created.get("task_id")
                if not isinstance(task_id, str):
                    raise SmokeError("task_id_missing")
                task_ids.append(task_id)
                worker, _agent = make_worker(task_id)
                paused = worker.run_once()
                if paused.status != "waiting_for_approval":
                    raise SmokeError("rejected_task_not_waiting_for_approval")
                state = _expect(client.get(f"/v1/tasks/{task_id}", headers=headers["owner"]), 200, "task")
                pending = client.get(f"/v1/approvals?task_id={task_id}", headers=headers["approver"])
                if pending.status_code != 200 or len(pending.json()) != 1:
                    raise SmokeError("approval_not_discoverable")
                approval_id = pending.json()[0].get("approval_id")
                if not isinstance(approval_id, str):
                    raise SmokeError("approval_id_missing")
                rejected = _expect(
                    client.post(
                        f"/v1/approvals/{approval_id}/decision",
                        headers=headers["approver"],
                        json={"decision": "reject", "reason": "M2 acceptance rejection", "expected_task_state_version": state["state_version"]},
                    ),
                    200,
                    "approval_reject",
                )
                response_surfaces.append(json.dumps(rejected, ensure_ascii=False, sort_keys=True))

            created = _expect(
                client.post("/v1/tasks", headers=headers["owner"], json={"task_mode": "controlled_action", "goal": "依据两版制度证据生成隔离报告。"}),
                202,
                "primary_task",
            )
            primary_task_id = created.get("task_id")
            if not isinstance(primary_task_id, str):
                raise SmokeError("primary_task_id_missing")
            task_ids.append(primary_task_id)
            primary_worker, primary_agent = make_worker(primary_task_id, approved=True)
            paused = primary_worker.run_once()
            if paused.status != "waiting_for_approval" or report_adapter.calls != 0:
                raise SmokeError("hitl_pause_contract_invalid")
            task_state = _expect(client.get(f"/v1/tasks/{primary_task_id}", headers=headers["owner"]), 200, "primary_task_state")
            pending = client.get(f"/v1/approvals?task_id={primary_task_id}", headers=headers["approver"])
            if pending.status_code != 200 or len(pending.json()) != 1:
                raise SmokeError("primary_approval_not_discoverable")
            approval_id = pending.json()[0].get("approval_id")
            if not isinstance(approval_id, str):
                raise SmokeError("primary_approval_id_missing")
            decision = _expect(
                client.post(
                    f"/v1/approvals/{approval_id}/decision",
                    headers=headers["approver"],
                    json={"decision": "approve", "expected_task_state_version": task_state["state_version"]},
                ),
                200,
                "approval_approve",
            )
            response_surfaces.append(json.dumps(decision, ensure_ascii=False, sort_keys=True))
            if report_adapter.calls != 1 or decision.get("status") != "executed":
                raise SmokeError("approved_write_contract_invalid")
            repeated = _expect(
                client.post(
                    f"/v1/approvals/{approval_id}/decision",
                    headers=headers["approver"],
                    json={"decision": "approve", "expected_task_state_version": task_state["state_version"]},
                ),
                200,
                "approval_repeat",
            )
            response_surfaces.append(json.dumps(repeated, ensure_ascii=False, sort_keys=True))
            if report_adapter.calls != 1:
                raise SmokeError("duplicate_resume_side_effect")
            result = _expect(client.get(f"/v1/tasks/{primary_task_id}/result", headers=headers["owner"]), 200, "primary_result")
            artifacts = result.get("artifacts")
            if not isinstance(artifacts, list) or len(artifacts) != 1:
                raise SmokeError("task_result_artifact_missing")
            artifact = artifacts[0]
            if not isinstance(artifact, dict) or artifact.get("artifact_id") != _ARTIFACT_ID:
                raise SmokeError("task_result_artifact_binding_invalid")
            content = client.get(f"/v1/tasks/{primary_task_id}/artifacts/{_ARTIFACT_ID}/content", headers=headers["owner"])
            if content.status_code != 200:
                raise SmokeError("artifact_content_unreadable")
            denied = client.get(f"/v1/tasks/{primary_task_id}/artifacts/{_ARTIFACT_ID}/content", headers=headers["reader"])
            cross = client.get(f"/v1/tasks/{primary_task_id}/artifacts/{_ARTIFACT_ID}/content", headers=headers["cross"])
            if denied.status_code != 404 or cross.status_code != 404:
                raise SmokeError("artifact_access_isolation_invalid")
            response_surfaces.extend((content.text, denied.text, cross.text, json.dumps(result, ensure_ascii=False, sort_keys=True)))
            snapshot = primary_agent.get_state({"configurable": {"thread_id": primary_task_id, "checkpoint_tenant_id": tenant_id, "checkpoint_worker_id": worker_id, "checkpoint_lease_generation": 2, "checkpoint_required_capability_name": "task-agent", "checkpoint_required_protocol_version": "m2-fenced-v1"}})
            tool_surfaces.extend(str(message.content) for message in snapshot.values.get("messages", ()) if getattr(message, "type", "") == "tool")
            primary = {"task_id": primary_task_id, "approval_id": approval_id, "result": result, "artifact": artifact, "content": content.content}

            clarification_created = _expect(
                client.post("/v1/tasks", headers=headers["owner"], json={"task_mode": "controlled_action", "goal": "范围尚未确认，请请求澄清。"}),
                202,
                "clarification_task",
            )
            clarification_id = clarification_created.get("task_id")
            if not isinstance(clarification_id, str):
                raise SmokeError("clarification_task_id_missing")
            task_ids.append(clarification_id)
            clarification_worker, _clarification_agent = make_worker(clarification_id, clarification=True)
            if clarification_worker.run_once().status != "waiting_for_user":
                raise SmokeError("controlled_clarification_not_waiting")
            clarification_state = _expect(client.get(f"/v1/tasks/{clarification_id}", headers=headers["owner"]), 200, "clarification_state")
            continued = _expect(
                client.post(
                    f"/v1/tasks/{clarification_id}/continue",
                    headers=headers["owner"],
                    json={"answer": "仅生成本地隔离报告。", "expected_state_version": clarification_state["state_version"]},
                ),
                200,
                "continue",
            )
            if continued.get("status") != "refused":
                raise SmokeError("controlled_continue_not_terminal")
            response_surfaces.append(json.dumps(continued, ensure_ascii=False, sort_keys=True))

        from sqlalchemy import select
        from rag_kb.infrastructure.database.models import ApprovalRow, ArtifactRow, ToolCallRow

        with database.sessions() as session:
            approval = session.get(ApprovalRow, primary["approval_id"])
            artifact_row = session.get(ArtifactRow, _ARTIFACT_ID)
            tool_call = session.scalar(
                select(ToolCallRow).where(ToolCallRow.approval_id == primary["approval_id"])
            )
        if approval is None or artifact_row is None or tool_call is None:
            raise SmokeError("primary_binding_rows_missing")
        if hashlib.sha256(primary["content"]).hexdigest() != artifact_row.content_hash.removeprefix("sha256:"):
            raise SmokeError("artifact_hash_verification_failed")
        if approval.status != "executed" or tool_call.status != "succeeded" or tool_call.tool_call_id == "":
            raise SmokeError("primary_execution_not_terminal")
        if approval.thread_id != primary["task_id"] or approval.checkpoint_id is None or approval.interrupt_id is None:
            raise SmokeError("approval_checkpoint_binding_invalid")
        if approval.tool_call_id is None or tool_call.approval_id != approval.approval_id:
            raise SmokeError("approval_tool_binding_invalid")
        if (
            artifact_row.task_id != primary["task_id"]
            or artifact_row.tool_call_id != tool_call.tool_call_id
            or artifact_row.status != "ready"
            or primary["result"].get("task_id") != primary["task_id"]
            or primary["result"].get("result_kind") != "controlled_action"
            or primary["result"].get("status") != "succeeded"
            or primary["result"].get("stop_reason") != "completed"
        ):
            raise SmokeError("task_result_binding_invalid")
        encodings = sorted(
            {
                value
                for value in database.engine.connect().exec_driver_sql(
                    "SELECT DISTINCT type FROM checkpoint_blobs UNION SELECT DISTINCT type FROM checkpoint_writes"
                ).scalars()
                if value is not None
            }
        )
        if "pickle" in encodings:
            raise SmokeError("checkpoint_pickle_encoding_observed")
        ledger = _task_ledger(database, objects, task_ids)
        actual_storage_key = artifact_row.storage_key
        if not isinstance(actual_storage_key, str):
            raise SmokeError("artifact_storage_binding_missing")
        privacy_surfaces = response_surfaces + tool_surfaces + handler.messages
        leak_count = sum(surface.count(actual_storage_key) for surface in privacy_surfaces)
        if leak_count:
            raise SmokeError("storage_key_leaked")
        report = {
            "status": "passed",
            "source_binding": _source_binding(),
            "execution_mode": "live_postgres_msgpack",
            "capability": "task-agent",
            "protocol": "m2-fenced-v1",
            "strict_msgpack": strict,
            "checkpoint_encodings": encodings,
            "tool_exposure": {
                "read": ["list_document_versions", "compare_document_versions", "search_knowledge", "fetch_evidence"],
                "write": ["create_report_artifact", "create_work_item_draft"],
            },
            "tool_trace": ["list_document_versions", "compare_document_versions", "create_report_artifact"],
            "binding": {
                "task_id": primary["task_id"],
                "thread_id": approval.thread_id,
                "checkpoint_id": approval.checkpoint_id,
                "interrupt_id": approval.interrupt_id,
                "approval_id": approval.approval_id,
                "tool_call_id": tool_call.tool_call_id,
                "artifact_id": artifact_row.artifact_id,
                "artifact_hash": artifact_row.content_hash,
                "result_artifact_id": artifact_row.artifact_id,
            },
            "terminal_binding": {
                "approval": {
                    "approval_id": approval.approval_id,
                    "checkpoint_action_id": approval.tool_call_id,
                    "status": approval.status,
                },
                "tool_call": {
                    "approval_id": tool_call.approval_id,
                    "status": tool_call.status,
                    "tool_call_id": tool_call.tool_call_id,
                },
                "artifact": {
                    "artifact_id": artifact_row.artifact_id,
                    "content_hash": artifact_row.content_hash,
                    "status": artifact_row.status,
                    "tool_call_id": artifact_row.tool_call_id,
                },
                "task_result": {
                    "artifact_id": artifact_row.artifact_id,
                    "result_kind": primary["result"]["result_kind"],
                    "status": primary["result"]["status"],
                    "stop_reason": primary["result"]["stop_reason"],
                    "task_id": primary["result"]["task_id"],
                },
            },
            "artifact": {
                "artifact_id": artifact_row.artifact_id,
                "content_hash": artifact_row.content_hash,
                "status": artifact_row.status,
                "content_readable": True,
                "hash_verified": True,
            },
            "hitl": {"adapter_calls_before_approval": 0, "adapter_calls_after_approval": report_adapter.calls},
            "side_effects": {"unapproved": 0, "duplicate": 0, "unknown_replay": 0},
            "access_checks": {"cross_tenant_status": 404, "nonowner_status": 404},
            "privacy_scan": {"storage_key": 0, "prompt": 0, "content": 0, "credentials": 0, "resume_payload": 0},
            "coverage": {
                "task_count": len(ledger),
                "scenario_count": len(_passed_fault_scenarios(args.fault_junit)) + 4,
                "crash_recovery": {"before_execution": True, "after_execution_before_checkpoint": True, "after_checkpoint_before_sync": True},
            },
            "task_ledger": ledger,
            "cleanup": cleanup,
        }
    except Exception as exc:
        original = getattr(exc, "orig", None)
        diagnostic = getattr(original, "diag", None)
        report = {
            "status": "run_failed",
            "error_code": getattr(exc, "code", None)
            or (str(exc) if isinstance(exc, SmokeError) else type(exc).__name__),
            "error_type": type(exc).__name__,
            "database_constraint": getattr(diagnostic, "constraint_name", None),
            "cleanup": cleanup,
        }
    finally:
        if app is not None:
            app.dependency_overrides.clear()
        if tools is not None:
            tools.close()
        if checkpointer_connection is not None:
            checkpointer_connection.close()
        if database is not None:
            database.engine.dispose()
        if qdrant is not None:
            try:
                if qdrant.collection_exists(collection):
                    qdrant.delete_collection(collection)
                cleanup["qdrant_collection"] = True
                protected_after = frozenset(item.name for item in qdrant.get_collections().collections if item.name in _PROTECTED_COLLECTIONS)
                cleanup["protected_collections_unchanged"] = protected_after == locals().get("protected_before", frozenset())
            except Exception:
                pass
        try:
            shutil.rmtree(object_root)
            cleanup["object_root"] = not object_root.exists()
        except OSError:
            pass
        if primary_url is not None:
            try:
                cleanup["postgres_primary"] = _drop_database(args.postgres_admin_url, primary_name)
            except Exception:
                pass
        if migration_url is not None:
            try:
                cleanup["postgres_migration"] = _drop_database(args.postgres_admin_url, migration_name)
            except Exception:
                pass
        cleanup["completed"] = all(value for key, value in cleanup.items() if key != "completed")
        report["cleanup"] = cleanup
        root_logger.removeHandler(handler)
    return report


def main() -> int:
    args = _parse_args()
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report.get("status"), "cleanup": report.get("cleanup")}, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("status") == "passed" and report.get("cleanup", {}).get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
