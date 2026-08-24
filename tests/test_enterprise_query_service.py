"""新版 KnowledgeQueryService 的证据账本与失败关闭契约测试。"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select, update

from rag_kb.answerer import DeepSeekAnswerModel
from rag_kb.application.query_service import KnowledgeQueryService, _SYNC_LEASE_SECONDS
from rag_kb.domain.identity import AuthContext
from rag_kb.domain.retrieval import RetrievalCandidate, RetrievalResult
from rag_kb.domain.tasks import TaskNode, TaskStatus
from rag_kb.infrastructure.database import Base, build_database
from rag_kb.infrastructure.database.models import (
    ACLGrantRow,
    AgentTaskRow,
    ChunkMetadataRow,
    DocumentRow,
    DocumentVersionRow,
    EvidenceRecordRow,
    IngestionJobRow,
    PrincipalRow,
    TenantRow,
)
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository


def _auth() -> AuthContext:
    return AuthContext(
        tenant_id="tenant-a",
        user_id="alice",
        attributes={"permissions": ["knowledge:read", "document:read"]},
        request_id="query-request",
    )


def _candidate() -> RetrievalCandidate:
    text = "授权的差旅政策正文"
    return RetrievalCandidate(
        tenant_id="tenant-a",
        document_id="policy",
        version_id="version-1",
        document_version=1,
        projection_generation=1,
        chunk_id="chunk-1",
        content_hash="sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_locator="section/1",
        source_file_name="policy.md",
        fused_score=1.0,
        text=text,
    )


@pytest.fixture()
def database(tmp_path: Path) -> Database:
    chunk_hash = "sha256:" + hashlib.sha256(
        "授权的差旅政策正文".encode("utf-8")
    ).hexdigest()
    db = build_database(f"sqlite+pysqlite:///{(tmp_path / 'query.sqlite3').as_posix()}")
    Base.metadata.create_all(db.engine)
    with db.sessions.begin() as session:
        session.add(TenantRow(tenant_id="tenant-a", name="租户甲"))
        session.flush()
        session.add(PrincipalRow(tenant_id="tenant-a", user_id="alice", roles=[], attributes={}))
        session.flush()
        session.add(
            DocumentRow(
                tenant_id="tenant-a", document_id="policy", title="差旅政策", department="财务",
                status="active", created_by="alice",
                projection_generation=1, applied_projection_generation=1,
                applied_projection_version_id="version-1",
            )
        )
        session.flush()
        session.add(
            DocumentVersionRow(
                version_id="version-1", tenant_id="tenant-a", document_id="policy", version=1,
                lifecycle_status="active", source_object_key="policy-v1", source_file_name="policy.md",
                content_hash="sha256:source", index_version="index-v1",
                desired_projection_generation=1, applied_projection_generation=1,
            )
        )
        session.flush()
        session.add(
            ChunkMetadataRow(
                chunk_id="chunk-1", version_id="version-1", tenant_id="tenant-a", document_id="policy",
                document_version=1, content_hash=chunk_hash, source_locator="section/1",
                source_file_name="policy.md", index_version="index-v1",
            )
        )
        session.add(
            IngestionJobRow(
                job_id="job-1", tenant_id="tenant-a", version_id="version-1",
                status="succeeded", indexed_chunk_count=1,
                index_manifest_digest="sha256:manifest",
                index_completed_at=datetime.now(UTC),
                projection_generation=1, projection_lease_generation=1,
            )
        )
        session.add(
            ACLGrantRow(
                grant_id="grant-alice", tenant_id="tenant-a", resource_type="document", resource_id="policy",
                subject_type="user", subject_id="alice",
                permissions=["knowledge:read", "document:read"],
            )
        )
    return db


@dataclass
class Verdict:
    verdict: str = "answer"
    answer: str = "根据授权政策，差旅需要按制度执行。"
    refusal_kind: str | None = None
    used_chunk_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.used_chunk_ids is None:
            self.used_chunk_ids = ["chunk-1"]


class FakeRetriever:
    def __init__(self, result: RetrievalResult):
        self.result = result
        self.calls = 0

    def search(self, question: str, auth: AuthContext) -> RetrievalResult:
        self.calls += 1
        return self.result


class FakeModel:
    def __init__(self, verdict: Verdict | None = None, error: Exception | None = None, on_answer=None):
        self.verdict = verdict or Verdict()
        self.error = error
        self.on_answer = on_answer
        self.calls: list[list[RetrievalCandidate]] = []

    def answer(self, question: str, chunks: list[RetrievalCandidate]) -> Verdict:
        self.calls.append(chunks)
        if self.error:
            raise self.error
        if self.on_answer:
            self.on_answer()
        return self.verdict


def _result(candidate: RetrievalCandidate | None = None) -> RetrievalResult:
    return RetrievalResult(
        query_fingerprint="query-fingerprint",
        candidates=() if candidate is None else (candidate,),
        context_token_estimate=3,
        sufficient=candidate is not None,
        insufficiency_code=None if candidate is not None else "no_authorized_evidence",
    )


def _tasks(db: Database) -> list[AgentTaskRow]:
    with db.sessions() as session:
        return list(session.scalars(select(AgentTaskRow)).all())


def test_answer_retrieves_once_records_ledger_and_returns_ledger_citations(database: Database) -> None:
    retriever = FakeRetriever(_result(_candidate()))
    model = FakeModel()
    result = KnowledgeQueryService(database, retriever, model).query("差旅怎么报销", _auth())

    assert result.verdict == "answer"
    assert retriever.calls == 1
    assert len(model.calls) == 1 and model.calls[0][0].chunk_id == "chunk-1"
    assert len(result.citations) == 1 and result.citations[0].chunk_id == "chunk-1"
    with database.sessions() as session:
        task = session.scalar(select(AgentTaskRow))
        evidence = session.scalars(select(EvidenceRecordRow)).all()
        assert task is not None and task.status == "succeeded"
        assert len(evidence) == 1 and evidence[0].version_id == "version-1"
        assert "text" not in EvidenceRecordRow.__table__.columns


@pytest.mark.parametrize(
    "verdict",
    [Verdict(used_chunk_ids=["forged-chunk"]), Verdict(used_chunk_ids=["chunk-1", "chunk-1"])],
)
def test_forged_or_duplicate_model_chunk_ids_fail_closed(database: Database, verdict: Verdict) -> None:
    retriever = FakeRetriever(_result(_candidate()))
    result = KnowledgeQueryService(database, retriever, FakeModel(verdict)).query("问题", _auth())

    assert result.verdict == "refusal"
    assert result.citations == ()
    assert retriever.calls == 1
    assert _tasks(database)[0].status == "failed"


def test_model_exception_fails_task_and_no_answer(database: Database) -> None:
    retriever = FakeRetriever(_result(_candidate()))
    result = KnowledgeQueryService(database, retriever, FakeModel(error=RuntimeError("model down"))).query(
        "问题", _auth()
    )

    assert result.verdict == "refusal"
    assert result.refusal is not None and result.refusal.kind == "retrieval_failure"
    assert retriever.calls == 1
    assert _tasks(database)[0].status == "failed"


def test_no_candidate_skips_model_and_task_creation(database: Database) -> None:
    retriever = FakeRetriever(_result())
    model = FakeModel()
    result = KnowledgeQueryService(database, retriever, model).query("无资料问题", _auth())

    assert result.verdict == "refusal"
    assert result.refusal is not None and result.refusal.kind == "no_authorized_evidence"
    assert retriever.calls == 1 and model.calls == []
    assert _tasks(database) == []


def test_sync_task_is_leased_during_request_and_released_at_terminal(database: Database) -> None:
    """同步问答任务在请求期间持有自租约，异步任务 worker 不得领取；
    终态写入时释放租约（lease 约束要求终态任务无租约）。"""

    def inspect_lease() -> None:
        with database.sessions() as session:
            task = session.scalar(select(AgentTaskRow))
            assert task is not None
            assert task.lease_owner == "api-sync-knowledge-query"
            assert task.lease_expires_at is not None
            assert task.lease_generation == 1
            assert task.status == "running"

    retriever = FakeRetriever(_result(_candidate()))
    result = KnowledgeQueryService(
        database, retriever, FakeModel(on_answer=inspect_lease)
    ).query("问题", _auth())

    assert result.verdict == "answer"
    task = _tasks(database)[0]
    assert task.status == "succeeded"
    assert task.current_node == "terminal"
    assert task.state_version == 2
    assert task.lease_owner is None and task.lease_expires_at is None


def test_sync_task_lease_released_on_failure(database: Database) -> None:
    retriever = FakeRetriever(_result(_candidate()))
    result = KnowledgeQueryService(
        database, retriever, FakeModel(error=RuntimeError("model down"))
    ).query("问题", _auth())

    assert result.verdict == "refusal"
    task = _tasks(database)[0]
    assert task.status == "failed"
    assert task.lease_owner is None and task.lease_expires_at is None


def test_late_sync_answer_cannot_overwrite_recovered_terminal_task(database: Database) -> None:
    def recover_task() -> None:
        with database.engine.begin() as connection:
            connection.execute(
                update(AgentTaskRow)
                .where(AgentTaskRow.task_mode == "knowledge_query")
                .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )
        repository = DatabaseTaskRepository(database)
        task_id = _tasks(database)[0].task_id
        lease = repository.claim_task(
            task_id,
            worker_id="query-recovery",
            lease_seconds=30,
        )
        assert lease is not None
        repository.checkpoint(
            lease,
            lease.state.model_copy(
                update={
                    "status": TaskStatus.FAILED,
                    "current_node": TaskNode.TERMINAL,
                    "last_error": "query_recovered",
                }
            ),
            event_type="query_recovered",
            trace_id="recovery",
            run_id="recovery",
        )

    result = KnowledgeQueryService(
        database,
        FakeRetriever(_result(_candidate())),
        FakeModel(on_answer=recover_task),
    ).query("问题", _auth())

    task = _tasks(database)[0]
    assert result.verdict == "refusal"
    assert task.status == "failed"
    assert task.last_error_code == "query_recovered"
    assert task.lease_generation == 2


def test_expired_sync_query_is_reaped_without_planner(database: Database) -> None:
    with database.sessions.begin() as session:
        session.add(
            AgentTaskRow(
                task_id="expired-query",
                tenant_id="tenant-a",
                user_id="alice",
                task_mode="knowledge_query",
                status="running",
                goal="knowledge_query:test",
                auth_context_ref="sha256:test",
                lease_owner="api-sync-knowledge-query",
                lease_expires_at=datetime.now(UTC) - timedelta(seconds=1),
                lease_generation=1,
            )
        )

    repository = DatabaseTaskRepository(database)
    assert repository.expire_stale_knowledge_queries() == 1
    task = next(item for item in _tasks(database) if item.task_id == "expired-query")
    assert task.status == "failed"
    assert task.current_node == "terminal"
    assert task.last_error_code == "knowledge_query_lease_expired"
    assert task.lease_owner is None and task.lease_expires_at is None


def test_deepseek_answer_timeout_is_shorter_than_sync_lease(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatDeepSeek:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def with_structured_output(self, _schema):
            return object()

    monkeypatch.setitem(
        sys.modules,
        "langchain_deepseek",
        SimpleNamespace(ChatDeepSeek=FakeChatDeepSeek),
    )
    DeepSeekAnswerModel(
        SimpleNamespace(
            deepseek_model="test",
            deepseek_api_key="test",
            deepseek_base_url="https://example.invalid",
        )
    )

    assert isinstance(captured.get("timeout"), int)
    assert captured["timeout"] < _SYNC_LEASE_SECONDS
    assert captured["max_retries"] == 0


def test_revoked_after_model_cannot_produce_citation(database: Database) -> None:
    retriever = FakeRetriever(_result(_candidate()))

    def revoke() -> None:
        with database.sessions.begin() as session:
            session.query(ACLGrantRow).filter_by(grant_id="grant-alice").delete()

    result = KnowledgeQueryService(database, retriever, FakeModel(on_answer=revoke)).query("问题", _auth())

    assert result.verdict == "refusal"
    assert result.refusal is not None and result.refusal.kind == "conflicting_evidence"
    assert result.citations == ()
    assert retriever.calls == 1
    assert _tasks(database)[0].status == "failed"
