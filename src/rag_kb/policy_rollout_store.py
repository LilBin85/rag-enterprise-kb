"""策略落地的最小 SQLite 持久化；不保存证据正文或模型输出。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rag_kb.models import (
    ExecutionStatus,
    PlanStatus,
    PolicyRolloutApprovalResponse,
    PolicyRolloutClarification,
    PolicyRolloutExecutionResponse,
    PolicyRolloutPlanVersion,
    PolicyRolloutTaskResponse,
    Role,
)


class PolicyRolloutTaskStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS policy_rollout_tasks (
                    task_id TEXT PRIMARY KEY, role TEXT NOT NULL, goal TEXT NOT NULL,
                    status TEXT NOT NULL, clarification_json TEXT, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS policy_rollout_plan_versions (
                    task_id TEXT NOT NULL, version INTEGER NOT NULL, plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, PRIMARY KEY (task_id, version),
                    FOREIGN KEY (task_id) REFERENCES policy_rollout_tasks(task_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS policy_rollout_approvals (
                    task_id TEXT NOT NULL, version INTEGER NOT NULL, role TEXT NOT NULL,
                    digest TEXT NOT NULL, approved_at TEXT NOT NULL, PRIMARY KEY (task_id, version),
                    FOREIGN KEY (task_id, version) REFERENCES policy_rollout_plan_versions(task_id, version) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS policy_rollout_executions (
                    execution_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, version INTEGER NOT NULL,
                    role TEXT NOT NULL, idempotency_key TEXT NOT NULL, action TEXT NOT NULL,
                    status TEXT NOT NULL, digest TEXT NOT NULL, work_item_id TEXT NOT NULL,
                    draft_path TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(task_id, idempotency_key),
                    FOREIGN KEY (task_id, version) REFERENCES policy_rollout_plan_versions(task_id, version) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS policy_rollout_decision_trace (
                    trace_id INTEGER PRIMARY KEY, task_id TEXT NOT NULL, action TEXT NOT NULL,
                    outcome TEXT NOT NULL, model_decisions INTEGER NOT NULL, search_calls INTEGER NOT NULL,
                    attempted_searches INTEGER NOT NULL DEFAULT 0,
                    executed_searches INTEGER NOT NULL DEFAULT 0,
                    deduplicated_searches INTEGER NOT NULL DEFAULT 0,
                    rejected_searches INTEGER NOT NULL DEFAULT 0,
                    repairs INTEGER NOT NULL, constraint_code TEXT, duration_ms REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES policy_rollout_tasks(task_id) ON DELETE CASCADE
                );
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(policy_rollout_tasks)")}
            if "claimed_at" not in columns:
                connection.execute("ALTER TABLE policy_rollout_tasks ADD COLUMN claimed_at TEXT")
            if "claim_token" not in columns:
                connection.execute("ALTER TABLE policy_rollout_tasks ADD COLUMN claim_token TEXT")
            task_progress_columns = {
                "model_decisions": "INTEGER NOT NULL DEFAULT 0",
                "repairs": "INTEGER NOT NULL DEFAULT 0",
                "attempted_searches": "INTEGER NOT NULL DEFAULT 0",
                "executed_searches": "INTEGER NOT NULL DEFAULT 0",
                "deduplicated_searches": "INTEGER NOT NULL DEFAULT 0",
                "rejected_searches": "INTEGER NOT NULL DEFAULT 0",
                "seen_query_hashes_json": "TEXT NOT NULL DEFAULT '[]'",
                "seen_query_structures_json": "TEXT NOT NULL DEFAULT '[]'",
                "seen_chunk_hashes_json": "TEXT NOT NULL DEFAULT '[]'",
                "query_rewrites": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, definition in task_progress_columns.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE policy_rollout_tasks ADD COLUMN {name} {definition}")
            trace_columns = {row[1] for row in connection.execute("PRAGMA table_info(policy_rollout_decision_trace)")}
            for name in ("attempted_searches", "executed_searches", "deduplicated_searches", "rejected_searches"):
                if name not in trace_columns:
                    connection.execute(
                        f"ALTER TABLE policy_rollout_decision_trace ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                    )
            for execution_id, work_item_id, draft_path in connection.execute(
                "SELECT execution_id, work_item_id, draft_path FROM policy_rollout_executions"
            ):
                filename = self._safe_draft_filename(draft_path, work_item_id)
                if filename != draft_path:
                    connection.execute(
                        "UPDATE policy_rollout_executions SET draft_path = ? WHERE execution_id = ?",
                        (filename, execution_id),
                    )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, task_id: str, role: Role, goal: str, status: PlanStatus, clarification: PolicyRolloutClarification | None, *, claim_token: str | None = None, now: datetime | None = None) -> None:
        if status == PlanStatus.PLANNING and not claim_token:
            raise ValueError("planning_claim_token_required")
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO policy_rollout_tasks(task_id, role, goal, status, clarification_json, created_at, claimed_at, claim_token) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (task_id, str(role), goal, str(status), self._json(clarification), timestamp, timestamp if status == PlanStatus.PLANNING else None, claim_token),
            )

    def update_state(self, task_id: str, role: Role, status: PlanStatus, clarification: PolicyRolloutClarification | None, *, claim_token: str | None = None, trace_event: dict | None = None) -> bool:
        if status in {PlanStatus.CLARIFICATION, PlanStatus.FAILED, PlanStatus.REFUSED} and not claim_token:
            raise ValueError("planning_claim_token_required")
        token_clause = " AND status = ? AND claim_token = ?" if claim_token else ""
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE policy_rollout_tasks SET status = ?, clarification_json = ?, claimed_at = NULL, claim_token = NULL WHERE task_id = ? AND role = ?" + token_clause,
                (str(status), self._json(clarification), task_id, str(role), *([str(PlanStatus.PLANNING), claim_token] if claim_token else [])),
            )
            if cursor.rowcount == 1 and trace_event:
                self._insert_trace_event(connection, task_id, trace_event)
        return cursor.rowcount == 1

    def planning_progress(self, task_id: str, role: Role, claim_token: str) -> dict[str, int | set[str]] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT model_decisions, repairs, attempted_searches, executed_searches, "
                "deduplicated_searches, rejected_searches, seen_query_hashes_json, "
                "seen_query_structures_json, seen_chunk_hashes_json, query_rewrites "
                "FROM policy_rollout_tasks WHERE task_id = ? AND role = ? AND status = ? AND claim_token = ?",
                (task_id, str(role), str(PlanStatus.PLANNING), claim_token),
            ).fetchone()
        if row is None:
            return None
        try:
            query_hashes = {item for item in json.loads(row[6]) if isinstance(item, str)}
        except (TypeError, ValueError):
            query_hashes = set()
        try:
            query_structures = {item for item in json.loads(row[7]) if isinstance(item, str)}
        except (TypeError, ValueError):
            query_structures = set()
        try:
            chunk_hashes = {item for item in json.loads(row[8]) if isinstance(item, str)}
        except (TypeError, ValueError):
            chunk_hashes = set()
        return {
            "model_decisions": int(row[0]),
            "repairs": int(row[1]),
            "attempted_searches": int(row[2]),
            "executed_searches": int(row[3]),
            "deduplicated_searches": int(row[4]),
            "rejected_searches": int(row[5]),
            "seen_query_hashes": query_hashes,
            "seen_query_structures": query_structures,
            "seen_chunk_hashes": chunk_hashes,
            "query_rewrites": int(row[9]),
        }

    def save_planning_progress(
        self, task_id: str, role: Role, claim_token: str, *, model_decisions: int,
        repairs: int, attempted_searches: int, executed_searches: int,
        deduplicated_searches: int, rejected_searches: int,
        seen_query_hashes: set[str], seen_query_structures: set[str] | None = None,
        seen_chunk_hashes: set[str] | None = None, query_rewrites: int = 0,
    ) -> bool:
        """以 CAS 保存任务级预算；查询仅保存不可逆哈希。"""
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE policy_rollout_tasks SET model_decisions = ?, repairs = ?, attempted_searches = ?, "
                "executed_searches = ?, deduplicated_searches = ?, rejected_searches = ?, "
                "seen_query_hashes_json = ?, seen_query_structures_json = ?, "
                "seen_chunk_hashes_json = ?, query_rewrites = ? "
                "WHERE task_id = ? AND role = ? AND status = ? AND claim_token = ?",
                (
                    model_decisions, repairs, attempted_searches, executed_searches,
                    deduplicated_searches, rejected_searches,
                    json.dumps(sorted(seen_query_hashes), ensure_ascii=True),
                    json.dumps(sorted(seen_query_structures or set()), ensure_ascii=True),
                    json.dumps(sorted(seen_chunk_hashes or set()), ensure_ascii=True),
                    query_rewrites,
                    task_id, str(role), str(PlanStatus.PLANNING), claim_token,
                ),
            )
        return cursor.rowcount == 1

    def append_trace_event(
        self, task_id: str, role: Role, claim_token: str, *, action: str, outcome: str,
        model_decisions: int, search_calls: int, repairs: int,
        constraint_code: str | None, duration_ms: float,
        attempted_searches: int | None = None, executed_searches: int | None = None,
        deduplicated_searches: int = 0, rejected_searches: int = 0,
    ) -> bool:
        """仅保存计数和代码，绝不保存目标、查询、正文、提示词或证据标识。"""
        with self._connection() as connection:
            active = connection.execute(
                "SELECT 1 FROM policy_rollout_tasks WHERE task_id = ? AND role = ? AND status = ? AND claim_token = ?",
                (task_id, str(role), str(PlanStatus.PLANNING), claim_token),
            ).fetchone()
            if active is None:
                return False
            self._insert_trace_event(connection, task_id, {
                "action": action, "outcome": outcome, "model_decisions": model_decisions,
                "search_calls": search_calls, "repairs": repairs,
                "attempted_searches": search_calls if attempted_searches is None else attempted_searches,
                "executed_searches": search_calls if executed_searches is None else executed_searches,
                "deduplicated_searches": deduplicated_searches,
                "rejected_searches": rejected_searches,
                "constraint_code": constraint_code, "duration_ms": duration_ms,
            })
        return True

    def _insert_trace_event(self, connection: sqlite3.Connection, task_id: str, event: dict) -> None:
        connection.execute(
            "INSERT INTO policy_rollout_decision_trace("
            "task_id, action, outcome, model_decisions, search_calls, attempted_searches, "
            "executed_searches, deduplicated_searches, rejected_searches, repairs, "
            "constraint_code, duration_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id, event["action"], event["outcome"], event["model_decisions"],
                event["search_calls"], event.get("attempted_searches", event["search_calls"]),
                event.get("executed_searches", event["search_calls"]),
                event.get("deduplicated_searches", 0), event.get("rejected_searches", 0),
                event["repairs"], event["constraint_code"], event["duration_ms"], self._timestamp(),
            ),
        )

    def decision_trace(self, task_id: str) -> list[dict[str, str | int | float | None]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT action, outcome, model_decisions, search_calls, attempted_searches, "
                "executed_searches, deduplicated_searches, rejected_searches, repairs, "
                "constraint_code, duration_ms FROM policy_rollout_decision_trace "
                "WHERE task_id = ? ORDER BY trace_id",
                (task_id,),
            ).fetchall()
        keys = (
            "action", "outcome", "model_decisions", "search_calls", "attempted_searches",
            "executed_searches", "deduplicated_searches", "rejected_searches", "repairs",
            "constraint_code", "duration_ms",
        )
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def claim_clarification(self, task_id: str, role: Role, clarification: PolicyRolloutClarification, *, claim_token: str, now: datetime | None = None, ttl_seconds: int = 300) -> bool:
        """CAS 领取澄清任务；只回收已过期的 planning 声明。"""
        if not claim_token:
            raise ValueError("planning_claim_token_required")
        timestamp = self._timestamp(now)
        cutoff = self._timestamp((now or datetime.now(UTC)) - timedelta(seconds=ttl_seconds))
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE policy_rollout_tasks SET status = ?, clarification_json = ?, claimed_at = ?, claim_token = ? "
                "WHERE task_id = ? AND role = ? AND (status = ? OR (status = ? AND claimed_at IS NOT NULL AND claimed_at <= ?))",
                (str(PlanStatus.PLANNING), self._json(clarification), timestamp, claim_token, task_id, str(role), str(PlanStatus.CLARIFICATION), str(PlanStatus.PLANNING), cutoff),
            )
        return cursor.rowcount == 1

    def complete_claim_and_append_plan(self, task_id: str, role: Role, plan: PolicyRolloutPlanVersion, clarification: PolicyRolloutClarification | None, claim_token: str, *, trace_event: dict | None = None) -> PolicyRolloutPlanVersion | None:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE policy_rollout_tasks SET status = ?, clarification_json = ?, claimed_at = NULL, claim_token = NULL WHERE task_id = ? AND role = ? AND status = ? AND claim_token = ?",
                (str(PlanStatus.PLANNED), self._json(clarification), task_id, str(role), str(PlanStatus.PLANNING), claim_token),
            )
            if cursor.rowcount != 1:
                return None
            row = connection.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM policy_rollout_plan_versions WHERE task_id = ?", (task_id,)).fetchone()
            completed = plan.model_copy(update={"version": int(row[0])})
            connection.execute("INSERT INTO policy_rollout_plan_versions(task_id, version, plan_json, created_at) VALUES (?, ?, ?, ?)", (task_id, completed.version, completed.model_dump_json(), completed.created_at.isoformat()))
            if trace_event:
                self._insert_trace_event(connection, task_id, trace_event)
        return completed

    def release_claim(self, task_id: str, role: Role, clarification: PolicyRolloutClarification | None, claim_token: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE policy_rollout_tasks SET status = ?, clarification_json = ?, claimed_at = NULL, claim_token = NULL WHERE task_id = ? AND role = ? AND status = ? AND claim_token = ?",
                (str(PlanStatus.CLARIFICATION), self._json(clarification), task_id, str(role), str(PlanStatus.PLANNING), claim_token),
            )
        return cursor.rowcount == 1

    def reclaim_stale_planning(self, task_id: str, role: Role, ttl_seconds: int, *, now: datetime | None = None) -> bool:
        """仅释放已超过 TTL 的规划声明，活动声明绝不回收。"""
        cutoff = self._timestamp((now or datetime.now(UTC)) - timedelta(seconds=ttl_seconds))
        with self._connection() as connection:
            row = connection.execute(
                "SELECT clarification_json FROM policy_rollout_tasks WHERE task_id = ? AND role = ? "
                "AND status = ? AND claimed_at IS NOT NULL AND claimed_at <= ?",
                (task_id, str(role), str(PlanStatus.PLANNING), cutoff),
            ).fetchone()
            if row is None:
                return False
            try:
                clarification = PolicyRolloutClarification.model_validate_json(row[0]) if row[0] else None
            except (TypeError, ValueError):
                clarification = None
            status = PlanStatus.CLARIFICATION if clarification and clarification.questions else PlanStatus.FAILED
            cursor = connection.execute(
                "UPDATE policy_rollout_tasks SET status = ?, claimed_at = NULL, claim_token = NULL WHERE task_id = ? AND role = ? "
                "AND status = ? AND claimed_at IS NOT NULL AND claimed_at <= ?",
                (str(status), task_id, str(role), str(PlanStatus.PLANNING), cutoff),
            )
        return cursor.rowcount == 1

    def get(self, task_id: str, role: Role, run_id: str) -> PolicyRolloutTaskResponse | None:
        with self._connection() as connection:
            row = connection.execute("SELECT role, goal, status, clarification_json FROM policy_rollout_tasks WHERE task_id = ? AND role = ?", (task_id, str(role))).fetchone()
            if row is None:
                return None
            plans = connection.execute("SELECT plan_json FROM policy_rollout_plan_versions WHERE task_id = ? ORDER BY version", (task_id,)).fetchall()
        return PolicyRolloutTaskResponse(task_id=task_id, role=Role(row[0]), goal=row[1], status=PlanStatus(row[2]), clarification=PolicyRolloutClarification.model_validate_json(row[3]) if row[3] else None, plan_versions=[PolicyRolloutPlanVersion.model_validate_json(item[0]) for item in plans], run_id=run_id)

    def get_plan(self, task_id: str, version: int, role: Role) -> tuple[str, PolicyRolloutClarification | None, PolicyRolloutPlanVersion] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT t.goal, t.clarification_json, p.plan_json FROM policy_rollout_tasks t JOIN policy_rollout_plan_versions p ON p.task_id = t.task_id WHERE t.task_id = ? AND p.version = ? AND t.role = ?",
                (task_id, version, str(role)),
            ).fetchone()
        if row is None:
            return None
        return row[0], PolicyRolloutClarification.model_validate_json(row[1]) if row[1] else None, PolicyRolloutPlanVersion.model_validate_json(row[2])

    def approve(self, task_id: str, version: int, role: Role, digest: str, *, now: datetime | None = None) -> PolicyRolloutApprovalResponse | None:
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            existing = connection.execute("SELECT role, digest, approved_at FROM policy_rollout_approvals WHERE task_id = ? AND version = ?", (task_id, version)).fetchone()
            if existing:
                if existing[0] != str(role) or existing[1] != digest:
                    return None
                return PolicyRolloutApprovalResponse(task_id=task_id, version=version, role=role, digest=digest, approved_at=datetime.fromisoformat(existing[2]))
            try:
                connection.execute("INSERT INTO policy_rollout_approvals(task_id, version, role, digest, approved_at) VALUES (?, ?, ?, ?, ?)", (task_id, version, str(role), digest, timestamp))
            except sqlite3.IntegrityError:
                return None
        return PolicyRolloutApprovalResponse(task_id=task_id, version=version, role=role, digest=digest, approved_at=datetime.fromisoformat(timestamp))

    def approval(self, task_id: str, version: int, role: Role) -> PolicyRolloutApprovalResponse | None:
        with self._connection() as connection:
            row = connection.execute("SELECT digest, approved_at FROM policy_rollout_approvals WHERE task_id = ? AND version = ? AND role = ?", (task_id, version, str(role))).fetchone()
        return PolicyRolloutApprovalResponse(task_id=task_id, version=version, role=role, digest=row[0], approved_at=datetime.fromisoformat(row[1])) if row else None

    def claim_execution(self, response: PolicyRolloutExecutionResponse) -> PolicyRolloutExecutionResponse:
        try:
            with self._connection() as connection:
                existing = connection.execute("SELECT * FROM policy_rollout_executions WHERE task_id = ? AND idempotency_key = ?", (response.task_id, response.idempotency_key)).fetchone()
                if existing:
                    return self._execution_from_row(existing)
                connection.execute(
                    "INSERT INTO policy_rollout_executions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (response.execution_id, response.task_id, response.version, str(response.role), response.idempotency_key, response.action, str(response.status), response.digest, response.work_item_id, response.draft_path, response.created_at.isoformat()),
                )
        except sqlite3.IntegrityError:
            with self._connection() as connection:
                winner = connection.execute("SELECT * FROM policy_rollout_executions WHERE task_id = ? AND idempotency_key = ?", (response.task_id, response.idempotency_key)).fetchone()
            if winner is not None:
                return self._execution_from_row(winner)
            raise
        return response

    def update_execution_status(self, execution_id: str, status: ExecutionStatus) -> PolicyRolloutExecutionResponse | None:
        with self._connection() as connection:
            connection.execute("UPDATE policy_rollout_executions SET status = ? WHERE execution_id = ?", (str(status), execution_id))
            row = connection.execute("SELECT * FROM policy_rollout_executions WHERE execution_id = ?", (execution_id,)).fetchone()
        return self._execution_from_row(row) if row else None

    def get_execution(self, task_id: str, role: Role, *, idempotency_key: str | None = None, execution_id: str | None = None) -> PolicyRolloutExecutionResponse | None:
        if bool(idempotency_key) == bool(execution_id):
            return None
        column, value = ("idempotency_key", idempotency_key) if idempotency_key else ("execution_id", execution_id)
        with self._connection() as connection:
            row = connection.execute(f"SELECT * FROM policy_rollout_executions WHERE task_id = ? AND role = ? AND {column} = ?", (task_id, str(role), value)).fetchone()
        return self._execution_from_row(row) if row else None

    @staticmethod
    def _execution_from_row(row: tuple) -> PolicyRolloutExecutionResponse:
        return PolicyRolloutExecutionResponse(execution_id=row[0], task_id=row[1], version=row[2], role=Role(row[3]), idempotency_key=row[4], action=row[5], status=ExecutionStatus(row[6]), digest=row[7], work_item_id=row[8], draft_path=row[9], created_at=datetime.fromisoformat(row[10]))

    @staticmethod
    def _json(value: PolicyRolloutClarification | None) -> str | None:
        return value.model_dump_json() if value else None

    @staticmethod
    def _timestamp(value: datetime | None = None) -> str:
        return (value or datetime.now(UTC)).astimezone(UTC).isoformat()

    @staticmethod
    def _safe_draft_filename(value: str, work_item_id: str) -> str:
        filename = value.replace("\\", "/").rsplit("/", 1)[-1]
        if filename and filename not in {".", ".."} and all(separator not in filename for separator in ("/", "\\", ":")):
            return filename
        return f"{work_item_id}.json"
