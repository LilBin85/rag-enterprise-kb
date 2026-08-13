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
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(policy_rollout_tasks)")}
            if "claimed_at" not in columns:
                connection.execute("ALTER TABLE policy_rollout_tasks ADD COLUMN claimed_at TEXT")
            if "claim_token" not in columns:
                connection.execute("ALTER TABLE policy_rollout_tasks ADD COLUMN claim_token TEXT")
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

    def update_state(self, task_id: str, role: Role, status: PlanStatus, clarification: PolicyRolloutClarification | None, *, claim_token: str | None = None) -> bool:
        token_clause = " AND claim_token = ?" if claim_token else ""
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE policy_rollout_tasks SET status = ?, clarification_json = ?, claimed_at = NULL, claim_token = NULL WHERE task_id = ? AND role = ?" + token_clause,
                (str(status), self._json(clarification), task_id, str(role), *([claim_token] if claim_token else [])),
            )
        return cursor.rowcount == 1

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

    def complete_claim_and_append_plan(self, task_id: str, role: Role, plan: PolicyRolloutPlanVersion, clarification: PolicyRolloutClarification | None, claim_token: str) -> PolicyRolloutPlanVersion | None:
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
