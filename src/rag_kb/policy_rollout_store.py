"""策略落地规划的最小 SQLite 持久化；不保存证据正文或模型输出。"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from rag_kb.models import (
    PlanStatus,
    PolicyRolloutClarification,
    PolicyRolloutPlanVersion,
    PolicyRolloutTaskResponse,
    Role,
)


class PolicyRolloutTaskStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS policy_rollout_tasks (
                    task_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    clarification_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS policy_rollout_plan_versions (
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, version),
                    FOREIGN KEY (task_id) REFERENCES policy_rollout_tasks(task_id) ON DELETE CASCADE
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def create(self, task_id: str, role: Role, goal: str, status: PlanStatus, clarification: PolicyRolloutClarification | None) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO policy_rollout_tasks(task_id, role, goal, status, clarification_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, str(role), goal, str(status), self._json(clarification), self._now()),
            )

    def update_state(
        self,
        task_id: str,
        role: Role,
        status: PlanStatus,
        clarification: PolicyRolloutClarification | None,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE policy_rollout_tasks SET status = ?, clarification_json = ? WHERE task_id = ? AND role = ?",
                (str(status), self._json(clarification), task_id, str(role)),
            )
        return cursor.rowcount == 1

    def claim_clarification(
        self, task_id: str, role: Role, clarification: PolicyRolloutClarification
    ) -> bool:
        """CAS：只有一个请求能从 clarification 领取本次同步规划。"""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE policy_rollout_tasks SET status = ?, clarification_json = ? "
                "WHERE task_id = ? AND role = ? AND status = ?",
                (str(PlanStatus.PLANNING), self._json(clarification), task_id, str(role), str(PlanStatus.CLARIFICATION)),
            )
        return cursor.rowcount == 1

    def complete_claim_and_append_plan(
        self,
        task_id: str,
        role: Role,
        plan: PolicyRolloutPlanVersion,
        clarification: PolicyRolloutClarification | None,
    ) -> PolicyRolloutPlanVersion | None:
        """同一事务内基于 DB 分配版本、追加计划并完成声明。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM policy_rollout_plan_versions WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            completed = plan.model_copy(update={"version": int(row[0])})
            connection.execute(
                "INSERT INTO policy_rollout_plan_versions(task_id, version, plan_json, created_at) VALUES (?, ?, ?, ?)",
                (task_id, completed.version, completed.model_dump_json(), completed.created_at.isoformat()),
            )
            cursor = connection.execute(
                "UPDATE policy_rollout_tasks SET status = ?, clarification_json = ? "
                "WHERE task_id = ? AND role = ? AND status = ?",
                (str(PlanStatus.PLANNED), self._json(clarification), task_id, str(role), str(PlanStatus.PLANNING)),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
        return completed

    def release_claim(self, task_id: str, role: Role, clarification: PolicyRolloutClarification | None) -> bool:
        """失败只释放仍属于当前请求的声明，不覆盖竞争者的已完成结果。"""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE policy_rollout_tasks SET status = ?, clarification_json = ? "
                "WHERE task_id = ? AND role = ? AND status = ?",
                (str(PlanStatus.CLARIFICATION), self._json(clarification), task_id, str(role), str(PlanStatus.PLANNING)),
            )
        return cursor.rowcount == 1

    def get(self, task_id: str, role: Role, run_id: str) -> PolicyRolloutTaskResponse | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT role, goal, status, clarification_json FROM policy_rollout_tasks WHERE task_id = ? AND role = ?",
                (task_id, str(role)),
            ).fetchone()
            if row is None:
                return None
            plans = connection.execute(
                "SELECT plan_json FROM policy_rollout_plan_versions WHERE task_id = ? ORDER BY version",
                (task_id,),
            ).fetchall()
        clarification = PolicyRolloutClarification.model_validate_json(row[3]) if row[3] else None
        return PolicyRolloutTaskResponse(
            task_id=task_id, role=Role(row[0]), goal=row[1], status=PlanStatus(row[2]),
            clarification=clarification,
            plan_versions=[PolicyRolloutPlanVersion.model_validate_json(item[0]) for item in plans],
            run_id=run_id,
        )

    @staticmethod
    def _json(value: PolicyRolloutClarification | None) -> str | None:
        return value.model_dump_json() if value else None

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
