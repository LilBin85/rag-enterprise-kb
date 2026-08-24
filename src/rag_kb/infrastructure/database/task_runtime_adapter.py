"""把数据库租约会话适配为应用层 TaskRuntime 的 CAS 仓储端口。"""

from __future__ import annotations

from datetime import datetime

from rag_kb.application.pending_action_store import (
    PendingActionStore,
    PendingActionStoreError,
)
from rag_kb.domain.tasks import PendingActionBinding, PersistableAction, TaskState
from rag_kb.infrastructure.database.task_repository import (
    DatabaseTaskRepository,
    TaskLease,
    TaskRepositoryError,
)


class LeasedTaskRuntimeRepository:
    def __init__(
        self,
        repository: DatabaseTaskRepository,
        lease: TaskLease,
        *,
        trace_id: str,
        run_id: str,
        pending_actions: PendingActionStore | None = None,
    ) -> None:
        if not trace_id or not run_id:
            raise ValueError("task_trace_binding_required")
        self._repository = repository
        self._lease = lease
        self._trace_id = trace_id
        self._run_id = run_id
        self._pending_actions = pending_actions

    @property
    def budgets(self):
        return self._lease.budgets

    @property
    def lease(self) -> TaskLease:
        return self._lease

    def load(self, task_id: str) -> TaskState:
        if task_id != self._lease.state.task_id:
            raise TaskRepositoryError("task_checkpoint_binding_invalid")
        return self._lease.state

    def save(
        self,
        state: TaskState,
        *,
        expected_state_version: int,
        event: dict[str, object] | None = None,
    ) -> TaskState:
        if expected_state_version != self._lease.state.state_version:
            raise TaskRepositoryError("task_state_conflict")
        event_type = str((event or {}).get("event_type") or "task_checkpointed")
        persisted = self._repository.checkpoint(
            self._lease,
            state,
            event_type=event_type,
            trace_id=self._trace_id,
            run_id=self._run_id,
        )
        self._lease = TaskLease(
            state=persisted,
            worker_id=self._lease.worker_id,
            lease_generation=self._lease.lease_generation,
            budgets=self._lease.budgets,
        )
        return persisted

    def stage_pending_action(
        self,
        state: TaskState,
        *,
        action: PersistableAction,
        expires_at: datetime,
    ) -> PendingActionBinding:
        """先发布对象；调用方随后可在自己的数据库事务中绑定该引用。"""

        if self._pending_actions is None:
            raise PendingActionStoreError("pending_action_store_unavailable")
        if state.task_id != self._lease.state.task_id:
            raise TaskRepositoryError("task_checkpoint_binding_invalid")
        return self._pending_actions.stage(
            task_id=state.task_id,
            auth_context_ref=state.auth_context_ref,
            plan_revision=state.plan_revision,
            source_state_version=self._lease.state.state_version,
            action=action,
            expires_at=expires_at,
        )

    def save_pending_action(
        self,
        state: TaskState,
        *,
        expected_state_version: int,
        action: PersistableAction,
        expires_at: datetime,
        event: dict[str, object] | None = None,
    ) -> TaskState:
        """对象先写、数据库后 CAS；失败时只可能留下未引用孤儿对象。"""

        binding = self.stage_pending_action(
            state,
            action=action,
            expires_at=expires_at,
        )
        return self.save(
            state.model_copy(update={"pending_action": binding}),
            expected_state_version=expected_state_version,
            event=event,
        )

    def load_pending_action(self) -> PersistableAction:
        if self._pending_actions is None:
            raise PendingActionStoreError("pending_action_store_unavailable")
        state = self._lease.state
        if state.pending_action is None:
            raise PendingActionStoreError("pending_action_binding_missing")
        return self._pending_actions.load(
            state.pending_action,
            task_id=state.task_id,
            auth_context_ref=state.auth_context_ref,
            plan_revision=state.plan_revision,
        )

    def refresh_lease(self, *, lease_seconds: int | None = None) -> TaskState:
        self._lease = self._repository.refresh_lease(
            self._lease,
            lease_seconds=lease_seconds,
        )
        return self._lease.state

    def save_plan_revision(self, state: TaskState, *, steps: tuple[str, ...]) -> TaskState:
        """将重规划写成真实计划行，再推进 runtime 的当前 revision。"""

        self._lease = self._repository.create_plan_revision(
            self._lease,
            state,
            steps=steps,
            trace_id=self._trace_id,
            run_id=self._run_id,
        )
        return self._lease.state

    def release_lease(self) -> None:
        self._repository.release_lease(self._lease)
