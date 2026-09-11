"""有界的普通问答 SSE 执行与传输。"""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from threading import BoundedSemaphore, Condition, Lock, Timer
from typing import Iterator

from rag_kb.application.conversation_service import (
    ConversationService,
    ConversationServiceError,
    PreparedAnswer,
)
from rag_kb.application.query_service import KnowledgeQueryExecution
from rag_kb.domain.identity import AuthContext


# ponytail: 进程内固定上限；多 worker 部署出现模型饱和时再换成共享准入控制。
_DEFAULT_MAX_EXECUTIONS = 2
_GENERATION_TIMEOUT_SECONDS = 60
_KEEPALIVE_SECONDS = 15


def _sse(event: str, data: dict[str, object]) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


class AnswerStream:
    """只保留最新草稿、最新状态和一个终态，慢消费者不会反压模型线程。"""

    def __init__(self) -> None:
        self._condition = Condition()
        self._status: str | None = None
        self._status_version = 0
        self._preview: str | None = None
        self._preview_version = 0
        self._terminal: tuple[str, dict[str, object]] | None = None

    def publish_status(self, status: str) -> None:
        with self._condition:
            if self._terminal is not None:
                return
            self._status = status
            self._status_version += 1
            self._condition.notify_all()

    def publish_preview(self, preview: str) -> None:
        with self._condition:
            if self._terminal is not None:
                return
            self._preview = preview
            self._preview_version += 1
            self._condition.notify_all()

    def finish_done(self, result: dict[str, object]) -> bool:
        return self._finish("done", result)

    def finish_error(self, code: str) -> bool:
        return self._finish("error", {"code": code, "refresh_required": True})

    def _finish(self, event: str, data: dict[str, object]) -> bool:
        with self._condition:
            if self._terminal is not None:
                return False
            self._terminal = (event, data)
            self._condition.notify_all()
            return True

    def iter_sse(self, accepted: dict[str, object]) -> Iterator[bytes]:
        yield _sse("turn.accepted", accepted)
        seen_status = 0
        seen_preview = 0
        while True:
            with self._condition:
                changed = self._condition.wait_for(
                    lambda: (
                        self._terminal is not None
                        or self._status_version != seen_status
                        or self._preview_version != seen_preview
                    ),
                    timeout=_KEEPALIVE_SECONDS,
                )
                status = self._status if self._status_version != seen_status else None
                status_version = self._status_version
                preview = self._preview if self._preview_version != seen_preview else None
                preview_version = self._preview_version
                terminal = self._terminal
            if not changed:
                yield b": keepalive\n\n"
                continue
            if status is not None:
                seen_status = status_version
                yield _sse("status", {"stage": status})
            if preview is not None:
                seen_preview = preview_version
                yield _sse("answer.preview", {"answer": preview})
            if terminal is not None:
                event, data = terminal
                yield _sse(event, data)
                return


class AnswerExecutionManager:
    """应用生命周期持有的有限模型执行器。"""

    def __init__(
        self,
        *,
        max_executions: int = _DEFAULT_MAX_EXECUTIONS,
        generation_timeout_seconds: int = _GENERATION_TIMEOUT_SECONDS,
    ) -> None:
        if max_executions < 1 or generation_timeout_seconds < 1:
            raise ValueError("answer_execution_manager_configuration_invalid")
        self._executor = ThreadPoolExecutor(
            max_workers=max_executions,
            thread_name_prefix="rag-kb-answer",
        )
        self._slots = BoundedSemaphore(max_executions)
        self._generation_timeout_seconds = generation_timeout_seconds
        self._lock = Lock()
        self._closed = False
        self._futures: set[Future] = set()
        self._timers: set[Timer] = set()

    def submit(
        self,
        *,
        service: ConversationService,
        auth: AuthContext,
        prepared: PreparedAnswer,
        reserved: bool = False,
    ) -> AnswerStream | None:
        if not reserved and not self.reserve():
            return None
        with self._lock:
            if self._closed:
                self._slots.release()
                return None
            stream = AnswerStream()
            try:
                context = copy_context()
                future = self._executor.submit(
                    context.run,
                    self._run,
                    service,
                    auth,
                    prepared,
                    stream,
                )
            except RuntimeError:
                self._slots.release()
                return None
            self._futures.add(future)
        # Future 可能已完成；必须在释放互斥锁后注册回调，避免同步回调自锁。
        future.add_done_callback(self._discard_future)
        return stream

    def reserve(self) -> bool:
        with self._lock:
            return not self._closed and self._slots.acquire(blocking=False)

    def release_reservation(self) -> None:
        self._slots.release()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        # 已接收的执行有 60 秒模型时限，退出时等待其完成而不是遗留后台线程。
        self._executor.shutdown(wait=True)
        with self._lock:
            timers = tuple(self._timers)
        for timer in timers:
            timer.cancel()
        for timer in timers:
            timer.join()

    def _discard_future(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)

    def _run(
        self,
        service: ConversationService,
        auth: AuthContext,
        prepared: PreparedAnswer,
        stream: AnswerStream,
    ) -> None:
        timer: Timer | None = None

        def generation_started(execution: KnowledgeQueryExecution) -> None:
            nonlocal timer
            with self._lock:
                if timer is not None:
                    return
                timer = Timer(
                    self._generation_timeout_seconds,
                    self._generation_timed_out,
                    args=(service, auth, prepared, execution, stream),
                )
                self._timers.add(timer)
            timer.start()

        try:
            result = service.execute_prepared_answer(
                auth=auth,
                prepared=prepared,
                preview=stream.publish_preview,
                status=stream.publish_status,
                on_generation_started=generation_started,
            )
            stream.finish_done(result.model_dump(mode="json"))
        except ConversationServiceError as exc:
            stream.finish_error(exc.code)
        except Exception:
            service.fail_prepared_answer(
                auth=auth,
                prepared=prepared,
                error_code="knowledge_query_failed",
            )
            stream.finish_error("conversation_content_unavailable")
        finally:
            if timer is not None:
                timer.cancel()
                with self._lock:
                    self._timers.discard(timer)
            self._slots.release()

    def _generation_timed_out(
        self,
        service: ConversationService,
        auth: AuthContext,
        prepared: PreparedAnswer,
        execution: KnowledgeQueryExecution,
        stream: AnswerStream,
    ) -> None:
        failed = service.fail_prepared_answer(
            auth=auth,
            prepared=prepared,
            error_code="answer_generation_timeout",
            execution=execution,
        )
        if failed:
            stream.finish_error("answer_generation_timeout")


def accepted_payload(prepared: PreparedAnswer) -> dict[str, object]:
    return {
        "conversation_id": prepared.turn.conversation_id,
        "turn_id": prepared.turn.turn_id,
        "task_id": prepared.turn.task_id,
        "user_message": {
            "message_id": prepared.user_message.message_id,
            "sequence": prepared.user_message.sequence,
        },
    }
