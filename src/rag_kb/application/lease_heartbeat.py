"""可由摄取与投影工作器共享的租约续期心跳。"""

from __future__ import annotations

import threading
from typing import Callable

from rag_kb.application.ports.indexing import RetryableIngestionError


class LeaseHeartbeat:
    def __init__(
        self,
        renew: Callable[[], None] | None,
        lease_seconds: int,
        *,
        on_failure: Callable[[Exception], None] | None = None,
    ) -> None:
        self._renew = renew
        self._on_failure = on_failure
        self._interval = max(0.1, lease_seconds / 3)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def __enter__(self) -> LeaseHeartbeat:
        if self._renew is not None:
            self.renew_now()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        try:
            self.stop()
        except Exception:
            if exc_type is None:
                raise

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._renew_once()
            except Exception as exc:
                self._set_error(exc)
                self._stop.set()

    def _renew_once(self) -> None:
        try:
            assert self._renew is not None
            self._renew()
        except Exception as exc:
            raise RetryableIngestionError("ingestion_lease_renewal_failed") from exc

    def check(self) -> None:
        if self._error is not None:
            raise RetryableIngestionError("ingestion_lease_renewal_failed") from self._error

    def renew_now(self) -> None:
        self.check()
        if self._renew is not None:
            try:
                self._renew_once()
            except Exception as exc:
                self._set_error(exc)
                raise

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
            self._thread = None
        self.check()

    def _set_error(self, exc: Exception) -> None:
        if self._error is not None:
            return
        self._error = exc
        if self._on_failure is not None:
            try:
                self._on_failure(exc)
            except Exception:
                pass
