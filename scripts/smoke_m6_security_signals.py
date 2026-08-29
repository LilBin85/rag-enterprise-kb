"""在临时 Worker 中以真实服务触发 UNKNOWN WRITE 与安全边界信号。"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, "/m6tests")

import test_approval_tool_execution as tool_test
import test_enterprise_retriever_security as retrieval_test
from rag_kb.application.enterprise_retriever import EnterpriseRetriever, RetrievalSecurityError
from rag_kb.application.tool_registry import ToolRegistry
from rag_kb.observability import Observability
from rag_kb.worker import _WorkerMetricsServer


def main() -> int:
    logging.disable(logging.CRITICAL)
    telemetry = Observability(
        enabled=True,
        metrics_enabled=True,
        service_name="rag-kb-m6-signal",
        otlp_endpoint="http://otel-collector:4317",
        trace_sample_ratio=1.0,
    )
    server = _WorkerMetricsServer(telemetry, port=9464)
    with tempfile.TemporaryDirectory(prefix="rag-kb-m6-signal-") as directory:
        stage = tool_test.stage5.__wrapped__(Path(directory))
        database, requester, approver, _, lease = next(stage)
        service, binding = tool_test._approved(database, requester, approver)
        executor = tool_test._executor(
            database,
            ToolRegistry(
                adapters={"create_report_artifact": tool_test._CrashingAdapter()}
            ),
            service,
        )
        try:
            if not server.start():
                return 1
            time.sleep(20)
            with telemetry.activate():
                result = executor.execute(
                    tool_test._request(requester, binding.approval_id, lease)
                )
                if result.status != "unknown":
                    return 1
                with pytest.raises(RetrievalSecurityError):
                    EnterpriseRetriever(
                        retrieval_test.Authorization(),
                        retrieval_test.Vectors(
                            retrieval_test.replace(
                                retrieval_test.chunk(), text="synthetic tamper"
                            )
                        ),
                    ).search("synthetic query", retrieval_test.auth())
            time.sleep(150)
            return 0
        finally:
            server.close()
            executor.close()
            try:
                next(stage)
            except StopIteration:
                pass
            telemetry.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
