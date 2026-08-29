"""运行隔离 controlled-action OTLP smoke，只输出无标识的汇总。"""

from __future__ import annotations

import io
import json
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from rag_kb.config import get_settings
from smoke_m2_controlled_write_e2e import run


def main() -> int:
    settings = get_settings()
    args = Namespace(
        allow_live=True,
        postgres_admin_url=settings.database_url,
        qdrant_url=settings.qdrant_url,
        object_root_parent=Path("/tmp"),
        fault_junit=None,
        report=Path("/tmp/unused-m6-controlled.json"),
        timeout=60.0,
        observability_smoke=True,
    )
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        report = run(args)
    terminal = report.get("terminal_binding", {})
    summary = {
        "status": report.get("status"),
        "error_code": report.get("error_code"),
        "error_type": report.get("error_type"),
        "cleanup": report.get("cleanup"),
        "hitl": report.get("hitl"),
        "side_effects": report.get("side_effects"),
        "privacy_scan": report.get("privacy_scan"),
        "terminal": {
            key: bool(terminal.get(key))
            for key in ("approval", "tool_call", "artifact", "task_result")
        },
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("status") == "passed" and report.get("cleanup", {}).get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
