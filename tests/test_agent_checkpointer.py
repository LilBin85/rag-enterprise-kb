"""M2 checkpoint 序列化必须保持最小 allowlist 且 strict 开关真实生效。"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from langgraph.checkpoint.serde._msgpack import STRICT_MSGPACK_ENABLED

from rag_kb.agent.checkpointer import (
    _TASK_CHECKPOINT_MSGPACK_TYPES,
    task_checkpoint_serializer,
)
from rag_kb.agent.result import TaskResultCandidate


ROOT_DIR = Path(__file__).resolve().parents[1]


def _run_clean_process(code: str) -> subprocess.CompletedProcess[str]:
    """在没有 LANGGRAPH_STRICT_MSGPACK 预设的干净进程中执行导入链验证。

    rag_kb 根包 bootstrap 必须在该进程内打开开关；否则 strict 常量冻结为 False。
    """
    env = {key: value for key, value in os.environ.items() if key != "LANGGRAPH_STRICT_MSGPACK"}
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
        env=env,
        timeout=120,
        check=False,
    )


def test_import_main_and_worker_enable_strict_msgpack_in_clean_process() -> None:
    for entrypoint in ("rag_kb.main", "rag_kb.worker"):
        result = _run_clean_process(
            "import rag_kb."
            + entrypoint.split(".", 1)[1]
            + ";"
            + "from langgraph.checkpoint.serde._msgpack import STRICT_MSGPACK_ENABLED;"
            + "assert STRICT_MSGPACK_ENABLED is True, STRICT_MSGPACK_ENABLED;"
            + "print('strict-ok')"
        )
        assert result.returncode == 0, result.stderr or result.stdout
        assert result.stdout.strip().endswith("strict-ok")


def test_langgraph_first_import_fails_closed_in_serializer() -> None:
    """先导入 LangGraph 再导入 checkpointer 时，序列化器必须 fail-close。"""
    result = _run_clean_process(
        "from langgraph.checkpoint.serde._msgpack import STRICT_MSGPACK_ENABLED;"
        "assert STRICT_MSGPACK_ENABLED is False, 'preset_env_leaked';"
        "from rag_kb.agent.checkpointer import task_checkpoint_serializer\n"
        "try:\n"
        "    task_checkpoint_serializer()\n"
        "except ValueError as exc:\n"
        "    assert 'langgraph_strict_msgpack_required' in str(exc), exc\n"
        "    print('fail-closed')\n"
        "else:\n"
        "    raise SystemExit('serializer_did_not_fail_close')"
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip().endswith("fail-closed")


def test_task_checkpoint_serializer_round_trips_structured_result_without_pickle() -> None:
    candidate = TaskResultCandidate(
        status="refused",
        refusal_code="insufficient_evidence",
        stop_reason="no_evidence",
    )

    serializer = task_checkpoint_serializer()

    assert serializer.pickle_fallback is False
    assert _TASK_CHECKPOINT_MSGPACK_TYPES == (
        ("rag_kb.agent.result", "TaskResultCandidate"),
    )
    assert serializer._allowed_msgpack_modules == set(_TASK_CHECKPOINT_MSGPACK_TYPES)
    assert serializer.loads_typed(serializer.dumps_typed(candidate)) == candidate


def test_pickle_payload_is_rejected_without_fallback() -> None:
    import pickle

    with pytest.raises(Exception):
        task_checkpoint_serializer().loads_typed(("pickle", pickle.dumps({"x": 1})))
