from __future__ import annotations

import pytest

from scripts.diagnose_retrieval import main, run


def test_diagnostic_rejects_unconfigured_role(monkeypatch: pytest.MonkeyPatch) -> None:
    class Settings:
        demo_role_tokens = {}

    monkeypatch.setattr("scripts.diagnose_retrieval.get_settings", lambda: Settings())

    with pytest.raises(RuntimeError, match="diagnostic_demo_role_unavailable"):
        run(["--query", "差旅政策", "--role", "missing"])


def test_diagnostic_main_returns_only_stable_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_argv) -> int:
        raise RuntimeError("secret query text")

    monkeypatch.setattr("scripts.diagnose_retrieval.run", fail)

    assert main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"status": "error", "error_code": "exception_runtimeerror"}\n'
