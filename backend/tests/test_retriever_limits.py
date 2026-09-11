import os
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rag_kb.config import Settings, get_settings, resolve_retrieval_limits


def test_limit_resolution_uses_common_default_or_explicit_values() -> None:
    assert resolve_retrieval_limits(SimpleNamespace(retrieval_top_k=5)) == (5, 5)
    assert resolve_retrieval_limits(
        SimpleNamespace(retrieval_top_k=5, retrieval_candidate_k=10, retrieval_context_k=3)
    ) == (10, 3)


def test_invalid_limit_relationship_fails_closed() -> None:
    with pytest.raises(ValueError, match="context_k_cannot_exceed_candidate_k"):
        resolve_retrieval_limits(
            SimpleNamespace(retrieval_top_k=5, retrieval_candidate_k=2, retrieval_context_k=3)
        )


@pytest.mark.parametrize("invalid", [0, False, 2.5])
def test_explicit_invalid_limits_do_not_fall_back_to_default(invalid) -> None:
    with pytest.raises(ValueError, match="retrieval_limits_must_be"):
        resolve_retrieval_limits(
            SimpleNamespace(retrieval_top_k=5, retrieval_candidate_k=invalid, retrieval_context_k=2)
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_settings_reject_non_finite_threshold(invalid) -> None:
    with pytest.raises(ValidationError):
        Settings(retrieval_score_threshold=invalid)


def test_empty_optional_retrieval_environment_values_use_defaults(monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", "")
    monkeypatch.setenv("RETRIEVAL_CONTEXT_K", "")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "")

    settings = Settings(_env_file=None)

    assert settings.retrieval_candidate_k is None
    assert settings.retrieval_context_k is None
    assert settings.retrieval_score_threshold is None


def test_settings_can_explicitly_skip_env_file(monkeypatch) -> None:
    monkeypatch.setenv("RAG_KB_DISABLE_ENV_FILE", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-only")
    get_settings.cache_clear()
    try:
        assert Settings().deepseek_api_key == "environment-only"
        assert get_settings().deepseek_api_key == "environment-only"
    finally:
        get_settings.cache_clear()


def test_runtime_environment_disables_torch_dynamo(monkeypatch) -> None:
    monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)

    Settings(_env_file=None).prepare_runtime_env()

    assert os.environ["TORCH_COMPILE_DISABLE"] == "1"
    assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
