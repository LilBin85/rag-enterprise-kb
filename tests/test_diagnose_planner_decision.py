import importlib.util
import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_planner_decision.py"
SPEC = importlib.util.spec_from_file_location("diagnose_planner_decision", SCRIPT)
assert SPEC and SPEC.loader
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_decision_schema_is_strict_and_action_specific():
    assert diagnostic.PlannerDecision(
        action="search", search_query="回滚时限"
    ).action == "search"
    with pytest.raises(ValidationError):
        diagnostic.PlannerDecision(action="search", search_query=1)
    with pytest.raises(ValidationError):
        diagnostic.PlannerDecision(action="search", search_query="x", refusal_reason="bad")
    with pytest.raises(ValidationError):
        diagnostic.PlannerDecision(action="propose")


class FakeStructuredModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        case = diagnostic.CASES[len(self.calls) - 1]
        action = case["expected"]
        if action == "search":
            payload = {"search_query": "offline query"}
        elif action == "clarify":
            payload = {"clarification_questions": ["offline question"]}
        elif action == "propose":
            payload = {"proposal": diagnostic.PlanningDraft.model_validate(case["proposal"])}
        else:
            payload = {"refusal_reason": "offline refusal"}
        return diagnostic.PlannerDecision(action=action, **payload)


def test_four_case_context_contract_and_safe_output(capsys):
    fake = FakeStructuredModel()
    assert diagnostic.run_diagnostic(fake)
    assert len(fake.calls) == 4
    for messages, case in zip(fake.calls, diagnostic.CASES):
        assert isinstance(messages[0], SystemMessage)
        assert diagnostic.SYSTEM_PROMPT in messages[0].content
        assert isinstance(messages[1], HumanMessage)
        payload = json.loads(messages[1].content)
        assert set(payload) == {"goal", "evidence", "access"}
        assert payload == {field: case[field] for field in ("goal", "evidence", "access")}
    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 4
    assert all("pass=true" in line and "error-type=none" in line for line in output)
    assert all(secret not in "\n".join(output) for secret in ("goal", "evidence", "offline"))


def test_main_returns_nonzero_on_failed_expectation(monkeypatch, capsys):
    fake = FakeStructuredModel()
    fake.invoke = lambda messages: diagnostic.PlannerDecision(
        action="refuse", refusal_reason="offline"
    )
    monkeypatch.setattr(diagnostic, "build_structured_model", lambda settings: fake)
    monkeypatch.setattr(diagnostic, "get_settings", lambda: object())
    assert diagnostic.main() == 1
    assert "pass=false" in capsys.readouterr().out


def test_explicit_local_trace_contains_diagnostics_but_stdout_remains_safe(tmp_path, capsys):
    trace = diagnostic.LocalDiagnosticTrace("planner.jsonl", root=tmp_path, ttl_seconds=60)
    fake = FakeStructuredModel()
    assert diagnostic.run_diagnostic(fake, cases=diagnostic.CASES[:1], trace=trace)
    trace.close()

    stdout = capsys.readouterr().out
    assert "offline query" not in stdout
    events = [
        json.loads(line)
        for line in (tmp_path / "planner.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[1]["prompt"]["input"]["goal"] == diagnostic.CASES[0]["goal"]
    assert events[1]["parsed_response"]["search_query"] == "offline query"
    assert events[1]["error_code"] == "none"


def test_raw_response_details_include_structured_tool_arguments() -> None:
    raw = type(
        "Raw",
        (),
        {
            "content": "",
            "tool_calls": [{"name": "PlannerDecision", "args": {"action": "search"}}],
            "invalid_tool_calls": [],
            "response_metadata": {"finish_reason": "tool_calls"},
            "usage_metadata": {"input_tokens": 1},
        },
    )()
    details = diagnostic._raw_response_details(raw)
    assert details is not None
    assert details["tool_calls"][0]["args"]["action"] == "search"
