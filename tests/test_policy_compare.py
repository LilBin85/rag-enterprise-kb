"""策略版本比较的安全边界单元测试,不连接 Qdrant 或真实 DeepSeek。"""

import json
import logging

import pytest
from langchain_core.messages import AIMessage

from rag_kb.models import Role
from rag_kb.policy_compare import (
    ALLOWED_TOOL_NAMES,
    BoundPolicyTools,
    PolicyComparisonService,
    PolicyToolError,
)


def _payload(document_id: str, version: int, chunk: str, section: str, text: str, status: str = "active") -> dict:
    return {
        "document_id": document_id,
        "document_version": version,
        "chunk_id": f"{document_id}:v{version}:chunk-{chunk}",
        "source_file_name": f"policy-v{version}.md",
        "source_locator": section,
        "content_hash": f"sha256:{version}{chunk}",
        "text": text,
        "status": status,
        "allowed_roles": ["hr", "admin"],
    }


class FakeStore:
    def __init__(self, data: dict[tuple[str, int, Role], list[dict]]) -> None:
        self.data = data
        self.calls: list[tuple[str, int, Role]] = []

    def scroll_authorized_version(self, document_id: str, version: int, role: Role) -> list[dict]:
        self.calls.append((document_id, version, role))
        return self.data.get((document_id, version, role), [])


class FakeAgentModel:
    """按固定顺序调用两个读取工具和比较工具,模拟真实 tool_calls。"""

    def __init__(self, from_version: int, to_version: int) -> None:
        self.from_version = from_version
        self.to_version = to_version
        self.turn = 0
        self.bound_names: set[str] = set()

    def bind_tools(self, tools: list, **_kwargs):
        self.bound_names = {tool.name for tool in tools}
        return self

    def invoke(self, _messages):
        self.turn += 1
        if self.turn == 1:
            return AIMessage(content="", tool_calls=[{"name": "fetch_authorized_version", "args": {"version": self.from_version}, "id": "a"}])
        if self.turn == 2:
            return AIMessage(content="", tool_calls=[{"name": "fetch_authorized_version", "args": {"version": self.to_version}, "id": "b"}])
        if self.turn == 3:
            return AIMessage(content="", tool_calls=[{"name": "compare_authorized_versions", "args": {}, "id": "c"}])
        return AIMessage(
            tool_calls=[],
            content=json.dumps({"summary": "按章节比较政策变化。", "used_chunk_ids": ["fabricated:v9:chunk-99"]}),
        )


def _service(from_version: int = 1, to_version: int = 2):
    data = {
        ("leave", 1, Role.HR): [_payload("leave", 1, "01", "年假", "每年 10 天")],
        ("leave", 2, Role.HR): [
            _payload("leave", 2, "01", "年假", "每年 15 天"),
            _payload("leave", 2, "02", "育儿假", "每年 10 天"),
        ],
    }
    store = FakeStore(data)
    return store, PolicyComparisonService(store, FakeAgentModel(from_version, to_version))


def test_authorized_comparison_reports_modified_and_added_sections() -> None:
    store, service = _service()
    response = service.compare("leave", 1, 2, Role.HR)

    assert response.verdict == "comparison"
    assert response.comparison.document_id == "leave"
    assert [(item.section, item.change) for item in response.comparison.changes] == [
        ("年假", "modified"),
        ("育儿假", "added"),
    ]
    assert {citation.document_version for citation in response.comparison.citations} == {1, 2}
    assert all(citation.document_id == "leave" for citation in response.comparison.citations)
    assert store.calls == [("leave", 1, Role.HR), ("leave", 2, Role.HR)]


def test_fabricated_model_chunk_id_never_becomes_citation() -> None:
    _, service = _service()
    response = service.compare("leave", 1, 2, Role.HR)

    assert response.comparison is not None
    assert all("fabricated" not in citation.chunk_id for citation in response.comparison.citations)


@pytest.mark.parametrize(
    ("role", "versions"),
    [
        (Role.EMPLOYEE, (1, 2)),
        (Role.HR, (3, 4)),
    ],
)
def test_unauthorized_or_missing_versions_fail_closed(role: Role, versions: tuple[int, int]) -> None:
    _, service = _service()
    response = service.compare("leave", *versions, role)

    assert response.verdict == "refusal"
    body = response.model_dump_json()
    assert "leave" not in body
    assert "v1" not in body and "v2" not in body
    assert response.refusal.message == "当前无法完成策略版本比较"


@pytest.mark.parametrize("versions", [(0, 2), (-1, 2), (2, 1), (1, 1)])
def test_invalid_reversed_or_identical_versions_fail_before_tool_access(versions: tuple[int, int]) -> None:
    store, service = _service()
    response = service.compare("leave", *versions, Role.HR)

    assert response.verdict == "refusal"
    assert store.calls == []


def test_tool_binding_rejects_unknown_version_and_extra_arguments() -> None:
    tools = BoundPolicyTools(lambda *_: [], "leave", 1, 2, Role.HR)
    assert {tool.name for tool in tools.langchain_tools()} == ALLOWED_TOOL_NAMES
    with pytest.raises(PolicyToolError):
        tools.fetch_authorized_version(3)


def test_production_adapter_accepts_create_agent_tool_keywords() -> None:
    from rag_kb.policy_compare import DeepSeekPolicyAgentModel

    class FakeLLM:
        def bind_tools(self, tools, **kwargs):
            return tools, kwargs

    adapter = DeepSeekPolicyAgentModel.__new__(DeepSeekPolicyAgentModel)
    adapter._llm = FakeLLM()
    _, kwargs = adapter.bind_tools([], tool_choice="any")
    assert kwargs == {"tool_choice": "any", "parallel_tool_calls": False}


def test_tool_calls_and_steps_are_bounded() -> None:
    _, service = _service()
    service.max_tool_calls = 2
    response = service.compare("leave", 1, 2, Role.HR)
    assert response.verdict == "refusal"


def test_agent_steps_are_bounded() -> None:
    _, service = _service()
    service.max_steps = 1
    response = service.compare("leave", 1, 2, Role.HR)
    assert response.verdict == "refusal"


def test_model_and_tool_errors_fail_closed_without_writes() -> None:
    store, service = _service()

    class BrokenModel:
        def bind_tools(self, _tools):
            raise RuntimeError("model unavailable")

    service.model = BrokenModel()
    response = service.compare("leave", 1, 2, Role.HR)
    assert response.verdict == "refusal"
    assert store.calls == []


@pytest.mark.parametrize("status", ["withdrawn", "deleted"])
def test_withdrawn_or_deleted_versions_fail_closed(status: str) -> None:
    store, service = _service()
    store.data[("leave", 1, Role.HR)] = [_payload("leave", 1, "01", "年假", "每年 10 天", status)]
    response = service.compare("leave", 1, 2, Role.HR)
    assert response.verdict == "refusal"


def test_policy_comparison_endpoint_contract() -> None:
    from fastapi.testclient import TestClient

    from rag_kb.api import app, build_policy_comparison_service, resolve_policy_role

    _, service = _service()
    app.dependency_overrides[build_policy_comparison_service] = lambda: service
    app.dependency_overrides[resolve_policy_role] = lambda: Role.HR
    try:
        response = TestClient(app).post(
            "/policy-comparisons",
            json={"document_id": "leave", "from_version": 1, "to_version": 2, "role": "hr"},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["verdict"] == "comparison"


def test_policy_endpoint_rejects_boolean_versions() -> None:
    from fastapi.testclient import TestClient

    from rag_kb.api import app, build_policy_comparison_service, resolve_policy_role

    _, service = _service()
    app.dependency_overrides[build_policy_comparison_service] = lambda: service
    app.dependency_overrides[resolve_policy_role] = lambda: Role.HR
    try:
        response = TestClient(app).post(
            "/policy-comparisons",
            json={"document_id": "leave", "from_version": True, "to_version": 2, "role": "hr"},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 422


def test_policy_role_requires_server_configured_demo_token(monkeypatch) -> None:
    from fastapi import HTTPException

    import rag_kb.api as api
    from rag_kb.models import PolicyComparisonRequest

    monkeypatch.setattr(
        api,
        "get_settings",
        lambda: type("SettingsStub", (), {"demo_role_tokens": {"hr": "hr-token"}})(),
    )
    req = PolicyComparisonRequest(document_id="leave", from_version=1, to_version=2, role=Role.HR)
    assert api.resolve_policy_role(req, "hr-token") == Role.HR
    with pytest.raises(HTTPException) as exc_info:
        api.resolve_policy_role(req, "wrong-token")
    assert exc_info.value.status_code == 403

    elevated = req.model_copy(update={"role": Role.ADMIN})
    with pytest.raises(HTTPException) as exc_info:
        api.resolve_policy_role(elevated, "hr-token")
    assert exc_info.value.status_code == 403


def test_policy_endpoint_checks_demo_token_header(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import rag_kb.api as api

    _, service = _service()
    monkeypatch.setattr(
        api,
        "get_settings",
        lambda: type("SettingsStub", (), {"demo_role_tokens": {"hr": "hr-token"}})(),
    )
    api.app.dependency_overrides[api.build_policy_comparison_service] = lambda: service
    try:
        client = TestClient(api.app)
        body = {"document_id": "leave", "from_version": 1, "to_version": 2, "role": "hr"}
        assert client.post("/policy-comparisons", json=body).status_code == 403
        assert client.post(
            "/policy-comparisons", json=body, headers={"X-Demo-Role-Token": "hr-token"}
        ).status_code == 200
    finally:
        api.app.dependency_overrides.clear()


def test_observability_is_structured_and_redacts_document_id(caplog) -> None:
    _, service = _service()
    with caplog.at_level(logging.INFO, logger="rag_kb.policy_comparison"):
        service.compare("leave", 1, 2, Role.HR)
    assert "leave" not in caplog.text
    event = json.loads(caplog.records[-1].message)
    assert {
        "run_id", "operation", "stage", "role", "document_hash", "step", "tool",
        "counts", "duration_ms", "status", "error_code",
    } == set(event)


def test_benchmark_report_shape_uses_actual_cases() -> None:
    from scripts.benchmark_policy_comparisons import run_benchmark

    _, service = _service()
    report = run_benchmark(
        service,
        {
            "policy_comparisons": [
                {
                    "name": "authorized",
                    "document_id": "leave",
                    "from_version": 1,
                    "to_version": 2,
                    "role": "hr",
                    "expect": "comparison",
                },
                {
                    "name": "unauthorized",
                    "document_id": "leave",
                    "from_version": 1,
                    "to_version": 2,
                    "role": "employee",
                    "expect": "refusal",
                },
            ]
        },
    )
    assert report["schema_version"] == 1
    assert report["sample_size"] == 2
    assert report["metrics"]["timing_sample_size"] == 2
    assert report["metrics"]["refusal_incorrect_count"] == 0
    assert report["cases"][0]["status"] == "pass"


def test_authorized_store_read_consumes_all_qdrant_pages() -> None:
    from rag_kb.store import ChunkStore

    payloads = [
        _payload("large", 1, f"{index:03d}", f"章节-{index}", f"正文-{index}")
        for index in range(257)
    ]

    class Point:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

    class Client:
        def scroll(self, *, offset=None, **_kwargs):
            if offset is None:
                return [Point(item) for item in payloads[:256]], "next"
            return [Point(payloads[256])], None

    store = ChunkStore.__new__(ChunkStore)
    store.client = Client()
    store.collection = "test"
    result = store.scroll_authorized_version("large", 1, Role.HR)
    assert len(result) == 257
