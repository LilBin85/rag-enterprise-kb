"""build_task_worker 组合根验收：capability 注册时序与关闭顺序。"""

from rag_kb import worker as worker_module


class _FakeCapability:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def start(self) -> None:
        self._events.append("capability.start")

    def close(self) -> None:
        self._events.append("capability.close")


def _patch_build_chain(monkeypatch, events: list[str], *, agent_build_error: Exception | None = None):
    """把 build_task_worker 的重依赖全部替换为记录事件的桩。"""

    class _StubTools:
        def close(self) -> None:
            events.append("tools.close")

    class _StubConnection:
        def close(self) -> None:
            events.append("connection.close")

    class _StubSaver:
        serde = None
        conn = None
        lock = None

        def setup(self) -> None: ...

    monkeypatch.setattr(worker_module, "build_database", lambda _url: object())
    monkeypatch.setattr(worker_module, "LocalObjectStore", lambda _root: object())
    monkeypatch.setattr(worker_module, "BgeM3Embedder", lambda: object())
    monkeypatch.setattr(
        worker_module,
        "QdrantAuthorizedSearch",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(worker_module, "resolve_retrieval_limits", lambda _s: (5, 5))
    monkeypatch.setattr(
        worker_module,
        "EnterpriseRetriever",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        worker_module,
        "build_tool_registry",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(worker_module, "ApprovalService", lambda *_a, **_kw: object())
    monkeypatch.setattr(
        worker_module,
        "ToolExecutionService",
        lambda *_a, **_kw: _StubTools(),
    )
    monkeypatch.setattr(
        worker_module,
        "open_postgres_checkpointer",
        lambda _url: (_StubSaver(), _StubConnection()),
    )

    class _StubAgent:
        def get_state(self, _config):
            return type("Snapshot", (), {"interrupts": (), "values": {}, "next": ()})()

    if agent_build_error is not None:
        def fail_agent(*_args, **_kwargs):
            raise agent_build_error
    else:
        def fail_agent(*_args, **_kwargs):
            return _StubAgent()

    monkeypatch.setattr(
        worker_module,
        "_build_research_model",
        lambda _settings: object(),
    )
    monkeypatch.setattr(
        worker_module,
        "build_research_agent",
        fail_agent,
    )
    monkeypatch.setattr(worker_module, "DatabaseTaskRepository", lambda *_a, **_kw: object())
    monkeypatch.setattr(
        worker_module,
        "DatabaseTaskAuthContextPort",
        lambda _db: object(),
    )
    monkeypatch.setattr(worker_module, "AgentResultStore", lambda _o: object())
    monkeypatch.setattr(worker_module, "EvidenceLedgerRepository", lambda: object())
    monkeypatch.setattr(
        worker_module,
        "_CapabilityLeaseHeartbeat",
        lambda repository, **_kwargs: _FakeCapability(events),
    )


def test_capability_starts_only_after_full_agent_build(monkeypatch) -> None:
    """默认预算 Agent 构建成功之前，capability.start 不得被调用。"""
    events: list[str] = []
    _patch_build_chain(monkeypatch, events)

    order: list[str] = []

    original_start = _FakeCapability.start

    def tracking_start(self) -> None:
        # build_task_worker 尾部才允许 start；此时尚无任何 close 事件。
        assert "tools.close" not in order and "connection.close" not in order
        original_start(self)
        order.append("capability.start")

    monkeypatch.setattr(_FakeCapability, "start", tracking_start)
    monkeypatch.setattr(
        worker_module,
        "build_research_tools",
        lambda tools: order.append("research_tools_built") or object(),
    )

    worker = worker_module.build_task_worker(worker_id="worker-a")
    assert worker is not None
    # start 只在完整 Agent 构建（含默认预算编译）成功之后发生一次。
    assert order == ["research_tools_built", "capability.start"]


def test_failed_agent_build_does_not_register_capability(monkeypatch) -> None:
    """Agent/checkpointer 构建失败时 capability 未注册，且资源按序清理。"""
    events: list[str] = []
    _patch_build_chain(
        monkeypatch,
        events,
        agent_build_error=worker_module.TaskAgentConfigurationError("task_agent_api_key_missing"),
    )
    try:
        worker_module.build_task_worker(worker_id="worker-a")
    except worker_module.TaskAgentConfigurationError:
        pass
    else:
        raise AssertionError("expected TaskAgentConfigurationError")

    assert events == ["tools.close", "connection.close"]


def test_close_order_stops_capability_then_tools_then_connection(monkeypatch) -> None:
    """close 顺序：停止接单 -> tools.close -> capability.close -> connection.close。"""
    events: list[str] = []
    _patch_build_chain(monkeypatch, events)

    started: list[bool] = []

    original_start = _FakeCapability.start

    def tracking_start(self) -> None:
        original_start(self)
        started.append(True)

    monkeypatch.setattr(_FakeCapability, "start", tracking_start)

    worker = worker_module.build_task_worker(worker_id="worker-a")
    assert started == [True]
    capability_closed_via_worker_close = events == ["capability.start"]

    worker.close()

    # TaskWorker.close 先关 tools 再执行外部 close（capability + connection）。
    assert capability_closed_via_worker_close
    expected = [
        "capability.start",
        "tools.close",
        "capability.close",
        "connection.close",
    ]
    assert events == expected
