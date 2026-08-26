"""真实 PostgreSQL 上的 LangGraph checkpointer 基础恢复协议。"""

from __future__ import annotations

import os
from typing import NotRequired, TypedDict
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

from rag_kb.agent.checkpointer import open_postgres_checkpointer


POSTGRES_TEST_URL = os.environ.get("RAG_KB_TEST_POSTGRES_URL")


class CheckpointState(TypedDict):
    messages: NotRequired[list[object]]
    answer: NotRequired[str]


@pytest.mark.skipif(not POSTGRES_TEST_URL, reason="postgres_test_url_unconfigured")
def test_postgres_checkpointer_persists_messages_and_resumes_same_thread() -> None:
    assert POSTGRES_TEST_URL is not None
    checkpointer, connection = open_postgres_checkpointer(POSTGRES_TEST_URL)
    thread_id = f"checkpoint-{uuid4().hex}"

    def ask_for_scope(_state: CheckpointState) -> dict[str, object]:
        resumed = interrupt({"kind": "user_clarification", "question": "需要比较范围？"})
        answer = resumed.get("answer") if isinstance(resumed, dict) else None
        assert isinstance(answer, str)
        return {"answer": answer, "messages": [ToolMessage("received", tool_call_id="scope")]}

    graph = StateGraph(CheckpointState)
    graph.add_node("ask_for_scope", ask_for_scope)
    graph.add_edge(START, "ask_for_scope")
    agent = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpointer.setup()
        paused = agent.invoke(
            {"messages": [HumanMessage("比较采购制度 V1 与 V2")]}, config
        )
        assert "__interrupt__" in paused
        snapshot = agent.get_state(config)
        assert len(snapshot.interrupts) == 1
        assert snapshot.interrupts[0].value == {
            "kind": "user_clarification",
            "question": "需要比较范围？",
        }

        resumed = agent.invoke(Command(resume={"answer": "仅比较审批规则"}), config)
        assert resumed["answer"] == "仅比较审批规则"
        latest = agent.get_state(config)
        assert latest.values["answer"] == "仅比较审批规则"
        assert any(isinstance(item, ToolMessage) for item in latest.values["messages"])
    finally:
        connection.close()
