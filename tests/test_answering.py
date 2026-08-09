"""阶段 4 受约束回答、引用与拒答集成测试(蓝图 7.2 / 7.3 / 7.4 / 10.3)。

除 API 用例外均直接使用 AnswerService;FakeAnswerModel 可编程,
验证:拒答分流不调用模型、引用只来自实际候选、失败关闭、拒答不泄露受限元数据。
"""

import pytest
from fastapi.testclient import TestClient

from rag_kb.answerer import AnswerService, ModelVerdict
from rag_kb.api import app, build_answer_service
from rag_kb.models import Role


@pytest.fixture()
def service(kb_and_retriever, fake_model) -> AnswerService:
    _, retriever = kb_and_retriever
    return AnswerService(retriever=retriever, model=fake_model)


# ---------- 正常回答与引用 ----------

def test_answer_includes_citations_from_actual_candidates(service, fake_model) -> None:
    fake_model.next = ModelVerdict(
        verdict="answer", answer="入职满 1 年后每年 15 天。",
        used_chunk_ids=["leave-policy:v2:chunk-01"],
    )
    resp = service.ask("年假有多少天?", Role.HR)
    assert resp.verdict == "answer"
    assert resp.answer == "入职满 1 年后每年 15 天。"
    assert len(resp.citations) == 1
    citation = resp.citations[0]
    # 引用五字段由应用层从实际候选构造,不是模型输出(蓝图 7.2)
    assert citation.chunk_id == "leave-policy:v2:chunk-01"
    assert citation.document_id == "leave-policy"
    assert citation.document_version == 2
    assert citation.source_file_name == "休假-v2.md"
    assert "年假" in citation.source_locator


def test_citations_intersected_with_candidates(service, fake_model) -> None:
    """模型声明的 used_chunk_ids 含编造 ID 时,该 ID 被剔除,不进入引用。"""
    fake_model.next = ModelVerdict(
        verdict="answer", answer="年假 15 天。",
        used_chunk_ids=["leave-policy:v2:chunk-01", "leave-policy:v9:chunk-99"],
    )
    resp = service.ask("年假有多少天?", Role.HR)
    assert resp.verdict == "answer"
    assert [c.chunk_id for c in resp.citations] == ["leave-policy:v2:chunk-01"]


def test_answer_with_only_fabricated_ids_fails_closed(service, fake_model) -> None:
    """引用构建失败:模型只给了编造 ID → 失败关闭,不交付无引用回答(蓝图 7.4)。"""
    fake_model.next = ModelVerdict(
        verdict="answer", answer="随口一说。", used_chunk_ids=["ghost:v1:chunk-99"]
    )
    resp = service.ask("年假有多少天?", Role.HR)
    assert resp.verdict == "refusal"
    assert resp.refusal.kind == "conflicting_evidence"
    assert resp.citations == []


# ---------- 拒答分流(不调用模型) ----------

def test_no_authorized_evidence_refusal_without_model_call(service, fake_model, kb_and_retriever) -> None:
    lifecycle, _ = kb_and_retriever
    lifecycle.delete("travel-policy")  # employee 从此无任何授权块
    resp = service.ask("住宿报销限额是多少?", Role.EMPLOYEE)
    assert resp.verdict == "refusal"
    assert resp.refusal.kind == "no_authorized_evidence"
    assert resp.refusal.message == "当前知识库中没有可用于回答的已授权资料"
    assert fake_model.calls == []  # 无证据时不调用模型


def test_insufficient_evidence_refusal_without_model_call(service, fake_model) -> None:
    """employee 查年假:授权块(travel)语义不相关被阈值裁掉 → 证据不足拒答。"""
    resp = service.ask("年假有多少天?", Role.EMPLOYEE)
    assert resp.verdict == "refusal"
    assert resp.refusal.kind == "insufficient_evidence"
    assert resp.refusal.message == "现有已授权资料不足以可靠回答该问题"
    assert fake_model.calls == []


# ---------- 模型层拒答 ----------

def test_model_refusal_insufficient(service, fake_model) -> None:
    fake_model.next = ModelVerdict(verdict="refusal", refusal_kind="insufficient")
    resp = service.ask("年假有多少天?", Role.HR)
    assert resp.refusal.kind == "insufficient_evidence"
    assert len(fake_model.calls) == 1


def test_model_refusal_conflict(service, fake_model) -> None:
    fake_model.next = ModelVerdict(verdict="refusal", refusal_kind="conflict")
    resp = service.ask("年假有多少天?", Role.HR)
    assert resp.refusal.kind == "conflicting_evidence"
    assert resp.refusal.message == "检索到的资料不足以形成可靠结论"


def test_model_exception_fails_closed(service, fake_model) -> None:
    fake_model.error = RuntimeError("deepseek api down")
    resp = service.ask("年假有多少天?", Role.HR)
    assert resp.verdict == "refusal"
    assert resp.refusal.kind == "retrieval_failure"


def test_retrieval_exception_fails_closed(fake_model) -> None:
    """检索阶段异常(向量库断连/embedding 失败)同样失败关闭,不向调用方抛 500(蓝图 7.4)。"""

    class BrokenRetriever:
        def search(self, question, role):
            raise RuntimeError("qdrant unreachable")

    service = AnswerService(retriever=BrokenRetriever(), model=fake_model)
    resp = service.ask("任何问题", Role.EMPLOYEE)
    assert resp.verdict == "refusal"
    assert resp.refusal.kind == "retrieval_failure"
    assert fake_model.calls == []  # 检索失败时不应再调用模型


# ---------- 拒答不泄露受限元数据(蓝图 7.4 用户无权限行) ----------

def test_refusal_leaks_no_restricted_metadata(service, fake_model) -> None:
    resp = service.ask("年假有多少天?", Role.EMPLOYEE)
    payload = resp.model_dump_json()
    for forbidden in ["leave-policy", "休假", "年假", "v1", "v2", '"hr"']:
        assert forbidden not in payload, f"拒答响应泄露了受限标识: {forbidden}"


# ---------- API 端点 ----------

@pytest.fixture()
def client(kb_and_retriever, fake_model):
    _, retriever = kb_and_retriever
    service = AnswerService(retriever=retriever, model=fake_model)
    app.dependency_overrides[build_answer_service] = lambda: service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_questions_endpoint(client, fake_model) -> None:
    fake_model.next = ModelVerdict(
        verdict="answer", answer="年假 15 天。", used_chunk_ids=["leave-policy:v2:chunk-01"]
    )
    resp = client.post("/questions", json={"question": "年假有多少天?", "role": "hr"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "answer"
    assert body["citations"][0]["document_version"] == 2


def test_questions_endpoint_refusal(client, fake_model) -> None:
    resp = client.post("/questions", json={"question": "年假有多少天?", "role": "employee"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "refusal"
    assert body["citations"] == []
    assert fake_model.calls == []
