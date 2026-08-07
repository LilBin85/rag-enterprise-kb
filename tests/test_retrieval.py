"""阶段 3 带 ACL 的 dense 检索集成测试(蓝图 9 节阶段 3 优先测试)。

连接真实 Qdrant;TopicEmbedder 使召回行为可控可断言:
- 授权角色召回期望 document_id
- 未授权角色候选为空——语义上查询与受限块相近,但 ACL 在检索期拦截(蓝图 6.2)
- 被替代版本(superseded)、撤回(withdrawn)、删除(deleted)的块不被召回
"""

from rag_kb.models import Role


# ---------- 授权召回 ----------

def test_authorized_role_recalls_expected_document(kb_and_retriever) -> None:
    _, retriever = kb_and_retriever
    candidates = retriever.search("发布窗口和回滚机制是什么?", Role.ENGINEER).candidates
    assert candidates
    assert all(c.document_id == "dev-process" for c in candidates)


def test_employee_recalls_public_document(kb_and_retriever) -> None:
    _, retriever = kb_and_retriever
    candidates = retriever.search("住宿报销限额是多少?", Role.EMPLOYEE).candidates
    assert candidates
    assert all(c.document_id == "travel-policy" for c in candidates)


# ---------- 越权隔离(核心) ----------

def test_unauthorized_role_gets_no_restricted_chunks(kb_and_retriever) -> None:
    """employee 查年假:语义上与休假块相近,但 ACL 必须在检索期拦截,候选为空。"""
    _, retriever = kb_and_retriever
    candidates = retriever.search("年假有多少天?", Role.EMPLOYEE).candidates
    assert candidates == []
    # 双重确认:任何候选都不得包含受限文档的 chunk_id
    assert all("leave-policy" not in c.chunk_id for c in candidates)


def test_engineer_cannot_recall_hr_document(kb_and_retriever) -> None:
    _, retriever = kb_and_retriever
    assert retriever.search("育儿假怎么休?", Role.ENGINEER).candidates == []


# ---------- 版本与生命周期隔离 ----------

def test_superseded_version_not_recalled(kb_and_retriever) -> None:
    _, retriever = kb_and_retriever
    candidates = retriever.search("年假有多少天?", Role.HR).candidates
    assert candidates
    assert all(c.document_version == 2 for c in candidates)
    assert all("v1" not in c.chunk_id for c in candidates)


def test_deleted_document_not_recalled(kb_and_retriever) -> None:
    lifecycle, retriever = kb_and_retriever
    lifecycle.delete("leave-policy")
    assert retriever.search("年假有多少天?", Role.HR).candidates == []


def test_withdrawn_document_not_recalled(kb_and_retriever) -> None:
    lifecycle, retriever = kb_and_retriever
    lifecycle.withdraw("travel-policy")
    assert retriever.search("住宿报销限额是多少?", Role.EMPLOYEE).candidates == []


# ---------- 候选完整性(引用组装的前置条件) ----------

def test_every_candidate_carries_citation_fields(kb_and_retriever) -> None:
    _, retriever = kb_and_retriever
    for c in retriever.search("年假有多少天?", Role.ADMIN).candidates:
        assert c.document_id and c.document_version >= 1
        assert c.chunk_id and c.source_file_name and c.source_locator
        assert c.text and c.score > 0
