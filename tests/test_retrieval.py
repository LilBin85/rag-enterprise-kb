"""阶段 3 带 ACL 的 dense 检索集成测试(蓝图 9 节阶段 3 优先测试)。

连接真实 Qdrant;TopicEmbedder 使召回行为可控可断言:
- 授权角色召回期望 document_id
- 未授权角色候选为空——语义上查询与受限块相近,但 ACL 在检索期拦截(蓝图 6.2)
- 被替代版本(superseded)、撤回(withdrawn)、删除(deleted)的块不被召回
"""

from pathlib import Path

import pytest

from rag_kb.lifecycle import DocumentLifecycle
from rag_kb.models import DocumentSpec, Role
from rag_kb.retriever import Retriever
from rag_kb.store import ChunkStore

TRAVEL_MD = "# 全员差旅报销制度\n## 住宿标准\n一线城市住宿每晚不超过 450 元,报销时限为返程后 15 个工作日。\n"
DEV_MD = "# 研发发布流程\n## 发布窗口\n每周二周四 14:00-17:00 为发布窗口,灰度放量。\n## 回滚机制\nP0 故障 30 分钟内回滚,RTO 15 分钟。\n"
LEAVE_V1_MD = "# 员工休假制度\n## 年假\n员工每年可享受带薪年假 10 天。\n"
LEAVE_V2_MD = "# 员工休假制度\n## 年假\n员工每年可享受带薪年假 15 天。\n## 育儿假\n子女未满 3 周岁的员工每年可享受育儿假 10 天。\n"


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def kb_and_retriever(store: ChunkStore, topic_embedder, tmp_path: Path):
    """摄取四类文档并把休假制度更新到 v2;检索阈值 0.3 仅用于分离 TopicEmbedder 的双峰分数。

    (命中主题的块余弦 ≈1,未命中 ≈0;生产阈值待冻结评测校准,蓝图 7.1。)
    """
    lifecycle = DocumentLifecycle(
        store=store, embedder=topic_embedder, upload_dir=tmp_path / "uploads", max_chars=800
    )
    lifecycle.add_document(
        DocumentSpec(document_id="travel-policy", file="差旅.md", title="全员差旅报销制度",
                     department="general",
                     allowed_roles=[Role.EMPLOYEE, Role.ENGINEER, Role.HR, Role.ADMIN], version=1),
        _write(tmp_path, "差旅.md", TRAVEL_MD),
    )
    lifecycle.add_document(
        DocumentSpec(document_id="dev-process", file="研发.md", title="研发发布流程",
                     department="engineering", allowed_roles=[Role.ENGINEER, Role.ADMIN], version=1),
        _write(tmp_path, "研发.md", DEV_MD),
    )
    lifecycle.add_document(
        DocumentSpec(document_id="leave-policy", file="休假-v1.md", title="员工休假制度",
                     department="hr", allowed_roles=[Role.HR, Role.ADMIN], version=1),
        _write(tmp_path, "休假-v1.md", LEAVE_V1_MD),
    )
    lifecycle.update_document(
        "leave-policy",
        DocumentSpec(document_id="leave-policy", file="休假-v2.md", title="员工休假制度",
                     department="hr", allowed_roles=[Role.HR, Role.ADMIN], version=0),
        _write(tmp_path, "休假-v2.md", LEAVE_V2_MD),
    )
    retriever = Retriever(store=store, embedder=topic_embedder, top_k=5, score_threshold=0.3)
    return lifecycle, retriever


# ---------- 授权召回 ----------

def test_authorized_role_recalls_expected_document(kb_and_retriever) -> None:
    _, retriever = kb_and_retriever
    candidates = retriever.search("发布窗口和回滚机制是什么?", Role.ENGINEER)
    assert candidates
    assert all(c.document_id == "dev-process" for c in candidates)


def test_employee_recalls_public_document(kb_and_retriever) -> None:
    _, retriever = kb_and_retriever
    candidates = retriever.search("住宿报销限额是多少?", Role.EMPLOYEE)
    assert candidates
    assert all(c.document_id == "travel-policy" for c in candidates)


# ---------- 越权隔离(核心) ----------

def test_unauthorized_role_gets_no_restricted_chunks(kb_and_retriever) -> None:
    """employee 查年假:语义上与休假块相近,但 ACL 必须在检索期拦截,候选为空。"""
    _, retriever = kb_and_retriever
    candidates = retriever.search("年假有多少天?", Role.EMPLOYEE)
    assert candidates == []
    # 双重确认:任何候选都不得包含受限文档的 chunk_id
    assert all("leave-policy" not in c.chunk_id for c in candidates)


def test_engineer_cannot_recall_hr_document(kb_and_retriever) -> None:
    _, retriever = kb_and_retriever
    assert retriever.search("育儿假怎么休?", Role.ENGINEER) == []


# ---------- 版本与生命周期隔离 ----------

def test_superseded_version_not_recalled(kb_and_retriever) -> None:
    _, retriever = kb_and_retriever
    candidates = retriever.search("年假有多少天?", Role.HR)
    assert candidates
    assert all(c.document_version == 2 for c in candidates)
    assert all("v1" not in c.chunk_id for c in candidates)


def test_deleted_document_not_recalled(kb_and_retriever) -> None:
    lifecycle, retriever = kb_and_retriever
    lifecycle.delete("leave-policy")
    assert retriever.search("年假有多少天?", Role.HR) == []


def test_withdrawn_document_not_recalled(kb_and_retriever) -> None:
    lifecycle, retriever = kb_and_retriever
    lifecycle.withdraw("travel-policy")
    assert retriever.search("住宿报销限额是多少?", Role.EMPLOYEE) == []


# ---------- 候选完整性(引用组装的前置条件) ----------

def test_every_candidate_carries_citation_fields(kb_and_retriever) -> None:
    _, retriever = kb_and_retriever
    for c in retriever.search("年假有多少天?", Role.ADMIN):
        assert c.document_id and c.document_version >= 1
        assert c.chunk_id and c.source_file_name and c.source_locator
        assert c.text and c.score > 0
