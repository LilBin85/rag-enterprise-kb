"""阶段 2 摄取管线单元测试(蓝图 9 节:先写失败测试,再实现最小代码使其通过)。

覆盖蓝图 5.3 前半段的确定性行为,全部脱离 LLM、网络与 Qdrant:
- 上传校验:类型与大小(蓝图 10.2 上传校验用例的前半)
- manifest 加载:文档注册信息是摄取的唯一事实来源
- 内容哈希:确定性、sha256 前缀(蓝图 6.1)
- 结构化切块:按章节边界成块、超长章节细分、chunk_id 稳定格式、source_locator 可追溯
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_kb.ingestion import (
    ParsedItem,
    UploadValidationError,
    chunk_items,
    compute_content_hash,
    load_manifest,
    parse_markdown,
    validate_upload,
)
from rag_kb.models import ChunkStatus, Role

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
NOW = datetime(2026, 8, 7, tzinfo=UTC)


# ---------- 上传校验 ----------

def test_validate_upload_accepts_md_and_pdf() -> None:
    validate_upload("全员差旅报销制度.md", 1024)
    validate_upload("员工休假制度-v1.pdf", 1024)


@pytest.mark.parametrize("filename", ["evil.exe", "notes.docx", "archive.zip", "无扩展名"])
def test_validate_upload_rejects_bad_extension(filename: str) -> None:
    with pytest.raises(UploadValidationError):
        validate_upload(filename, 1024)


def test_validate_upload_rejects_oversize() -> None:
    with pytest.raises(UploadValidationError):
        validate_upload("big.md", 10 * 1024 * 1024 + 1)


# ---------- 内容哈希 ----------

def test_compute_content_hash_deterministic() -> None:
    assert compute_content_hash("同一段文本") == compute_content_hash("同一段文本")
    assert compute_content_hash("甲") != compute_content_hash("乙")


def test_compute_content_hash_format() -> None:
    assert compute_content_hash("x").startswith("sha256:")


# ---------- manifest 加载 ----------

def test_load_manifest_reads_all_documents() -> None:
    specs = load_manifest(CORPUS_DIR)
    assert len(specs) == 14
    leave = sorted((s for s in specs if s.document_id == "hr-leave-policy"), key=lambda s: s.version)
    assert [s.version for s in leave] == [1, 2]
    assert set(leave[0].allowed_roles) == {Role.HR, Role.ADMIN}


def test_load_manifest_roles_are_enums() -> None:
    specs = load_manifest(CORPUS_DIR)
    travel = next(s for s in specs if s.document_id == "travel-reimbursement-policy")
    assert set(travel.allowed_roles) == {Role.EMPLOYEE, Role.ENGINEER, Role.HR, Role.ADMIN}


# ---------- 结构化切块 ----------

def _spec(**overrides):
    base = dict(
        document_id="hr-leave-policy",
        file="员工休假制度-v2.pdf",
        title="员工休假制度",
        department="hr",
        allowed_roles=[Role.HR, Role.ADMIN],
        version=2,
    )
    base.update(overrides)
    from rag_kb.models import DocumentSpec

    return DocumentSpec(**base)


def test_chunk_items_splits_by_heading() -> None:
    items = [
        ParsedItem(kind="title", text="员工休假制度(2026 版)", page=1),
        ParsedItem(kind="heading", text="年假", page=1),
        ParsedItem(kind="text", text="员工入职满 1 年后,每年可享受带薪年假 15 天。", page=1),
        ParsedItem(kind="heading", text="病假", page=2),
        ParsedItem(kind="text", text="员工每年可享受带薪病假 15 天。", page=2),
    ]
    chunks = chunk_items(items, _spec(), max_chars=800, now=NOW)
    assert len(chunks) == 2
    assert chunks[0].source_locator == "第 1 页 / 年假"
    assert chunks[1].source_locator == "第 2 页 / 病假"
    assert "15 天" in chunks[0].text


def test_chunk_items_long_section_splits_at_paragraph() -> None:
    long_text = "。".join(f"第 {i} 条规定了非常详细的报销细节与注意事项" for i in range(80))
    items = [
        ParsedItem(kind="heading", text="住宿标准", page=None),
        ParsedItem(kind="text", text=long_text, page=None),
    ]
    chunks = chunk_items(items, _spec(), max_chars=200, now=NOW)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.source_locator.startswith("住宿标准") for c in chunks)


def test_chunk_ids_follow_stable_format() -> None:
    items = [
        ParsedItem(kind="heading", text="年假", page=1),
        ParsedItem(kind="text", text="内容", page=1),
    ]
    chunks = chunk_items(items, _spec(), max_chars=800, now=NOW)
    assert chunks[0].chunk_id == "hr-leave-policy:v2:chunk-01"
    assert chunks[0].document_version == 2
    assert chunks[0].status == ChunkStatus.ACTIVE


def test_chunk_record_payload_covers_blueprint_fields() -> None:
    items = [
        ParsedItem(kind="heading", text="年假", page=1),
        ParsedItem(kind="text", text="内容", page=1),
    ]
    payload = chunk_items(items, _spec(), max_chars=800, now=NOW)[0].to_payload()
    for field in [
        "document_id", "document_version", "chunk_id", "source_file_name", "source_locator",
        "title", "department", "allowed_roles", "status", "content_hash",
        "created_at", "updated_at", "text",
    ]:
        assert field in payload, f"payload 缺少蓝图字段: {field}"
    assert payload["allowed_roles"] == ["hr", "admin"]
    assert payload["status"] == "active"
    assert payload["created_at"] in (NOW.isoformat(), NOW.isoformat().replace("+00:00", "Z"))


# ---------- 真实语料的 Markdown 解析 ----------

def test_parse_markdown_real_corpus_travel_policy() -> None:
    items = parse_markdown((CORPUS_DIR / "全员差旅报销制度.md").read_text(encoding="utf-8"))
    headings = [i.text for i in items if i.kind == "heading"]
    assert "第三章 交通标准" in headings
    assert "第四章 住宿标准" in headings
    assert items[0].kind == "title"


def test_real_travel_document_chunks() -> None:
    items = parse_markdown((CORPUS_DIR / "全员差旅报销制度.md").read_text(encoding="utf-8"))
    chunks = chunk_items(items, _spec(
        document_id="travel-reimbursement-policy", file="全员差旅报销制度.md",
        title="全员差旅报销制度", department="general",
        allowed_roles=[Role.EMPLOYEE, Role.ENGINEER, Role.HR, Role.ADMIN], version=1,
    ), max_chars=800, now=NOW)
    assert len(chunks) >= 6  # 六章各自至少成块
    lodging = [c for c in chunks if "住宿标准" in c.source_locator]
    assert lodging and "450" in lodging[0].text
