from pathlib import Path

import pytest

from rag_kb.ingestion import (
    ParsedItem,
    UploadValidationError,
    items_from_docling_document,
    load_manifest,
    parse_markdown,
    validate_upload,
)
from rag_kb.models import Role
from types import SimpleNamespace

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


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


def test_load_manifest_reads_document_versions_and_roles() -> None:
    specs = load_manifest(CORPUS_DIR)

    assert len(specs) == 14
    leave = sorted((spec for spec in specs if spec.document_id == "hr-leave-policy"), key=lambda spec: spec.version)
    assert [spec.version for spec in leave] == [1, 2]
    assert set(leave[0].allowed_roles) == {Role.HR, Role.ADMIN}


def test_parse_markdown_real_corpus_travel_policy() -> None:
    items = parse_markdown((CORPUS_DIR / "全员差旅报销制度.md").read_text(encoding="utf-8"))

    assert items[0] == ParsedItem(kind="title", text="全员差旅报销制度")
    assert "第四章 住宿标准" in [item.text for item in items if item.kind == "heading"]


class _FakeTableItem:
    """模拟 Docling TableItem：没有可用 text，内容由导出接口提供。"""

    def __init__(self, markdown: str) -> None:
        self.label = "table"
        self.prov: list = []
        self._markdown = markdown

    def export_to_markdown(self, _document=None) -> str:
        return self._markdown


class _FakeTextItem:
    def __init__(self, label: str, text: str, page: int | None) -> None:
        self.label = label
        self.text = text
        self.prov = [SimpleNamespace(page_no=page)] if page is not None else []


class _FakeDocument:
    def __init__(self, items: list) -> None:
        self._items = items

    def iterate_items(self):
        return iter((item, 0) for item in self._items)


def test_pdf_items_keep_tables_instead_of_skipping_them() -> None:
    table = "| 假期 | 天数 |\n| --- | --- |\n| 年假 | 15 |"
    document = _FakeDocument(
        [
            _FakeTextItem("title", "员工休假制度", 1),
            _FakeTextItem("section_header", "假期标准", 1),
            _FakeTextItem("text", "正文段落", 1),
            _FakeTableItem(table),
        ]
    )

    items = items_from_docling_document(document)

    assert [item.kind for item in items] == ["title", "heading", "text", "table"]
    assert items[-1].text == table
    assert [item.page for item in items] == [1, 1, 1, None]
