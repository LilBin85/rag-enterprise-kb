"""共享的上传校验、语料加载与文档解析辅助逻辑。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from rag_kb.config import get_settings
from rag_kb.models import DocumentSpec

_SENTENCE_ENDINGS = "。!?\n"


class UploadValidationError(ValueError):
    """上传文件未通过类型或大小校验。"""


def validate_upload(filename: str, size_bytes: int, *, max_bytes: int | None = None) -> None:
    settings = get_settings()
    limit = max_bytes if max_bytes is not None else settings.max_upload_bytes
    suffix = Path(filename).suffix.lower()
    if suffix not in set(settings.allowed_extensions):
        raise UploadValidationError(
            f"不支持的文件类型: {suffix or '(无扩展名)'};仅允许 {sorted(settings.allowed_extensions)}"
        )
    if size_bytes > limit:
        raise UploadValidationError(f"文件超出大小上限: {size_bytes} > {limit} 字节")


def load_manifest(corpus_dir: Path) -> list[DocumentSpec]:
    manifest = yaml.safe_load((corpus_dir / "manifest.yaml").read_text(encoding="utf-8"))
    return [DocumentSpec(**entry) for entry in manifest["documents"]]


@dataclass(frozen=True)
class ParsedItem:
    kind: Literal["title", "heading", "text"]
    text: str
    page: int | None = None


def parse_markdown(md_text: str) -> list[ParsedItem]:
    items: list[ParsedItem] = []
    for raw_line in md_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            items.append(ParsedItem("heading", line[3:]))
        elif line.startswith("# "):
            items.append(ParsedItem("title", line[2:]))
        else:
            items.append(ParsedItem("text", line))
    return items


def parse_document(path: Path) -> list[ParsedItem]:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return parse_markdown(path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        return _parse_pdf_with_docling(path)
    raise UploadValidationError(f"不支持的文件类型: {suffix or '(无扩展名)'}")


def _parse_pdf_with_docling(path: Path) -> list[ParsedItem]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    get_settings().prepare_runtime_env()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=PdfPipelineOptions(do_ocr=False))
        }
    )
    items: list[ParsedItem] = []
    for item, _level in converter.convert(str(path)).document.iterate_items():
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        raw_label = getattr(item, "label", None)
        label = str(getattr(raw_label, "value", raw_label or "text"))
        prov = getattr(item, "prov", None)
        if label == "title":
            kind: Literal["title", "heading", "text"] = "title"
        elif label == "section_header":
            kind = "heading"
        else:
            kind = "text"
        items.append(
            ParsedItem(
                kind,
                text,
                prov[0].page_no if prov else None,
            )
        )
    return items


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences: list[str] = []
    buffer = ""
    for character in paragraph:
        buffer += character
        if character in _SENTENCE_ENDINGS:
            sentences.append(buffer)
            buffer = ""
    if buffer:
        sentences.append(buffer)

    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(sentence[index : index + max_chars] for index in range(0, len(sentence), max_chars))
        elif current and len(current) + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        pieces.append(current)
    return pieces
