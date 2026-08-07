"""摄取管线(蓝图 5.3 前半段):上传校验 → manifest 加载 → 结构化解析 → 切块 → 内容哈希。

本模块只做确定性处理:不生成向量(工作单元 3)、不写 Qdrant(工作单元 3/4)、
不调用任何模型。切块遵循"结构化边界优先,超长再细分"(蓝图 7.1)。
"""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml

from rag_kb.config import get_settings
from rag_kb.models import ChunkRecord, ChunkStatus, DocumentSpec

_SENTENCE_ENDINGS = "。!?\n"


class UploadValidationError(ValueError):
    """上传文件未通过类型或大小校验(蓝图 10.2 上传校验用例)。"""


def validate_upload(filename: str, size_bytes: int, *, max_bytes: int | None = None) -> None:
    """校验扩展名与大小;不合法即抛 UploadValidationError,调用方不得写入任何存储。"""
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
    """读取 manifest.yaml 的 documents 段;它是摄取时角色与版本信息的唯一事实来源。"""
    manifest = yaml.safe_load((corpus_dir / "manifest.yaml").read_text(encoding="utf-8"))
    return [DocumentSpec(**entry) for entry in manifest["documents"]]


def compute_content_hash(text: str) -> str:
    """蓝图 6.1:内容哈希用于变更识别与重复内容检查。"""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParsedItem:
    """解析产出的结构化条目。page 为 None 表示该格式无页码(如 Markdown)。"""

    kind: Literal["title", "heading", "text"]
    text: str
    page: int | None = None


def parse_markdown(md_text: str) -> list[ParsedItem]:
    """Markdown 的结构化解析:仅识别 # 标题、## 章节标题与正文段落,与语料生成器对称。

    PDF 的统一 Docling 解析入口在工作单元 3 接入(需下载布局分析模型);
    本函数同时充当切块逻辑的测试夹具。
    """
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
    """统一解析入口:Markdown 走轻量解析,PDF 走 Docling(蓝图 3.1)。"""
    suffix = path.suffix.lower()
    if suffix == ".md":
        return parse_markdown(path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        return _parse_pdf_with_docling(path)
    raise UploadValidationError(f"不支持的文件类型: {suffix or '(无扩展名)'}")


def _parse_pdf_with_docling(path: Path) -> list[ParsedItem]:
    """Docling 解析 PDF,懒加载(首次转换需经 HF_ENDPOINT 下载布局分析模型)。

    蓝图 3.2:OCR 不属于 MVP——语料 PDF 均为数字原生文本层,故显式禁用 OCR 管线;
    这也避免了 RapidOCR 模型下载这一额外外部依赖(2026-08-07 实测 modelscope 不可达)。

    标题/章节标题保留页码,供 source_locator 生成"第 N 页 / 章节"格式(蓝图 6.1)。
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    get_settings().prepare_runtime_env()  # 须在 Docling 模型加载前禁用 torch.compile
    pipeline_options = PdfPipelineOptions(do_ocr=False)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(str(path))
    items: list[ParsedItem] = []
    for item, _level in result.document.iterate_items():
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        raw_label = getattr(item, "label", None)
        label = str(getattr(raw_label, "value", raw_label or "text"))
        prov = getattr(item, "prov", None)
        page = prov[0].page_no if prov else None
        if label == "title":
            kind: Literal["title", "heading", "text"] = "title"
        elif label == "section_header":
            kind = "heading"
        else:
            kind = "text"
        items.append(ParsedItem(kind, text, page))
    return items


def _split_long_paragraph(para: str, max_chars: int) -> list[str]:
    """段落超过 max_chars 时先按句子边界装箱;极端长句按字符硬切,保证每片不超限。"""
    if len(para) <= max_chars:
        return [para]
    sentences: list[str] = []
    buf = ""
    for ch in para:
        buf += ch
        if ch in _SENTENCE_ENDINGS:
            sentences.append(buf)
            buf = ""
    if buf:
        sentences.append(buf)

    pieces: list[str] = []
    cur = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
        elif cur and len(cur) + len(sentence) > max_chars:
            pieces.append(cur)
            cur = sentence
        else:
            cur += sentence
    if cur:
        pieces.append(cur)
    return pieces


def chunk_items(
    items: list[ParsedItem],
    spec: DocumentSpec,
    *,
    max_chars: int,
    now: datetime | None = None,
) -> list[ChunkRecord]:
    """把结构化条目切成文档块。

    规则:
    - title 条目不成块(文档标题由 manifest 提供);
    - 以 heading 为块边界,正文聚合到所属章节;
    - 章节内容超过 max_chars 时在段落/句子边界细分,细分块共享同一 source_locator;
    - chunk_id 采用蓝图 6.1 的稳定格式 ``{document_id}:v{version}:chunk-{seq:02d}``。
    """
    timestamp = now or datetime.now(UTC)

    sections: list[tuple[str, int | None, list[str]]] = []
    heading: str | None = None
    page: int | None = None
    paras: list[str] = []

    def flush() -> None:
        nonlocal paras
        if heading is not None and paras:
            sections.append((heading, page, paras))
        paras = []

    for item in items:
        if item.kind == "title":
            continue
        if item.kind == "heading":
            flush()
            heading = item.text
            page = item.page
        else:
            paras.append(item.text)
    flush()

    chunks: list[ChunkRecord] = []
    seq = 0
    for sec_heading, sec_page, sec_paras in sections:
        locator = f"第 {sec_page} 页 / {sec_heading}" if sec_page is not None else sec_heading
        buffer = ""
        for para in sec_paras:
            for piece in _split_long_paragraph(para, max_chars):
                if buffer and len(buffer) + len(piece) + 1 > max_chars:
                    seq += 1
                    chunks.append(_make_record(spec, seq, locator, buffer, timestamp))
                    buffer = piece
                else:
                    buffer = f"{buffer}\n{piece}" if buffer else piece
        if buffer:
            seq += 1
            chunks.append(_make_record(spec, seq, locator, buffer, timestamp))
    return chunks


def _make_record(
    spec: DocumentSpec, seq: int, locator: str, text: str, timestamp: datetime
) -> ChunkRecord:
    return ChunkRecord(
        document_id=spec.document_id,
        document_version=spec.version,
        chunk_id=f"{spec.document_id}:v{spec.version}:chunk-{seq:02d}",
        source_file_name=spec.file,
        source_locator=locator,
        title=spec.title,
        department=spec.department,
        allowed_roles=[str(role) for role in spec.allowed_roles],
        status=ChunkStatus.ACTIVE,
        content_hash=compute_content_hash(text),
        text=text,
        created_at=timestamp,
        updated_at=timestamp,
    )
