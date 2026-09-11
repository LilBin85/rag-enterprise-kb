"""冻结语料 PDF 生成器:将 data/corpus/sources/ 下的 Markdown 源渲染为数字原生 PDF。

设计约束(蓝图 3.2 / 11.1):
- 生成的 PDF 必须带完整文本层(非扫描件),使 Docling 可直接解析,无需 OCR 或多模态模型。
- 语料内容全部自造,不含真实企业敏感数据。
- 重复运行产生相同结构的 PDF,保证语料可复现。

用法: python scripts/make_corpus_pdfs.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# 语料位于仓库根 data/corpus；脚本移入 backend/scripts 后取上两级。
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES_DIR = REPO_ROOT / "data" / "corpus" / "sources"
OUTPUT_DIR = REPO_ROOT / "data" / "corpus"
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
FONT_NAME = "SimHei"

TITLE_STYLE = ParagraphStyle("title", fontName=FONT_NAME, fontSize=18, leading=26, spaceAfter=10 * mm)
HEADING_STYLE = ParagraphStyle("heading", fontName=FONT_NAME, fontSize=13, leading=20, spaceBefore=6 * mm, spaceAfter=3 * mm)
BODY_STYLE = ParagraphStyle("body", fontName=FONT_NAME, fontSize=10.5, leading=17, spaceAfter=2 * mm)


def md_to_story(md_text: str) -> list:
    """极简 Markdown 解析:仅支持 # 标题、## 章节标题与普通段落,对冻结语料足够。"""
    story = []
    for raw_line in md_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            story.append(Paragraph(line[3:], HEADING_STYLE))
        elif line.startswith("# "):
            story.append(Paragraph(line[2:], TITLE_STYLE))
        else:
            story.append(Paragraph(line, BODY_STYLE))
            story.append(Spacer(1, 1 * mm))
    return story


def build_pdf(source_md: Path, output_pdf: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=source_md.stem,
    )
    doc.build(md_to_story(source_md.read_text(encoding="utf-8")))


def main() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
    for source_md in sorted(SOURCES_DIR.glob("*.md")):
        output_pdf = OUTPUT_DIR / f"{source_md.stem}.pdf"
        build_pdf(source_md, output_pdf)
        print(f"generated: {output_pdf.name}")


if __name__ == "__main__":
    main()
