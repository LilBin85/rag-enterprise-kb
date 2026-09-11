"""阶段 1 语料完整性测试(蓝图 10.3:确定性验证完全脱离 LLM 与网络)。

覆盖:
- 语料文件存在且 manifest 与磁盘一致
- PDF 为数字原生文本层,关键版本差异可被机器断言(v1 年假 10 天 / v2 年假 15 天 + 育儿假)
- allowed_roles 合法(均为已声明角色)、同文档版本号递增
"""

from pathlib import Path

import pypdfium2 as pdfium
import pytest
import yaml

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.yaml"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def pdf_text(path: Path) -> str:
    doc = pdfium.PdfDocument(str(path))
    return "".join(page.get_textpage().get_text_range() for page in doc)


def pdf_text_compact(path: Path) -> str:
    """去除全部空白字符。

    PDF 文本层的物理换行位置由排版决定(如『年假 10\\r\\n天』),不等于语义边界;
    数值类断言必须在规范化文本上进行,避免断言结果随排版抖动。
    """
    return "".join(pdf_text(path).split())


def test_corpus_files_exist(manifest: dict) -> None:
    for entry in manifest["documents"]:
        assert (CORPUS_DIR / entry["file"]).is_file(), f"缺少语料文件: {entry['file']}"


def test_manifest_matches_disk(manifest: dict) -> None:
    declared = {entry["file"] for entry in manifest["documents"]}
    on_disk = {p.name for p in CORPUS_DIR.glob("*.md")} | {p.name for p in CORPUS_DIR.glob("*.pdf")}
    assert declared == on_disk, f"manifest 与磁盘不一致: 多出 {on_disk - declared}, 缺少 {declared - on_disk}"


def test_allowed_roles_are_known(manifest: dict) -> None:
    known = set(manifest["roles"])
    for entry in manifest["documents"]:
        assert set(entry["allowed_roles"]) <= known, f"{entry['document_id']} 含未声明角色"
        assert entry["allowed_roles"], f"{entry['document_id']} allowed_roles 为空"


def test_leave_policy_versions(manifest: dict) -> None:
    versions = sorted(
        entry["version"] for entry in manifest["documents"] if entry["document_id"] == "hr-leave-policy"
    )
    assert versions == [1, 2], "hr-leave-policy 应有 v1 与 v2 两个版本"


def test_leave_policy_v1_text_layer() -> None:
    compact = pdf_text_compact(CORPUS_DIR / "员工休假制度-v1.pdf")
    assert "10天" in compact, "v1 应包含『年假 10 天』"
    assert "V1" in compact
    assert "育儿假" not in compact, "v1 不应包含育儿假条款"


def test_leave_policy_v2_text_layer() -> None:
    compact = pdf_text_compact(CORPUS_DIR / "员工休假制度-v2.pdf")
    assert "15天" in compact, "v2 应包含『年假 15 天』"
    assert "育儿假" in compact, "v2 应包含育儿假条款"
    assert "替代" in compact, "v2 应声明替代旧版"


def test_cases_reference_known_documents(manifest: dict) -> None:
    known_ids = {entry["document_id"] for entry in manifest["documents"]}
    for case in manifest["cases"]:
        for doc_id in case.get("expect_documents", []):
            assert doc_id in known_ids, f"用例 {case['name']} 引用了未知文档 {doc_id}"
        assert case["role"] in manifest["roles"], f"用例 {case['name']} 角色未声明"
