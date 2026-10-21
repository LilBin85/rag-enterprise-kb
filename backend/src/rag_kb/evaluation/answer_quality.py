"""答案质量的离屏确定性评测：引用合法 ≠ 事实被支持。

证据账本引用复核（``query_service``）只能保证模型引用了本轮真实授权片段；它挡不住
"引用『年假 15 天』的真实片段，却回答『年假 200 天』"。本模块按检索层同一
"数字+单位"口径（``QUANTITY_CLAIM``）提取答案断言，并在被引用块中核对等价数量：
阿拉伯/中文数字互认，"20 万元"与"二十万元"刻度互认。

这是离屏评测而不是运行时硬门——与检索层证据门同一教训：字面规则会把可纠正的
前提再次误拒。调用方在拿到真实回答样本后逐条调用 ``evaluate_answer_support``。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from rag_kb.application.evidence_sufficiency import QUANTITY_CLAIM

_CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_SMALL = {"十": 10, "百": 100, "千": 1000}
_CN_SCALE = {"万": 10_000, "亿": 100_000_000}


@dataclass(frozen=True)
class AnswerQualityReport:
    """答案正文数量断言的支持情况；``supported`` 为 False 表示存在无据数字。"""

    claims: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    has_citations: bool

    @property
    def supported(self) -> bool:
        return self.has_citations and not self.unsupported_claims


def evaluate_answer_support(
    cited_texts: Sequence[str],
    answer: str,
) -> AnswerQualityReport:
    """答案中每个"数字+单位"断言都必须被引用块支持（原文一致或数值等价）。"""
    evidence = "\n".join(cited_texts)
    evidence_keys = [
        key
        for key in (_claim_key(number, unit) for number, unit in QUANTITY_CLAIM.findall(evidence))
        if key is not None
    ]
    claims: list[str] = []
    unsupported: list[str] = []
    for number, unit in QUANTITY_CLAIM.findall(answer):
        label = f"{number}{unit}"
        claims.append(label)
        key = _claim_key(number, unit)
        if key is None or not _covered(key, evidence_keys):
            unsupported.append(label)
    return AnswerQualityReport(
        claims=tuple(claims),
        unsupported_claims=tuple(unsupported),
        has_citations=bool(evidence.strip()),
    )


def _claim_key(number: str, unit: str) -> tuple[float, str] | None:
    value = canonical_number(number)
    if value is None:
        return None
    normalized = unit.lower()
    if normalized.startswith("万"):
        value *= 10_000
        normalized = normalized[1:]
    return (value, normalized)


def _covered(key: tuple[float, str], evidence_keys: Sequence[tuple[float, str]]) -> bool:
    value, unit = key
    return any(
        candidate_unit == unit
        and math.isclose(candidate_value, value, rel_tol=1e-9, abs_tol=1e-9)
        for candidate_value, candidate_unit in evidence_keys
    )


def canonical_number(text: str) -> float | None:
    """阿拉伯/中文数字归一（含万/亿刻度："一万五" → 15000，"1.5" → 1.5）。"""
    try:
        return float(text)
    except ValueError:
        pass
    total = section = value = 0
    for character in text:
        if character in _CN_DIGITS:
            value = _CN_DIGITS[character]
        elif character in _CN_SMALL:
            section += (value or 1) * _CN_SMALL[character]
            value = 0
        elif character in _CN_SCALE:
            section = (section + value) * _CN_SCALE[character]
            total += section
            section = value = 0
        else:
            return None
    return float(total + section + value)
