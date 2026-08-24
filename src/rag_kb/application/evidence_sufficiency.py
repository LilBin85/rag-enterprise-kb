"""确定性的证据充分性门；只拦截字面可验证的缺失。

单位/数量与 ASCII 锚点是字面事实（证据中要么存在"15 天"要么不存在，不存在词汇
鸿沟），适合确定性拦截；主谓语义是否足以回答交由回答模型的 verdict 判定（蓝图
7.2 SystemPrompt 已要求"上下文不足必须 refusal"），防幻觉兜底由证据账本引用复核
承担。这与 RAGFlow / txtai 等主流 RAG 的分层一致：检索层只做字面与分数过滤，
语义充分性在生成层判定。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_kb.domain.retrieval import RetrievalCandidate


_ASCII_ANCHOR = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9._-]{1,}(?![A-Za-z0-9])")
_NUMBER = r"[0-9零一二两三四五六七八九十百千万亿]+"
_UNITS = (
    "个工作日",
    "工作日",
    "个字符",
    "分钟",
    "小时",
    "万元",
    "字符",
    "GB",
    "天",
    "日",
    "次",
    "月",
    "年",
    "%",
    "元",
)


class DeterministicEvidenceSufficiencyPolicy:
    """检查问题中的可验证锚点，避免仅凭主题相似度回答缺失事实。"""

    def select_candidates(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> tuple[RetrievalCandidate, ...]:
        """问题点名文档时，只保留名称匹配度最高的文档候选。"""

        normalized_query = _normalize_name(query)
        scored = tuple(
            (candidate, _document_name_match(normalized_query, candidate.source_file_name))
            for candidate in candidates
        )
        best = max((score for _candidate, score in scored), default=0)
        scoped = (
            tuple(candidates)
            if best < 4
            else tuple(candidate for candidate, score in scored if score == best)
        )
        if not _has_explicit_requirements(query):
            return scoped
        groups = _document_groups(scoped)
        supported = {
            binding
            for binding, group in groups.items()
            if _group_insufficiency_code(query, group) is None
        }
        if not supported:
            return scoped
        return tuple(
            candidate
            for candidate in scoped
            if (candidate.document_id, candidate.version_id) in supported
        )

    def insufficiency_code(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> str | None:
        if not candidates:
            return "no_candidates"
        codes = tuple(
            _group_insufficiency_code(query, group)
            for group in _document_groups(candidates).values()
        )
        if any(code is None for code in codes):
            return None
        return codes[0] if codes else "no_candidates"


def _normalize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", value).lower()


def _document_name_match(normalized_query: str, source_file_name: str) -> int:
    stem = _normalize_name(source_file_name.rsplit(".", 1)[0])
    stem = re.sub(r"v[0-9]+$", "", stem, flags=re.IGNORECASE)
    if not stem.endswith(("制度", "规定", "协议", "规范", "流程", "政策")):
        return 0
    return max(
        (
            len(stem[index:])
            for index in range(max(0, len(stem) - 24), len(stem) - 3)
            if stem[index:] in normalized_query
        ),
        default=0,
    )


def _document_groups(
    candidates: Sequence[RetrievalCandidate],
) -> dict[tuple[str, str], tuple[RetrievalCandidate, ...]]:
    groups: dict[tuple[str, str], list[RetrievalCandidate]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.document_id, candidate.version_id), []).append(candidate)
    return {binding: tuple(group) for binding, group in groups.items()}


def _requested_units(query: str) -> tuple[str, ...]:
    return tuple(
        unit
        for unit in _UNITS
        if re.search(rf"(?:多少|几)\s*.{{0,4}}{re.escape(unit)}", query, re.IGNORECASE)
    )


def _has_explicit_requirements(query: str) -> bool:
    return bool(_requested_units(query) or "多久" in query or _ASCII_ANCHOR.search(query))


def _group_insufficiency_code(
    query: str,
    candidates: Sequence[RetrievalCandidate],
) -> str | None:
    evidence = "\n".join(candidate.text for candidate in candidates)
    bound_metadata = "\n".join(candidate.source_file_name for candidate in candidates)
    for unit in _requested_units(query):
        if re.search(rf"{_NUMBER}\s*{re.escape(unit)}", evidence, re.IGNORECASE) is None:
            return "evidence_quantitative_fact_missing"
    if "多久" in query and re.search(
        rf"{_NUMBER}\s*(?:分钟|小时|天|日|个工作日|工作日|月|年)",
        evidence,
    ) is None:
        return "evidence_quantitative_fact_missing"
    anchors = tuple(dict.fromkeys(token.lower() for token in _ASCII_ANCHOR.findall(query)))
    anchor_material = f"{evidence}\n{bound_metadata}".lower()
    if anchors and any(anchor not in anchor_material for anchor in anchors):
        return "evidence_anchor_missing"
    return None
