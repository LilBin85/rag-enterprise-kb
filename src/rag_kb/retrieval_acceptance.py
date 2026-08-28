"""V2 企业检索基准数据集加载与校验。"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

from rag_kb.models import Role

_REQUIRED_CATEGORY_MINIMUMS = {
    "semantic": 2,
    "exact_term": 2,
    "acl": 2,
    "no_answer": 2,
    "multi_chunk": 2,
}


@dataclass(frozen=True)
class AcceptanceCase:
    case_id: str
    split: str
    category: str
    question: str
    role: Role
    answerable: bool
    gold_chunk_ids: frozenset[str]
    gold_versions: frozenset[tuple[str, int]]
    exact_terms: tuple[str, ...]
    acl_filtered: bool
    forbidden_document_ids: frozenset[str]
    no_answer_kind: str | None = None


@dataclass(frozen=True)
class AcceptanceDataset:
    path: Path
    version: int
    name: str
    description: str
    corpus_manifest: str
    required_top_k: int
    cases: tuple[AcceptanceCase, ...]

    @property
    def category_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(case.category for case in self.cases).items()))

    @property
    def split_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(case.split for case in self.cases).items()))


def load_acceptance_dataset(dataset_path: Path, schema_path: Path) -> AcceptanceDataset:
    raw = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(raw)
    dataset = AcceptanceDataset(
        path=dataset_path,
        version=raw["version"],
        name=raw["name"],
        description=raw["description"],
        corpus_manifest=raw["corpus_manifest"],
        required_top_k=raw["required_top_k"],
        cases=tuple(_parse_case(item) for item in raw["cases"]),
    )
    validate_dataset_semantics(dataset)
    return dataset


def _parse_case(raw: Mapping[str, Any]) -> AcceptanceCase:
    return AcceptanceCase(
        case_id=raw["case_id"],
        split=raw["split"],
        category=raw["category"],
        question=raw["query"],
        role=Role(raw["role"]),
        answerable=raw["answerable"],
        gold_chunk_ids=frozenset(raw["gold_chunk_ids"]),
        gold_versions=frozenset(
            (item["document_id"], item["version"]) for item in raw["gold_versions"]
        ),
        exact_terms=tuple(raw["exact_terms"]),
        acl_filtered=raw["acl_filtered"],
        forbidden_document_ids=frozenset(raw["forbidden_document_ids"]),
        no_answer_kind=raw.get("no_answer_kind"),
    )


def validate_dataset_semantics(dataset: AcceptanceDataset) -> None:
    if dataset.version != 2:
        raise ValueError("enterprise_retrieval_dataset_must_be_v2")
    if len(dataset.cases) < 20:
        raise ValueError("retrieval_acceptance_dataset_requires_at_least_20_cases")
    case_ids = [case.case_id for case in dataset.cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("retrieval_acceptance_dataset_case_ids_must_be_unique")
    for category, minimum in _REQUIRED_CATEGORY_MINIMUMS.items():
        actual = dataset.category_counts.get(category, 0)
        if actual < minimum:
            raise ValueError(f"retrieval_acceptance_dataset_category_too_small:{category}:{actual}<{minimum}")
    if not {"development", "calibration", "test"} <= set(dataset.split_counts):
        raise ValueError("retrieval_acceptance_dataset_requires_development_calibration_and_test_splits")

    for case in dataset.cases:
        if case.answerable:
            if not case.gold_chunk_ids or not case.gold_versions:
                raise ValueError(f"answerable_case_requires_gold:{case.case_id}")
            if not {
                _chunk_document_version(chunk_id) for chunk_id in case.gold_chunk_ids
            }.issubset(case.gold_versions):
                raise ValueError(f"gold_chunk_version_not_declared:{case.case_id}")
        elif case.gold_chunk_ids or case.gold_versions:
            raise ValueError(f"no_answer_case_cannot_have_gold:{case.case_id}")
        if case.category == "exact_term" and not case.exact_terms:
            raise ValueError(f"exact_term_case_requires_terms:{case.case_id}")
        if case.category == "multi_chunk" and len(case.gold_chunk_ids) < 2:
            raise ValueError(f"multi_chunk_case_requires_multiple_gold_chunks:{case.case_id}")
        if case.category == "acl" and not case.acl_filtered:
            raise ValueError(f"acl_category_requires_acl_filter_marker:{case.case_id}")
        if case.no_answer_kind == "acl_denied" and not case.forbidden_document_ids:
            raise ValueError(f"acl_denied_case_requires_forbidden_documents:{case.case_id}")


def _chunk_document_version(chunk_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"(?P<document_id>[a-z0-9-]+):v(?P<version>[1-9][0-9]*):chunk-[0-9]{2,}", chunk_id)
    if match is None:
        raise ValueError(f"invalid_gold_chunk_id:{chunk_id}")
    return match.group("document_id"), int(match.group("version"))
