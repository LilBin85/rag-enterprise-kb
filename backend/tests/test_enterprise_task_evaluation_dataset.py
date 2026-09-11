from pathlib import Path

import pytest

from rag_kb.evaluation import load_task_evaluation_dataset


# 语料/评测数据锚定仓库根 data；契约锚定后端工程根 schemas。
DATASET = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "enterprise_task_evaluation.yaml"
SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "enterprise-task-evaluation.schema.json"


def test_task_evaluation_dataset_is_frozen_and_split_before_test() -> None:
    dataset = load_task_evaluation_dataset(DATASET, SCHEMA)

    assert 60 <= len(dataset.cases) <= 100
    assert len(dataset.sha256) == 64
    assert dataset.cases_for("development")
    assert dataset.cases_for("calibration")
    assert len(dataset.cases_for("test")) >= 12
    assert all(case.case_id for case in dataset.cases_for("test"))
    focus_counts = {
        focus: sum(focus in case.focus_areas for case in dataset.cases)
        for focus in {
            "tool_selection",
            "clarification",
            "unauthorized_write",
            "prompt_injection",
            "citation_correctness",
            "fault_recovery",
        }
    }
    assert all(count >= 8 for count in focus_counts.values()), focus_counts


def test_task_evaluation_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    text = DATASET.read_text(encoding="utf-8")
    duplicate = text.replace("case_id: two-direction-research", "case_id: single-search-answer", 1)
    path = tmp_path / "duplicate.yaml"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate_task_evaluation_case_id"):
        load_task_evaluation_dataset(path, SCHEMA)
