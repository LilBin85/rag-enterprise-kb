"""加载独立任务评测集；测试标签不会反馈给运行时策略。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_kb.domain.tasks import TaskMode, TaskStatus


class TaskEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    split: Literal["development", "calibration", "test"]
    scenario: str
    mode: TaskMode
    goal: str
    scripted_actions: tuple[str, ...] = Field(min_length=1)
    required_events: tuple[str, ...] = Field(min_length=1)
    expected_terminal_status: TaskStatus
    security_gates: tuple[str, ...]


class TaskEvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    dataset_version: str
    description: str
    cases: tuple[TaskEvaluationCase, ...] = Field(min_length=20)
    sha256: str

    @model_validator(mode="after")
    def validate_identity_and_splits(self) -> "TaskEvaluationDataset":
        identifiers = [case.case_id for case in self.cases]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate_task_evaluation_case_id")
        splits = {case.split for case in self.cases}
        if splits != {"development", "calibration", "test"}:
            raise ValueError("task_evaluation_requires_three_splits")
        return self

    def cases_for(self, split: Literal["development", "calibration", "test"]):
        return tuple(case for case in self.cases if case.split == split)


def load_task_evaluation_dataset(path: Path, schema_path: Path) -> TaskEvaluationDataset:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(raw)
    normalized = {
        **raw,
        "cases": [
            {
                **case,
                "mode": TaskMode.RESEARCH.value
                if case["mode"] == "knowledge_research"
                else case["mode"],
            }
            for case in raw["cases"]
        ],
    }
    return TaskEvaluationDataset.model_validate(
        {**normalized, "sha256": hashlib.sha256(raw_bytes).hexdigest()}
    )
