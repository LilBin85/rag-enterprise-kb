"""运行冻结企业任务集的离线确定性契约验收。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_kb.evaluation import load_task_evaluation_dataset, run_dataset


# 默认语料/输出锚定仓库根 data；默认契约锚定后端工程根 schemas。
BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "benchmarks" / "enterprise_task_evaluation.yaml"
DEFAULT_SCHEMA = BACKEND_ROOT / "schemas" / "enterprise-task-evaluation.schema.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "local" / "benchmarks" / "enterprise-task-evaluation.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="运行离线确定性企业任务验收")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_dataset(load_task_evaluation_dataset(args.dataset, args.schema))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
