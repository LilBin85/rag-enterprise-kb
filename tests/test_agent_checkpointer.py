"""M1 checkpoint 序列化必须显式 allowlist 项目结果类型。"""

from rag_kb.agent.checkpointer import task_checkpoint_serializer
from rag_kb.agent.result import TaskResultCandidate


def test_task_checkpoint_serializer_round_trips_structured_result_without_pickle() -> None:
    candidate = TaskResultCandidate(
        status="refused",
        refusal_code="insufficient_evidence",
        stop_reason="no_evidence",
    )

    serializer = task_checkpoint_serializer()

    assert serializer.loads_typed(serializer.dumps_typed(candidate)) == candidate
