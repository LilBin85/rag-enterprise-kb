from pathlib import Path

import yaml


WORKFLOWS = Path(".github/workflows")


def _workflow(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _run_commands(workflow: dict) -> list[str]:
    return [
        step["run"]
        for step in workflow["jobs"]["pytest"]["steps"]
        if "run" in step
    ]


def test_unit_and_integration_workflows_are_separate_and_publish_junit() -> None:
    unit = _workflow("tests.yml")
    integration = _workflow("integration-tests.yml")

    assert any('-m "not integration"' in command for command in _run_commands(unit))
    assert set(integration["jobs"]["pytest"]["services"]) == {"postgres", "qdrant"}
    assert any("-m integration" in command for command in _run_commands(integration))
    assert all(
        any("--junitxml=" in command for command in _run_commands(workflow))
        for workflow in (unit, integration)
    )
    assert all(
        any(step.get("uses") == "actions/upload-artifact@v4" for step in workflow["jobs"]["pytest"]["steps"])
        for workflow in (unit, integration)
    )
