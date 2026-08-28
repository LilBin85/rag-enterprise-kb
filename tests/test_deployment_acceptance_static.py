"""部署与验收入口的机械静态回归；不要求本机安装 Docker 或启动外部组件。"""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_docker_image_installs_runtime_model_dependencies() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY pyproject.toml README.md requirements.txt ./" in dockerfile
    assert "python -m pip install --no-cache-dir -r requirements.txt" in dockerfile
    assert "COPY scripts/init_demo_principals.py ./scripts/init_demo_principals.py" in dockerfile
    assert "COPY scripts/init_agent_checkpointer.py ./scripts/init_agent_checkpointer.py" in dockerfile
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "\n-e .\n" in requirements
    assert "langgraph-checkpoint-postgres==3.1.2" in requirements
    assert "psycopg-pool==3.3.1" in requirements
    assert 'pywin32==312; sys_platform == "win32"' in requirements


def test_compose_contains_migration_api_worker_and_health_dependencies() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert {"postgres", "qdrant", "qdrant-init", "migrate", "api", "worker"}.issubset(
        compose["services"]
    )
    services = compose["services"]
    assert services["migrate"]["command"] == [
        "/bin/sh",
        "-c",
        "alembic upgrade head && python scripts/init_agent_checkpointer.py",
    ]
    assert services["migrate"]["environment"]["DATABASE_URL"] == services["api"][
        "environment"
    ]["DATABASE_URL"]
    assert services["api"]["environment"]["DEMO_ROLE_TOKENS"] == "${DEMO_ROLE_TOKENS:-{}}"
    assert "DEMO_ROLE_TOKENS" not in services["migrate"]["environment"]
    assert "DEEPSEEK_API_KEY" not in services["migrate"]["environment"]
    assert services["worker"]["command"] == ["python", "-m", "rag_kb.worker", "--watch", "--include-tasks"]
    assert services["qdrant-init"]["command"] == ["python", "-m", "rag_kb.qdrant_init"]
    assert services["qdrant-init"]["depends_on"]["qdrant"]["condition"] == "service_healthy"
    assert services["postgres"]["ports"] == ["127.0.0.1:5432:5432"]
    assert services["qdrant"]["ports"] == [
        "127.0.0.1:6333:6333",
        "127.0.0.1:6334:6334",
    ]
    assert services["api"]["ports"] == ["127.0.0.1:8000:8000"]
    expected_runtime_config = {
        "DEEPSEEK_BASE_URL": "${DEEPSEEK_BASE_URL:-https://api.deepseek.com}",
        "DEEPSEEK_MODEL": "${DEEPSEEK_MODEL:-deepseek-v4-flash}",
        "EMBEDDING_DIM": "${EMBEDDING_DIM:-1024}",
        "RETRIEVAL_TOP_K": "${RETRIEVAL_TOP_K:-5}",
        "RETRIEVAL_CANDIDATE_K": "${RETRIEVAL_CANDIDATE_K:-}",
        "RETRIEVAL_CONTEXT_K": "${RETRIEVAL_CONTEXT_K:-}",
        "RETRIEVAL_SCORE_THRESHOLD": "${RETRIEVAL_SCORE_THRESHOLD:-}",
        "RETRIEVAL_CONTEXT_TOKEN_BUDGET": "${RETRIEVAL_CONTEXT_TOKEN_BUDGET:-4000}",
        "RETRIEVAL_RRF_K": "${RETRIEVAL_RRF_K:-60}",
        "ENABLE_HYBRID_RETRIEVAL": "${ENABLE_HYBRID_RETRIEVAL:-false}",
        "AGENT_MAX_MODEL_CALLS": "${AGENT_MAX_MODEL_CALLS:-12}",
        "AGENT_MAX_TOOL_CALLS": "${AGENT_MAX_TOOL_CALLS:-20}",
        "AGENT_MAX_RUNTIME_SECONDS": "${AGENT_MAX_RUNTIME_SECONDS:-600}",
        "AGENT_TIMEOUT_SECONDS": "${AGENT_TIMEOUT_SECONDS:-20}",
        "AGENT_CAPABILITY_LEASE_SECONDS": "${AGENT_CAPABILITY_LEASE_SECONDS:-30}",
        "TASK_AGENT_CAPABILITY_NAME": "${TASK_AGENT_CAPABILITY_NAME:-task-agent}",
        "TASK_AGENT_PROTOCOL_VERSION": "${TASK_AGENT_PROTOCOL_VERSION:-m2-fenced-v1}",
        "LANGGRAPH_STRICT_MSGPACK": "true",
        "TOOL_RESULT_TTL_SECONDS": "${TOOL_RESULT_TTL_SECONDS:-604800}",
        "TASK_RESULT_RETENTION_DAYS": "${TASK_RESULT_RETENTION_DAYS:-7}",
        "RESULT_OBJECT_ORPHAN_GRACE_SECONDS": "${RESULT_OBJECT_ORPHAN_GRACE_SECONDS:-3600}",
    }
    for service in ("api", "worker"):
        assert {
            name: services[service]["environment"][name]
            for name in expected_runtime_config
        } == expected_runtime_config
    assert services["qdrant-init"]["environment"]["EMBEDDING_DIM"] == "${EMBEDDING_DIM:-1024}"
    for service in ("api", "worker"):
        assert services[service]["depends_on"]["qdrant-init"]["condition"] == (
            "service_completed_successfully"
        )
    assert services["api"]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
    assert services["qdrant"]["healthcheck"]["test"]
    for service in ("api", "worker"):
        assert services[service]["environment"]["HF_HOME"] == "/home/ragkb/.cache/huggingface"
        assert services[service]["environment"]["HF_HUB_OFFLINE"] == "1"
        model_mount = next(
            mount
            for mount in services[service]["volumes"]
            if isinstance(mount, dict)
            and mount.get("target") == "/home/ragkb/.cache/huggingface"
        )
        assert model_mount == {
            "type": "bind",
            "source": "${HF_HOME:-D:/DevelopmentTools/hf-cache}",
            "target": "/home/ragkb/.cache/huggingface",
            "read_only": True,
        }


def test_example_agent_budget_configuration_matches_runtime_defaults() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "AGENT_MAX_STEPS" not in example
    for line in (
        "AGENT_MAX_MODEL_CALLS=12",
        "AGENT_MAX_TOOL_CALLS=20",
        "AGENT_MAX_RUNTIME_SECONDS=600",
        "AGENT_TIMEOUT_SECONDS=20",
        "AGENT_CAPABILITY_LEASE_SECONDS=30",
        "TASK_AGENT_CAPABILITY_NAME=task-agent",
        "TASK_AGENT_PROTOCOL_VERSION=m2-fenced-v1",
        "LANGGRAPH_STRICT_MSGPACK=true",
        "TOOL_RESULT_TTL_SECONDS=604800",
        "TASK_RESULT_RETENTION_DAYS=7",
        "RESULT_OBJECT_ORPHAN_GRACE_SECONDS=3600",
    ):
        assert line in example


def test_readme_and_usage_use_same_v1_startup_and_acceptance_targets() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "使用指南.md").read_text(encoding="utf-8")
    for text in (readme, guide):
        assert "alembic upgrade head" in text
        assert "rag_kb.worker" in text and "--include-tasks" in text
        assert "docs/acceptance/latest.json" in text
        assert "/v1/knowledge/query" in text
    assert "检索质量门" in readme and "unverified" in readme


def test_acceptance_schema_has_explicit_unverified_and_hash_contracts() -> None:
    schema = json.loads((ROOT / "schemas/project-acceptance-report.schema.json").read_text(encoding="utf-8"))
    statuses = schema["$defs"]["status_object"]["properties"]["status"]["enum"]
    assert {"passed", "failed", "skipped", "unverified"}.issubset(statuses)
    assert schema["properties"]["report_hash"]["pattern"] == "^[0-9a-f]{64}$"
    assert "evidence_artifacts" in schema["required"]
    assert "m2" in schema["properties"]
    assert "cost" not in schema["required"]
