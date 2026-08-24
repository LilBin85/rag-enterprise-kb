"""部署与验收入口的机械静态回归；不要求本机安装 Docker 或启动外部组件。"""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_docker_image_installs_runtime_model_dependencies() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert '"torch>=2.13,<3" "torchvision>=0,<1"' in dockerfile
    assert 'python -m pip install --no-cache-dir ".[models]"' in dockerfile


def test_compose_contains_migration_api_worker_and_health_dependencies() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert {"postgres", "qdrant", "qdrant-init", "migrate", "api", "worker"}.issubset(
        compose["services"]
    )
    services = compose["services"]
    assert services["migrate"]["command"] == ["alembic", "upgrade", "head"]
    assert services["migrate"]["environment"]["RAG_KB_DATABASE_URL"] == services["api"][
        "environment"
    ]["DATABASE_URL"]
    assert services["api"]["environment"]["DEMO_ROLE_TOKENS"] == "${DEMO_ROLE_TOKENS:-{}}"
    assert "DEMO_ROLE_TOKENS" not in services["migrate"]["environment"]
    assert "DEEPSEEK_API_KEY" not in services["migrate"]["environment"]
    assert services["worker"]["command"] == ["python", "-m", "rag_kb.worker", "--watch", "--include-tasks"]
    assert services["qdrant-init"]["command"] == ["python", "-m", "rag_kb.qdrant_init"]
    assert services["qdrant-init"]["depends_on"]["qdrant"]["condition"] == "service_healthy"
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
