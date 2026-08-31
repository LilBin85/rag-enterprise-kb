"""部署与验收入口的机械静态回归；不要求本机安装 Docker 或启动外部组件。"""

import json
import re
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
    worker_names = ("task-worker", "ingestion-worker", "projection-worker")
    assert {
        "postgres",
        "qdrant",
        "qdrant-init",
        "migrate",
        "api",
        *worker_names,
    }.issubset(compose["services"])
    services = compose["services"]
    assert "worker" not in services
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
    for name, queue in zip(worker_names, ("task", "ingestion", "projection"), strict=True):
        assert services[name]["command"] == [
            "python",
            "-m",
            "rag_kb.worker",
            "--queue",
            queue,
            "--watch",
        ]
        assert services[name]["restart"] == "unless-stopped"
        assert services[name]["depends_on"] == {
            "postgres": {"condition": "service_healthy"},
            "qdrant": {"condition": "service_healthy"},
            "qdrant-init": {"condition": "service_completed_successfully"},
            "migrate": {"condition": "service_completed_successfully"},
        }
        assert "object_storage:/app/data/local/objects" in services[name]["volumes"]
    assert services["qdrant-init"]["command"] == ["python", "-m", "rag_kb.qdrant_init"]
    assert services["qdrant-init"]["depends_on"]["qdrant"]["condition"] == "service_healthy"
    assert services["postgres"]["ports"] == ["127.0.0.1:5432:5432"]
    assert services["qdrant"]["ports"] == [
        "127.0.0.1:6333:6333",
        "127.0.0.1:6334:6334",
    ]
    assert services["api"]["ports"] == ["127.0.0.1:8000:8000"]
    expected_runtime_config = {
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
        "OBSERVABILITY_ENABLED": "${OBSERVABILITY_ENABLED:-true}",
        "METRICS_ENABLED": "${METRICS_ENABLED:-true}",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "${OTEL_EXPORTER_OTLP_ENDPOINT:-http://otel-collector:4317}",
        "OTEL_TRACE_SAMPLE_RATIO": "${OTEL_TRACE_SAMPLE_RATIO:-1.0}",
    }
    for service in ("api", *worker_names):
        assert {
            name: services[service]["environment"][name]
            for name in expected_runtime_config
        } == expected_runtime_config
    deepseek_config = {
        "DEEPSEEK_BASE_URL": "${DEEPSEEK_BASE_URL:-https://api.deepseek.com}",
        "DEEPSEEK_MODEL": "${DEEPSEEK_MODEL:-deepseek-v4-flash}",
        "DEEPSEEK_API_KEY": "${DEEPSEEK_API_KEY:-}",
    }
    for service in ("api", "task-worker"):
        assert {
            name: services[service]["environment"][name]
            for name in deepseek_config
        } == deepseek_config
    for service in ("ingestion-worker", "projection-worker", "migrate"):
        assert not deepseek_config.keys() & services[service]["environment"].keys()
    assert services["qdrant-init"]["environment"]["EMBEDDING_DIM"] == "${EMBEDDING_DIM:-1024}"
    for service in ("api", *worker_names):
        assert services[service]["depends_on"]["qdrant-init"]["condition"] == (
            "service_completed_successfully"
        )
    assert services["api"]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
    assert not set(worker_names).intersection(services["api"]["depends_on"])
    assert services["qdrant"]["healthcheck"]["test"]
    for service in ("api", *worker_names):
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


def test_observability_profile_is_internal_and_provisioned() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    for name in ("otel-collector", "jaeger", "prometheus", "grafana"):
        assert services[name]["profiles"] == ["observability"]
    assert services["otel-collector"]["expose"] == ["4317", "8888"]
    assert services["jaeger"]["ports"] == ["127.0.0.1:16686:16686"]
    assert services["jaeger"]["expose"] == ["4317"]
    assert services["prometheus"]["ports"] == ["127.0.0.1:9090:9090"]
    assert services["grafana"]["ports"] == ["127.0.0.1:3000:3000"]
    assert services["grafana"]["volumes"] == [
        "./ops/observability/grafana/provisioning/alerting:/etc/grafana/provisioning/alerting:ro",
        "./ops/observability/grafana/provisioning/dashboards:/etc/grafana/provisioning/dashboards:ro",
        "./ops/observability/grafana/provisioning/datasources:/etc/grafana/provisioning/datasources:ro",
        "./ops/observability/grafana/dashboards:/var/lib/grafana/dashboards:ro",
    ]
    worker_ports = {
        "task-worker": "9464",
        "ingestion-worker": "9465",
        "projection-worker": "9466",
    }
    for name, port in worker_ports.items():
        assert "ports" not in services[name]
        assert services[name]["expose"] == [port]
        assert services[name]["environment"]["WORKER_METRICS_PORT"] == port

    prometheus = yaml.safe_load(
        (ROOT / "ops/observability/prometheus/prometheus.yml").read_text(encoding="utf-8")
    )
    scrape_targets = {
        scrape["job_name"]: {
            target
            for config in scrape["static_configs"]
            for target in config["targets"]
        }
        for scrape in prometheus["scrape_configs"]
    }
    assert scrape_targets == {
        "rag-kb-api": {"api:8000"},
        "rag-kb-task-worker": {"task-worker:9464"},
        "rag-kb-ingestion-worker": {"ingestion-worker:9465"},
        "rag-kb-projection-worker": {"projection-worker:9466"},
        "rag-kb-collector": {"otel-collector:8888"},
    }
    recording = yaml.safe_load(
        (ROOT / "ops/observability/prometheus/rules/recording.yml").read_text(encoding="utf-8")
    )
    recording_rules = {
        rule["record"]: rule["expr"] for rule in recording["groups"][0]["rules"]
    }
    for quantile in ("0.50", "0.95", "0.99"):
        assert f"histogram_quantile({quantile}" in "\n".join(recording_rules.values())
    for name, metric in (
        ("rag_kb:llm_failure_ratio:rate5m", "rag_kb_llm_calls_total"),
        (
            "rag_kb:retrieval_dependency_failure_ratio:rate5m",
            "rag_kb_retrieval_stage_duration_seconds_count",
        ),
        ("rag_kb:task_dependency_failure_ratio:rate5m", "rag_kb_task_runs_total"),
        ("rag_kb:ingestion_dependency_failure_ratio:rate5m", "rag_kb_ingestion_jobs_total"),
        ("rag_kb:tool_timeout_ratio:rate15m", "rag_kb_tool_calls_total"),
        ("rag_kb:approval_expiry_ratio:rate15m", "rag_kb_approvals_total"),
        ("rag_kb:retention_cleanup_failure_ratio:rate15m", "rag_kb_recovery_actions_total"),
    ):
        expr = recording_rules[name]
        assert "/" in expr and "clamp_min(" in expr
        assert expr.count(metric) == 2
    projection_proxy = recording_rules["rag_kb:projection_unconverged_proxy:10m"]
    assert 'outcome=~"retry_scheduled|failure_recording_failed"' in projection_proxy
    assert 'outcome="succeeded"' in projection_proxy
    assert ">= 2" in projection_proxy and "or vector(0)" in projection_proxy
    assert "otelcol_exporter_queue_size" in recording_rules[
        "rag_kb:collector_exporter_queue_utilization"
    ]
    assert "otelcol_exporter_send_failed_spans" in recording_rules[
        "rag_kb:collector_export_failures_per_second:rate5m"
    ]

    collector = yaml.safe_load(
        (ROOT / "ops/observability/otel-collector.yml").read_text(encoding="utf-8")
    )
    assert collector["receivers"]["otlp"]["protocols"]["grpc"]["endpoint"] == "0.0.0.0:4317"
    assert collector["service"]["telemetry"]["metrics"] == {
        "address": "0.0.0.0:8888",
        "level": "normal",
    }
    assert collector["exporters"]["otlp/jaeger"]["sending_queue"] == {
        "enabled": True,
        "queue_size": 1000,
    }
    assert collector["service"]["pipelines"]["traces"]["exporters"] == ["otlp/jaeger"]

    datasource = yaml.safe_load(
        (ROOT / "ops/observability/grafana/provisioning/datasources/prometheus.yml").read_text(
            encoding="utf-8"
        )
    )
    assert datasource["datasources"][0]["url"] == "http://prometheus:9090"

    dashboards = ROOT / "ops/observability/grafana/dashboards"
    assert {
        "system-overview.json",
        "rag.json",
        "agent.json",
        "ingestion-projection.json",
        "security.json",
    } == {path.name for path in dashboards.glob("*.json")}
    dashboard_configs = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in dashboards.glob("*.json")
    }
    assert len({dashboard["title"] for dashboard in dashboard_configs.values()}) == 5
    dashboard_exprs = {
        name: [
            target["expr"]
            for panel in dashboard["panels"]
            for target in panel.get("targets", [])
        ]
        for name, dashboard in dashboard_configs.items()
    }
    for exprs in dashboard_exprs.values():
        assert all("histogram_quantile(" in expr for expr in exprs if "_bucket" in expr)
    system_exprs = "\n".join(dashboard_exprs["system-overview.json"])
    for contract in (
        "histogram_quantile(0.50",
        "histogram_quantile(0.95",
        "histogram_quantile(0.99",
        "otelcol_exporter_queue_size",
        "otelcol_exporter_send_failed_spans",
    ):
        assert contract in system_exprs
    expected_up_jobs = {
        "rag-kb-api",
        "rag-kb-task-worker",
        "rag-kb-ingestion-worker",
        "rag-kb-projection-worker",
        "rag-kb-collector",
    }
    assert set(re.findall(r'up\{job="([^"]+)"\}', system_exprs)) == expected_up_jobs
    rag_exprs = "\n".join(dashboard_exprs["rag.json"])
    for contract in (
        "sum by (outcome, reason)",
        "rag_kb_retrieval_stage_duration_seconds_bucket",
        "rag_kb_retrieval_candidates_bucket",
        "rag_kb_retrieval_context_token_estimate_bucket",
        "rag_kb_llm_calls_total",
        "rag_kb_llm_tokens_total",
        "rag_kb_llm_usage_missing_total",
    ):
        assert contract in rag_exprs
    agent_exprs = "\n".join(dashboard_exprs["agent.json"])
    for metric in (
        "rag_kb_task_claims_total",
        "rag_kb_task_runs_total",
        "rag_kb_llm_calls_total",
        "rag_kb_tool_calls_total",
        "rag_kb_approvals_total",
        "rag_kb_approval_wait_duration_seconds_bucket",
        "rag_kb_checkpoint_writes_total",
        "rag_kb_lease_failures_total",
        "rag_kb_task_queue_depth",
        "rag_kb_task_queue_oldest_age_seconds",
    ):
        assert metric in agent_exprs
    ingestion_exprs = "\n".join(dashboard_exprs["ingestion-projection.json"])
    for metric in (
        "rag_kb_ingestion_queue_depth",
        "rag_kb_ingestion_queue_oldest_age_seconds",
        "rag_kb_ingestion_phase_duration_seconds_bucket",
        "rag_kb_ingestion_jobs_total",
        "rag_kb_projection_reconciliations_total",
        "rag_kb_projection_reconcile_duration_seconds_bucket",
        "rag_kb:projection_unconverged_proxy:10m",
    ):
        assert metric in ingestion_exprs
    security_exprs = "\n".join(dashboard_exprs["security.json"])
    for contract in (
        "rag_kb_security_boundary_violations_total",
        'mode="write",outcome="unknown"',
        'decision=~"rejected|expired"',
        'kind="content_hash"',
        'outcome="mismatched"',
    ):
        assert contract in security_exprs

    alert_path = ROOT / "ops/observability/grafana/provisioning/alerting/rag-kb.yml"
    alert_rules = {
        rule["uid"]: rule
        for rule in yaml.safe_load(alert_path.read_text(encoding="utf-8"))["groups"][0]["rules"]
    }
    assert set(alert_rules) == {
        "rag-kb-security-boundary",
        "rag-kb-write-unknown",
        "rag-kb-queue-without-capability",
        "rag-kb-target-down",
        "rag-kb-lease-failures",
        "rag-kb-projection-unconverged",
        "rag-kb-dependency-errors",
        "rag-kb-api-degradation",
        "rag-kb-queue-age",
        "rag-kb-operation-reliability",
    }
    alert_exprs = {
        uid: next(query for query in rule["data"] if query["refId"] == "A")["model"]["expr"]
        for uid, rule in alert_rules.items()
    }
    for uid in ("rag-kb-security-boundary", "rag-kb-write-unknown", "rag-kb-lease-failures"):
        assert "or vector(0)" in alert_exprs[uid]
        assert alert_rules[uid]["noDataState"] == "OK"
    target_down = alert_exprs["rag-kb-target-down"]
    assert set(re.findall(r'up\{job="([^"]+)"\}', target_down)) == expected_up_jobs
    for job in expected_up_jobs:
        assert f'up{{job="{job}"}}' in target_down
        assert f'absent(up{{job="{job}"}})' in target_down
    assert alert_rules["rag-kb-target-down"]["noDataState"] == "Alerting"
    assert alert_rules["rag-kb-target-down"]["for"] == "2m"
    assert all(
        rule["noDataState"] == "OK"
        for uid, rule in alert_rules.items()
        if uid != "rag-kb-target-down"
    )
    assert "[1m]" in alert_exprs["rag-kb-lease-failures"]
    assert alert_rules["rag-kb-lease-failures"]["for"] == "5m"
    assert alert_exprs["rag-kb-projection-unconverged"] == (
        "rag_kb:projection_unconverged_proxy:10m"
    )
    dependency_expr = alert_exprs["rag-kb-dependency-errors"]
    assert "rag_kb:llm_failure_ratio:rate5m" in dependency_expr
    assert "rag_kb:retrieval_dependency_failure_ratio:rate5m" in dependency_expr
    assert "rag_kb:task_dependency_failure_ratio:rate5m" in dependency_expr
    assert "rag_kb:ingestion_dependency_failure_ratio:rate5m" in dependency_expr
    assert dependency_expr.count("> bool 0.05") == 4
    api_expr = alert_exprs["rag-kb-api-degradation"]
    assert "rag_kb:http_5xx_ratio:rate5m > bool 0.05" in api_expr
    assert "rag_kb:http_p95_seconds:5m > bool 2" in api_expr
    reliability_expr = alert_exprs["rag-kb-operation-reliability"]
    for ratio in ("tool_timeout_ratio", "approval_expiry_ratio", "retention_cleanup_failure_ratio"):
        assert ratio in reliability_expr
    assert reliability_expr.count("> bool 0.1") == 3


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
    queue_commands = (
        "python -m rag_kb.worker --queue task --watch",
        "python -m rag_kb.worker --queue ingestion --watch",
        "python -m rag_kb.worker --queue projection --watch",
    )
    for text in (readme, guide):
        assert "alembic upgrade head" in text
        assert all(command in text for command in queue_commands)
        assert "--include-tasks" not in text
        assert "docs/acceptance/latest.json" in text
        assert "/v1/knowledge/query" in text
    assert "检索质量门" in readme and "unverified" in readme

    script = (ROOT / "scripts/start_worker.ps1").read_text(encoding="utf-8")
    assert '[ValidateSet("task", "ingestion", "projection")]' in script
    assert "[Parameter(Mandatory = $true)]" in script
    assert "python -m rag_kb.worker --queue $Queue --watch" in script


def test_acceptance_schema_has_explicit_unverified_and_hash_contracts() -> None:
    schema = json.loads((ROOT / "schemas/project-acceptance-report.schema.json").read_text(encoding="utf-8"))
    statuses = schema["$defs"]["status_object"]["properties"]["status"]["enum"]
    assert {"passed", "failed", "skipped", "unverified"}.issubset(statuses)
    assert schema["properties"]["report_hash"]["pattern"] == "^[0-9a-f]{64}$"
    assert "evidence_artifacts" in schema["required"]
    assert "m2" in schema["properties"]
    assert "cost" not in schema["required"]
