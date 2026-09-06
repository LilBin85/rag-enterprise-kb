"""显式运行一次隔离 M1 Research HTTP -> worker -> DeepSeek smoke。"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import threading
import time
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn


class SmokeError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行会调用 DeepSeek 的隔离 M1 Research E2E；只允许临时资源。"
    )
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-collection", required=True)
    parser.add_argument("--object-store-root", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--previous-source", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180)
    return parser.parse_args()


def _require_isolated_inputs(args: argparse.Namespace) -> None:
    if not args.allow_live:
        raise SmokeError("live_execution_requires_allow_live")
    if not args.qdrant_collection.startswith("m1_e2e_"):
        raise SmokeError("qdrant_collection_must_use_m1_e2e_prefix")
    if any(
        not source.is_file() or source.suffix.lower() != ".md"
        for source in (args.previous_source, args.source)
    ):
        raise SmokeError("source_markdown_required")
    snapshots = args.hf_home / "hub" / "models--BAAI--bge-m3" / "snapshots"
    if not snapshots.exists() or not any(
        (item / "pytorch_model.bin").is_file() for item in snapshots.iterdir()
    ):
        raise SmokeError("bge_m3_cache_required")


def _configure_environment(args: argparse.Namespace, demo_token: str) -> None:
    os.environ.update(
        {
            "DATABASE_URL": args.database_url,
            "QDRANT_URL": args.qdrant_url,
            "QDRANT_V2_COLLECTION": args.qdrant_collection,
            "OBJECT_STORE_ROOT": str(args.object_store_root),
            "HF_HOME": str(args.hf_home),
            "HF_HUB_OFFLINE": "1",
            "DEMO_ROLE_TOKENS": json.dumps({"admin": demo_token}),
        }
    )


def _pick_port(port: int) -> int:
    if port:
        return port
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _expect(response: httpx.Response, status_code: int, endpoint: str) -> dict[str, object]:
    if response.status_code != status_code:
        raise SmokeError(f"http_{endpoint}_{response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise SmokeError(f"http_{endpoint}_payload_invalid")
    return payload


def _wait_for_api(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/health/live", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise SmokeError("api_start_timeout")


def _initialize() -> None:
    from alembic import command
    from alembic.config import Config
    from qdrant_client import QdrantClient

    from rag_kb.agent.checkpointer import open_postgres_checkpointer
    from rag_kb.config import get_settings
    from rag_kb.infrastructure.database import build_database
    from rag_kb.infrastructure.vectorstore.qdrant_collection import ensure_document_collection
    from init_demo_principals import sync_demo_principals

    settings = get_settings()
    config = Config("alembic.ini")
    config.attributes["database_url"] = settings.database_url
    command.upgrade(config, "head")
    checkpointer, connection = open_postgres_checkpointer(settings.database_url)
    try:
        checkpointer.setup()
    finally:
        connection.close()
    ensure_document_collection(
        QdrantClient(url=settings.qdrant_url),
        collection=settings.qdrant_v2_collection,
        dimension=settings.embedding_dim,
    )
    sync_demo_principals(build_database(settings.database_url), settings.demo_role_tokens)


def _observed_task_facts(task_id: str, document_id: str) -> dict[str, object]:
    """从真实数据库观测工具轨迹、版本数与 checkpoint 编码；不做任何推断。"""
    from sqlalchemy import select, text

    from rag_kb.config import get_settings
    from rag_kb.infrastructure.database import build_database
    from rag_kb.infrastructure.database.models import (
        DocumentRow,
        DocumentVersionRow,
        ToolCallRow,
    )

    database = build_database(get_settings().database_url)
    try:
        with database.sessions() as session:
            rows = session.execute(
                select(ToolCallRow.tool_name, ToolCallRow.status)
                .where(ToolCallRow.task_id == task_id)
                .order_by(ToolCallRow.created_at, ToolCallRow.tool_call_id)
            ).all()
            versions = session.scalar(
                select(DocumentVersionRow.version)
                .join(
                    DocumentRow,
                    (DocumentRow.tenant_id == DocumentVersionRow.tenant_id)
                    & (DocumentRow.document_id == DocumentVersionRow.document_id),
                )
                .where(
                    DocumentRow.document_id == document_id,
                    DocumentVersionRow.lifecycle_status.in_(["active", "superseded"]),
                )
                .order_by(DocumentVersionRow.version.desc())
                .limit(1)
            )
            encodings = sorted(
                {
                    value
                    for value in session.execute(
                        text(
                            "SELECT DISTINCT type FROM ("
                            "SELECT type FROM checkpoint_blobs WHERE thread_id = :tid "
                            "UNION SELECT type FROM checkpoint_writes WHERE thread_id = :tid"
                            ") AS encodings WHERE type IS NOT NULL"
                        ),
                        {"tid": task_id},
                    ).scalars()
                }
            )
        return {
            "tool_trace": [row.tool_name for row in rows],
            "tool_statuses": [row.status for row in rows],
            "document_versions": int(versions) if versions is not None else 0,
            "checkpoint_encodings": encodings,
        }
    finally:
        database.engine.dispose()


def _check_serializer_is_strict() -> dict[str, object]:
    """证明 LANGGRAPH_STRICT_MSGPACK 与显式禁用 pickle 的实际运行配置。"""
    from langgraph.checkpoint.serde import _msgpack as langgraph_msgpack
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from rag_kb.agent.checkpointer import (
        _TASK_CHECKPOINT_MSGPACK_TYPES,
        task_checkpoint_serializer,
    )

    configured_flag = os.environ.get("LANGGRAPH_STRICT_MSGPACK", "").strip().lower()
    if configured_flag not in ("1", "true", "yes"):
        raise SmokeError("langgraph_strict_msgpack_env_required")
    if not langgraph_msgpack.STRICT_MSGPACK_ENABLED:
        raise SmokeError("langgraph_runtime_strict_not_enabled")
    serializer = task_checkpoint_serializer()
    if not isinstance(serializer, JsonPlusSerializer) or serializer.pickle_fallback:
        raise SmokeError("checkpoint_pickle_fallback_must_be_disabled")
    allowlist = sorted(f"{module}.{name}" for module, name in _TASK_CHECKPOINT_MSGPACK_TYPES)
    if allowlist != [
        "rag_kb.agent.result.TaskResultCandidate",
        "rag_kb.application.agent_contract.TaskResultCandidate",
    ]:
        raise SmokeError("checkpoint_allowlist_invalid")
    return {
        "runtime_flag": True,
        "serializer": type(serializer).__name__,
        "pickle_fallback": False,
        "allowed_msgpack_modules": allowlist,
        "saver_package": "langgraph-checkpoint-postgres==3.1.2",
    }


def _source_binding() -> dict[str, str]:
    """复用最终汇总器的源码身份计算，避免 M1 证据被历史运行替代。"""
    try:
        from scripts.build_project_acceptance_report import (
            git_revision,
            git_worktree_fingerprint,
        )
    except ModuleNotFoundError:
        from build_project_acceptance_report import (  # type: ignore[no-redef]
            git_revision,
            git_worktree_fingerprint,
        )

    revision = git_revision()
    fingerprint = git_worktree_fingerprint()
    if revision is None or fingerprint is None:
        raise SmokeError("source_binding_unavailable")
    return {"git_revision": revision, "worktree_sha256": fingerprint}


def run(args: argparse.Namespace) -> dict[str, object]:
    _require_isolated_inputs(args)
    demo_token = secrets.token_urlsafe(24)
    # 配置必须先于任何 rag_kb 导入，Settings 与 strict msgpack 均会缓存运行时环境。
    _configure_environment(args, demo_token)
    serializer_evidence = _check_serializer_is_strict()
    _initialize()

    from rag_kb.application.conversation_service import ConversationService
    from rag_kb.config import get_settings
    from rag_kb.infrastructure.database import build_database
    from rag_kb.infrastructure.database.conversation_repository import (
        DatabaseConversationRepository,
    )
    from rag_kb.infrastructure.database.task_repository import DatabaseTaskRepository
    from rag_kb.web.app import create_app
    from rag_kb.web.dependencies import build_conversation_service

    settings = get_settings()
    database = build_database(settings.database_url)
    tasks = DatabaseTaskRepository(
        database,
        task_capability_name=settings.task_agent_capability_name,
        task_capability_protocol_version=settings.task_agent_protocol_version,
    )
    app = create_app()
    app.dependency_overrides[build_conversation_service] = lambda: ConversationService(
        DatabaseConversationRepository(database), tasks
    )
    port = _pick_port(args.port)
    base_url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    api_thread = threading.Thread(target=server.run, daemon=True)
    api_thread.start()
    try:
        _wait_for_api(base_url, args.timeout)
        headers = {"X-Demo-Role-Token": demo_token}
        document_id = f"m1-e2e-{uuid4().hex}"
        with httpx.Client(base_url=base_url, timeout=args.timeout) as client:
            upload = _expect(
                client.post(
                    "/v1/documents",
                    headers={**headers, "Idempotency-Key": uuid4().hex},
                    data={
                        "metadata": json.dumps(
                            {
                                "document_id": document_id,
                                "title": "M1 隔离年假制度",
                                "department": "hr",
                            }
                        )
                    },
                    files={
                        "file": (
                            args.previous_source.name,
                            args.previous_source.read_bytes(),
                            "text/markdown",
                        )
                    },
                ),
                202,
                "documents",
            )
            job_id = upload.get("ingestion_job_id")
            if not isinstance(job_id, str):
                raise SmokeError("ingestion_job_id_missing")
            if upload.get("version") != 1:
                raise SmokeError("initial_document_version_invalid")

            from rag_kb.worker import build_ingestion_worker

            ingestion = build_ingestion_worker(worker_id="m1-e2e-ingestion")
            if ingestion.run_once().status != "succeeded":
                raise SmokeError("ingestion_not_succeeded")
            job = _expect(
                client.get(f"/v1/ingestion-jobs/{job_id}", headers=headers),
                200,
                "ingestion_job",
            )
            if job.get("status") != "succeeded":
                raise SmokeError("ingestion_status_invalid")

            update = _expect(
                client.post(
                    f"/v1/documents/{document_id}/versions",
                    headers={**headers, "Idempotency-Key": uuid4().hex},
                    data={
                        "metadata": json.dumps(
                            {
                                "document_id": document_id,
                                "title": "M1 隔离年假制度",
                                "department": "hr",
                            }
                        )
                    },
                    files={
                        "file": (
                            args.source.name,
                            args.source.read_bytes(),
                            "text/markdown",
                        )
                    },
                ),
                202,
                "document_version",
            )
            update_job_id = update.get("ingestion_job_id")
            if not isinstance(update_job_id, str):
                raise SmokeError("updated_ingestion_job_id_missing")
            if update.get("version") != 2:
                raise SmokeError("updated_document_version_invalid")
            if ingestion.run_once().status != "succeeded":
                raise SmokeError("updated_ingestion_not_succeeded")
            update_job = _expect(
                client.get(f"/v1/ingestion-jobs/{update_job_id}", headers=headers),
                200,
                "updated_ingestion_job",
            )
            if update_job.get("status") != "succeeded":
                raise SmokeError("updated_ingestion_status_invalid")

            # capability 必须先于任务创建注册，否则 m1-fenced-v2 门会拒绝新任务。
            from rag_kb.worker import build_task_worker

            worker = build_task_worker(worker_id="m1-e2e-task")
            try:
                conversation = _expect(
                    client.post(
                        "/v1/conversations",
                        headers=headers,
                        json={},
                    ),
                    201,
                    "conversations",
                )
                conversation_id = conversation.get("conversation_id")
                if not isinstance(conversation_id, str):
                    raise SmokeError("conversation_id_missing")
                submitted = _expect(
                    client.post(
                        f"/v1/conversations/{conversation_id}/messages",
                        headers={**headers, "Idempotency-Key": uuid4().hex},
                        json={
                            "kind": "new",
                            "mode": "research",
                            "content": (
                                f"第一轮只调用 list_document_versions 确认文档 {document_id} 的版本，"
                                "等待它的 ToolMessage 后，第二轮只调用 compare_document_versions 比较版本 1 和 2 的年假变化；"
                                "第三轮只调用 search_knowledge 找出当前版本员工入职满 1 年后的带薪年假天数；"
                                "第四轮只调用 fetch_evidence 获取支撑结论的证据。禁止并行或同轮调用多个工具。"
                                "完成这些步骤后，仅依据工具内容回答，并返回 evidence_ids。"
                            ),
                        },
                    ),
                    202,
                    "conversation_message",
                )
                task = submitted.get("task")
                task_id = task.get("task_id") if isinstance(task, dict) else None
                if not isinstance(task_id, str):
                    raise SmokeError("task_id_missing")
                outcome = worker.run_once()
            finally:
                worker.close()
            if outcome.status != "finished":
                raise SmokeError(f"task_worker_{outcome.status}")
            state = _expect(client.get(f"/v1/tasks/{task_id}", headers=headers), 200, "task")
            if state.get("status") != "succeeded" or not state.get("result_available"):
                raise SmokeError("task_not_succeeded")
            result = _expect(
                client.get(f"/v1/tasks/{task_id}/result", headers=headers),
                200,
                "task_result",
            )

        evidence_ids = result.get("evidence_ids")
        citations = result.get("citations")
        answer = result.get("answer")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise SmokeError("result_evidence_missing")
        if not isinstance(citations, list) or len(citations) != len(evidence_ids):
            raise SmokeError("result_citations_invalid")
        if not isinstance(answer, str) or not answer.strip():
            raise SmokeError("result_answer_missing")
        facts = _observed_task_facts(task_id, document_id)
        tool_names = tuple(facts["tool_trace"])
        tool_statuses = facts["tool_statuses"]
        expected_tool_trace = (
            "list_document_versions",
            "compare_document_versions",
            "search_knowledge",
            "fetch_evidence",
        )
        if tool_names != expected_tool_trace:
            raise SmokeError("research_tool_trace_invalid")
        if any(status != "succeeded" for status in tool_statuses):
            raise SmokeError("research_tool_not_succeeded")
        if facts["document_versions"] != 2:
            raise SmokeError(f"document_versions_observed_{facts['document_versions']}")
        encodings = facts["checkpoint_encodings"]
        if any(encoding == "pickle" for encoding in encodings):
            raise SmokeError("checkpoint_pickle_encoding_observed")
        return {
            "status": "passed",
            "source_binding": _source_binding(),
            "execution_mode": "live_postgres_msgpack",
            "strict_msgpack": serializer_evidence,
            "checkpoint_encodings": sorted(encodings),
            "document_version_endpoint": "/v1/documents/{document_id}/versions",
            "document_versions": facts["document_versions"],
            "tool_trace": list(tool_names),
            "task_id": task_id,
            "document_id": document_id,
            "tool_calls": len(tool_names),
            "citation_count": len(citations),
            "result_readable": True,
        }
    finally:
        server.should_exit = True
        api_thread.join(timeout=10)
        app.dependency_overrides.clear()
        database.engine.dispose()


def main() -> int:
    try:
        args = _parse_args()
        report = run(args)
    except SmokeError as exc:
        print(f"m1_research_e2e_failed:{exc}")
        return 1
    except (OSError, ValueError):
        print("m1_research_e2e_failed:runtime_configuration_invalid")
        return 1
    except Exception as exc:
        print(f"m1_research_e2e_failed:{type(exc).__name__}")
        return 1
    encoded_report = json.dumps(report, ensure_ascii=False, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded_report + "\n", encoding="utf-8")
    print(encoded_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
