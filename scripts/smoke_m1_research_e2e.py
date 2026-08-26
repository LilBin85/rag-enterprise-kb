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


def _tool_call_names(task_id: str) -> tuple[str, ...]:
    from sqlalchemy import select

    from rag_kb.config import get_settings
    from rag_kb.infrastructure.database import build_database
    from rag_kb.infrastructure.database.models import ToolCallRow

    database = build_database(get_settings().database_url)
    try:
        with database.sessions() as session:
            return tuple(
                session.scalars(
                    select(ToolCallRow.tool_name)
                    .where(ToolCallRow.task_id == task_id)
                    .order_by(ToolCallRow.created_at)
                )
            )
    finally:
        database.engine.dispose()


def run(args: argparse.Namespace) -> dict[str, object]:
    _require_isolated_inputs(args)
    demo_token = secrets.token_urlsafe(24)
    _configure_environment(args, demo_token)
    _initialize()

    port = _pick_port(args.port)
    base_url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config("rag_kb.main:app", host="127.0.0.1", port=port, log_level="warning")
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

            task = _expect(
                client.post(
                    "/v1/tasks",
                    headers=headers,
                    json={
                        "task_mode": "research",
                        "goal": (
                            f"第一轮只调用 list_document_versions 确认文档 {document_id} 的版本，"
                            "等待它的 ToolMessage 后，第二轮只调用 compare_document_versions 比较版本 1 和 2 的年假变化；"
                            "第三轮只调用 search_knowledge 找出当前版本员工入职满 1 年后的带薪年假天数；"
                            "第四轮只调用 fetch_evidence 获取支撑结论的证据。禁止并行或同轮调用多个工具。"
                            "完成这些步骤后，仅依据工具内容回答，并返回 evidence_ids。"
                        ),
                    },
                ),
                202,
                "tasks",
            )
            task_id = task.get("task_id")
            if not isinstance(task_id, str):
                raise SmokeError("task_id_missing")

            from rag_kb.worker import build_task_worker

            worker = build_task_worker(worker_id="m1-e2e-task")
            try:
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
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise SmokeError("result_evidence_missing")
        if not isinstance(citations, list) or len(citations) != len(evidence_ids):
            raise SmokeError("result_citations_invalid")
        tool_names = _tool_call_names(task_id)
        expected_tool_trace = (
            "list_document_versions",
            "compare_document_versions",
            "search_knowledge",
            "fetch_evidence",
        )
        if tool_names != expected_tool_trace:
            raise SmokeError("research_tool_trace_invalid")
        return {
            "status": "passed",
            "execution_mode": "live_postgres_msgpack",
            "document_version_endpoint": "/v1/documents/{document_id}/versions",
            "document_versions": 2,
            "tool_trace": list(tool_names),
            "task_id": task_id,
            "document_id": document_id,
            "tool_calls": len(tool_names),
            "citation_count": len(citations),
        }
    finally:
        server.should_exit = True
        api_thread.join(timeout=10)


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
