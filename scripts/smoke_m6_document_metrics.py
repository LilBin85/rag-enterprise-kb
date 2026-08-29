"""在当前 Compose 中创建并清理一个隔离文档，验证真实摄取 Metrics。"""

from __future__ import annotations

import json
import time
from uuid import uuid4

import httpx
from sqlalchemy import select

from rag_kb.config import get_settings
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.models import DocumentRow, DocumentVersionRow


API = "http://api:8000"
PROMETHEUS = "http://prometheus:9090"


def _scalar(client: httpx.Client, query: str) -> float:
    payload = client.get(f"{PROMETHEUS}/api/v1/query", params={"query": query}).json()
    rows = payload.get("data", {}).get("result", [])
    return float(rows[0]["value"][1]) if rows else 0.0


def _metrics(client: httpx.Client) -> dict[str, float]:
    return {
        "jobs": _scalar(client, "sum(rag_kb_ingestion_jobs_total)"),
        "phases": _scalar(client, "sum(rag_kb_ingestion_phase_duration_seconds_count)"),
        "queue_depth": _scalar(client, "sum(rag_kb_ingestion_queue_depth)"),
        "projection": _scalar(client, "sum(rag_kb_projection_reconciliations_total)"),
    }


def main() -> int:
    settings = get_settings()
    tokens = tuple(settings.demo_role_tokens.values())
    if not tokens:
        print(json.dumps({"status": "failed", "reason": "demo_identity_unavailable"}))
        return 1
    document_id = f"m6-observability-{uuid4().hex}"
    idempotency_key = uuid4().hex
    content = b"# Synthetic M6 document\n\nThis isolated document validates ingestion telemetry.\n"
    database = build_database(settings.database_url)
    report: dict[str, object] = {"status": "failed", "cleanup": False}
    with httpx.Client(timeout=30) as client:
        baseline = _metrics(client)
        token: str | None = None
        try:
            for candidate in tokens:
                response = client.post(
                    f"{API}/v1/documents",
                    headers={"X-Demo-Role-Token": candidate, "Idempotency-Key": idempotency_key},
                    data={
                        "metadata": json.dumps(
                            {
                                "document_id": document_id,
                                "title": f"M6 isolated {document_id[-12:]}",
                                "department": "m6-observability",
                            }
                        )
                    },
                    files={"file": (f"{document_id}.md", content, "text/markdown")},
                )
                if response.status_code == 202:
                    token = candidate
                    job_id = response.json()["ingestion_job_id"]
                    break
            if token is None:
                raise RuntimeError("document_writer_unavailable")

            deadline = time.monotonic() + 180
            job_status = "pending"
            while time.monotonic() < deadline:
                response = client.get(
                    f"{API}/v1/ingestion-jobs/{job_id}",
                    headers={"X-Demo-Role-Token": token},
                )
                if response.status_code == 200:
                    job_status = response.json()["status"]
                    if job_status in {"succeeded", "failed"}:
                        break
                time.sleep(1)
            if job_status != "succeeded":
                raise RuntimeError("ingestion_not_succeeded")

            with database.sessions() as session:
                document = session.scalar(
                    select(DocumentRow).where(DocumentRow.document_id == document_id)
                )
                version = session.scalar(
                    select(DocumentVersionRow).where(
                        DocumentVersionRow.document_id == document_id,
                        DocumentVersionRow.lifecycle_status == "active",
                    )
                )
                converged = bool(
                    document is not None
                    and version is not None
                    and document.applied_projection_version_id == version.version_id
                    and document.projection_generation
                    == document.applied_projection_generation
                    == version.applied_projection_generation
                )
            time.sleep(20)
            after = _metrics(client)
            deltas = {key: after[key] - baseline[key] for key in baseline}
            report = {
                "status": "passed" if converged and deltas["jobs"] >= 1 and deltas["phases"] >= 5 else "failed",
                "business": {"ingestion": job_status, "projection_converged": converged},
                "metric_deltas": deltas,
                "cleanup": False,
            }
        except Exception as exc:
            report["reason"] = str(exc) if str(exc).startswith(("document_", "ingestion_")) else type(exc).__name__
        finally:
            if token is not None:
                deleted = client.delete(
                    f"{API}/v1/documents/{document_id}",
                    headers={"X-Demo-Role-Token": token},
                )
                report["cleanup"] = deleted.status_code == 200
            database.engine.dispose()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("status") == "passed" and report.get("cleanup") else 1


if __name__ == "__main__":
    raise SystemExit(main())
