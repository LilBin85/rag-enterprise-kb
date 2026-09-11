"""隔离编排的真实 HTTP 场景；数据库仅用于读取验收证据。"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from prometheus_client.parser import text_string_to_metric_families

from rag_kb.domain.indexing import compute_chunk_manifest_digest
from rag_kb.infrastructure.vectorstore.qdrant_document_index import projection_key


def metric_value(text, name, **labels):
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items())
    )


def _success_metric(h, service):
    names = {
        "ingestion-worker": "rag_kb_ingestion_jobs_total",
        "task-worker": "rag_kb_task_runs_total",
        "projection-worker": "rag_kb_projection_reconciliations_total",
    }
    def probe():
        try:
            return h.metrics(service)
        except RuntimeError:
            return None

    return metric_value(h.poll("metrics_listener:" + service, probe, timeout=60), names[service], outcome="succeeded")


def _metric_increased(h, service, before):
    def probe():
        value = _success_metric(h, service)
        return value if value > before else None

    value = h.poll("metric:" + service, probe, timeout=30)
    h.record("worker_metric", {"service": service, "before": before, "after": value,
                               "identity": h.container_identity(service)})


def _one(h, query, **params):
    rows = h.sql(query, params)
    h.check(len(rows) == 1, "expected_one_database_row")
    return rows[0]


def _upload(h, doc_id, version):
    days = 7 if version == 1 else 11
    content = (f"# 隔离验收制度 {doc_id} V{version}\n\n"
               f"本合成制度只适用于验收项目 {doc_id}。\n"
               f"第 {version} 版规定：材料提交期限为 {days} 个工作日。\n"
               "审批责任人为项目负责人；仅允许生成 Markdown 制度摘要报告。\n")
    result = h.request(
        "POST", "/v1/documents" if version == 1 else f"/v1/documents/{doc_id}/versions",
        expected=202, headers={"Idempotency-Key": f"{doc_id}-v{version}"},
        data={"metadata": json.dumps({"document_id": doc_id, "title": f"隔离验收制度 {doc_id}",
                                     "department": "engineering"}, ensure_ascii=False)},
        files={"file": (f"{doc_id}-v{version}.md", content.encode(), "text/markdown")},
    ).json()
    h.check(result["document_id"] == doc_id and result["version"] == version,
            "upload_binding_mismatch")
    h.record("upload", result)
    return result


def _job(h, upload):
    return h.request("GET", f"/v1/ingestion-jobs/{upload['ingestion_job_id']}").json()


def _observe_stopped(h, label, check, seconds=6):
    started = time.monotonic()

    def probe():
        check()
        return time.monotonic() - started >= seconds

    h.poll(label, probe, timeout=seconds + 15)


def _pending_ingestion(h, upload):
    first = _job(h, upload)
    h.check(first["status"] == "pending" and first["attempts"] == 0,
            "ingestion_consumed_without_worker")

    def check():
        row = _job(h, upload)
        h.check(row["status"] == "pending" and row["attempts"] == first["attempts"],
                "ingestion_consumed_without_worker")

    _observe_stopped(h, "ingestion_backlog", check)
    h.record("ingestion_backlog", first)


def _wait_ingestion(h, upload):
    def probe():
        row = _job(h, upload)
        h.check(row["status"] not in {"failed", "cancelled"}, "ingestion_terminal_failure")
        return row if row["status"] == "succeeded" else None

    row = h.poll("ingestion:" + upload["ingestion_job_id"], probe, timeout=300)
    h.check(row["attempts"] >= 1, "ingestion_not_claimed")
    h.record("ingestion_completed", row)
    return row


def _document_row(h, doc_id):
    return _one(h, "document_projection", id=doc_id)


def _points(h, binding, *, current_only=False):
    must = [{"key": key, "match": {"value": binding[key]}}
            for key in ("tenant_id", "document_id")]
    if current_only:
        must.append({"key": "current", "match": {"value": True}})
    else:
        must.extend([
            {"key": "version_id", "match": {"value": binding["applied_projection_version_id"]}},
            {"key": "projection_generation", "match": {
                "value": binding["applied_projection_generation"]}},
        ])
    points, offset = [], None
    for _ in range(100):
        body = {"filter": {"must": must}, "limit": 100, "with_payload": True,
                "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        page = h.qdrant("POST", f"/collections/{h.collection}/points/scroll", json=body)["result"]
        points.extend(page["points"])
        offset = page.get("next_page_offset")
        if offset is None:
            return points
    raise RuntimeError("synthetic_projection_exceeds_page_limit")


def _verify_projection(h, upload):
    row = _document_row(h, upload["document_id"])
    h.check(row["applied_projection_version_id"] == upload["version_id"],
            "projection_version_mismatch")
    version = _one(h, "document_version_projection", id=upload["version_id"])
    h.check(version["lifecycle_status"] == "active" and
            version["applied_projection_generation"] == row["applied_projection_generation"],
            "projection_database_binding_mismatch")
    receipt = _one(h, "ingestion_receipt", id=upload["ingestion_job_id"])
    chunks = h.sql("chunks_by_version", {"id": upload["version_id"]})
    points = _points(h, row)
    h.check(len(points) == len(chunks) == receipt["indexed_chunk_count"] and len(points) > 0,
            "projection_point_count_mismatch")
    h.check(receipt["index_completed_at"] is not None, "index_receipt_missing")
    entries = []
    for point in points:
        payload = point["payload"]
        h.check(payload.get("current") is True and payload.get("lifecycle_status") == "active",
                "projection_not_current")
        h.check(payload.get("content_hash") == "sha256:" + hashlib.sha256(
            payload["text"].encode()).hexdigest(), "projection_content_hash_mismatch")
        entries.append(SimpleNamespace(**{k: payload[k] for k in (
            "chunk_id", "content_hash", "source_locator")}))
    digest = compute_chunk_manifest_digest(entries)
    h.check(digest == receipt["index_manifest_digest"] == compute_chunk_manifest_digest(
        [SimpleNamespace(**chunk) for chunk in chunks]), "projection_manifest_mismatch")
    current = _points(h, row, current_only=True)
    h.check({point["id"] for point in current} == {point["id"] for point in points},
            "old_projection_still_current")
    h.record("projection_binding", {**row, "point_count": len(points), "manifest_digest": digest})
    return row


def _submit(h, mode, goal, suffix):
    conversation = h.request("POST", "/v1/conversations", expected=201,
                             json={"title": f"e2e-{h.run_id}-{suffix}"}).json()
    result = h.request("POST", f"/v1/conversations/{conversation['conversation_id']}/messages",
                       expected=202, headers={"Idempotency-Key": f"{h.run_id}-{suffix}"},
                       json={"kind": "new", "mode": mode, "content": goal}).json()
    h.check(result["task"] is not None and result["selected_mode"] == mode,
            "async_task_not_accepted")
    h.record("task_accepted", {key: result[key] for key in ("conversation_id", "turn_id", "task")})
    return result["task"]["task_id"]


def _wait_task(h, task_id, status="succeeded"):
    def probe():
        row = h.request("GET", f"/v1/tasks/{task_id}").json()
        h.check(row["status"] not in {"failed", "cancelled", "unknown", "refused", "waiting_for_user"},
                "task_unexpected_terminal_or_clarification:" + row["status"])
        h.check(not (row["status"] == "succeeded" and status != "succeeded"),
                "task_completed_without_required_approval")
        h.check(not (row["status"] == "waiting_for_approval" and status == "succeeded"),
                "task_unexpected_extra_approval")
        return row if row["status"] == status and (status != "succeeded" or row["result_available"]) else None

    row = h.poll("task:" + task_id, probe, timeout=660)
    h.record("task_state", {key: row[key] for key in ("task_id", "status", "state_version", "result_available")})
    return row


def _task_evidence(h, task_id):
    row = _one(h, "task_evidence", id=task_id)
    checkpoint = _one(h, "task_checkpoints", id=task_id)
    h.check(row["lease_generation"] > 0 and checkpoint["count"] > 0, "task_claim_or_checkpoint_missing")
    events = h.request("GET", f"/v1/tasks/{task_id}/events").json()
    h.check(bool(events), "task_events_missing")
    h.record("task_persistence", {**row, "checkpoint_count": checkpoint["count"], "events": events})


def run_worker_scenarios(h):
    """生产 Worker 自己消费；harness 不持有任何 Worker 对象。"""
    doc_id = f"e2e-{h.run_id}-workers"
    with h.scenario("ingestion_backlog_and_restart"):
        h.start("task-worker")
        h.start("projection-worker")
        v1 = _upload(h, doc_id, 1)
        _pending_ingestion(h, v1)
        h.start("ingestion-worker")
        _wait_ingestion(h, v1)
        _verify_projection(h, v1)
        _metric_increased(h, "ingestion-worker", 0)
        before_identity = h.container_identity("ingestion-worker")
        h.stop("ingestion-worker")
        v2 = _upload(h, doc_id, 2)
        _pending_ingestion(h, v2)
        h.start("ingestion-worker")
        _wait_ingestion(h, v2)
        _verify_projection(h, v2)
        _metric_increased(h, "ingestion-worker", 0)
        versions = h.request("GET", f"/v1/documents/{doc_id}").json()["versions"]
        h.check(len(versions) == 2 and {v["version_id"]: v["lifecycle_status"] for v in versions}
                == {v1["version_id"]: "superseded", v2["version_id"]: "active"},
                "version_transition_or_duplicate_error")
        h.record("ingestion_restart", {"before": before_identity,
                                       "after": h.container_identity("ingestion-worker")})

    h.poll("task_capability", lambda: h.sql("task_capability"), timeout=300)
    with h.scenario("task_research"):
        before = _success_metric(h, "task-worker")
        task_id = _submit(h, "research", f"请研究并比较文档 {doc_id} 的 V1 与 V2，"
                          "明确两个版本材料提交期限（工作日）的变化。请用版本比较工具读取历史版本，"
                          "引用两个版本的原文证据；不需要澄清，不要生成文件。", "research")
        _wait_task(h, task_id)
        result = h.request("GET", f"/v1/tasks/{task_id}/result").json()
        citations = result["citations"]
        h.check(result["status"] == "succeeded" and "7" in result["answer"] and "11" in result["answer"],
                "research_version_facts_missing")
        h.check(bool(citations) and all(c["document_id"] == doc_id for c in citations)
                and {c["document_version"] for c in citations} == {1, 2}, "research_version_citations_missing")
        _task_evidence(h, task_id)
        _metric_increased(h, "task-worker", before)
        h.record("research_result", result)

    with h.scenario("task_approval_restart_idempotency"):
        task_id = _submit(h, "controlled_action", f"根据文档 {doc_id} 当前 V2，"
                           "请调用 create_report_artifact（format=markdown）生成且仅生成一份 Markdown 制度摘要报告，"
                            "包含材料提交期限11个工作日及审批责任人。格式和需求已经确定，"
                            "不需要澄清；严格按轮次串行调用工具，每轮只调用一个工具；"
                            "请提出一次导出审批，获批后导出，不执行其他写操作。首次受控写成功回执后，"
                            "立即调用 TaskResultCandidate 返回 succeeded，使用该回执的 artifact_id 和此前只读"
                            "工具回执中的 evidence_ids；不再调用任何读取或受控写工具，也不再请求审批。", "approval")
        state = _wait_task(h, task_id, "waiting_for_approval")
        approvals = h.request("GET", "/v1/approvals", role="approver", params={"task_id": task_id}).json()
        h.check(len(approvals) == 1, "expected_single_approval")
        approval = approvals[0]
        h.check(approval["tool_name"] == "create_report_artifact", "unexpected_approved_tool")
        before_identity = h.container_identity("task-worker")
        _task_evidence(h, task_id)
        h.stop("task-worker")
        body = {"decision": "approve", "expected_task_state_version": state["state_version"]}
        path = f"/v1/approvals/{approval['approval_id']}/decision"
        accepted = h.request("POST", path, role="approver", expected=202, json=body).json()
        replay = h.request("POST", path, role="approver", expected=202, json=body).json()
        h.check(accepted["resume_command"] == replay["resume_command"], "approval_replay_created_command")
        h.request("POST", path, role="approver", expected=409, json={**body, "decision": "reject"})

        def pending():
            commands = h.sql("resume_commands_by_task", {"id": task_id})
            h.check(len(commands) == 1 and commands[0]["status"] == "pending"
                    and commands[0]["attempt_count"] == 0
                    and commands[0]["command_id"] == accepted["resume_command"]["command_id"],
                    "resume_consumed_without_task_worker")
            h.check(not h.sql("artifact_ids_by_task", {"id": task_id}),
                    "artifact_created_without_task_worker")

        _observe_stopped(h, "approval_persisted_not_executed", pending)
        h.record("approval_accepted", {"task_id": task_id, "approval_id": approval["approval_id"],
                                       "resume_command": accepted["resume_command"]})
        h.start("task-worker")
        # 恢复命令先落地；之后若出现另一次审批，必须失败，不能悄悄批准第二次写入。
        def consumed():
            command = _one(h, "resume_command", id=accepted["resume_command"]["command_id"])
            h.check(command["status"] not in {"failed", "expired"}, "resume_command_failed")
            return command if command["status"] == "consumed" else None

        command = h.poll("resume_consumed", consumed, timeout=660)
        h.check(command["applied_checkpoint_id"] is not None, "resume_checkpoint_not_applied")
        _wait_task(h, task_id)
        result = h.request("GET", f"/v1/tasks/{task_id}/result").json()
        artifacts = h.sql("artifacts_by_task", {"id": task_id})
        writes = h.sql("write_tool_calls_by_task", {"id": task_id})
        h.check(len(artifacts) == len(writes) == len(result["artifacts"]) == 1,
                "duplicate_or_missing_controlled_write")
        artifact, write = artifacts[0], writes[0]
        h.check(write["status"] == "succeeded" and write["verified_at"] is not None
                and artifact["status"] == "ready"
                and write["approval_id"] == approval["approval_id"]
                and artifact["tool_call_id"] == write["tool_call_id"], "controlled_write_binding_mismatch")
        content = h.request("GET", f"/v1/tasks/{task_id}/artifacts/{artifact['artifact_id']}/content").content
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        h.check(digest == artifact["content_hash"] == result["artifacts"][0]["content_hash"],
                "artifact_content_hash_mismatch")
        h.check("11" in content.decode("utf-8"), "artifact_synthetic_fact_missing")
        final = h.request("GET", f"/v1/approvals/{approval['approval_id']}", role="approver").json()
        h.check(final["status"] == "executed", "approval_not_executed")
        _task_evidence(h, task_id)
        _metric_increased(h, "task-worker", 0)
        h.record("approval_recovered", {"command": command, "writes": writes, "artifacts": artifacts,
                                       "before": before_identity, "after": h.container_identity("task-worker")})


def _date(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_projection_scenario(h):
    with h.scenario("projection_natural_audit_and_restart_repair"):
        upload = _upload(h, f"e2e-{h.run_id}-projection", 1)
        _wait_ingestion(h, upload)
        initial = _verify_projection(h, upload)
        metric = _success_metric(h, "projection-worker")

        def audited():
            row = _document_row(h, upload["document_id"])
            return row if (row["projection_reconcile_lease_generation"] > initial["projection_reconcile_lease_generation"]
                           and _date(row["projection_verified_at"]) > _date(initial["projection_verified_at"])
                           and _date(row["projection_reconcile_after"]) > _date(initial["projection_reconcile_after"])
                           and row["projection_reconcile_owner"] is None) else None

        due = _date(initial["projection_reconcile_after"])
        audited_row = h.poll("projection_natural_audit", audited,
                            timeout=max(1, (due - datetime.now(timezone.utc)).total_seconds()) + 120)
        _metric_increased(h, "projection-worker", metric)
        before_identity = h.container_identity("projection-worker")
        h.stop("projection-worker")
        # harness 每次检查项目标签；再核对租户、文档、版本、世代和精确 point ID。
        h.container_identity("projection-worker")
        binding = _verify_projection(h, upload)
        point = _points(h, binding)[0]
        payload = point["payload"]
        h.check(payload["tenant_id"] == binding["tenant_id"]
                and payload["document_id"] == upload["document_id"]
                and payload["version_id"] == upload["version_id"]
                and payload["projection_generation"] == binding["applied_projection_generation"]
                and payload["projection_key"] == projection_key(binding["tenant_id"], upload["version_id"], binding["applied_projection_generation"])
                and payload["current"] is True and payload["lifecycle_status"] == "active",
                "fault_point_binding_mismatch")
        h.record("projection_fault", {"point_id": point["id"], "collection": h.collection,
                                      **{k: payload[k] for k in ("tenant_id", "document_id", "version_id", "projection_generation")}})
        h.qdrant("POST", f"/collections/{h.collection}/points/delete?wait=true", json={"points": [point["id"]]},
                  expected_binding={"point_id": point["id"], "tenant_id": binding["tenant_id"],
                                    "document_id": upload["document_id"], "version_id": upload["version_id"],
                                    "projection_generation": binding["applied_projection_generation"],
                                    "projection_key": payload["projection_key"], "current": True,
                                    "lifecycle_status": "active"})

        def still_missing():
            points = _points(h, binding)
            h.check(all(p["id"] != point["id"] for p in points), "projection_repaired_without_worker")
            row = _document_row(h, upload["document_id"])
            h.check(row["applied_projection_generation"] == binding["applied_projection_generation"],
                    "projection_generation_changed_without_worker")

        due = _date(audited_row["projection_reconcile_after"])
        seconds = max(6, (due - datetime.now(timezone.utc)).total_seconds() + 6)
        _observe_stopped(h, "projection_overdue_without_worker", still_missing, seconds=seconds)
        h.start("projection-worker")

        def repaired():
            row = _document_row(h, upload["document_id"])
            return row if (row["applied_projection_generation"] > binding["applied_projection_generation"]
                           and row["projection_reconcile_owner"] is None) else None

        h.poll("projection_repaired", repaired, timeout=420)
        final = _verify_projection(h, upload)
        _metric_increased(h, "projection-worker", 0)
        # 公开问答必须重新检索修复后的资料，不读取 Qdrant payload 充当回答。
        conversation = h.request("POST", "/v1/conversations", expected=201,
                                 json={"title": "投影修复检索验证"}).json()
        answer = h.request("POST", f"/v1/conversations/{conversation['conversation_id']}/messages",
                           headers={"Idempotency-Key": f"{h.run_id}-projection-answer"},
                           json={"kind": "new", "mode": "answer", "content":
                                 f"根据制度 {upload['document_id']}，材料提交期限是多少个工作日？"},
                           timeout=180).json()
        message = answer.get("assistant_message") or {}
        h.check("7" in message.get("content", "") and any(
            c["document_id"] == upload["document_id"] for c in message.get("citations", [])),
            "repaired_projection_not_queryable")
        h.record("projection_recovered", {"initial": initial, "audited": audited_row, "final": final,
                                         "before": before_identity, "after": h.container_identity("projection-worker")})
