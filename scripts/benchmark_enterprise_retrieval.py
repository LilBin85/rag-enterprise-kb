"""Stage 3 企业检索链路与冻结验收集的隔离对比评测。

本脚本不改写旧检索 benchmark，也不把新切块 ID 当作旧基线的 gold。它创建随机
PostgreSQL tenant、随机 Qdrant collection 与临时对象根目录，使用生产的文档服务、
摄取工作器和企业检索器完成一次真实 dense/hybrid 管线对比评测。无论成功或失败，
finally 均只清理本次随机创建的资源。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from sqlalchemy import delete

from rag_kb.application.document_service import DocumentService
from rag_kb.application.enterprise_retriever import EnterpriseRetriever
from rag_kb.application.ingestion_worker import (
    DocumentRegistryIngestionJobPort,
    IngestionWorker,
)
from rag_kb.domain.identity import AuthContext, Permission, Principal
from rag_kb.embedder import BgeM3Embedder
from rag_kb.ingestion import load_manifest
from rag_kb.infrastructure.database.acl_repository import DatabaseAuthorizationPort
from rag_kb.infrastructure.database.base import Base
from rag_kb.infrastructure.database.index_projection_metadata import (
    DatabaseDocumentIndexMetadataPort,
)
from rag_kb.infrastructure.database.models import ACLGrantRow, PrincipalRow, TenantRow
from rag_kb.infrastructure.database.session import build_database
from rag_kb.infrastructure.ingestion.local_document_parser import LocalDocumentParser
from rag_kb.infrastructure.object_store import LocalObjectStore
from rag_kb.infrastructure.vectorstore.qdrant_authorized_search import QdrantAuthorizedSearch
from rag_kb.infrastructure.vectorstore.qdrant_document_index import QdrantDocumentIndex
from rag_kb.retrieval_acceptance import AcceptanceCase, load_acceptance_dataset


@dataclass(frozen=True)
class CaseObservation:
    """不含正文的单 case 检索观测，gold 只匹配 document/version。"""

    case_id: str
    answerable: bool
    candidates: tuple[tuple[str, int, float | None], ...]
    latency_ms: float
    sufficient: bool = True
    insufficiency_code: str | None = None


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _corpus_digest(corpus_dir: Path, specs: Sequence[object]) -> str:
    """按 manifest 顺序和文件字节构造可复现语料指纹，不把正文写进报告。"""
    digest = hashlib.sha256()
    for spec in specs:
        source = corpus_dir / str(getattr(spec, "file"))
        digest.update(str(getattr(spec, "document_id")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(getattr(spec, "version")).encode("ascii"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _git_baseline_hash(repo_root: Path) -> str | None:
    """尽力记录基线提交；不可用时报告 null，不影响隔离清理。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _read_old_baseline(path: Path) -> dict[str, Any]:
    """仅引用旧报告已有指标；不会修改、重跑或把其 chunk 标签用于新链。"""
    reference: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256_file(path) if path.is_file() else None,
        "metrics": None,
    }
    if not path.is_file():
        return reference
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = payload.get("test") if isinstance(payload, dict) else None
        metrics = test.get("metrics") if isinstance(test, dict) else None
        reference["metrics"] = metrics if isinstance(metrics, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        reference["read_error_code"] = "old_baseline_unreadable"
    return reference


def _candidate_bindings(result) -> tuple[tuple[str, int, float | None], ...]:
    return tuple(
        (candidate.document_id, candidate.document_version, candidate.dense_score)
        for candidate in result.candidates
    )


def _evaluate_document_version_cases(
    cases: Iterable[AcceptanceCase],
    observations: Mapping[str, CaseObservation],
) -> dict[str, Any]:
    """以 document+version 为 gold 计算 Recall@5/MRR，明确不是旧 chunk-level 指标。"""
    rows: list[dict[str, Any]] = []
    answerable_count = 0
    recall_hits = 0
    reciprocal_rank_sum = 0.0
    no_answer_count = 0
    no_answer_correct = 0
    acl_denied_count = 0
    acl_leak_case_count = 0
    acl_leak_count = 0

    for case in cases:
        observation = observations[case.case_id]
        returned = tuple((document_id, version) for document_id, version, _ in observation.candidates)
        first_gold_rank = next(
            (
                index
                for index, candidate in enumerate(returned, start=1)
                if candidate in case.gold_versions
            ),
            None,
        )
        leaks = tuple(sorted({document_id for document_id, _version in returned if document_id in case.forbidden_document_ids}))
        if case.answerable:
            answerable_count += 1
            if first_gold_rank is not None:
                recall_hits += 1
                reciprocal_rank_sum += 1.0 / first_gold_rank
        else:
            # 分层考核口径：拒答准确率只统计 out_of_scope 负样本——它们的确定性
            # 拦截（数量/锚点）仍在检索层生效。acl_denied 负样本考核"不许把无权限
            # 文档检索出来"，由 acl_leak_rate 门承担；其最终拒答由回答模型 verdict
            # 层完成，不属于检索层评分范围。
            if case.no_answer_kind != "acl_denied":
                no_answer_count += 1
                if not returned:
                    no_answer_correct += 1
        if case.forbidden_document_ids:
            acl_denied_count += 1
            acl_leak_count += len(leaks)
            if leaks:
                acl_leak_case_count += 1
        rows.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "category": case.category,
                "answerable": case.answerable,
                "gold_document_versions": [
                    {"document_id": document_id, "version": version}
                    for document_id, version in sorted(case.gold_versions)
                ],
                "returned_document_versions": [
                    {"document_id": document_id, "version": version, "dense_score": score}
                    for document_id, version, score in observation.candidates
                ],
                "first_gold_rank": first_gold_rank,
                "no_answer_correct": (
                    # acl_denied 样本不在检索层考核拒答正确性（见计数注释），置 None
                    (not returned)
                    if not case.answerable and case.no_answer_kind != "acl_denied"
                    else None
                ),
                "sufficient": observation.sufficient,
                "insufficiency_code": observation.insufficiency_code,
                "acl_leak_document_ids": list(leaks),
                "latency_ms": round(observation.latency_ms, 3),
            }
        )

    metrics = {
        "metric_contract": "document_version_level_not_legacy_chunk_level",
        "k": 5,
        "case_count": len(rows),
        "answerable_case_count": answerable_count,
        "no_answer_case_count": no_answer_count,
        "acl_denied_case_count": acl_denied_count,
        "recall_at_5": (recall_hits / answerable_count) if answerable_count else None,
        "mrr": (reciprocal_rank_sum / answerable_count) if answerable_count else None,
        "no_answer_accuracy": (no_answer_correct / no_answer_count) if no_answer_count else None,
        "acl_leak_rate": (acl_leak_case_count / acl_denied_count) if acl_denied_count else None,
        "acl_leak_case_count": acl_leak_case_count,
        "acl_leak_count": acl_leak_count,
    }
    return {"metrics": metrics, "cases": rows}


_DENSE_THRESHOLD_CANDIDATES: tuple[float | None, ...] = (None, 0.0, 0.2, 0.4, 0.6, 0.8)


def _selection_objective(configuration: Mapping[str, Any]) -> tuple[float, float, float, int, float]:
    """仅依 calibration 指标选管线；数值并列时稳定偏向 dense-only 与较低阈值。

    检索层选择器只优化安全（无 ACL 泄漏）与质量（召回/MRR）。拒答正确率
    不再参与选择：out_of_scope 问题的最终拒答由回答模型 verdict 层保证
    （见 _evaluate_document_version_cases 的分层考核注释），用检索层阈值
    硬压拒答会以牺牲召回为代价（2026-08-24 实测：为拦截语义不相关但含
    数量词的候选而选 0.6 阈值，导致合法低分正样本被误杀）。
    """
    metrics = configuration["evaluation"]["metrics"]
    threshold = configuration["dense_score_threshold"]
    return (
        float(metrics["acl_leak_count"] == 0),
        float(metrics["recall_at_5"] or 0.0),
        float(metrics["mrr"] or 0.0),
        int(not configuration["hybrid_enabled"]),
        -(float(threshold) if threshold is not None else float("-inf")),
    )


def _select_pipeline_configuration(
    configurations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """从已真实执行的 calibration 配置中挑选唯一、可审计的生产管线。"""
    if not configurations:
        raise ValueError("calibration_configurations_required")
    return dict(max(configurations, key=_selection_objective))


def _auth_for_role(*, tenant_id: str, user_id: str, role: str, request_id: str) -> AuthContext:
    principal = Principal(
        tenant_id=tenant_id,
        user_id=user_id,
        roles=frozenset({role}),
        attributes={"permissions": [Permission.KNOWLEDGE_READ.value]},
    )
    return AuthContext.from_principal(principal, request_id=request_id)


def _create_tenant_principals(database, *, tenant_id: str, run_id: str) -> tuple[AuthContext, dict[str, AuthContext]]:
    writer_id = f"benchmark-writer-{run_id}"
    role_auths: dict[str, AuthContext] = {}
    with database.sessions.begin() as session:
        session.add(TenantRow(tenant_id=tenant_id, name=tenant_id))
        session.flush()
        session.add(
            PrincipalRow(
                tenant_id=tenant_id,
                user_id=writer_id,
                roles=["admin"],
                attributes={
                    "permissions": [
                        Permission.DOCUMENT_WRITE.value,
                        Permission.DOCUMENT_READ.value,
                        Permission.KNOWLEDGE_READ.value,
                    ]
                },
            )
        )
        for role in ("employee", "engineer", "hr", "admin"):
            user_id = f"benchmark-{role}-{run_id}"
            session.add(
                PrincipalRow(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    roles=[role],
                    attributes={"permissions": [Permission.KNOWLEDGE_READ.value]},
                )
            )
            role_auths[role] = _auth_for_role(
                tenant_id=tenant_id,
                user_id=user_id,
                role=role,
                request_id=f"benchmark-query-{role}-{run_id}",
            )
    writer = AuthContext.from_principal(
        Principal(
            tenant_id=tenant_id,
            user_id=writer_id,
            roles=frozenset({"admin"}),
            attributes={
                "permissions": [
                    Permission.DOCUMENT_WRITE.value,
                    Permission.DOCUMENT_READ.value,
                    Permission.KNOWLEDGE_READ.value,
                ]
            },
        ),
        request_id=f"benchmark-ingest-{run_id}",
    )
    return writer, role_auths


def _write_manifest_acl_grants(database, *, tenant_id: str, specs: Sequence[object]) -> None:
    """在 worker 建索引前写入 document ACL，使 payload 与 SQL 事实同源。"""
    roles_by_document: dict[str, set[str]] = {}
    for spec in specs:
        roles_by_document.setdefault(str(getattr(spec, "document_id")), set()).update(
            str(role) for role in getattr(spec, "allowed_roles")
        )
    with database.sessions.begin() as session:
        for document_id, roles in roles_by_document.items():
            for role in sorted(roles):
                session.add(
                    ACLGrantRow(
                        grant_id=uuid4().hex,
                        tenant_id=tenant_id,
                        resource_type="document",
                        resource_id=document_id,
                        subject_type="role",
                        subject_id=role,
                        permissions=[Permission.KNOWLEDGE_READ.value],
                    )
                )


def _run_queries(
    retriever: EnterpriseRetriever,
    cases: Iterable[AcceptanceCase],
    role_auths: Mapping[str, AuthContext],
) -> dict[str, CaseObservation]:
    observations: dict[str, CaseObservation] = {}
    for case in cases:
        started = time.perf_counter()
        result = retriever.search(case.question, role_auths[str(case.role)])
        observations[case.case_id] = CaseObservation(
            case_id=case.case_id,
            answerable=case.answerable,
            candidates=_candidate_bindings(result),
            latency_ms=(time.perf_counter() - started) * 1000,
            sufficient=result.sufficient,
            insufficiency_code=result.insufficiency_code,
        )
    return observations


def _delete_random_tenant(database, tenant_id: str) -> None:
    """按外键依赖逆序删除本次随机租户的全部事实，不触碰其他租户。"""
    with database.sessions.begin() as session:
        for table in reversed(Base.metadata.sorted_tables):
            if "tenant_id" in table.c:
                session.execute(delete(table).where(table.c.tenant_id == tenant_id))


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def run_benchmark(
    *,
    dataset_path: Path,
    schema_path: Path,
    corpus_dir: Path,
    output_path: Path,
    old_baseline_path: Path,
    qdrant_url: str,
    database_url: str,
    collection_prefix: str = "enterprise_retrieval_stage3",
    chunk_max_chars: int = 800,
) -> tuple[dict[str, Any], int]:
    """执行隔离真实链路；调用方负责决定是否在本机花费模型推理时间。"""
    if not database_url:
        raise ValueError("rag_kb_database_url_required")
    if not collection_prefix or chunk_max_chars <= 0:
        raise ValueError("invalid_benchmark_configuration")

    dataset = load_acceptance_dataset(dataset_path, schema_path)
    specs = load_manifest(corpus_dir)
    candidate_k = max(20, dataset.required_top_k)
    run_id = uuid4().hex
    tenant_id = f"enterprise-retrieval-{run_id}"
    collection = f"{collection_prefix}_{run_id}"
    object_root = Path(tempfile.mkdtemp(prefix="rag-kb-enterprise-retrieval-"))
    report: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256_file(dataset_path),
            "name": dataset.name,
            "version": dataset.version,
            "case_count": len(dataset.cases),
        },
        "corpus": {"manifest": dataset.corpus_manifest, "sha256": _corpus_digest(corpus_dir, specs)},
        "git_baseline_hash": _git_baseline_hash(Path.cwd()),
        "old_baseline_reference": _read_old_baseline(old_baseline_path),
        "configuration": {
            "database_source": "RAG_KB_DATABASE_URL",
            "qdrant_source": "argument_or_RAG_KB_QDRANT_URL",
            "collection": collection,
            "embedding_model": "BAAI/bge-m3",
            "calibration_hybrid_candidates": [False, True],
            "dense_threshold_candidates": list(_DENSE_THRESHOLD_CANDIDATES),
            "candidate_k": candidate_k,
            "context_k": dataset.required_top_k,
            "chunk_max_chars": chunk_max_chars,
        },
        "metric_contract": "gold matches document_id+document_version; not comparable as legacy chunk-level identity",
        "cleanup": {"collection": False, "tenant": False, "object_root": False},
    }
    database = None
    client = None
    tenant_created = False
    phase = "connect_components"
    try:
        from qdrant_client import QdrantClient

        database = build_database(database_url)
        if database.engine.dialect.name != "postgresql":
            raise RuntimeError("enterprise_benchmark_requires_postgresql")
        client = QdrantClient(url=qdrant_url, timeout=30)
        phase = "create_tenant_principals"
        writer_auth, role_auths = _create_tenant_principals(database, tenant_id=tenant_id, run_id=run_id)
        tenant_created = True
        service = DocumentService(database, LocalObjectStore(object_root / "objects"))
        seen_documents: set[str] = set()
        ingested: list[dict[str, Any]] = []
        phase = "register_uploads"
        for spec in specs:
            document_id = str(spec.document_id)
            upload = service.register_upload(
                document_id=document_id,
                title=str(spec.title),
                department=str(spec.department),
                source_file_name=Path(str(spec.file)).name,
                content=(corpus_dir / str(spec.file)).read_bytes(),
                idempotency_key=f"benchmark-upload-{run_id}-{len(ingested)}",
                auth=writer_auth,
                expect_existing=document_id in seen_documents,
            )
            seen_documents.add(document_id)
            ingested.append(upload.model_dump())

        phase = "write_acl_grants"
        _write_manifest_acl_grants(database, tenant_id=tenant_id, specs=specs)
        # BGE-M3 实例在这里才创建；--help 与纯测试不会加载模型。
        phase = "load_embedding_model"
        embedder = BgeM3Embedder()
        indexer = QdrantDocumentIndex(
            client,
            embedder,
            DatabaseDocumentIndexMetadataPort(database),
            collection=collection,
        )
        worker = IngestionWorker(
            DocumentRegistryIngestionJobPort(database),
            LocalDocumentParser(LocalObjectStore(object_root / "objects"), max_chars=chunk_max_chars),
            indexer,
            worker_id=f"enterprise-retrieval-worker-{run_id}",
            lease_seconds=300,
        )
        phase = "ingest_documents"
        worker_outcomes = [asdict(worker.run_once()) for _ in ingested]
        if any(outcome["status"] != "succeeded" for outcome in worker_outcomes):
            report["ingestion"] = {"uploads": ingested, "worker_outcomes": worker_outcomes}
            raise RuntimeError("enterprise_benchmark_ingestion_failed")
        authorization = DatabaseAuthorizationPort(database)
        vectors = QdrantAuthorizedSearch(client, embedder, collection=collection)
        calibration_cases = tuple(case for case in dataset.cases if case.split == "calibration")
        test_cases = tuple(case for case in dataset.cases if case.split == "test")
        pipeline_reports: dict[str, dict[str, Any]] = {}
        calibration_configurations: list[dict[str, Any]] = []
        phase = "calibrate_retrieval"
        for hybrid_enabled, pipeline_name in ((False, "dense_only"), (True, "hybrid")):
            evaluations: list[dict[str, Any]] = []
            for threshold in _DENSE_THRESHOLD_CANDIDATES:
                # 每种候选配置都真实经过 production EnterpriseRetriever；不能用离线
                # 裁切模拟 hybrid，因为 dense cutoff 会改变后续 RRF 的候选集合和顺序。
                calibration_retriever = EnterpriseRetriever(
                    authorization,
                    vectors,
                    candidate_k=candidate_k,
                    context_k=dataset.required_top_k,
                    hybrid_enabled=hybrid_enabled,
                    dense_score_threshold=threshold,
                )
                observations = _run_queries(calibration_retriever, calibration_cases, role_auths)
                evaluation = _evaluate_document_version_cases(calibration_cases, observations)
                configuration = {
                    "pipeline": pipeline_name,
                    "hybrid_enabled": hybrid_enabled,
                    "dense_score_threshold": threshold,
                    "evaluation": evaluation,
                    "observations": observations,
                }
                calibration_configurations.append(configuration)
                evaluations.append(
                    {
                        "dense_score_threshold": threshold,
                        "metrics": evaluation["metrics"],
                        "cases": evaluation["cases"],
                    }
                )
            pipeline_reports[pipeline_name] = {
                "hybrid_enabled": hybrid_enabled,
                "threshold_evaluations": evaluations,
            }
        selected = _select_pipeline_configuration(calibration_configurations)

        # 选参冻结后，test 只执行一次真实生产检索，不把 test 标签反馈给阈值选择。
        frozen_retriever = EnterpriseRetriever(
            authorization,
            vectors,
            candidate_k=candidate_k,
            context_k=dataset.required_top_k,
            hybrid_enabled=selected["hybrid_enabled"],
            dense_score_threshold=selected["dense_score_threshold"],
        )
        phase = "evaluate_frozen_test_split"
        test_observations = _run_queries(frozen_retriever, test_cases, role_auths)
        selected_calibration = selected["evaluation"]
        evaluated_cases = (*calibration_cases, *test_cases)
        all_observations = selected["observations"] | test_observations
        report.update(
            {
                "status": "completed",
                "ingestion": {"uploads": ingested, "worker_outcomes": worker_outcomes},
                "threshold_calibration": {
                    "selection_split": "calibration",
                    "test_labels_used_for_selection": False,
                    "selection_rule": "ACL leak count=0, no-answer accuracy, Recall@5, MRR; exact ties prefer dense-only then lower threshold",
                    "selected_pipeline": selected["pipeline"],
                    "selected_hybrid_enabled": selected["hybrid_enabled"],
                    "selected_dense_score_threshold": selected["dense_score_threshold"],
                    "pipeline_configurations": pipeline_reports,
                },
                "calibration": selected_calibration,
                "test": _evaluate_document_version_cases(test_cases, test_observations),
                "development": {
                    "status": "not_evaluated",
                    "case_count": sum(case.split == "development" for case in dataset.cases),
                    "reason": "development split is excluded from calibration and holdout metrics",
                },
                "all_cases": _evaluate_document_version_cases(
                    evaluated_cases, all_observations
                ),
            }
        )
        return report, 0
    except Exception as exc:
        original = getattr(exc, "orig", None)
        diagnostic = getattr(original, "diag", None)
        report.update(
            {
                "status": "run_failed",
                "error_code": type(exc).__name__,
                "failure_phase": phase,
                "database_error_code": getattr(original, "sqlstate", None),
                "database_constraint": getattr(diagnostic, "constraint_name", None),
            }
        )
        return report, 3
    finally:
        if client is not None:
            try:
                if client.collection_exists(collection):
                    client.delete_collection(collection)
                report["cleanup"]["collection"] = True
            except Exception:
                report["cleanup"]["collection_error_code"] = "collection_cleanup_failed"
        if database is not None and tenant_created:
            try:
                _delete_random_tenant(database, tenant_id)
                report["cleanup"]["tenant"] = True
            except Exception:
                report["cleanup"]["tenant_error_code"] = "tenant_cleanup_failed"
        try:
            shutil.rmtree(object_root)
            report["cleanup"]["object_root"] = True
        except OSError:
            report["cleanup"]["object_root_error_code"] = "object_cleanup_failed"
        if database is not None:
            database.engine.dispose()
        report["completed_at"] = datetime.now(UTC).isoformat()
        _write_report(output_path, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/benchmarks/retrieval_acceptance_v2.yaml"),
    )
    parser.add_argument(
        "--schema", type=Path, default=Path("schemas/retrieval-acceptance-v2.schema.json")
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/local/benchmarks/enterprise-retrieval-stage3.json")
    )
    parser.add_argument(
        "--old-baseline",
        type=Path,
        default=Path("data/local/benchmarks/retrieval-acceptance-baseline-20260822.json"),
    )
    parser.add_argument("--qdrant-url", default=os.environ.get("RAG_KB_QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection-prefix", default="enterprise_retrieval_stage3")
    parser.add_argument("--chunk-max-chars", type=int, default=800)
    args = parser.parse_args(argv)
    database_url = os.environ.get("RAG_KB_DATABASE_URL", "")
    if not database_url:
        parser.error("必须通过 RAG_KB_DATABASE_URL 提供 PostgreSQL 连接；脚本不会打印该值")
    report, code = run_benchmark(
        dataset_path=args.dataset,
        schema_path=args.schema,
        corpus_dir=args.corpus,
        output_path=args.output,
        old_baseline_path=args.old_baseline,
        qdrant_url=args.qdrant_url,
        database_url=database_url,
        collection_prefix=args.collection_prefix,
        chunk_max_chars=args.chunk_max_chars,
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
