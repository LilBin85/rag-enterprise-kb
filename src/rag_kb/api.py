"""FastAPI 接口层(蓝图 8 节契约)。

设计要点:
- 权限元数据(allowed_roles)由管理员在上传时显式指定,不让模型或系统猜测(蓝图 6.2);
- 更新(PUT)沿用原文档的权限元数据,只接收新文件;
- 生产 lifecycle 经 Depends 注入,测试用 dependency_overrides 替换为假 embedder 版本。
"""

import json
import logging
import secrets
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient

from rag_kb.config import ROOT_DIR, get_settings
from rag_kb.answerer import AnswerService, DeepSeekAnswerModel
from rag_kb.embedder import BgeM3Embedder
from rag_kb.ingestion import UploadValidationError, validate_upload
from rag_kb.lifecycle import (
    DocumentConflictError,
    DocumentLifecycle,
    DocumentNotFoundError,
)
from rag_kb.models import (
    DocumentSpec,
    PolicyComparisonRequest,
    PolicyComparisonResponse,
    PolicyRolloutTaskContinueRequest,
    PolicyRolloutTaskCreateRequest,
    PolicyRolloutTaskResponse,
    QuestionResponse,
    Role,
)
from rag_kb.policy_compare import DeepSeekPolicyAgentModel, PolicyComparisonService
from rag_kb.policy_rollout_planning import PolicyRolloutPlanningService
from rag_kb.policy_rollout_store import PolicyRolloutTaskStore
from rag_kb.retriever import Retriever
from rag_kb.store import ChunkStore

app = FastAPI(title="RAG 企业知识库智能体", version="0.1.0")
logger = logging.getLogger(__name__)


def _build_store() -> ChunkStore:
    settings = get_settings()
    store = ChunkStore(settings.qdrant_url, settings.qdrant_collection)
    store.ensure_collection(settings.embedding_dim)
    return store


def _build_read_store() -> ChunkStore:
    """只读工作流专用：绝不创建 collection 或 payload index。"""
    settings = get_settings()
    return ChunkStore(settings.qdrant_url, settings.qdrant_collection)


class DocumentMetadata(BaseModel):
    """POST /documents 随文件提交的元数据表单(JSON 字符串)。"""

    document_id: str
    title: str
    department: str
    allowed_roles: list[Role]


@lru_cache
def build_lifecycle() -> DocumentLifecycle:
    """生产 lifecycle 单例;embedder 懒加载,不会在健康检查时触发模型下载。"""
    settings = get_settings()
    return DocumentLifecycle(
        store=_build_store(),
        embedder=BgeM3Embedder(),
        upload_dir=ROOT_DIR / "data" / "local" / "uploads",
        max_chars=settings.chunk_max_chars,
    )


LifecycleDep = Annotated[DocumentLifecycle, Depends(build_lifecycle)]


@app.get("/health", response_model=None)
def health() -> dict | JSONResponse:
    settings = get_settings()
    try:
        QdrantClient(url=settings.qdrant_url, timeout=3).get_collections()
        return {"status": "ok", "qdrant": "ok"}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable", "qdrant": "unreachable"})


@app.post("/documents")
async def create_document(
    lifecycle: LifecycleDep,
    file: Annotated[UploadFile, File()],
    metadata: Annotated[str, Form()],
) -> dict:
    """上传新文档(蓝图 8:仅 admin;首版角色身份为固定演示身份,见 8 节末注)。"""
    try:
        meta = DocumentMetadata.model_validate(json.loads(metadata))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"metadata 非法: {exc}") from exc

    filename = file.filename or ""
    content = await file.read()
    try:
        validate_upload(filename, len(content))
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    spec = DocumentSpec(**meta.model_dump(), file=filename, version=1)
    saved = None
    try:
        saved = lifecycle.save_upload(meta.document_id, filename, content)
        return lifecycle.add_document(spec, saved)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DocumentConflictError as exc:
        if saved is not None:
            saved.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        if saved is not None:
            saved.unlink(missing_ok=True)
        raise


@app.put("/documents/{document_id}")
async def update_document(document_id: str, lifecycle: LifecycleDep, file: Annotated[UploadFile, File()]) -> dict:
    """上传新版本:版本号由服务端计算,权限元数据沿用原文档。"""
    filename = file.filename or ""
    content = await file.read()
    try:
        validate_upload(filename, len(content))
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    existing = lifecycle.store.scroll_document(document_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"文档不存在: {document_id}")
    first = existing[0]
    spec = DocumentSpec(
        document_id=document_id,
        file=filename,
        title=first["title"],
        department=first["department"],
        allowed_roles=first["allowed_roles"],
        version=0,  # 占位,由 update_document 按当前最高版本 +1 覆写
    )
    saved = None
    try:
        saved = lifecycle.save_upload(document_id, filename, content)
        return lifecycle.update_document(document_id, spec, saved)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        if saved is not None:
            saved.unlink(missing_ok=True)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        if saved is not None:
            saved.unlink(missing_ok=True)
        raise


@app.delete("/documents/{document_id}")
def remove_document(document_id: str, lifecycle: LifecycleDep, mode: str = "delete") -> dict:
    """删除或撤回文档(蓝图 6.3:后续默认检索不得返回相关块)。"""
    if not lifecycle.versions(document_id):
        raise HTTPException(status_code=404, detail=f"文档不存在: {document_id}")
    if mode == "delete":
        lifecycle.delete(document_id)
    elif mode == "withdraw":
        lifecycle.withdraw(document_id)
    else:
        raise HTTPException(status_code=422, detail="mode 仅支持 delete / withdraw")
    return {"document_id": document_id, "mode": mode}


@app.get("/documents/{document_id}/versions")
def get_versions(document_id: str, lifecycle: LifecycleDep) -> dict:
    """查看版本状态(蓝图 8:仅用于调试与演示版本生命周期)。"""
    return {"document_id": document_id, "versions": lifecycle.versions(document_id)}


# ---------- 问答(蓝图 8:POST /questions) ----------

class QuestionRequest(BaseModel):
    """提问请求;首版角色身份为显式传入的固定演示身份,不是真实企业认证(蓝图 8 末注)。"""

    question: str
    role: Role


@lru_cache
def build_answer_service() -> AnswerService:
    """生产问答服务单例;模型懒加载,不会在健康检查或非问答路径触发。"""
    settings = get_settings()
    retriever = Retriever(
        _build_store(),
        BgeM3Embedder(),
        top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
    )
    return AnswerService(retriever, DeepSeekAnswerModel(settings))


@app.post("/questions", response_model=QuestionResponse)
def ask_question(
    req: QuestionRequest,
    service: Annotated[AnswerService, Depends(build_answer_service)],
) -> QuestionResponse:
    """提问并检索:返回回答或标准化拒答及引用(蓝图 8 / 7.3 / 7.4)。"""
    return service.ask(req.question, req.role)


@lru_cache
def build_policy_comparison_service() -> PolicyComparisonService:
    """生产比较服务;ChatDeepSeek 仅在首次调用新端点时懒加载。"""
    settings = get_settings()
    try:
        model = DeepSeekPolicyAgentModel(settings)
    except Exception:
        # 缺少可选运行时或 API key 时由服务统一失败关闭,不影响 /questions 导入。
        logger.warning(
            "策略比较 Agent 初始化失败(error_code=agent_runtime_unavailable)",
            exc_info=True,
        )
        model = None
    return PolicyComparisonService(
        _build_store(),
        model,
        max_steps=settings.agent_max_steps,
        max_tool_calls=settings.agent_max_tool_calls,
    )


def resolve_policy_role(
    req: PolicyComparisonRequest,
    demo_role_token: Annotated[str | None, Header(alias="X-Demo-Role-Token")] = None,
) -> Role:
    """用服务端配置的演示令牌确认请求角色,不把 body role 当作可信认证。"""
    expected = get_settings().demo_role_tokens.get(str(req.role))
    if not demo_role_token or not expected or not secrets.compare_digest(demo_role_token, expected):
        raise HTTPException(status_code=403, detail="身份验证失败")
    return req.role


@app.post("/policy-comparisons", response_model=PolicyComparisonResponse)
def compare_policy_versions(
    req: PolicyComparisonRequest,
    role: Annotated[Role, Depends(resolve_policy_role)],
    service: Annotated[PolicyComparisonService, Depends(build_policy_comparison_service)],
) -> PolicyComparisonResponse:
    """在服务端 ACL 下比较两个明确请求的策略版本。"""
    return service.compare(req.document_id, req.from_version, req.to_version, role)


# ---------- 策略落地规划（v1：持久化只读规划） ----------

@lru_cache
def build_policy_rollout_planning_service() -> PolicyRolloutPlanningService:
    settings = get_settings()
    try:
        model = DeepSeekPolicyAgentModel(settings)
    except Exception:
        logger.warning("策略落地规划 Agent 初始化失败(error_code=agent_runtime_unavailable)", exc_info=True)
        model = None
    store = _build_read_store()
    retriever = Retriever(
        store, BgeM3Embedder(), top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
    )
    return PolicyRolloutPlanningService(
        PolicyRolloutTaskStore(settings.policy_rollout_db_path), retriever, model,
        citation_checker=store.citations_currently_authorized,
        max_steps=settings.agent_max_steps, max_tool_calls=settings.agent_max_tool_calls,
    )


def resolve_policy_rollout_role(
    demo_role_token: Annotated[str | None, Header(alias="X-Demo-Role-Token")] = None,
) -> Role:
    """从令牌解析服务端身份；续作和读取没有可被伪造的 body role。"""
    if demo_role_token:
        for raw_role, expected in get_settings().demo_role_tokens.items():
            if secrets.compare_digest(demo_role_token, expected):
                try:
                    return Role(raw_role)
                except ValueError:
                    continue
    raise HTTPException(status_code=403, detail="身份验证失败")


@app.post("/policy-rollout-tasks", response_model=PolicyRolloutTaskResponse)
def create_policy_rollout_task(
    req: PolicyRolloutTaskCreateRequest,
    role: Annotated[Role, Depends(resolve_policy_rollout_role)],
    service: Annotated[PolicyRolloutPlanningService, Depends(build_policy_rollout_planning_service)],
) -> PolicyRolloutTaskResponse:
    if req.role != role:
        raise HTTPException(status_code=403, detail="身份验证失败")
    return service.create(req.goal, role)


@app.post("/policy-rollout-tasks/{task_id}/continue", response_model=PolicyRolloutTaskResponse)
def continue_policy_rollout_task(
    task_id: str,
    req: PolicyRolloutTaskContinueRequest,
    role: Annotated[Role, Depends(resolve_policy_rollout_role)],
    service: Annotated[PolicyRolloutPlanningService, Depends(build_policy_rollout_planning_service)],
) -> PolicyRolloutTaskResponse:
    result = service.continue_task(task_id, role, req.clarification_answers)
    if result is None:
        raise HTTPException(status_code=403, detail="任务不可访问")
    return result


@app.get("/policy-rollout-tasks/{task_id}", response_model=PolicyRolloutTaskResponse)
def get_policy_rollout_task(
    task_id: str,
    role: Annotated[Role, Depends(resolve_policy_rollout_role)],
    service: Annotated[PolicyRolloutPlanningService, Depends(build_policy_rollout_planning_service)],
) -> PolicyRolloutTaskResponse:
    result = service.get(task_id, role)
    if result is None:
        raise HTTPException(status_code=403, detail="任务不可访问")
    return result
