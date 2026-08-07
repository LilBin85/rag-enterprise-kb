"""FastAPI 接口层(蓝图 8 节契约)。

设计要点:
- 权限元数据(allowed_roles)由管理员在上传时显式指定,不让模型或系统猜测(蓝图 6.2);
- 更新(PUT)沿用原文档的权限元数据,只接收新文件;
- 生产 lifecycle 经 Depends 注入,测试用 dependency_overrides 替换为假 embedder 版本。
"""

import json
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
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
from rag_kb.models import DocumentSpec, QuestionResponse, Role
from rag_kb.retriever import Retriever
from rag_kb.store import ChunkStore

app = FastAPI(title="RAG 企业知识库智能体", version="0.1.0")


def _build_store() -> ChunkStore:
    settings = get_settings()
    store = ChunkStore(settings.qdrant_url, settings.qdrant_collection)
    store.ensure_collection(settings.embedding_dim)
    return store


class DocumentMetadata(BaseModel):
    """POST /documents 随文件提交的元数据表单(JSON 字符串)。"""

    document_id: str
    title: str
    department: str
    allowed_roles: list[Role]


_lifecycle_singleton: DocumentLifecycle | None = None


def build_lifecycle() -> DocumentLifecycle:
    """生产 lifecycle 单例;embedder 懒加载,不会在健康检查时触发模型下载。"""
    global _lifecycle_singleton
    if _lifecycle_singleton is None:
        store = _build_store()
        _lifecycle_singleton = DocumentLifecycle(
            store=store,
            embedder=BgeM3Embedder(),
            upload_dir=ROOT_DIR / "data" / "local" / "uploads",
            max_chars=settings.chunk_max_chars,
        )
    return _lifecycle_singleton


LifecycleDep = Annotated[DocumentLifecycle, Depends(build_lifecycle)]


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    try:
        QdrantClient(url=settings.qdrant_url, timeout=3).get_collections()
        qdrant_status = "ok"
    except Exception:
        qdrant_status = "unreachable"
    return {"status": "ok", "qdrant": qdrant_status}


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
    saved = lifecycle.save_upload(meta.document_id, filename, content)
    try:
        return lifecycle.add_document(spec, saved)
    except DocumentConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
    saved = lifecycle.save_upload(document_id, filename, content)
    try:
        return lifecycle.update_document(document_id, spec, saved)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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


_answer_service_singleton: AnswerService | None = None


def build_answer_service() -> AnswerService:
    """生产问答服务单例;模型懒加载,不会在健康检查或非问答路径触发。"""
    global _answer_service_singleton
    if _answer_service_singleton is None:
        store = _build_store()
        retriever = Retriever(
            store,
            BgeM3Embedder(),
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        )
        _answer_service_singleton = AnswerService(retriever, DeepSeekAnswerModel(settings))
    return _answer_service_singleton


@app.post("/questions", response_model=QuestionResponse)
def ask_question(
    req: QuestionRequest,
    service: Annotated[AnswerService, Depends(build_answer_service)],
) -> QuestionResponse:
    """提问并检索:返回回答或标准化拒答及引用(蓝图 8 / 7.3 / 7.4)。"""
    return service.ask(req.question, req.role)
