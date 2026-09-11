"""版本化文档注册与摄取任务接口。"""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from rag_kb.application.document_service import (
    DocumentAccessDenied,
    DocumentDeletionFailed,
    DocumentDeletionResult,
    DocumentDeletionUnavailable,
    DocumentListResult,
    DocumentResult,
    DocumentService,
    DocumentUploadResult,
    IngestionJobResult,
)
from rag_kb.application.ports.documents import DocumentRegistryConflict, IdempotencyConflict
from rag_kb.config import get_settings
from rag_kb.domain.identity import AuthContext
from rag_kb.web.dependencies import build_document_service, resolve_auth_context
from rag_kb.web.schemas.documents import DocumentUploadMetadata


router = APIRouter(prefix="/v1", tags=["documents"])

MAX_UPLOAD_METADATA_CHARS = 16 * 1024


def parse_metadata(raw: str) -> DocumentUploadMetadata:
    if len(raw) > MAX_UPLOAD_METADATA_CHARS:
        raise HTTPException(status_code=413, detail="metadata 超出大小上限")
    try:
        return DocumentUploadMetadata.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail="metadata 非法") from exc


async def read_upload_limited(file: UploadFile) -> bytes:
    """有界读取上传内容，避免在业务校验前把任意大小请求完整载入内存。"""
    limit = get_settings().max_upload_bytes
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail="文件超出大小上限")
    return content


async def _register_upload(
    *,
    parsed: DocumentUploadMetadata,
    file: UploadFile,
    idempotency_key: str,
    auth: AuthContext,
    service: DocumentService,
    expect_existing: bool,
) -> DocumentUploadResult:
    content = await read_upload_limited(file)
    try:
        return await run_in_threadpool(
            service.register_upload,
            document_id=parsed.document_id,
            title=parsed.title,
            department=parsed.department,
            source_file_name=file.filename or "",
            content=content,
            idempotency_key=idempotency_key,
            auth=auth,
            expect_existing=expect_existing,
        )
    except DocumentAccessDenied as exc:
        raise HTTPException(status_code=403, detail="无权写入文档") from exc
    except DocumentRegistryConflict as exc:
        detail: dict[str, object] = {
            "code": exc.code,
            "message": "文档已经存在，未重复创建处理任务"
            if exc.code == "document_already_exists"
            else str(exc),
        }
        if exc.recovery is not None:
            detail["recovery"] = {
                "document_id": exc.recovery.document_id,
                "version_id": exc.recovery.version_id,
                "ingestion_job_id": exc.recovery.job_id,
                "lifecycle_status": exc.recovery.lifecycle_status,
                "ingestion_status": exc.recovery.ingestion_status,
            }
        raise HTTPException(status_code=409, detail=detail) from exc
    except IdempotencyConflict as exc:
        code = str(exc)
        message = (
            "同一个 Idempotency-Key 已经绑定了另一份请求；保持原请求重试，"
            "或为新请求生成新的 Key"
            if code == "idempotency_key_reused_with_different_request"
            else code
        )
        raise HTTPException(status_code=409, detail={"code": code, "message": message}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/documents", response_model=DocumentUploadResult, status_code=202)
async def create_document(
    file: Annotated[UploadFile, File()],
    metadata: Annotated[str, Form()],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[DocumentService, Depends(build_document_service)],
) -> DocumentUploadResult:
    parsed = parse_metadata(metadata)
    return await _register_upload(
        parsed=parsed,
        file=file,
        idempotency_key=idempotency_key,
        auth=auth,
        service=service,
        expect_existing=False,
    )


@router.get("/documents", response_model=DocumentListResult)
def list_documents(
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[DocumentService, Depends(build_document_service)],
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = None,
) -> DocumentListResult:
    try:
        return service.list_documents(auth=auth, limit=limit, cursor=cursor)
    except DocumentAccessDenied as exc:
        raise HTTPException(status_code=403, detail="无权读取文档") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="文档分页参数非法") from exc


@router.post(
    "/documents/{document_id}/versions",
    response_model=DocumentUploadResult,
    status_code=202,
)
async def create_document_version(
    document_id: str,
    file: Annotated[UploadFile, File()],
    metadata: Annotated[str, Form()],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[DocumentService, Depends(build_document_service)],
) -> DocumentUploadResult:
    parsed = parse_metadata(metadata)
    if parsed.document_id != document_id:
        raise HTTPException(status_code=422, detail="metadata 文档标识不匹配")
    return await _register_upload(
        parsed=parsed,
        file=file,
        idempotency_key=idempotency_key,
        auth=auth,
        service=service,
        expect_existing=True,
    )


@router.get("/documents/{document_id}", response_model=DocumentResult)
def get_document(
    document_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[DocumentService, Depends(build_document_service)],
) -> DocumentResult:
    try:
        result = service.get_document(document_id, auth)
    except DocumentAccessDenied as exc:
        raise HTTPException(status_code=403, detail="无权读取文档") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return result


@router.delete("/documents/{document_id}", response_model=DocumentDeletionResult)
def delete_document(
    document_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[DocumentService, Depends(build_document_service)],
) -> DocumentDeletionResult:
    try:
        result = service.delete_document(document_id, auth)
    except DocumentAccessDenied as exc:
        raise HTTPException(status_code=403, detail="无权删除文档") from exc
    except DocumentRegistryConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": "文档仍在处理中，暂时不能删除"},
        ) from exc
    except DocumentDeletionUnavailable as exc:
        raise HTTPException(status_code=503, detail="文档向量清理服务不可用") from exc
    except DocumentDeletionFailed as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": str(exc), "message": "外部存储清理未完成，可使用同一接口重试"},
        ) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return result


@router.get("/ingestion-jobs/{job_id}", response_model=IngestionJobResult)
def get_ingestion_job(
    job_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[DocumentService, Depends(build_document_service)],
) -> IngestionJobResult:
    try:
        result = service.get_ingestion_job(job_id, auth)
    except DocumentAccessDenied as exc:
        raise HTTPException(status_code=403, detail="无权读取摄取任务") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="摄取任务不存在")
    return result


@router.get(
    "/documents/{document_id}/ingestion-jobs",
    response_model=list[IngestionJobResult],
)
def list_document_ingestion_jobs(
    document_id: str,
    auth: Annotated[AuthContext, Depends(resolve_auth_context)],
    service: Annotated[DocumentService, Depends(build_document_service)],
) -> list[IngestionJobResult]:
    try:
        result = service.list_ingestion_jobs(document_id, auth)
    except DocumentAccessDenied as exc:
        raise HTTPException(status_code=403, detail="无权读取摄取任务") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return list(result)
