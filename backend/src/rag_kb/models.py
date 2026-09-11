"""共享的旧语料基准与当前问答契约。"""

from enum import StrEnum

from pydantic import BaseModel


class Role(StrEnum):
    EMPLOYEE = "employee"
    ENGINEER = "engineer"
    HR = "hr"
    ADMIN = "admin"


class DocumentSpec(BaseModel):
    document_id: str
    file: str
    title: str
    department: str
    allowed_roles: list[Role]
    version: int
    note: str | None = None


class Citation(BaseModel):
    document_id: str
    document_version: int
    source_file_name: str
    source_locator: str
    chunk_id: str
    document_title: str | None = None


class Refusal(BaseModel):
    kind: str
    message: str
