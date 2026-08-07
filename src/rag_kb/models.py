"""领域模型:对应蓝图 6.1 的 payload 字段、6.2 的 ACL 角色与 6.3 的生命周期状态。"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class Role(StrEnum):
    """蓝图 2.1 的四个演示角色。"""

    EMPLOYEE = "employee"
    ENGINEER = "engineer"
    HR = "hr"
    ADMIN = "admin"


class ChunkStatus(StrEnum):
    """蓝图 6.3 的文档块生命周期状态。"""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"
    DELETED = "deleted"


class DocumentSpec(BaseModel):
    """manifest.yaml 中一份文档(单个版本)的注册信息,是摄取的唯一事实来源。"""

    document_id: str
    file: str
    title: str
    department: str
    allowed_roles: list[Role]
    version: int
    note: str | None = None


class ChunkRecord(BaseModel):
    """一个文档块写入 Qdrant 前的完整记录:蓝图 6.1 全部字段 + 正文文本。

    向量本身不能脱离版本、来源与权限单独存在(蓝图 6.1);正文存入 payload 的
    text 字段,供检索后组装授权上下文与引用。
    """

    document_id: str
    document_version: int
    chunk_id: str
    source_file_name: str
    source_locator: str
    title: str
    department: str
    allowed_roles: list[str]
    status: ChunkStatus
    content_hash: str
    text: str
    created_at: datetime
    updated_at: datetime

    def to_payload(self) -> dict:
        """序列化为 Qdrant payload:JSON 兼容、枚举转值、时间转 ISO 8601。"""
        return {
            "document_id": self.document_id,
            "document_version": self.document_version,
            "chunk_id": self.chunk_id,
            "source_file_name": self.source_file_name,
            "source_locator": self.source_locator,
            "title": self.title,
            "department": self.department,
            "allowed_roles": list(self.allowed_roles),
            "status": str(self.status),
            "content_hash": self.content_hash,
            "text": self.text,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
