"""领域模型:对应蓝图 6.1 的 payload 字段、6.2 的 ACL 角色与 6.3 的生命周期状态。"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

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
        return self.model_dump(mode="json")


class Citation(BaseModel):
    """蓝图 7.3 的引用格式:每条引用必须指向实际检索到的、已授权且 active 的块。"""

    document_id: str
    document_version: int
    source_file_name: str
    source_locator: str
    chunk_id: str


class Refusal(BaseModel):
    """标准化拒答(蓝图 7.4):kind 为机器可读原因码,message 为对外统一文案。"""

    kind: str
    message: str


class QuestionResponse(BaseModel):
    """POST /questions 的响应契约。不回显问题原文,避免响应意外携带敏感查询上下文。"""

    verdict: Literal["answer", "refusal"]
    answer: str | None = None
    refusal: Refusal | None = None
    citations: list[Citation] = []
    role: Role
