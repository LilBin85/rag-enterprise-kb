"""领域模型:对应蓝图 6.1 的 payload 字段、6.2 的 ACL 角色与 6.3 的生命周期状态。"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, StrictInt


class Role(StrEnum):
    """蓝图 2.1 的四个演示角色,用于 ACL 权限判定。"""

    EMPLOYEE = "employee"
    ENGINEER = "engineer"
    HR = "hr"
    ADMIN = "admin"


class ChunkStatus(StrEnum):
    """蓝图 6.3 的文档块生命周期状态,控制块在检索中的可见性。"""

    ACTIVE = "active"          # 当前有效,可被检索
    SUPERSEDED = "superseded"  # 已被新版本替代
    WITHDRAWN = "withdrawn"    # 主动撤回,不再对外
    DELETED = "deleted"        # 已删除,仅保留审计痕迹


class DocumentSpec(BaseModel):
    """manifest.yaml 中一份文档(单个版本)的注册信息,是摄取的唯一事实来源。"""

    document_id: str          # 文档唯一标识
    file: str                 # 文档文件路径(相对于 manifest)
    title: str                # 文档标题
    department: str           # 所属部门
    allowed_roles: list[Role] # 允许访问的角色列表
    version: int              # 文档版本号(从 1 递增)
    note: str | None = None   # 版本说明(可选)


class ChunkRecord(BaseModel):
    """一个文档块写入 Qdrant 前的完整记录:蓝图 6.1 全部字段 + 正文文本。

    向量本身不能脱离版本、来源与权限单独存在(蓝图 6.1);正文存入 payload 的
    text 字段,供检索后组装授权上下文与引用。
    """

    document_id: str        # 所属文档标识
    document_version: int   # 所属文档版本号
    chunk_id: str           # 块唯一标识(跨版本不变)
    source_file_name: str   # 来源文件名
    source_locator: str     # 来源定位符(如页码、段落号),用于精确引用
    title: str              # 文档标题
    department: str         # 所属部门
    allowed_roles: list[str] # 授权角色列表(序列化为字符串数组)
    status: ChunkStatus     # 生命周期状态
    content_hash: str       # 正文内容的哈希值,用于变更检测
    text: str               # 正文文本(存入 Qdrant payload)
    created_at: datetime    # 创建时间
    updated_at: datetime    # 最后更新时间

    def to_payload(self) -> dict:
        """序列化为 Qdrant payload:枚举转值、时间转 ISO 8601,确保 JSON 兼容。"""
        return self.model_dump(mode="json")


class Citation(BaseModel):
    """蓝图 7.3 的引用格式:每条引用必须指向实际检索到的、已授权且 active 的块。"""

    document_id: str        # 来源文档标识
    document_version: int   # 来源文档版本号
    source_file_name: str   # 来源文件名
    source_locator: str     # 来源定位符,指向文档内具体位置
    chunk_id: str           # 引用块的唯一标识


class Refusal(BaseModel):
    """标准化拒答模型(蓝图 7.4):kind 为机器可读原因码,message 为对外统一文案。"""

    kind: str     # 拒答原因码(如 "unauthorized"、"out_of_scope")
    message: str  # 面向用户的拒答文案


class QuestionResponse(BaseModel):
    """POST /questions 的响应契约。不回显问题原文,避免响应意外携带敏感查询上下文。"""

    verdict: Literal["answer", "refusal"]  # 判定结果:正常回答 / 拒答
    answer: str | None = None              # 回答正文(verdict=answer 时非空)
    refusal: Refusal | None = None         # 拒答信息(verdict=refusal 时非空)
    citations: list[Citation] = []         # 引用来源列表
    role: Role                             # 请求用户的角色身份


class PolicyComparisonRequest(BaseModel):
    """策略版本比较请求;版本必须由服务端按原样绑定,不接受 latest 等别名。"""

    document_id: str
    from_version: StrictInt
    to_version: StrictInt
    role: Role


class PolicyChange(BaseModel):
    """应用层确定性 diff 的单个章节变化。"""

    section: str
    change: Literal["added", "modified", "removed"]
    from_text: str | None = None
    to_text: str | None = None


class PolicyComparison(BaseModel):
    """策略比较结果;文本来自已授权证据,不是模型自由生成的引用字段。"""

    document_id: str
    from_version: int
    to_version: int
    changes: list[PolicyChange]
    summary: str
    citations: list[Citation]


class PolicyComparisonResponse(BaseModel):
    """POST /policy-comparisons 的稳定响应;拒答不回显文档或版本标识。"""

    verdict: Literal["comparison", "refusal"]
    comparison: PolicyComparison | None = None
    refusal: Refusal | None = None
    role: Role
    run_id: str
