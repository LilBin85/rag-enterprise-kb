"""应用配置:所有环境相关取值集中于此,密钥只从环境变量 / .env 读取(蓝图 11.1)。"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_kb.domain.tasks import (
    DEFAULT_MAX_MODEL_CALLS,
    DEFAULT_MAX_RUNTIME_SECONDS,
    DEFAULT_MAX_TOOL_CALLS,
    TaskBudgets,
)
from rag_kb.domain.tools import TOOL_TIMEOUT_CEILING_SECONDS

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # DeepSeek(阶段 4 才消费;此处只负责读取与保管)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # 本地策略比较接口的演示身份令牌;不等同于生产认证系统
    demo_role_tokens: dict[str, str] = {}
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "kb_chunks"
    # 新版企业投影与旧 role-only collection 物理隔离，迁移评测通过前不切别名。
    qdrant_v2_collection: str = "kb_chunks_v2"

    # PostgreSQL 是业务与任务状态的生产事实源。SQLite URL 只用于本地轻量测试；
    # schema 创建和升级必须由 Alembic 显式执行，普通请求不得建表或 ALTER。
    database_url: str = "postgresql+psycopg://rag_kb:rag_kb@localhost:5432/rag_kb"

    # HuggingFace 下载端点(国内镜像;可直连时留空)
    hf_endpoint: str | None = None

    # HF 缓存根目录(模型冻结位置)。.env 兜底:用户级 HF_HOME 变量在旧终端会话里
    # 可能读不到,导致缓存检测误判、误走联网探测(2026-08-08 实测)。
    hf_home: Path | None = None

    # 摄取限制(蓝图 8 节上传校验)
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_extensions: tuple[str, ...] = (".md", ".pdf")

    # 切块上限(按字符数;结构化边界优先,超长章节再细分)
    chunk_max_chars: int = 800

    # 检索(蓝图 7.1):retrieval_top_k 是兼容旧配置的共同默认值。灰度时可分别
    # 覆盖候选池和最终上下文上限；未设置新变量时行为仍严格等同于旧版 Top-K=5。
    retrieval_top_k: int = Field(default=5, ge=1, le=100)
    retrieval_candidate_k: int | None = Field(default=None, ge=1, le=100)
    retrieval_context_k: int | None = Field(default=None, ge=1, le=100)
    retrieval_score_threshold: float | None = Field(default=None, allow_inf_nan=False)
    retrieval_context_token_budget: int = Field(default=4000, ge=1, le=32000)
    retrieval_rrf_k: int = Field(default=60, ge=1, le=1000)
    enable_hybrid_retrieval: bool = False

    # M1 Research Agent：Settings 只决定新任务的预算默认值。
    agent_max_model_calls: int = Field(
        default=DEFAULT_MAX_MODEL_CALLS, ge=1, le=100
    )
    agent_max_tool_calls: int = Field(default=DEFAULT_MAX_TOOL_CALLS, ge=1, le=100)
    agent_max_runtime_seconds: int = Field(default=DEFAULT_MAX_RUNTIME_SECONDS, ge=1)
    agent_timeout_seconds: int = Field(default=20, ge=1, le=120)
    agent_capability_lease_seconds: int = Field(default=30, ge=3, le=3600)

    # 敏感正文保留策略（M1_FIX_PLAN 修复 5）：工具结果与任务结果/checkpoint 过期即清理。
    tool_result_ttl_seconds: int = Field(default=7 * 24 * 3600, ge=1)
    task_result_retention_days: int = Field(default=7, ge=1)
    result_object_orphan_grace_seconds: int = Field(default=3600, ge=1)

    # 规划复用检索 tool 上限；模型决策和结构修复另行有界，避免隐式循环。
    planner_max_model_decisions: int = Field(default=6, ge=1, le=10)
    planner_max_repairs: int = Field(default=2, ge=0, le=5)
    # 已点名制度时先由服务端做一次 ACL 受权检索；空集直接拒绝，避免模型把
    # 权限/证据问题误判成需要用户澄清。可在测试或灰度环境显式关闭。
    planner_preflight_named_policy_search: bool = True

    # 策略落地规划的本地恢复状态；SQLite 不承载知识证据正文。
    policy_rollout_db_path: Path = ROOT_DIR / "data" / "local" / "policy_rollout_tasks.sqlite3"
    policy_rollout_draft_dir: Path = ROOT_DIR / "data" / "local" / "policy_rollout_drafts"
    policy_rollout_claim_ttl_seconds: int = Field(default=300, ge=1, le=3600)

    # 服务端生成对象键；请求和模型均不得指定任意文件路径。
    object_store_root: Path = ROOT_DIR / "data" / "local" / "objects"

    # bge-m3 dense 向量维度(工作单元 3 使用)
    embedding_dim: int = 1024

    corpus_dir: Path = ROOT_DIR / "data" / "corpus"

    @model_validator(mode="after")
    def validate_retention_windows(self) -> "Settings":
        if self.result_object_orphan_grace_seconds <= TOOL_TIMEOUT_CEILING_SECONDS + 60:
            raise ValueError("result_object_orphan_grace_too_short")
        if self.tool_result_ttl_seconds <= TOOL_TIMEOUT_CEILING_SECONDS:
            raise ValueError("tool_result_ttl_too_short")
        return self

    def prepare_runtime_env(self) -> None:
        """运行时环境准备;必须在相关三方库被 import 之前调用(其常量于导入时读取)。

        - HF_ENDPOINT 指向国内镜像(huggingface.co 直连超时,2026-08-07 实测);
        - HF_HUB_DISABLE_XET=1:镜像下 xet 协议会回跳官方 CAS 服务器报 401,
          禁用后回退普通 HTTP 下载,与镜像兼容(同日实测);
        - TORCH_COMPILE_DISABLE=1:本机无 MSVC(cl.exe),Docling/FlagEmbedding
          部分模型默认启用 torch.compile,须显式禁用(同日实测 InvalidCxxCompiler);
        - HF_HUB_OFFLINE=1(仅当 bge-m3 已完整缓存):transformers 5.x 加载 tokenizer
          时会向 HF 发起 list_repo_templates 远程探测,即使模型已在本地——该探测在
          镜像抖动时会以 ConnectTimeout 拖垮知识查询(2026-08-08 实测)。
          模型冻结后运行期不应依赖外网;缓存不完整时不设此值,允许走镜像下载。
        """
        if self.hf_home:
            os.environ.setdefault("HF_HOME", str(self.hf_home))
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        if self._bge_m3_fully_cached():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def _bge_m3_fully_cached(self) -> bool:
        """检测 bge-m3 主权重是否已落地本地 HF 缓存。"""
        hf_home = self.hf_home or Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        snapshots = hf_home / "hub" / "models--BAAI--bge-m3" / "snapshots"
        return snapshots.exists() and any(
            (s / "pytorch_model.bin").is_file() for s in snapshots.iterdir() if s.is_dir()
        )


def task_budgets_from_settings(settings: Settings) -> TaskBudgets:
    """把当前环境默认值完整持久化到新任务，运行中的任务不再读取 Settings。"""
    return TaskBudgets(
        max_model_calls=settings.agent_max_model_calls,
        max_tool_calls=settings.agent_max_tool_calls,
        max_runtime_seconds=settings.agent_max_runtime_seconds,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_retrieval_limits(settings: object) -> tuple[int, int]:
    """解析兼容的新旧检索 K 配置，并在服务构建前拒绝歧义配置。

    ``RETRIEVAL_TOP_K`` 继续作为两者的默认值；只有显式提供
    ``RETRIEVAL_CANDIDATE_K`` / ``RETRIEVAL_CONTEXT_K`` 时才覆盖。候选池小于
    最终上下文没有明确语义，因此失败关闭，而不是静默放大候选池。
    """
    legacy_top_k = getattr(settings, "retrieval_top_k", 5)
    candidate_value = getattr(settings, "retrieval_candidate_k", None)
    context_value = getattr(settings, "retrieval_context_k", None)
    candidate_k = legacy_top_k if candidate_value is None else candidate_value
    context_k = legacy_top_k if context_value is None else context_value
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (legacy_top_k, candidate_k, context_k)
    ):
        raise ValueError("retrieval_limits_must_be_integers")
    if candidate_k <= 0 or context_k <= 0:
        raise ValueError("retrieval_limits_must_be_positive")
    if context_k > candidate_k:
        raise ValueError("retrieval_context_k_cannot_exceed_candidate_k")
    return candidate_k, context_k
