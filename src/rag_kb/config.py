"""应用配置:所有环境相关取值集中于此,密钥只从环境变量 / .env 读取(蓝图 11.1)。"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # DeepSeek(阶段 4 才消费;此处只负责读取与保管)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # 本地策略比较接口的演示身份令牌;不等同于生产认证系统
    demo_role_tokens: dict[str, str] = {}

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "kb_chunks"

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

    # 检索(蓝图 7.1):Top-K 候选数;相似度阈值待冻结评测校准,None 表示不裁切
    retrieval_top_k: int = 5
    retrieval_score_threshold: float | None = None

    # 策略比较 Agent:只读工具循环的有界运行参数
    agent_max_steps: int = Field(default=4, ge=1, le=10)
    agent_max_tool_calls: int = Field(default=3, ge=1, le=10)
    agent_timeout_seconds: int = Field(default=20, ge=1, le=120)

    # 策略落地规划的本地恢复状态；SQLite 不承载知识证据正文。
    policy_rollout_db_path: Path = ROOT_DIR / "data" / "local" / "policy_rollout_tasks.sqlite3"
    policy_rollout_draft_dir: Path = ROOT_DIR / "data" / "local" / "policy_rollout_drafts"
    policy_rollout_claim_ttl_seconds: int = Field(default=300, ge=1, le=3600)

    # bge-m3 dense 向量维度(工作单元 3 使用)
    embedding_dim: int = 1024

    corpus_dir: Path = ROOT_DIR / "data" / "corpus"

    def prepare_runtime_env(self) -> None:
        """运行时环境准备;必须在相关三方库被 import 之前调用(其常量于导入时读取)。

        - HF_ENDPOINT 指向国内镜像(huggingface.co 直连超时,2026-08-07 实测);
        - HF_HUB_DISABLE_XET=1:镜像下 xet 协议会回跳官方 CAS 服务器报 401,
          禁用后回退普通 HTTP 下载,与镜像兼容(同日实测);
        - TORCH_COMPILE_DISABLE=1:本机无 MSVC(cl.exe),Docling/FlagEmbedding
          部分模型默认启用 torch.compile,须显式禁用(同日实测 InvalidCxxCompiler);
        - HF_HUB_OFFLINE=1(仅当 bge-m3 已完整缓存):transformers 5.x 加载 tokenizer
          时会向 HF 发起 list_repo_templates 远程探测,即使模型已在本地——该探测在
          镜像抖动时会以 ConnectTimeout 拖垮 /questions(2026-08-08 实测)。
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
