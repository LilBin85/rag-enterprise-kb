"""应用配置:所有环境相关取值集中于此,密钥只从环境变量 / .env 读取(蓝图 11.1)。"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # DeepSeek(阶段 4 才消费;此处只负责读取与保管)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "kb_chunks"

    # HuggingFace 下载端点(国内镜像;可直连时留空)
    hf_endpoint: str | None = None

    # 摄取限制(蓝图 8 节上传校验)
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_extensions: tuple[str, ...] = (".md", ".pdf")

    # 切块上限(按字符数;结构化边界优先,超长章节再细分)
    chunk_max_chars: int = 800

    # 检索(蓝图 7.1):Top-K 候选数;相似度阈值待冻结评测校准,None 表示不裁切
    retrieval_top_k: int = 5
    retrieval_score_threshold: float | None = None

    # bge-m3 dense 向量维度(工作单元 3 使用)
    embedding_dim: int = 1024

    corpus_dir: Path = ROOT_DIR / "data" / "corpus"

    def prepare_runtime_env(self) -> None:
        """运行时环境准备;必须在相关三方库被 import 之前调用(其常量于导入时读取)。

        - HF_ENDPOINT 指向国内镜像(huggingface.co 直连超时,2026-08-07 实测);
        - HF_HUB_DISABLE_XET=1:镜像下 xet 协议会回跳官方 CAS 服务器报 401,
          禁用后回退普通 HTTP 下载,与镜像兼容(同日实测);
        - TORCH_COMPILE_DISABLE=1:本机无 MSVC(cl.exe),Docling/FlagEmbedding
          部分模型默认启用 torch.compile,须显式禁用(同日实测 InvalidCxxCompiler)。
        """
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    def apply_hf_endpoint(self) -> None:
        """兼容别名;统一转发到 prepare_runtime_env。"""
        self.prepare_runtime_env()


@lru_cache
def get_settings() -> Settings:
    return Settings()
