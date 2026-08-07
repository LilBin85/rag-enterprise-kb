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

    # bge-m3 dense 向量维度(工作单元 3 使用)
    embedding_dim: int = 1024

    corpus_dir: Path = ROOT_DIR / "data" / "corpus"

    def apply_hf_endpoint(self) -> None:
        """huggingface_hub 在导入时读取 HF_ENDPOINT;下载模型前调用一次。"""
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)


@lru_cache
def get_settings() -> Settings:
    return Settings()
