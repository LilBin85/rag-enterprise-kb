"""显式预下载 bge-m3 模型(经 HF_ENDPOINT 镜像,禁用 xet),打印落地路径。

手动运行: python scripts/download_bge_m3.py

说明:
- 约 2.3GB,落地到 HF_HOME(见 .env;当前指向 D 盘),不占用 C 盘。
- allow_patterns 只取加载必需文件:2026-08-07 实测 hf-mirror 对仓库内 imgs/.DS_Store
  等无关文件返回 403,会拖死整仓快照下载;权重文件本身不受影响。
- FlagEmbedding 1.4 底层按 sentence-transformers 架构加载,故需包含 modules.json
  与 1_Pooling/ 目录。
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# 大文件(主权重约 2.27GB)在不稳定链路上的宽容超时;默认 10 秒易中途 ReadTimeout
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from huggingface_hub import snapshot_download

ALLOW_PATTERNS = ["*.json", "*.model", "*.txt", "*.bin", "*.safetensors", "1_Pooling/*"]

# bge-m3 是 dense/sparse 检索必需权重；bge-reranker-v2-m3 在 ENABLE_RERANKER=true
# (默认)时使用。Docker 强制 HF_HUB_OFFLINE=1，未预下载时重排会回落融合顺序。
MODELS = ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3")

if __name__ == "__main__":
    for model_name in MODELS:
        for attempt in range(1, 4):
            try:
                path = snapshot_download(model_name, allow_patterns=ALLOW_PATTERNS)
                print(f"{model_name} downloaded to:", path)
                break
            except Exception as exc:  # 断点续传:重试不会丢失已下载部分
                print(f"{model_name} attempt {attempt} failed: {type(exc).__name__}: {exc};断点续传重试…")
        else:
            raise SystemExit(f"{model_name} 多次重试仍失败;请开启 VPN 后重跑本脚本")
