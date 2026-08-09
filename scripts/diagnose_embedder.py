"""诊断:完整复现 POST /questions 的内部路径(env 状态 → embedder 加载 → 真实检索)。

手动运行: python scripts/diagnose_embedder.py
定位 2026-08-08 的 retrieval_failure:确认 HF_HUB_OFFLINE 是否生效、模型是否可加载、
Qdrant 是否可达、检索是否返回候选。
"""

import os
import traceback

from rag_kb.config import get_settings


def main() -> None:
    settings = get_settings()
    settings.prepare_runtime_env()
    print("HF_HOME =", os.environ.get("HF_HOME"))
    print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"))
    print("HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE"))
    print("bge-m3 缓存完整 =", settings._bge_m3_fully_cached())

    from rag_kb.models import Role
    from rag_kb.retriever import Retriever
    from rag_kb.store import ChunkStore
    from rag_kb.embedder import BgeM3Embedder

    try:
        store = ChunkStore(settings.qdrant_url, settings.qdrant_collection)
        cols = [c.name for c in store.client.get_collections().collections]
        print("Qdrant collections:", cols)
    except Exception:
        print("Qdrant 连接失败:")
        traceback.print_exc()
        return

    try:
        retriever = Retriever(store, BgeM3Embedder(), top_k=5)
        result = retriever.search("常规发布窗口是什么时间", Role.ENGINEER)
        print("检索候选:", [(c.chunk_id, round(c.score, 3)) for c in result.candidates])
    except Exception:
        print("检索/模型加载失败:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
