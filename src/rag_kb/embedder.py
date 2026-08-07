"""向量生成(蓝图 7.1 dense 检索基线)。

生产路径使用 bge-m3(FlagEmbedding,CPU);测试路径注入 DeterministicEmbedder,
使确定性工程行为与模型下载/推理完全解耦(蓝图 10.3)。
"""

import hashlib
from typing import Protocol

import numpy as np

from rag_kb.config import get_settings


class Embedder(Protocol):
    """embedding 抽象:生命周期与存储只依赖此协议,不关心下游是真模型还是测试桩。"""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DeterministicEmbedder:
    """确定性伪向量:同一文本永远得到同一单位向量,维度与 bge-m3 dense 对齐。"""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim)
            vectors.append((vec / np.linalg.norm(vec)).astype(np.float32).tolist())
        return vectors


class BgeM3Embedder:
    """bge-m3 dense 向量封装。懒加载:首次 embed 时才下载/加载模型(约 2.3GB)。"""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            get_settings().apply_hf_endpoint()
            from FlagEmbedding import FlagModel

            # CPU 推理必须 fp32;fp16 仅 GPU 可用
            self._model = FlagModel(self.model_name, use_fp16=False)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        result = model.encode(texts)
        dense = result["dense_vecs"] if isinstance(result, dict) else result
        return np.asarray(dense, dtype=np.float32).tolist()
