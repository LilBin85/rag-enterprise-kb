"""向量生成(蓝图 7.1 dense 检索基线)。生产使用 bge-m3(FlagEmbedding,CPU)。"""

import numpy as np

from rag_kb.config import get_settings


class BgeM3Embedder:
    """bge-m3 dense 向量封装。懒加载:首次 embed 时才下载/加载模型(约 2.3GB)。"""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            get_settings().prepare_runtime_env()
            from FlagEmbedding import FlagModel

            # CPU 推理必须 fp32;fp16 仅 GPU 可用
            self._model = FlagModel(self.model_name, use_fp16=False)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        result = model.encode(texts)
        dense = result["dense_vecs"] if isinstance(result, dict) else result
        return np.asarray(dense, dtype=np.float32).tolist()
