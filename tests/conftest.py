"""pytest 共享夹具。

集成测试策略(蓝图 10.3):
- 生命周期与 API 测试连接真实 Qdrant(验证真实 filter/状态语义),但使用
  DeterministicEmbedder(确定性伪向量)脱离 bge-m3 模型下载;
- Qdrant 不可达时集成测试整体 skip,干净环境仍可跑纯单元测试;
- 每个测试函数使用独立 collection,测完即删,互不污染。
"""

import hashlib
import uuid
from collections.abc import Iterator

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_kb.config import get_settings
from rag_kb.embedder import DeterministicEmbedder
from rag_kb.store import ChunkStore


class TopicEmbedder:
    """主题可控的确定性 embedder,用于检索集成测试(蓝图 10.3:脱离真实模型)。

    注册关键词映射为固定基向量;文本向量 = 命中关键词基向量的均值 + 微噪声。
    命中主题的查询与块余弦 ≈1,未命中 ≈0(双峰分布),配合 score_threshold
    即可断言召回行为;而 ACL/生命周期过滤由 Qdrant 真实执行,不是测试假设。
    """

    def __init__(self, dim: int = 1024, noise: float = 0.01) -> None:
        self.dim = dim
        self.noise = noise
        self._bases: dict[str, np.ndarray] = {}

    def register(self, *keywords: str) -> "TopicEmbedder":
        for kw in keywords:
            seed = int.from_bytes(hashlib.sha256(kw.encode("utf-8")).digest()[:8], "big")
            vec = np.random.default_rng(seed).standard_normal(self.dim)
            self._bases[kw] = vec / np.linalg.norm(vec)
        return self

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            hits = [base for kw, base in self._bases.items() if kw in text]
            base = np.mean(hits, axis=0) if hits else np.zeros(self.dim)
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            noise = np.random.default_rng(seed + 1).standard_normal(self.dim) * self.noise
            vec = base + noise
            norm = np.linalg.norm(vec)
            out.append((vec / norm if norm else vec).astype(np.float32).tolist())
        return out


def _qdrant_reachable(url: str) -> bool:
    try:
        QdrantClient(url=url, timeout=3).get_collections()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def qdrant_url() -> str:
    url = get_settings().qdrant_url
    if not _qdrant_reachable(url):
        pytest.skip(f"Qdrant 不可达({url});跳过集成测试")
    return url


@pytest.fixture()
def store(qdrant_url: str) -> Iterator[ChunkStore]:
    s = ChunkStore(url=qdrant_url, collection=f"test_{uuid.uuid4().hex[:12]}")
    s.ensure_collection(dim=1024)
    yield s
    try:
        s.client.delete_collection(s.collection)
    except UnexpectedResponse:
        pass


@pytest.fixture()
def embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder(dim=1024)


@pytest.fixture()
def topic_embedder() -> TopicEmbedder:
    """注册冻结语料主题词;摄取与检索必须共享同一实例(基向量表一致)。"""
    return TopicEmbedder().register("报销", "住宿", "发布", "回滚", "年假", "育儿假")
