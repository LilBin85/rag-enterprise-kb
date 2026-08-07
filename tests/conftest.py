"""pytest 共享夹具。

集成测试策略(蓝图 10.3):
- 生命周期与 API 测试连接真实 Qdrant(验证真实 filter/状态语义),但使用
  DeterministicEmbedder(确定性伪向量)脱离 bge-m3 模型下载;
- Qdrant 不可达时集成测试整体 skip,干净环境仍可跑纯单元测试;
- 每个测试函数使用独立 collection,测完即删,互不污染。
"""

import uuid
from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_kb.config import get_settings
from rag_kb.embedder import DeterministicEmbedder
from rag_kb.store import ChunkStore


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
