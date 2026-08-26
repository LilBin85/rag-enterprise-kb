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
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_kb.config import get_settings
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
        pytest.skip("qdrant_unavailable")
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
def embedder() -> TopicEmbedder:
    return TopicEmbedder()


@pytest.fixture()
def topic_embedder() -> TopicEmbedder:
    """注册冻结语料主题词;摄取与检索必须共享同一实例(基向量表一致)。"""
    return TopicEmbedder().register("报销", "住宿", "发布", "回滚", "年假", "育儿假")


# ---------- 知识库夹具(检索与回答测试共享) ----------

TRAVEL_MD = "# 全员差旅报销制度\n## 住宿标准\n一线城市住宿每晚不超过 450 元,报销时限为返程后 15 个工作日。\n"
DEV_MD = "# 研发发布流程\n## 发布窗口\n每周二周四 14:00-17:00 为发布窗口,灰度放量。\n## 回滚机制\nP0 故障 30 分钟内回滚,RTO 15 分钟。\n"
LEAVE_V1_MD = "# 员工休假制度\n## 年假\n员工每年可享受带薪年假 10 天。\n"
LEAVE_V2_MD = "# 员工休假制度\n## 年假\n员工每年可享受带薪年假 15 天。\n## 育儿假\n子女未满 3 周岁的员工每年可享受育儿假 10 天。\n"


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def kb_and_retriever(store: ChunkStore, topic_embedder: TopicEmbedder, tmp_path: Path):
    """摄取四类文档并把休假制度更新到 v2;检索阈值 0.3 仅用于分离 TopicEmbedder 的双峰分数。

    (命中主题的块余弦 ≈1,未命中 ≈0;生产阈值待冻结评测校准,蓝图 7.1。)
    """
    from rag_kb.lifecycle import DocumentLifecycle
    from rag_kb.models import DocumentSpec, Role
    from rag_kb.retriever import Retriever

    lifecycle = DocumentLifecycle(
        store=store, embedder=topic_embedder, upload_dir=tmp_path / "uploads", max_chars=800
    )
    lifecycle.add_document(
        DocumentSpec(document_id="travel-policy", file="差旅.md", title="全员差旅报销制度",
                     department="general",
                     allowed_roles=[Role.EMPLOYEE, Role.ENGINEER, Role.HR, Role.ADMIN], version=1),
        _write(tmp_path, "差旅.md", TRAVEL_MD),
    )
    lifecycle.add_document(
        DocumentSpec(document_id="dev-process", file="研发.md", title="研发发布流程",
                     department="engineering", allowed_roles=[Role.ENGINEER, Role.ADMIN], version=1),
        _write(tmp_path, "研发.md", DEV_MD),
    )
    lifecycle.add_document(
        DocumentSpec(document_id="leave-policy", file="休假-v1.md", title="员工休假制度",
                     department="hr", allowed_roles=[Role.HR, Role.ADMIN], version=1),
        _write(tmp_path, "休假-v1.md", LEAVE_V1_MD),
    )
    lifecycle.update_document(
        "leave-policy",
        DocumentSpec(document_id="leave-policy", file="休假-v2.md", title="员工休假制度",
                     department="hr", allowed_roles=[Role.HR, Role.ADMIN], version=0),
        _write(tmp_path, "休假-v2.md", LEAVE_V2_MD),
    )
    retriever = Retriever(store=store, embedder=topic_embedder, top_k=5, score_threshold=0.3)
    return lifecycle, retriever


# ---------- 可编程假模型(蓝图 10.3:脱离真实 LLM) ----------

class FakeAnswerModel:
    """记录调用并返回预设 verdict;设置 error 可模拟生成服务故障。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.next = None
        self.error: Exception | None = None

    def answer(self, question: str, chunks: list):
        self.calls.append({"question": question, "chunks": chunks})
        if self.error is not None:
            raise self.error
        return self.next


@pytest.fixture()
def fake_model() -> FakeAnswerModel:
    return FakeAnswerModel()
