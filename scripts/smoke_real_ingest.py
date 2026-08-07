"""真实摄取冒烟:一次性验证 bge-m3(真模型)与 Docling(真 PDF)两条生产路径。

手动运行(不进 pytest 默认集:首次需经 HF_ENDPOINT 下载约 2.3GB 模型,CPU 推理较慢):
    python scripts/smoke_real_ingest.py

通过标准:休假 v1.pdf 被解析为多个章节块(带页码 locator)、向量维度 1024、payload 含蓝图 6.1 字段;
结束后自动清理冒烟 collection。
"""

import uuid

from rag_kb.config import ROOT_DIR, get_settings
from rag_kb.embedder import BgeM3Embedder
from rag_kb.lifecycle import DocumentLifecycle
from rag_kb.models import DocumentSpec, Role
from rag_kb.store import ChunkStore


def main() -> None:
    settings = get_settings()
    settings.apply_hf_endpoint()

    store = ChunkStore(settings.qdrant_url, f"smoke_{uuid.uuid4().hex[:8]}")
    store.ensure_collection(settings.embedding_dim)
    lifecycle = DocumentLifecycle(
        store=store,
        embedder=BgeM3Embedder(),
        upload_dir=ROOT_DIR / "data" / "local" / "uploads",
        max_chars=settings.chunk_max_chars,
    )

    spec = DocumentSpec(
        document_id="hr-leave-policy",
        file="员工休假制度-v1.pdf",
        title="员工休假制度",
        department="hr",
        allowed_roles=[Role.HR, Role.ADMIN],
        version=1,
    )
    pdf_path = ROOT_DIR / "data" / "corpus" / "员工休假制度-v1.pdf"

    result = lifecycle.add_document(spec, pdf_path)
    print("摄取结果:", result)

    chunks = store.scroll_document("hr-leave-policy")
    print("块数:", len(chunks))
    print("locator 样例:", [c["source_locator"] for c in chunks][:3])

    point_id = store.point_id(chunks[0]["chunk_id"])
    record = store.client.retrieve(store.collection, [point_id], with_vectors=True)[0]
    print("向量维度:", len(record.vector))

    store.client.delete_collection(store.collection)
    print("冒烟 collection 已清理")


if __name__ == "__main__":
    main()
