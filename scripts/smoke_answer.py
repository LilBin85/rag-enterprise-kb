"""真实问答冒烟:真 bge-m3 + 真 Docling PDF + 真 DeepSeek API 的端到端问答。

手动运行: python scripts/smoke_answer.py
蓝图 10.3 允许的少量真实 API 调用;通过标准:
- hr 角色问年假 → verdict=answer,回答引用 v2(15 天),citations 指向 v2 块
- employee 同问 → 标准化拒答,且不触发 DeepSeek 调用
结束后自动清理冒烟 collection。
"""

import uuid

from rag_kb.answerer import AnswerService, DeepSeekAnswerModel
from rag_kb.config import ROOT_DIR, get_settings, resolve_retrieval_limits
from rag_kb.embedder import BgeM3Embedder
from rag_kb.lifecycle import DocumentLifecycle
from rag_kb.models import DocumentSpec, Role
from rag_kb.retriever import Retriever
from rag_kb.store import ChunkStore


def main() -> None:
    settings = get_settings()
    settings.prepare_runtime_env()

    store = ChunkStore(settings.qdrant_url, f"smoke_qa_{uuid.uuid4().hex[:8]}")
    store.ensure_collection(settings.embedding_dim)
    embedder = BgeM3Embedder()
    lifecycle = DocumentLifecycle(
        store, embedder, ROOT_DIR / "data" / "local" / "uploads", settings.chunk_max_chars
    )

    # 摄取休假制度 v1 并更新到 v2(真实 Docling PDF 路径 + 真实向量)
    base = dict(document_id="hr-leave-policy", title="员工休假制度", department="hr",
                allowed_roles=[Role.HR, Role.ADMIN])
    lifecycle.add_document(
        DocumentSpec(**base, file="员工休假制度-v1.pdf", version=1),
        ROOT_DIR / "data" / "corpus" / "员工休假制度-v1.pdf",
    )
    lifecycle.update_document(
        "hr-leave-policy",
        DocumentSpec(**base, file="员工休假制度-v2.pdf", version=0),
        ROOT_DIR / "data" / "corpus" / "员工休假制度-v2.pdf",
    )

    candidate_k, context_k = resolve_retrieval_limits(settings)
    retriever = Retriever(
        store, embedder,
        top_k=settings.retrieval_top_k, candidate_k=candidate_k, context_k=context_k,
        score_threshold=settings.retrieval_score_threshold,
    )
    service = AnswerService(retriever, DeepSeekAnswerModel(settings))

    question = "员工入职满一年后,每年有多少天年假?"
    preview = retriever.search(question, Role.HR)
    print("候选块:", [(c.chunk_id, round(c.score, 3)) for c in preview.candidates])

    resp = service.ask(question, Role.HR)
    print("hr 问年假 → verdict:", resp.verdict, "| refusal:", resp.refusal)
    print("answer:", resp.answer)
    for c in resp.citations:
        print("citation:", c.model_dump_json())

    resp2 = service.ask("员工年假有多少天?", Role.EMPLOYEE)
    print("employee 同问 → verdict:", resp2.verdict,
          "|", resp2.refusal.message if resp2.refusal else "")

    store.client.delete_collection(store.collection)
    print("冒烟 collection 已清理")


if __name__ == "__main__":
    main()
