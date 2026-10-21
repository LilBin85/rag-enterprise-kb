"""bge-reranker 交叉编码重排适配器；加载/推理失败时回落融合顺序（fail-soft）。

重排只允许改变顺序和 rerank_score，不能增删候选——
``EnterpriseRetriever._validate_reranker_boundary`` 会拒绝任何绑定变化。

权重约 2.1GB 且懒加载；Docker 明确设置 ``HF_HUB_OFFLINE=1``，权重未缓存时加载
必然失败。默认启用（ENABLE_RERANKER=true）后，失败必须回落而不是把每次问答拖成
检索失败，因此这里 catch 一切并保留原融合顺序，本进程内不再重试。

实现说明：不走 FlagEmbedding.FlagReranker——其 1.4.x 推理路径调用已被
transformers 5.x 移除的 ``prepare_for_model``（2026-10 实测
``XLMRobertaTokenizer has no attribute prepare_for_model``）。bge-reranker 本就是
单 logit 序列分类模型，直接用 AutoModelForSequenceClassification + sigmoid 打分。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

from rag_kb.application.ports.retrieval import RankedChunk
from rag_kb.config import get_settings
from rag_kb.observability import log_event, stable_error_code

_MAX_PAIR_LENGTH = 512


class BgeReranker:
    """懒加载 bge-reranker；失败后本进程内跳过重排（保留融合顺序）。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._unavailable = False

    def _load(self):
        if self._model is None:
            get_settings().prepare_runtime_env()
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # CPU 推理必须 fp32；fp16 仅 GPU 可用（与 BgeM3Embedder 同约束）。
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            model.eval()
            self._model = model
        return self._model

    def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> tuple[RankedChunk, ...]:
        if not candidates or self._unavailable:
            return tuple(candidates)
        try:
            self._load()
            scores = self._scores(query, candidates)
            return self._order(candidates, scores)
        except Exception as exc:
            self._unavailable = True
            log_event(
                "retrieval.rerank.unavailable",
                level=30,
                component="retrieval",
                operation="retrieval.rerank",
                outcome="failure",
                error_category="model",
                error_code=stable_error_code(exc),
                reranker_model=self._model_name,
            )
            return tuple(candidates)

    def _scores(self, query: str, candidates: Sequence[RankedChunk]) -> list[float]:
        import torch

        inputs = self._tokenizer(
            [query] * len(candidates),
            [candidate.text for candidate in candidates],
            padding=True,
            truncation=True,
            max_length=_MAX_PAIR_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits.view(-1).float()
        return torch.sigmoid(logits).tolist()

    @staticmethod
    def _order(
        candidates: Sequence[RankedChunk],
        scores: object,
    ) -> tuple[RankedChunk, ...]:
        values = [scores] if isinstance(scores, (int, float)) else list(scores)  # type: ignore[arg-type]
        if len(values) != len(candidates) or any(
            isinstance(value, bool) or not math.isfinite(float(value)) for value in values
        ):
            raise ValueError("reranker_score_vector_invalid")
        return tuple(
            replace(candidate, rerank_score=float(score))
            for candidate, score in sorted(
                zip(candidates, values, strict=True), key=lambda pair: -float(pair[1])
            )
        )
