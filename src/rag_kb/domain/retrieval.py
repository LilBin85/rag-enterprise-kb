"""可插拔检索管线的输入、候选和输出契约。"""

from pydantic import BaseModel, ConfigDict, Field


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=4000)
    fingerprint: str
    candidate_k: int = Field(default=20, ge=1, le=100)
    context_k: int = Field(default=5, ge=1, le=20)


class RetrievalCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    document_id: str
    version_id: str
    document_version: int
    projection_generation: int = Field(default=1, ge=1)
    chunk_id: str
    content_hash: str
    source_locator: str
    source_file_name: str
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float
    rerank_score: float | None = None
    text: str


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_fingerprint: str
    candidates: tuple[RetrievalCandidate, ...]
    context_token_estimate: int = Field(ge=0)
    sufficient: bool
    insufficiency_code: str | None = None
