"""证据账本契约；只保存来源映射和哈希，不复制授权正文。"""

from pydantic import AwareDatetime, BaseModel, ConfigDict

from rag_kb.domain.fingerprints import canonical_fingerprint


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    task_id: str
    query_fingerprint: str
    tenant_id: str
    document_id: str
    version_id: str
    document_version: int
    chunk_id: str
    content_hash: str
    source_locator: str
    retrieved_at: AwareDatetime
    permission_granted: bool
    supported_claim_ids: tuple[str, ...] = ()


class EvidenceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ids: tuple[str, ...]
    digest: str


def canonical_evidence_snapshot(
    evidence_ids: tuple[str, ...] | list[str],
    *,
    require_non_empty: bool = False,
) -> EvidenceSnapshot:
    """构造顺序无关的证据精确集合；重复或空标识一律失败关闭。"""

    if any(not isinstance(item, str) or not item.strip() for item in evidence_ids):
        raise ValueError("evidence_id_invalid")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("duplicate_evidence_id")
    normalized = tuple(sorted(evidence_ids))
    if require_non_empty and not normalized:
        raise ValueError("evidence_required")
    return EvidenceSnapshot(
        evidence_ids=normalized,
        digest=canonical_fingerprint("evidence_exact_set", list(normalized)),
    )
