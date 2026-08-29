from rag_kb.application.evidence_sufficiency import DeterministicEvidenceSufficiencyPolicy
from rag_kb.domain.retrieval import RetrievalCandidate


def candidate(
    text: str,
    *,
    source_file_name: str = "doc.md",
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        tenant_id="tenant-a",
        document_id=document_id,
        version_id=f"version-{document_id}",
        document_version=1,
        chunk_id=chunk_id,
        content_hash="sha256:chunk",
        source_locator="doc/1",
        source_file_name=source_file_name,
        dense_score=0.7,
        fused_score=1.0,
        text=text,
    )


def test_rejects_missing_acronym_even_when_topic_is_related() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    assert policy.insufficiency_code(
        "灾难恢复 RTO 必须是多少分钟？",
        (candidate("P0 事件应在五分钟内响应，但未规定恢复时限。"),),
    ) == "evidence_anchor_missing"


def test_rejects_missing_requested_numeric_unit() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    assert policy.insufficiency_code(
        "企业 VPN 每月有多少 GB 流量？",
        (candidate("内部数据必须使用企业 VPN。"),),
    ) == "evidence_quantitative_fact_missing"


def test_accepts_covered_acronym_and_numeric_fact() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    assert policy.insufficiency_code(
        "核心 API 可用性是多少 %？",
        (candidate("核心 API 月度可用性为 99.95%。"),),
    ) is None


def test_accepts_non_quantitative_semantic_evidence() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    assert policy.insufficiency_code(
        "居家处理内部数据时对设备和网络有什么要求？",
        (candidate("处理内部数据时必须使用公司设备和企业 VPN。"),),
    ) is None


def test_named_document_scope_excludes_related_but_wrong_rto() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    scoped = policy.select_candidates(
        "产品服务等级协议规定灾难恢复 RTO 是多少分钟？",
        (
            candidate("回滚恢复时间目标（RTO）为十五分钟。", source_file_name="研发发布流程.md"),
            candidate("响应目标不等同于恢复时限。", source_file_name="产品服务等级协议.md", chunk_id="chunk-2"),
        ),
    )
    assert [item.source_file_name for item in scoped] == ["产品服务等级协议.md"]
    assert policy.insufficiency_code(
        "产品服务等级协议规定灾难恢复 RTO 是多少分钟？",
        scoped,
    ) == "evidence_quantitative_fact_missing"


def test_named_document_scope_accepts_unambiguous_title_suffix() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    scoped = policy.select_candidates(
        "访问管理制度规定什么？",
        (
            candidate("访问要求", source_file_name="信息安全访问管理制度.md"),
            candidate("其他要求", source_file_name="供应商管理规范.md", chunk_id="chunk-2"),
        ),
    )
    assert [item.source_file_name for item in scoped] == ["信息安全访问管理制度.md"]


def test_requirements_cannot_be_satisfied_across_documents() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    query = "灾难恢复 RTO 必须是多少分钟？"
    candidates = (
        candidate("恢复时间目标缩写为 RTO。", source_file_name="术语规范.md"),
        candidate("事故须在五分钟内响应。", source_file_name="事故规范.md", chunk_id="chunk-2", document_id="doc-2"),
    )
    assert policy.insufficiency_code(query, candidates) is not None


def test_exact_quantity_scopes_candidates_to_the_supporting_document() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    query = "二十万元采购在 V2 中还需要谁批准？"
    procurement = candidate(
        "二十万元及以上还须采购委员会批准。",
        source_file_name="采购审批制度-v2.md",
        document_id="procurement",
    )
    unrelated = candidate(
        "员工每年可享受带薪年假十五天。",
        source_file_name="员工休假制度-v2.pdf",
        chunk_id="chunk-2",
        document_id="leave",
    )

    scoped = policy.select_candidates(query, (unrelated, procurement))

    assert scoped == (procurement,)
    assert policy.insufficiency_code(query, scoped) is None
    assert policy.insufficiency_code(query, (unrelated,)) == "evidence_quantitative_fact_missing"


def test_semantic_mismatch_is_deferred_to_verdict_layer() -> None:
    """主谓语义不匹配不再由词面滑窗拦截，交由回答模型 verdict 判定。"""

    policy = DeterministicEvidenceSufficiencyPolicy()
    assert policy.insufficiency_code(
        "关键供应商必须购买多少万元的责任保险？",
        (candidate("二十万元以上采购须由委员会批准。", source_file_name="采购审批制度.md"),),
    ) is None


def test_natural_annual_leave_phrasing_is_not_lexically_gated() -> None:
    """自然问法（员工年假/申请年假）与文档原文（带薪年假）存在词汇鸿沟，
    不得被确定性门误拒；证据含数量事实（15 天/1 年）时必须放行。"""

    policy = DeterministicEvidenceSufficiencyPolicy()
    assert (
        policy.insufficiency_code(
            "员工年假有多少天？申请年假需要满足什么条件？",
            (
                candidate(
                    "员工入职满 1 年后，每年可享受带薪年假 15 天。年假按自然年度计算，"
                    "当年度未休完的年假最多可结转 5 天至次年度 3 月底前使用。",
                    source_file_name="员工休假制度-v2.pdf",
                ),
            ),
        )
        is None
    )


def test_version_anchor_may_be_satisfied_by_bound_source_metadata() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    assert policy.insufficiency_code(
        "采购审批制度 V2 从哪一天开始生效？",
        (candidate("本版自 2026 年 7 月 1 日生效。", source_file_name="采购审批制度-v2.md"),),
    ) is None
    assert policy.insufficiency_code(
        "人力资源人员可以查阅单一来源采购的书面论证要求吗？",
        (candidate("人力资源在入职日前创建入职单。", source_file_name="新员工入职流程.md"),),
    ) is None
