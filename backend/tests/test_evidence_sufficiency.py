from rag_kb.application.evidence_sufficiency import DeterministicEvidenceSufficiencyPolicy
from rag_kb.domain.retrieval import RetrievalCandidate


def candidate(
    text: str,
    *,
    source_file_name: str = "doc.md",
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    document_title: str = "",
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
        document_title=document_title,
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


def test_document_name_does_not_implicitly_hard_scope_candidates() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    scoped = policy.select_candidates(
        "产品服务等级协议规定灾难恢复 RTO 是多少分钟？",
        (
            candidate("回滚恢复时间目标（RTO）为十五分钟。", source_file_name="研发发布流程.md"),
            candidate(
                "响应目标不等同于恢复时限。",
                source_file_name="产品服务等级协议.md",
                chunk_id="chunk-2",
                document_id="doc-2",
            ),
        ),
    )
    assert [item.source_file_name for item in scoped] == ["研发发布流程.md"]
    assert policy.insufficiency_code(
        "产品服务等级协议规定灾难恢复 RTO 是多少分钟？", scoped
    ) is None


def test_document_name_without_explicit_requirements_keeps_all_candidates() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    scoped = policy.select_candidates(
        "访问管理制度规定什么？",
        (
            candidate("访问要求", source_file_name="信息安全访问管理制度.md"),
            candidate("其他要求", source_file_name="供应商管理规范.md", chunk_id="chunk-2"),
        ),
    )
    assert [item.source_file_name for item in scoped] == [
        "信息安全访问管理制度.md",
        "供应商管理规范.md",
    ]


def test_document_title_satisfies_ascii_anchor() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    assert policy.insufficiency_code(
        "demo-onboarding-20260911 的要求是什么？",
        (
            candidate(
                "入职前创建账号。",
                document_title="demo-onboarding-20260911",
            ),
        ),
    ) is None


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


def test_annual_leave_question_does_not_require_a_number_of_years() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()

    assert policy.insufficiency_code(
        "正式员工每年有多少天带薪年假？",
        (candidate("正式员工每年享有 17 天带薪年假。"),),
    ) is None


def test_requested_working_days_accepts_the_classifier_in_evidence() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()

    assert policy.insufficiency_code(
        "正式员工每年有多少天专项假，申请须提前几个工作日？",
        (candidate("正式员工每年享有 17 天专项假，申请必须提前 3 个工作日提交。"),),
    ) is None


def test_requested_working_days_accepts_both_classifier_forms() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()

    assert policy.insufficiency_code(
        "人力资源必须提前几个工作日创建入职单？",
        (candidate("人力资源必须在员工入职日前三个工作日创建入职单。"),),
    ) is None
    assert policy.insufficiency_code(
        "人力资源必须提前几个工作日创建入职单？",
        (candidate("人力资源必须在员工入职日前三工作日创建入职单。"),),
    ) is None
    assert policy.insufficiency_code(
        "人力资源必须提前多少个工作日创建入职单？",
        (candidate("人力资源必须在员工入职日前三个工作日创建入职单。"),),
    ) is None


def test_requested_characters_accepts_the_classifier_in_evidence() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()

    assert policy.insufficiency_code(
        "标题最多允许几个字符？",
        (candidate("标题最多允许一百个字符。"),),
    ) is None


def test_non_classifier_unit_keeps_its_original_form() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()

    assert policy.insufficiency_code(
        "审批需要几天？",
        (candidate("审批需要三天。"),),
    ) is None


def test_requested_working_days_rejects_evidence_without_a_quantity() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()

    assert policy.insufficiency_code(
        "人力资源必须提前几个工作日创建入职单？",
        (candidate("人力资源必须在员工入职日前创建入职单。"),),
    ) == "evidence_quantitative_fact_missing"


def test_quantity_requirement_may_be_satisfied_by_any_chunk_in_document() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    query = "人力资源必须提前几个工作日创建入职单？"
    candidates = (
        candidate("本制度适用于所有新员工。", chunk_id="chunk-1"),
        candidate(
            "人力资源必须在员工入职日前三个工作日创建入职单。",
            chunk_id="chunk-2",
        ),
    )

    assert policy.select_candidates(query, candidates) == candidates
    assert policy.insufficiency_code(query, candidates) is None


def test_exact_quantity_with_classifier_accepts_either_evidence_form() -> None:
    policy = DeterministicEvidenceSufficiencyPolicy()
    query = "人力资源是否必须提前三个工作日创建入职单？"

    assert policy.insufficiency_code(query, (candidate("必须提前三个工作日创建。"),)) is None
    assert policy.insufficiency_code(query, (candidate("必须提前三工作日创建。"),)) is None


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
