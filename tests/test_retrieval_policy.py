"""检索策略只做确定性路由和结构判断，不调用模型或改变向量检索器。"""

from rag_kb.retrieval_policy import QueryIntent, QueryPolicy, TopicRouter


def test_topic_router_adds_policy_topic_to_evidence_facet() -> None:
    router = TopicRouter()
    assert router.topics("请根据新版年假制度制定计划。") == ["年假制度"]
    assert router.route("试点范围制度依据", "请根据新版年假制度制定计划。") == "年假制度 试点范围制度依据"
    assert router.route("年假制度", "请根据新版年假制度制定计划。") == "年假制度"


def test_preflight_queries_cover_explicit_implementation_but_respect_exclusion() -> None:
    policy = QueryPolicy()
    assert policy.preflight_queries("请协调新版休假政策的落地工作，并确认执行安排。") == [
        "休假政策", "休假政策 实施安排",
    ]
    assert policy.preflight_queries("请根据新版年假制度核对清单；不需要系统实施安排。") == [
        "年假制度",
    ]


def test_query_policy_blocks_business_input_before_routing() -> None:
    decision = QueryPolicy().classify(
        "研发部首批试点部门", "请根据年假制度制定计划。",
        business_terms={"研发部", "首批试点部门"},
        missing_variables=[], answers=["研发部"],
    )
    assert decision.intent == QueryIntent.BUSINESS_INPUT
    assert decision.query is None


def test_structural_fingerprint_ignores_synonyms_but_detects_new_facet() -> None:
    policy = QueryPolicy()
    goal = "请根据年假制度制定计划。"
    first = policy.classify(
        "公司年假政策", goal, business_terms=set(), missing_variables=[], answers=[],
    )
    synonym = policy.classify(
        "员工休假规则", goal, business_terms=set(), missing_variables=[], answers=[],
    )
    implementation = policy.classify(
        "实施安排", goal, business_terms=set(), missing_variables=[], answers=[],
    )
    assert first.structure_fingerprint == synonym.structure_fingerprint
    assert implementation.structure_fingerprint != first.structure_fingerprint


def test_explicit_missing_variable_cannot_be_searched() -> None:
    decision = QueryPolicy().classify(
        "年假制度", "首批试点部门尚未确定。", business_terms=set(),
        missing_variables=["首批试点部门"], answers=[],
    )
    assert decision.intent == QueryIntent.MISSING_VARIABLE


def test_query_policy_blocks_facet_explicitly_excluded_by_goal() -> None:
    policy = QueryPolicy()
    decision = policy.classify(
        "实施安排", "请核对年假资格并形成清单；不需要系统实施安排。",
        business_terms=set(), missing_variables=[], answers=[],
    )
    assert decision.intent == QueryIntent.OUT_OF_SCOPE
    assert decision.query is None
    assert policy.classify(
        "年假制度", "请核对年假资格并形成清单；不需要系统实施安排。",
        business_terms=set(), missing_variables=[], answers=[],
    ).intent == QueryIntent.EVIDENCE_QUERY
