"""答案质量评测：引用合法但答错数字的确定性检查。"""

from rag_kb.evaluation.answer_quality import canonical_number, evaluate_answer_support

CITED = ("员工入职满 1 年后，每年可享受带薪年假 15 天。",)


def test_reported_hallucination_is_caught() -> None:
    """引用『年假 15 天』的真实片段却回答 200 天，必须被判定 unsupported。"""

    report = evaluate_answer_support(CITED, "员工年假是 200 天。")

    assert not report.supported
    assert report.unsupported_claims == ("200天",)


def test_supported_numbers_pass_including_chinese_numeral_equivalence() -> None:
    assert evaluate_answer_support(CITED, "员工每年享有 15 天带薪年假。").supported
    assert evaluate_answer_support(
        ("每年可享受带薪年假十五天。",), "员工每年享有 15 天带薪年假。"
    ).supported
    assert evaluate_answer_support(CITED, "员工每年享有十五天带薪年假。").supported


def test_missing_citations_never_count_as_supported() -> None:
    report = evaluate_answer_support((), "员工每年享有 15 天带薪年假。")

    assert report.has_citations is False
    assert not report.supported


def test_scale_and_decimal_equivalence() -> None:
    procurement = ("二十万元以上采购须由采购委员会批准。",)
    availability = ("核心 API 月度可用性为 99.95%。",)

    assert evaluate_answer_support(procurement, "20 万元以上的采购需要委员会批准。").supported
    assert evaluate_answer_support(availability, "月度可用性是 99.95%。").supported
    assert not evaluate_answer_support(availability, "月度可用性是 99.9%。").supported


def test_unit_mismatch_is_not_support() -> None:
    report = evaluate_answer_support(("申请必须提前 3 个工作日提交。",), "申请必须提前 3 天提交。")

    assert not report.supported
    assert report.unsupported_claims == ("3天",)


def test_canonical_number_accepts_arabic_decimal_and_chinese_scales() -> None:
    assert canonical_number("99.95") == 99.95
    assert canonical_number("15") == 15.0
    assert canonical_number("十五") == 15.0
    assert canonical_number("二十万") == 200_000.0
    assert canonical_number("一万五千") == 15_000.0
    assert canonical_number("两") == 2.0
