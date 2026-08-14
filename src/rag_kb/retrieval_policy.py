"""Planner 检索前后的确定性策略：路由、结构化改写与新证据停止。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum


class QueryIntent(StrEnum):
    EVIDENCE_QUERY = "evidence_query"
    BUSINESS_INPUT = "business_input"
    MISSING_VARIABLE = "missing_variable"
    OUT_OF_SCOPE = "out_of_scope"
    INVALID_QUERY = "invalid_query"


@dataclass(frozen=True)
class QueryPolicyDecision:
    intent: QueryIntent
    query: str | None
    structure_fingerprint: str | None


_POLICY_SUFFIXES = ("制度", "政策", "规则", "办法", "规定")
_FACETS = {
    "scope": ("适用范围", "适用对象", "覆盖人员", "资格", "对象"),
    "implementation": ("实施", "落地", "执行", "上线", "配置", "发布", "通知"),
    "balance": ("余额", "天数", "期限", "有效期", "额度"),
    "approval": ("审批", "申请", "批准", "权限"),
    "version": ("版本", "新版", "旧版", "现行", "生效"),
    "source": ("依据", "条款", "制度编号", "文件编号", "正文"),
}
_EXCLUSION_MARKERS = ("不需要", "无需", "不要", "不涉及", "不包括", "排除")


def _normalized(value: str) -> str:
    return re.sub(r"[\s，,。！？；;：:、()（）\[\]【】]", "", value).casefold()


def hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class TopicRouter:
    """用目标中的制度主题补全检索查询，不改变 ACL 或底层检索器。"""

    @staticmethod
    def topics(goal: str) -> list[str]:
        topics: list[str] = []
        for clause in re.split(r"[。！？；;，,\n]+", goal):
            for match in re.finditer(rf"[A-Za-z0-9_\-\u4e00-\u9fff]{{1,12}}(?:{'|'.join(_POLICY_SUFFIXES)})", clause):
                topic = match.group(0)
                previous = None
                while previous != topic:
                    previous = topic
                    topic = re.sub(
                        r"^(?:请|为|hr|人力资源|根据|仅根据|关于|协调|制定|落地|核对|实施|新版|旧版)+",
                        "", topic, flags=re.IGNORECASE,
                    )
                if topic not in topics:
                    topics.append(topic)
        return topics

    def route(self, query: str, goal: str) -> str:
        topics = self.topics(goal)
        if not topics or any(topic in query for topic in topics):
            return query
        # 只补主题，不让路由器选择角色、租户、文档或版本过滤条件。
        return f"{topics[0]} {query}".strip()


class QueryPolicy:
    """固定 Python 规则；不增加模型调用，也不承担 ACL 判定。"""

    def __init__(self, router: TopicRouter | None = None) -> None:
        self.router = router or TopicRouter()

    def classify(
        self, query: str, goal: str, *, business_terms: set[str],
        missing_variables: list[str], answers: list[str],
    ) -> QueryPolicyDecision:
        normalized = " ".join(query.split()) if isinstance(query, str) else ""
        if not normalized or len(normalized) > 4000:
            return QueryPolicyDecision(QueryIntent.INVALID_QUERY, None, None)
        query_key = _normalized(normalized)
        if any(_normalized(term) and _normalized(term) in query_key for term in business_terms):
            return QueryPolicyDecision(QueryIntent.BUSINESS_INPUT, None, None)
        if missing_variables and not answers:
            return QueryPolicyDecision(QueryIntent.MISSING_VARIABLE, None, None)
        routed = self.router.route(normalized, goal)
        if self.excluded_facets(goal) & self.query_facets(routed):
            return QueryPolicyDecision(QueryIntent.OUT_OF_SCOPE, None, None)
        return QueryPolicyDecision(
            QueryIntent.EVIDENCE_QUERY,
            routed,
            self.structure_fingerprint(routed, goal),
        )

    @staticmethod
    def query_facets(value: str) -> set[str]:
        normalized = _normalized(value)
        return {
            name for name, terms in _FACETS.items()
            if any(_normalized(term) in normalized for term in terms)
        }

    def excluded_facets(self, goal: str) -> set[str]:
        excluded: set[str] = set()
        for clause in re.split(r"[。！？；;\n]+", goal):
            marker_positions = [clause.find(marker) for marker in _EXCLUSION_MARKERS if marker in clause]
            if marker_positions:
                excluded.update(self.query_facets(clause[min(marker_positions):]))
        return excluded

    def preflight_queries(self, goal: str) -> list[str]:
        """从已写明的目标维度派生有限的受控检索，不调用模型。"""
        topics = self.router.topics(goal)
        if not topics:
            return []
        queries = [topics[0]]
        # 实施安排是唯一需要额外制度证据的复合动作；显式排除时绝不补检索。
        required_facets = self.query_facets(goal) - self.excluded_facets(goal)
        if "implementation" in required_facets:
            queries.append(f"{topics[0]} 实施安排")
        return queries

    def structure_fingerprint(self, query: str, goal: str) -> str:
        """只对检索结构签名，纯近义改写不会制造新的可执行查询。"""
        normalized = _normalized(query)
        facets = sorted(self.query_facets(normalized))
        exact_constraints: list[str] = []
        patterns = (
            r"v\d+(?:\.\d+)*", r"第[一二三四五六七八九十百千万\d]+版",
            r"第[一二三四五六七八九十百千万\d]+条", r"\d{4}[-年]\d{1,2}(?:[-月]\d{1,2}日?)?",
        )
        for pattern in patterns:
            exact_constraints.extend(re.findall(pattern, normalized, flags=re.IGNORECASE))
        canonical_topics = self.router.topics(goal)
        structure = {
            "topic": canonical_topics[0] if canonical_topics else "",
            "facets": facets,
            "exact": sorted(set(exact_constraints)),
        }
        serialized = repr(structure)
        return hash_identifier(serialized)
