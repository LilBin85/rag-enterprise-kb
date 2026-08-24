"""旧制度比较响应的薄适配；事实读取完全复用新版 ACL 版本工具。"""

from uuid import uuid4

from rag_kb.application.tool_registry import ToolExecutionContext
from rag_kb.domain.identity import AuthContext, Permission
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.tools.read_tools import DatabaseReadToolAdapter
from rag_kb.models import Citation, PolicyChange, PolicyComparison, PolicyComparisonResponse, Refusal, Role


class CompatibilityPolicyComparisonService:
    def __init__(self, database: Database) -> None:
        self._adapter = DatabaseReadToolAdapter(database, operation="compare_document_versions")

    def compare(
        self,
        *,
        auth: AuthContext,
        document_id: str,
        from_version: int,
        to_version: int,
        role: Role,
    ) -> PolicyComparisonResponse:
        run_id = uuid4().hex
        if (
            not auth.has_permission(Permission.DOCUMENT_READ)
            or (str(role) not in auth.roles and "admin" not in auth.roles)
            or from_version <= 0
            or to_version <= from_version
        ):
            return self._refusal(role, run_id)
        context = ToolExecutionContext(
            auth=auth,
            task_id=f"compat-comparison-{run_id}",
            tool_call_id=f"compat-comparison-{run_id}",
            plan_revision=1,
            plan_digest="compatibility",
            evidence_digest="compatibility",
        )
        try:
            result = self._adapter.execute(
                {
                    "document_id": document_id,
                    "left_version": from_version,
                    "right_version": to_version,
                },
                context,
            )
            if not self._adapter.verify(result, context):
                return self._refusal(role, run_id)
            left, right = result["items"]
            changes, citations = self._diff(document_id, left, right)
            if not changes or not citations:
                return self._refusal(role, run_id)
            return PolicyComparisonResponse(
                verdict="comparison",
                comparison=PolicyComparison(
                    document_id=document_id,
                    from_version=from_version,
                    to_version=to_version,
                    changes=changes,
                    summary=f"检测到 {len(changes)} 个章节发生变化。",
                    citations=citations,
                ),
                role=role,
                run_id=run_id,
            )
        except (PermissionError, RuntimeError, ValueError, KeyError, TypeError):
            return self._refusal(role, run_id)

    @staticmethod
    def _diff(document_id: str, left, right):
        before = {item["source_locator"]: item for item in left["chunks"]}
        after = {item["source_locator"]: item for item in right["chunks"]}
        changes: list[PolicyChange] = []
        citations: list[Citation] = []
        for section in sorted(set(before) | set(after)):
            if section not in before:
                change = "added"
            elif section not in after:
                change = "removed"
            elif before[section]["content_hash"] != after[section]["content_hash"]:
                change = "modified"
            else:
                continue
            changes.append(PolicyChange(section=section, change=change))
            for version, item in ((left["version"], before.get(section)), (right["version"], after.get(section))):
                if item is not None:
                    citations.append(
                        Citation(
                            document_id=document_id,
                            document_version=version,
                            source_file_name=item["source_file_name"],
                            source_locator=section,
                            chunk_id=item["chunk_id"],
                        )
                    )
        return changes, citations

    @staticmethod
    def _refusal(role: Role, run_id: str) -> PolicyComparisonResponse:
        return PolicyComparisonResponse(
            verdict="refusal",
            refusal=Refusal(kind="comparison_refused", message="当前无法完成策略版本比较"),
            role=role,
            run_id=run_id,
        )
