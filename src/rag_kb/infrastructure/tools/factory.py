"""生产工具白名单装配；不存在动态导入、任意网络或进程执行入口。"""

from pathlib import Path

from rag_kb.application.enterprise_retriever import EnterpriseRetriever
from rag_kb.application.tool_registry import ToolRegistry
from rag_kb.infrastructure.database.session import Database
from rag_kb.infrastructure.object_store.local import LocalObjectStore
from rag_kb.infrastructure.tools.artifacts import LocalArtifactToolAdapter
from rag_kb.infrastructure.tools.read_tools import DatabaseReadToolAdapter


def build_tool_registry(
    *, database: Database, artifact_root: Path, retriever: EnterpriseRetriever
) -> ToolRegistry:
    store = LocalObjectStore(artifact_root)
    return ToolRegistry(
        adapters={
            name: DatabaseReadToolAdapter(
                database,
                operation=name,
                retriever=retriever if name == "search_knowledge" else None,
            )
            for name in (
                "search_knowledge",
                "fetch_evidence",
                "list_document_versions",
                "compare_document_versions",
            )
        }
        | {
            "create_report_artifact": LocalArtifactToolAdapter(
                database, store, artifact_type="report"
            ),
            "create_work_item_draft": LocalArtifactToolAdapter(
                database, store, artifact_type="work_item_draft"
            ),
        }
    )
