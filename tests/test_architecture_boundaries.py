"""架构依赖棘轮：既有边界违规只能减少，不能新增。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "rag_kb"

FORBIDDEN_IMPORTS = {
    "domain": (
        "rag_kb.application",
        "rag_kb.infrastructure",
        "rag_kb.agent",
        "rag_kb.web",
        "rag_kb.worker",
        "rag_kb.main",
    ),
    "application": (
        "rag_kb.infrastructure",
        "rag_kb.web",
        "rag_kb.agent",
        "rag_kb.worker",
        "rag_kb.main",
        "sqlalchemy",
        "qdrant",
        "qdrant_client",
        "psycopg",
    ),
    "infrastructure": (
        "rag_kb.web",
        "rag_kb.worker",
        "rag_kb.main",
    ),
    "agent": (
        "rag_kb.web",
        "rag_kb.worker",
        "rag_kb.main",
    ),
}

# 每项按源模块和目标模块精确允许，迁移后必须立即删除对应项。
TEMPORARY_ALLOWLIST: dict[str, frozenset[str]] = {}


@dataclass(frozen=True, order=True)
class Violation:
    source: str
    target: str
    line: int


def _module_name(path: Path, source_root: Path) -> str:
    parts = list(path.relative_to(source_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return "rag_kb" if not parts else "rag_kb." + ".".join(parts)


def _package_name(path: Path, source_root: Path) -> str:
    module = _module_name(path, source_root)
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def _is_forbidden(module: str, roots: tuple[str, ...]) -> bool:
    return any(module == root or module.startswith(f"{root}.") for root in roots)


def _imported_modules(node: ast.Import | ast.ImportFrom, package: str, roots: tuple[str, ...]):
    if isinstance(node, ast.Import):
        yield from (alias.name for alias in node.names)
        return

    relative = "." * node.level + (node.module or "")
    base = resolve_name(relative, package) if node.level else node.module
    if base:
        yield base
    if base and not _is_forbidden(base, roots):
        for alias in node.names:
            if alias.name != "*":
                yield f"{base}.{alias.name}"


def _violations(source_root: Path = SOURCE_ROOT) -> set[Violation]:
    violations: set[Violation] = set()
    for layer, roots in FORBIDDEN_IMPORTS.items():
        for path in (source_root / layer).rglob("*.py"):
            source = _module_name(path, source_root)
            package = _package_name(path, source_root)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                for target in _imported_modules(node, package, roots):
                    if _is_forbidden(target, roots):
                        violations.add(Violation(source, target, node.lineno))
    return violations


def _allowlisted_violations() -> set[tuple[str, str]]:
    return {
        (source, target)
        for source, targets in TEMPORARY_ALLOWLIST.items()
        for target in targets
    }


def _assert_boundary_ratchet(source_root: Path = SOURCE_ROOT) -> None:
    actual = {(item.source, item.target) for item in _violations(source_root)}
    allowed = _allowlisted_violations()
    unexpected = sorted(actual - allowed)
    stale = sorted(allowed - actual)
    assert not unexpected, f"新增架构边界违规: {unexpected}"
    assert not stale, f"已迁移依赖仍留在 allowlist: {stale}"


def _absolute_from_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None
    }


def test_architecture_boundary_violations_are_exactly_allowlisted() -> None:
    _assert_boundary_ratchet()


def test_retrieval_adapters_depend_on_stable_retrieval_contract() -> None:
    adapters = (
        "infrastructure/database/acl_repository.py",
        "infrastructure/tools/factory.py",
        "infrastructure/tools/read_tools.py",
        "infrastructure/vectorstore/qdrant_authorized_search.py",
    )
    for relative_path in adapters:
        imports = _absolute_from_imports(SOURCE_ROOT / relative_path)
        assert "rag_kb.application.ports.retrieval" in imports
        assert "rag_kb.application.enterprise_retriever" not in imports
        assert "rag_kb.application.retrieval_fusion" not in imports


def test_fast_query_persistence_depends_on_stable_lifecycle_contract() -> None:
    service_imports = _absolute_from_imports(SOURCE_ROOT / "application/query_service.py")
    adapter_imports = _absolute_from_imports(
        SOURCE_ROOT / "infrastructure/database/knowledge_query_lifecycle.py"
    )
    assert "rag_kb.application.ports.knowledge_query" in service_imports
    assert "rag_kb.application.ports.knowledge_query" in adapter_imports


def test_tool_execution_persistence_depends_on_stable_lifecycle_contract() -> None:
    service_imports = _absolute_from_imports(
        SOURCE_ROOT / "application/tool_execution_service.py"
    )
    adapter_imports = _absolute_from_imports(
        SOURCE_ROOT / "infrastructure/database/tool_execution_lifecycle.py"
    )
    assert "rag_kb.application.ports.tool_execution" in service_imports
    assert "rag_kb.application.ports.tool_execution" in adapter_imports


def test_task_worker_persistence_depends_on_stable_worker_contract() -> None:
    worker_imports = _absolute_from_imports(SOURCE_ROOT / "application/task_worker.py")
    adapter_imports = _absolute_from_imports(
        SOURCE_ROOT / "infrastructure/database/task_execution.py"
    )
    assert "rag_kb.application.ports.task_worker" in worker_imports
    assert "rag_kb.application.ports.task_worker" in adapter_imports


def test_agent_layer_consumes_stable_application_contract() -> None:
    worker_imports = _absolute_from_imports(SOURCE_ROOT / "application/task_worker.py")
    assert "rag_kb.application.agent_contract" in worker_imports
    for shim in ("agent/context.py", "agent/result.py", "agent/tools.py"):
        imports = _absolute_from_imports(SOURCE_ROOT / shim)
        assert "rag_kb.application.agent_contract" in imports


def test_document_ingestion_adapters_depend_on_stable_contracts() -> None:
    adapters = {
        "infrastructure/database/document_repository.py": {
            "rag_kb.application.ports.documents",
            "rag_kb.application.ports.indexing",
        },
        "infrastructure/database/document_lifecycle.py": {
            "rag_kb.application.ports.documents",
            "rag_kb.application.ports.indexing",
        },
        "infrastructure/database/index_projection_metadata.py": {
            "rag_kb.application.ports.indexing",
        },
        "infrastructure/database/projection_repository.py": {
            "rag_kb.application.ports.documents",
            "rag_kb.application.ports.indexing",
        },
        "infrastructure/ingestion/local_document_parser.py": {
            "rag_kb.application.ports.indexing",
            "rag_kb.application.ports.object_store",
        },
        "infrastructure/object_store/local.py": {
            "rag_kb.application.ports.object_store",
        },
        "infrastructure/vectorstore/qdrant_collection.py": {
            "rag_kb.application.ports.indexing",
        },
        "infrastructure/vectorstore/qdrant_document_index.py": {
            "rag_kb.application.ports.indexing",
        },
    }
    implementations = {
        "rag_kb.application.document_service",
        "rag_kb.application.ingestion_worker",
        "rag_kb.application.upload_reconciler",
        "rag_kb.application.projection_reconciler",
    }
    for relative_path, ports in adapters.items():
        imports = _absolute_from_imports(SOURCE_ROOT / relative_path)
        assert ports <= imports
        assert not implementations.intersection(imports)


def test_architecture_ratchet_rejects_temporary_reverse_dependency(tmp_path: Path) -> None:
    temporary_root = tmp_path / "src" / "rag_kb"
    application = temporary_root / "application"
    application.mkdir(parents=True)
    (application / "new_dependency.py").write_text(
        "from rag_kb.infrastructure.database import session\n",
        encoding="utf-8",
    )

    try:
        _assert_boundary_ratchet(temporary_root)
    except AssertionError as error:
        assert "rag_kb.application.new_dependency" in str(error)
    else:
        raise AssertionError("临时 application -> infrastructure 依赖未被架构棘轮拒绝")
