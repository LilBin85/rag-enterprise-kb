"""在测试直接导入 LangGraph 前触发应用的严格 MsgPack bootstrap。"""

import rag_kb


def pytest_collection_modifyitems(items) -> None:
    integration_files = {
        "test_database_migrations.py",
        "test_stage3_qdrant_component.py",
    }
    for item in items:
        name = item.path.name
        if name.startswith("test_postgres_") or name in integration_files:
            item.add_marker("integration")
