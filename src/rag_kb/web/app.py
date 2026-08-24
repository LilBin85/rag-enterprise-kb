"""FastAPI 应用工厂与旧路由兼容装配。"""

from fastapi import FastAPI

from rag_kb.web.routers.health import router as health_router
from rag_kb.web.routers.knowledge import router as knowledge_router
from rag_kb.web.routers.documents import router as documents_router
from rag_kb.web.routers.tasks import router as tasks_router


def create_app(*, include_legacy_routes: bool = False) -> FastAPI:
    app = FastAPI(title="企业知识任务智能体", version="1.0.0")
    app.include_router(health_router)
    app.include_router(knowledge_router)
    app.include_router(documents_router)
    app.include_router(tasks_router)

    if include_legacy_routes:
        from rag_kb.web.routers.compatibility import router as compatibility_router

        app.include_router(compatibility_router)
    return app
