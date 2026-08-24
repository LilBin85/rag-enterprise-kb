"""FastAPI 应用工厂。"""

from fastapi import FastAPI

from rag_kb.web.routers.health import router as health_router
from rag_kb.web.routers.knowledge import router as knowledge_router
from rag_kb.web.routers.documents import router as documents_router
from rag_kb.web.routers.tasks import router as tasks_router


def create_app() -> FastAPI:
    app = FastAPI(title="企业知识任务智能体", version="1.0.0")
    app.include_router(health_router)
    app.include_router(knowledge_router)
    app.include_router(documents_router)
    app.include_router(tasks_router)

    return app
