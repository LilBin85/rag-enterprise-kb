"""应用组合根；迁移和大型模型初始化不得发生在导入期。"""

from fastapi import FastAPI

from rag_kb.web.app import create_app


def create_application() -> FastAPI:
    return create_app()
