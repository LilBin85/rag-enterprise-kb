"""新 HTTP 接口层；旧 ``rag_kb.api`` 在迁移期继续作为兼容入口。"""

from rag_kb.web.app import create_app

__all__ = ["create_app"]
