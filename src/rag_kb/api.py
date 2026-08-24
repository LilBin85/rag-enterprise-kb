"""弃用入口：只装配新版核心与薄兼容路由，不再承载业务实现。"""

from rag_kb.web.app import create_app


app = create_app(include_legacy_routes=True)


__all__ = ["app"]
