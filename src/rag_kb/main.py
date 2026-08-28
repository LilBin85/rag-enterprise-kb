"""生产入口：``uvicorn rag_kb.main:app``。"""

from rag_kb.web.app import create_app

app = create_app()
