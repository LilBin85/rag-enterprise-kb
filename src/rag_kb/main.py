"""新版生产入口：``uvicorn rag_kb.main:app``。"""

from rag_kb.bootstrap import create_application

app = create_application()
