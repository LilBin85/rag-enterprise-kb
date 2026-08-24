"""摄取基础设施适配器。"""

from rag_kb.infrastructure.ingestion.local_document_parser import (
    DocumentParseError,
    LocalDocumentParser,
)

__all__ = ["DocumentParseError", "LocalDocumentParser"]
