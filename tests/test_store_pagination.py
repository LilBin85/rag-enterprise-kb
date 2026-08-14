from types import SimpleNamespace

from rag_kb.store import ChunkStore


class PagedClient:
    def __init__(self) -> None:
        self.calls = []
        self.pages = [
            ([SimpleNamespace(payload={"chunk_id": f"chunk-{index}"}) for index in range(256)], "next"),
            ([SimpleNamespace(payload={"chunk_id": f"chunk-{index}"}) for index in range(256, 300)], None),
        ]

    def scroll(self, **kwargs):
        self.calls.append(kwargs)
        return self.pages[len(self.calls) - 1]


def test_scroll_document_consumes_every_qdrant_page() -> None:
    store = object.__new__(ChunkStore)
    store.collection = "offline"
    store.client = PagedClient()

    payloads = store.scroll_document("large-document")

    assert len(payloads) == 300
    assert payloads[0]["chunk_id"] == "chunk-0"
    assert payloads[-1]["chunk_id"] == "chunk-299"
    assert [call["offset"] for call in store.client.calls] == [None, "next"]
    assert all(call["limit"] == 256 for call in store.client.calls)
    condition = store.client.calls[0]["scroll_filter"].must[0]
    assert condition.key == "document_id" and condition.match.value == "large-document"
