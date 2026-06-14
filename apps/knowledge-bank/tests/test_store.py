"""Tests for the Chroma-backed vector store."""

from knowledge_bank.chunker import Chunk
from knowledge_bank.config import EmbeddingConfig, VectorStoreConfig
from knowledge_bank.store import VectorStore


def test_store_add_and_query():
    emb = EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    store_cfg = VectorStoreConfig(path=":memory:", collection="test")
    store = VectorStore(emb, store_cfg)

    chunks = [
        Chunk(text="The cat sat on the mat.", source="doc.md", page_index=0, chunk_index=0),
        Chunk(text="Dogs are loyal animals.", source="doc.md", page_index=0, chunk_index=1),
    ]
    assert store.add(chunks) == 2
    assert store.count() == 2

    results = store.query("cat", top_k=2)
    assert len(results) == 2
    texts = [r[0] for r in results]
    assert "The cat sat on the mat." in texts

    store.close()
