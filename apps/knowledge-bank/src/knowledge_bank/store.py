"""Embedding + Chroma vector-store ingestion."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.config import DEFAULT_TENANT, Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from knowledge_bank.chunker import Chunk
from knowledge_bank.config import EmbeddingConfig, VectorStoreConfig


class VectorStore:
    """Thin wrapper around ChromaDB with local sentence-transformer embeddings."""

    def __init__(
        self,
        embedding_config: EmbeddingConfig,
        store_config: VectorStoreConfig,
    ):
        self.embedding_config = embedding_config
        self.store_config = store_config
        self._model: SentenceTransformer | None = None
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    def _lazy_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self.embedding_config.model,
                device=self.embedding_config.device,
            )
        return self._model

    def _lazy_client(self) -> chromadb.ClientAPI:
        if self._client is None:
            if self.store_config.path == ":memory:":
                self._client = chromadb.EphemeralClient(
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            else:
                self._client = chromadb.PersistentClient(
                    path=str(Path(self.store_config.path).expanduser().resolve()),
                    settings=ChromaSettings(anonymized_telemetry=False),
                    tenant=DEFAULT_TENANT,
                )
        return self._client

    def close(self) -> None:
        """Release ChromaDB resources."""
        self._collection = None
        if self._client is not None:
            del self._client
            self._client = None
            gc.collect()

    def _lazy_collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self._lazy_client().get_or_create_collection(
                name=self.store_config.collection,
                metadata={"hnsw:space": self.store_config.distance},
            )
        return self._collection

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        model = self._lazy_model()
        return model.encode(
            texts,
            normalize_embeddings=self.embedding_config.normalize_embeddings,
        ).tolist()

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        """Stable ID derived from source + page + chunk content."""
        payload = f"{chunk.source}:{chunk.page_index}:{chunk.chunk_index}:{chunk.text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def add(self, chunks: List[Chunk]) -> int:
        """Embed and upsert chunks into the collection."""
        if not chunks:
            return 0

        collection = self._lazy_collection()
        ids = [self._chunk_id(c) for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = self.embed(texts)
        metadatas = [
            {
                "source": c.source,
                "page_index": c.page_index,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,  # type: ignore[arg-type]
        )
        return len(chunks)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float, Any]]:
        """Return (text, distance, metadata) tuples for the query."""
        collection = self._lazy_collection()
        embedding = self.embed([text])[0]
        results = collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        out: List[Tuple[str, float, Any]] = []
        docs = results.get("documents") or [[]]
        dists = results.get("distances") or [[]]
        metas = results.get("metadatas") or [[]]
        for doc, dist, meta in zip(docs[0], dists[0], metas[0]):
            out.append((doc, float(dist), meta or {}))
        return out

    def count(self) -> int:
        return self._lazy_collection().count()
