"""Markdown-aware chunking for OCR output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter

from knowledge_bank.config import ChunkingConfig


@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""

    text: str
    source: str
    page_index: int = 0
    chunk_index: int = 0


class Chunker:
    """Split Markdown OCR text into semantically useful chunks."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._md_splitter = MarkdownTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, markdown: str, source: str, page_index: int = 0) -> List[Chunk]:
        """Chunk a Markdown string and tag each chunk with source metadata."""
        text = (markdown or "").strip()
        if not text:
            return []

        try:
            docs = self._md_splitter.create_documents([text])
        except Exception:
            docs = self._fallback_splitter.create_documents([text])

        chunks: List[Chunk] = []
        for idx, doc in enumerate(docs):
            chunks.append(
                Chunk(
                    text=doc.page_content,
                    source=source,
                    page_index=page_index,
                    chunk_index=idx,
                )
            )
        return chunks
