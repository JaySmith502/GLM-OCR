"""Tests for Markdown chunking."""

from knowledge_bank.chunker import Chunker
from knowledge_bank.config import ChunkingConfig


def test_chunker_splits_markdown():
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
    chunker = Chunker(config)
    md = "# Heading\n\nParagraph one.\n\n## Subheading\n\nParagraph two."
    chunks = chunker.chunk(md, source="test.md")
    assert len(chunks) > 0
    assert all(c.source == "test.md" for c in chunks)
    assert all(isinstance(c.text, str) for c in chunks)


def test_chunker_returns_empty_for_blank():
    chunker = Chunker(ChunkingConfig())
    assert chunker.chunk("", source="empty.md") == []
