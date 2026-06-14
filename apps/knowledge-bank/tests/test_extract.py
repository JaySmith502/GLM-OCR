"""Tests for OCR artifact extraction."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from knowledge_bank.config import Settings
from knowledge_bank.ocr_service import OcrPage, OcrResult
from knowledge_bank.pipeline import KnowledgeBank


def test_extract_saves_markdown_and_json():
    settings = Settings()
    bank = KnowledgeBank(settings)

    result = OcrResult(
        source="doc.pdf",
        pages=[
            OcrPage(
                source="doc.pdf",
                page_index=0,
                markdown="# Title\n\nBody text.",
                raw={"blocks": [{"text": "Body text."}]},
            )
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "out"
        written = bank._save_ocr_result(result, output_dir)

        assert len(written) == 2
        md_path = output_dir / "doc" / "doc.md"
        json_path = output_dir / "doc" / "doc.json"
        assert md_path.exists()
        assert json_path.exists()
        assert "# Title" in md_path.read_text(encoding="utf-8")
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["blocks"][0]["text"] == "Body text."


def test_extract_command_uses_pipeline():
    settings = Settings()
    bank = KnowledgeBank(settings)

    mock_result = OcrResult(
        source="doc.pdf",
        pages=[OcrPage(source="doc.pdf", page_index=0, markdown="text", raw={})],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        doc_path = Path(tmpdir) / "doc.pdf"
        doc_path.write_text("dummy")
        with patch.object(bank, "_ocr", return_value=[mock_result]):
            written = bank.extract(doc_path, output_dir=tmpdir)
            assert len(written) == 2
            assert all(str(w).startswith(tmpdir) for w in written)
