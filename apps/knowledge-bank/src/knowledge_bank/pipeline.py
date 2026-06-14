"""High-level pipeline: OCR → chunk → embed → store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from knowledge_bank.chunker import Chunk, Chunker
from knowledge_bank.config import Settings
from knowledge_bank.ocr_service import OcrResult, OcrService
from knowledge_bank.store import VectorStore


class KnowledgeBank:
    """End-to-end knowledge bank backed by GLM-OCR and Chroma."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunker = Chunker(settings.chunking)
        self.store = VectorStore(settings.embedding, settings.vector_store)

    def _ocr(self, path: Path) -> List[OcrResult]:
        """Run OCR on a single file or directory."""
        with OcrService(self.settings.ocr) as ocr:
            if path.is_file():
                return [ocr.parse(path)]
            return ocr.parse_directory(path)

    def ingest(
        self,
        path: Path | str,
        extract_to: Path | str | None = None,
    ) -> int:
        """OCR, chunk, embed and store a single file or directory.

        If *extract_to* is provided, also save the raw Markdown and JSON OCR
        artifacts alongside ingestion.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        ocr_results = self._ocr(path)

        if extract_to:
            extract_dir = Path(extract_to)
            extract_dir.mkdir(parents=True, exist_ok=True)
            for result in ocr_results:
                self._save_ocr_result(result, extract_dir)

        all_chunks: List[Chunk] = []
        for result in tqdm(ocr_results, desc="Chunking", unit="doc"):
            for page in result.pages:
                all_chunks.extend(
                    self.chunker.chunk(
                        page.markdown, source=result.source, page_index=page.page_index
                    )
                )

        if not all_chunks:
            return 0

        return self.store.add(all_chunks)

    def extract(
        self,
        path: Path | str,
        output_dir: Path | str = "./extracted",
    ) -> List[Path]:
        """OCR a file or directory and save Markdown + JSON artifacts."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        ocr_results = self._ocr(path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written: List[Path] = []
        for result in ocr_results:
            written.extend(self._save_ocr_result(result, output_dir))
        return written

    @staticmethod
    def _save_ocr_result(result: OcrResult, output_dir: Path) -> List[Path]:
        """Save Markdown and JSON for one OCR result."""
        source_path = Path(result.source)
        base_name = source_path.stem
        out_dir = output_dir / base_name
        out_dir.mkdir(parents=True, exist_ok=True)

        written: List[Path] = []

        md_path = out_dir / f"{base_name}.md"
        md_path.write_text(result.markdown, encoding="utf-8")
        written.append(md_path)

        json_path = out_dir / f"{base_name}.json"
        page_raws = [page.raw for page in result.pages]
        # If there is only one page, flatten to the raw dict/list for convenience.
        payload = page_raws[0] if len(page_raws) == 1 else page_raws
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written.append(json_path)

        return written

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """Search the knowledge bank."""
        return self.store.query(text, top_k=top_k)

    def count(self) -> int:
        return self.store.count()
