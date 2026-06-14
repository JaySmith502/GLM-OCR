"""Thin wrapper around GLM-OCR that handles both self-hosted and cloud backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from glmocr import GlmOcr

from knowledge_bank.config import OCRConfig


@dataclass
class OcrPage:
    """One page of OCR output."""

    source: str
    page_index: int
    markdown: str
    raw: Union[dict, list]


@dataclass
class OcrResult:
    """OCR output for a single document."""

    source: str
    pages: List[OcrPage]

    @property
    def markdown(self) -> str:
        return "\n\n".join(page.markdown for page in self.pages)


class OcrService:
    """Wrap GLM-OCR with pipeline-friendly defaults."""

    def __init__(self, config: OCRConfig):
        self.config = config
        dotted: dict = {}
        if config.mode == "selfhosted":
            dotted["pipeline.maas.enabled"] = False
            dotted["pipeline.ocr_api.api_host"] = config.api_host
            dotted["pipeline.ocr_api.api_port"] = config.api_port
        else:
            dotted["pipeline.maas.enabled"] = True
            if config.maas.api_url:
                dotted["pipeline.maas.api_url"] = config.maas.api_url
            if config.maas.api_key:
                dotted["pipeline.maas.api_key"] = config.maas.api_key
            dotted["pipeline.maas.verify_ssl"] = config.maas.verify_ssl

        self._parser = GlmOcr(mode=config.mode, _dotted=dotted)

    def __enter__(self) -> "OcrService":
        self._parser.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._parser.__exit__(exc_type, exc_val, exc_tb)

    def parse(
        self,
        source: Union[str, Path],
    ) -> OcrResult:
        """Run OCR on a single PDF/image or a directory of files.

        A list of paths is treated by GLM-OCR as pages of one document, so we
        process one file at a time to keep documents separate.
        """
        source = str(source)
        result = self._parser.parse(source)
        pages: List[OcrPage] = []
        # GlmOcr returns a single result object for one path. For multi-page
        # PDFs the json_result may be a list; markdown_result is one string.
        raw_json = result.json_result
        if isinstance(raw_json, list):
            raw = raw_json
        else:
            raw = [raw_json]
        pages.append(
            OcrPage(source=source, page_index=0, markdown=result.markdown_result or "", raw=raw)
        )
        return OcrResult(source=source, pages=pages)

    def parse_directory(
        self,
        directory: Union[str, Path],
        extensions: tuple[str, ...] = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"),
    ) -> List[OcrResult]:
        """Run OCR on every supported file under *directory*."""
        directory = Path(directory)
        files: List[Path] = []
        seen: set = set()
        for ext in extensions:
            for path in directory.rglob(f"*{ext}"):
                if path not in seen:
                    seen.add(path)
                    files.append(path)
            for path in directory.rglob(f"*{ext.upper()}"):
                if path not in seen:
                    seen.add(path)
                    files.append(path)
        files.sort()
        return [self.parse(f) for f in files]
