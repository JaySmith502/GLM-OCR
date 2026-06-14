"""Tests for the OCR service wrapper."""

from unittest.mock import MagicMock, patch

from knowledge_bank.config import OCRConfig
from knowledge_bank.ocr_service import OcrService


def test_ocr_service_uses_selfhosted_defaults():
    config = OCRConfig(mode="selfhosted", api_host="localhost", api_port=8080)
    with patch("knowledge_bank.ocr_service.GlmOcr") as MockGlm:
        instance = MockGlm.return_value
        instance.__enter__ = MagicMock(return_value=instance)
        instance.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.markdown_result = "# Hello\n\nWorld"
        mock_result.json_result = {"blocks": []}
        instance.parse.return_value = mock_result

        with OcrService(config) as svc:
            result = svc.parse("doc.pdf")

        assert result.source == "doc.pdf"
        assert len(result.pages) == 1
        assert "# Hello" in result.markdown
        MockGlm.assert_called_once()
        _, kwargs = MockGlm.call_args
        assert kwargs["mode"] == "selfhosted"
        assert kwargs["_dotted"]["pipeline.maas.enabled"] is False
        assert kwargs["_dotted"]["pipeline.ocr_api.api_port"] == 8080
