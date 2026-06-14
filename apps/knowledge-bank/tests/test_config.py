"""Tests for settings loading."""

import os
import tempfile
from pathlib import Path

from knowledge_bank.config import Settings


def test_default_settings():
    settings = Settings()
    assert settings.ocr.mode == "selfhosted"
    assert settings.ocr.api_port == 8080
    assert settings.chunking.chunk_size == 512
    assert settings.vector_store.collection == "default"


def test_settings_from_yaml():
    yaml_content = """
ocr:
  mode: maas
  api_host: 127.0.0.1
  api_port: 9000
chunking:
  chunk_size: 256
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        path = f.name

    try:
        settings = Settings.from_yaml(path)
        assert settings.ocr.mode == "maas"
        assert settings.ocr.api_port == 9000
        assert settings.chunking.chunk_size == 256
    finally:
        os.unlink(path)


def test_env_overrides():
    os.environ["KB_OCR__MODE"] = "maas"
    os.environ["KB_CHUNKING__CHUNK_SIZE"] = "128"
    try:
        settings = Settings()
        assert settings.ocr.mode == "maas"
        assert settings.chunking.chunk_size == 128
    finally:
        del os.environ["KB_OCR__MODE"]
        del os.environ["KB_CHUNKING__CHUNK_SIZE"]
