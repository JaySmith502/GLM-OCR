"""Pydantic settings for the knowledge-bank pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Type

import yaml
from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class OCRMaasConfig(BaseSettings):
    """Cloud API (Zhipu MaaS) settings."""

    model_config = SettingsConfigDict(env_prefix="KB_OCR__MAAS__", extra="ignore")

    api_key: Optional[str] = Field(default=None, description="Zhipu MaaS API key")
    api_url: Optional[str] = Field(default=None, description="Optional custom MaaS endpoint")
    verify_ssl: bool = Field(default=True)


class OCRConfig(BaseSettings):
    """OCR backend settings."""

    model_config = SettingsConfigDict(env_prefix="KB_OCR__", extra="ignore")

    mode: Literal["selfhosted", "maas"] = Field(default="selfhosted")
    api_host: str = Field(default="localhost")
    api_port: int = Field(default=8080)
    maas: OCRMaasConfig = Field(default_factory=OCRMaasConfig)


class ChunkingConfig(BaseSettings):
    """Text chunking settings."""

    model_config = SettingsConfigDict(env_prefix="KB_CHUNKING__", extra="ignore")

    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)


class EmbeddingConfig(BaseSettings):
    """Embedding model settings."""

    model_config = SettingsConfigDict(env_prefix="KB_EMBEDDING__", extra="ignore")

    model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    device: str = Field(default="cpu")
    normalize_embeddings: bool = Field(default=True)


class VectorStoreConfig(BaseSettings):
    """Vector store settings."""

    model_config = SettingsConfigDict(env_prefix="KB_VECTOR_STORE__", extra="ignore")

    path: str = Field(default="./kb_chroma")
    collection: str = Field(default="default")
    distance: Literal["cosine", "l2", "ip"] = Field(default="cosine")


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="KB_",
        env_nested_delimiter="__",
        extra="ignore",
        yaml_file="config.yaml",
    )

    ocr: OCRConfig = Field(default_factory=OCRConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    @field_validator("ocr", "chunking", "embedding", "vector_store", mode="before")
    @classmethod
    def _none_to_default(cls, value):
        return value if value is not None else {}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = getattr(settings_cls.model_config, "yaml_file", "config.yaml")
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Settings":
        """Load settings from a YAML file, then overlay env vars."""
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
