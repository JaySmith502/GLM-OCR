# Knowledge Bank

A local OCR-to-knowledge-bank pipeline. It converts PDFs and images into clean Markdown using [GLM-OCR](https://github.com/zai-org/GLM-OCR), then chunks, embeds, and ingests them into a local vector store so downstream RAG / search applications spend fewer tokens on raw PDFs and images.

## Pipeline

```
PDF / image
    ↓
GLM-OCR (self-hosted vLLM/SGLang by default, cloud API configurable)
    ↓
Markdown + layout-aware text
    ↓
Markdown-aware chunker
    ↓
Sentence-transformer embeddings
    ↓
Chroma vector store (knowledge bank)
```

## Install

```bash
cd apps/knowledge-bank

# Local GLM-OCR backend (recommended for token savings)
pip install -e ".[dev]"

# If you want GLM-OCR's layout-detection extras locally
pip install -e ".[dev]" "glmocr[selfhosted]"
```

## Configure

Create `config.yaml` (or set environment variables):

```yaml
ocr:
  mode: selfhosted          # "selfhosted" | "maas"
  api_host: localhost       # vLLM/SGLang host
  api_port: 8080            # vLLM/SGLang port
  # maas:
  #   api_key: ${ZHIPU_API_KEY}

chunking:
  chunk_size: 512
  chunk_overlap: 64

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2

vector_store:
  path: ./kb_chroma
  collection: default
```

Environment variables override YAML values:

```bash
export KB_OCR__MODE=maas
export KB_OCR__MAAS__API_KEY=sk-xxx
export KB_VECTOR_STORE__PATH=./my_kb
```

## Start a local GLM-OCR backend

```bash
# vLLM
vllm serve zai-org/GLM-OCR --port 8080 --served-model-name glm-ocr

# SGLang
SGLANG_ENABLE_SPEC_V2=1 sglang serve --model-path zai-org/GLM-OCR --port 8080 --served-model-name glm-ocr
```

## CLI Usage

```bash
# Ingest a single PDF
kb ingest document.pdf

# Ingest while also saving OCR .md/.json artifacts
kb ingest document.pdf --extract-to ./extracted

# Extract OCR artifacts only (no embedding)
kb extract document.pdf --output ./extracted

# Ingest a directory of PDFs and images
kb ingest ./documents/

# Use a custom config
kb ingest document.pdf --config config.yaml

# Query the knowledge bank
kb query "What are the revenue figures for Q3?"

# Start an interactive chat over the bank
kb chat
```

## Python API

```python
from knowledge_bank.pipeline import KnowledgeBank
from knowledge_bank.config import Settings

settings = Settings()
bank = KnowledgeBank(settings)

bank.ingest("document.pdf")
results = bank.query("What are the revenue figures for Q3?", top_k=5)
for doc, score in results:
    print(score, doc.page_content)
```
