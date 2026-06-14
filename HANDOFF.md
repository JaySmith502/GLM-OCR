# Handoff: Local OCR Pipeline for Knowledge Banks

## Goal
Build a local OCR application that preprocesses PDFs before chunking + embedding them into a knowledge bank. The purpose is to reduce API token usage by converting PDFs/images into clean text first, since embedding and LLM operations over text are far cheaper than processing raw PDFs or images.

## Core Pipeline
```
PDF / scanned document
    ↓
GLM-OCR (local or self-hosted)
    ↓
Markdown + layout-aware text
    ↓
chunking
    ↓
embeddings
    ↓
vector store (knowledge bank)
```

## Chosen OCR Engine
**GLM-OCR** (https://github.com/zai-org/GLM-OCR)
- 0.9B parameter multimodal OCR model for complex documents
- Outputs Markdown + JSON layout details
- Strong on tables, formulas, code, seals, complex layouts
- Ranked #1 on OmniDocBench V1.5
- Supports local deployment via vLLM, SGLang, Ollama, MLX (Apple Silicon)
- Also offers a hosted cloud API via Zhipu MaaS

## Recommended Deployment Modes
1. **Self-hosted (recommended for token savings)**
   - `vllm serve zai-org/GLM-OCR --port 8080`
   - Requires GPU; BF16 weights, ~reasonable VRAM for a 0.9B model
   - One-time or local compute cost instead of per-page API fees

2. **Cloud API (quick start)**
   - Get key from https://open.bigmodel.cn
   - SDK acts as thin wrapper; no GPU needed

## SDK / Usage
```bash
pip install glmocr              # cloud/MaaS + local files
pip install "glmocr[selfhosted]" # full local pipeline with layout detection
```

```python
from glmocr import parse

result = parse("document.pdf")
print(result.markdown)   # or result.json_result
result.save(output_dir="./ocr_output")
```

CLI also available:
```bash
glmocr parse document.pdf --output ./ocr_output
```

## Suggested Tech Stack (to decide in next chat)
- **OCR:** GLM-OCR via self-hosted vLLM/SGLang or cloud API
- **Chunking:** langchain-text-splitters, semantic chunker, or custom Markdown splitter
- **Embeddings:** local model (e.g., sentence-transformers, nomic-embed-text, bge) or a small API model
- **Vector store:** Chroma, FAISS, LanceDB, or pgvector
- **CLI / app:** Python CLI with Typer or a small FastAPI service
- **Config:** YAML or pydantic-settings for switching local vs cloud OCR

## Initial Tasks for Next Agent
1. Scaffold a new repo with `pyproject.toml`, README, and source layout.
2. Add GLM-OCR as a dependency and wrap it in a clean OCR service/module.
3. Implement PDF → text extraction (handle single files and directories).
4. Add chunking module.
5. Add embedding module + vector store ingestion.
6. Add CLI entrypoint.
7. Add basic tests and example config.

## Open Decisions
- Local self-hosted vs cloud API for GLM-OCR?
- Which embedding model and vector store?
- CLI-only, API service, or both?
- How to handle multi-page PDFs and output file naming?
- Whether to preserve layout structure in chunks or flatten to plain text?

## Notes
- The current repo (`LocalOCR`) is actually Miso TTS 8B, a text-to-speech model — not OCR. Do not use it for text extraction.
- GLM-OCR was selected because it is open-source, layout-aware, and designed for document understanding.
