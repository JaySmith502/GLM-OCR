"""CLI for the knowledge-bank pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from knowledge_bank.config import Settings
from knowledge_bank.pipeline import KnowledgeBank

app = typer.Typer(
    name="knowledge-bank",
    help="OCR PDFs/images with GLM-OCR, chunk, embed, and store in a local knowledge bank.",
)


def _load_settings(config: Optional[Path]) -> Settings:
    if config:
        return Settings.from_yaml(config)
    return Settings()


@app.command()
def ingest(
    input_path: Path = typer.Argument(..., help="PDF/image file or directory to ingest"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
    extract_to: Optional[Path] = typer.Option(
        None, "--extract-to", "-e", help="Also save .md + .json OCR artifacts to this directory"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print progress details"),
):
    """Ingest documents into the knowledge bank."""
    settings = _load_settings(config)
    bank = KnowledgeBank(settings)

    typer.echo(f"Ingesting {input_path} ...")
    count = bank.ingest(input_path, extract_to=extract_to)
    total = bank.count()
    typer.echo(
        f"Added {count} chunks ({total} total in collection '{settings.vector_store.collection}')."
    )
    if extract_to:
        typer.echo(f"OCR artifacts saved to {extract_to}")


@app.command()
def extract(
    input_path: Path = typer.Argument(..., help="PDF/image file or directory to OCR"),
    output_dir: Path = typer.Option("./extracted", "--output", "-o", help="Output directory"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
):
    """OCR documents and save clean Markdown + JSON artifacts (no embedding)."""
    settings = _load_settings(config)
    bank = KnowledgeBank(settings)

    typer.echo(f"Extracting {input_path} ...")
    written = bank.extract(input_path, output_dir=output_dir)
    typer.echo(f"Saved {len(written)} artifact(s) under {output_dir}:")
    for path in written:
        typer.echo(f"  - {path}")


@app.command()
def query(
    text: str = typer.Argument(..., help="Question or query text"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
):
    """Search the knowledge bank."""
    settings = _load_settings(config)
    bank = KnowledgeBank(settings)

    results = bank.query(text, top_k=top_k)
    if not results:
        typer.echo("No results found.")
        raise typer.Exit(0)

    for idx, (doc, score, meta) in enumerate(results, 1):
        source = Path(meta.get("source", "unknown")).name
        typer.echo(f"\n[{idx}] score={score:.4f} source={source} page={meta.get('page_index', 0)}")
        typer.echo("-" * 40)
        typer.echo(doc)


@app.command()
def chat(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of context chunks per turn"),
):
    """Interactive search over the knowledge bank."""
    settings = _load_settings(config)
    bank = KnowledgeBank(settings)
    typer.echo("Knowledge Bank Chat (empty line to quit)\n")

    while True:
        question = typer.prompt("You")
        if not question.strip():
            break

        results = bank.query(question, top_k=top_k)
        if not results:
            typer.echo("No relevant context found.\n")
            continue

        typer.echo("\nContext:")
        for idx, (doc, score, meta) in enumerate(results, 1):
            source = Path(meta.get("source", "unknown")).name
            typer.echo(f"  [{idx}] {source} p{meta.get('page_index', 0)} ({score:.3f})")
        typer.echo("")


@app.command()
def stats(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
):
    """Show knowledge-bank statistics."""
    settings = _load_settings(config)
    bank = KnowledgeBank(settings)
    typer.echo(f"Collection: {settings.vector_store.collection}")
    typer.echo(f"Store path: {settings.vector_store.path}")
    typer.echo(f"Total chunks: {bank.count()}")


if __name__ == "__main__":
    app()
