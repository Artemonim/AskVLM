from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from core.pipelines import LocalPipeline
from editing.text_model import Document
from utils.exporters import export_document


app = typer.Typer(help="Artemonim's Speech Kit CLI")


@app.command()
def transcribe(
    input_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Input media file or directory"
    ),
    output_dir: Path = typer.Option(
        Path("transcriptions"), "--output", "-o", help="Output directory"
    ),
    whisper_model: str = typer.Option("base", help="Whisper/WhisperX model name"),
    engine: str = typer.Option(
        "whisper", help="Engine: whisper | whisperx | auto (prefers faster-whisper)"
    ),
    language: Optional[str] = typer.Option(None, help="Language code (optional)"),
    diarization: bool = typer.Option(True, help="Enable speaker diarization"),
    dialog_blocks: bool = typer.Option(False, help="Format text with LLM"),
    export: str = typer.Option("txt", help="Export format: txt|srt|vtt|json"),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Process directories recursively"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing output files"
    ),
    device: str = typer.Option(
        "auto", help="Device: auto|cuda|cpu (passed to engines where applicable)"
    ),
    compute_type: str = typer.Option(
        "auto", help="Compute type for faster-whisper: auto|float16|int8|float32"
    ),
) -> None:
    """Transcribe a file or directory and export results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # * Normalize engine when auto is requested
    chosen_engine = "whisperx" if engine == "auto" else engine
    pipeline = LocalPipeline(
        whisper_model=whisper_model,
        engine=chosen_engine,
        enable_diarization=diarization,
        enable_dialog_blocks=dialog_blocks,
        language=language,
        device=device,
        compute_type=compute_type,
    )

    files = [input_path]
    if input_path.is_dir():
        if recursive:
            files = [p for p in input_path.rglob("*") if p.is_file()]
        else:
            files = [p for p in input_path.iterdir() if p.is_file()]

    for media in files:
        typer.echo(f"Processing {media}...")
        doc: Document = pipeline.process(media, output_dir)
        out_file = output_dir / f"{media.stem}.{export.lower()}"
        if out_file.exists() and not overwrite:
            typer.echo(f"Exists, skip: {out_file}")
            continue
        export_document(doc, export, out_file)
        typer.echo(f"Saved to {out_file}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()


