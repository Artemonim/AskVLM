from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from core.pipelines import LocalPipeline
from core.ffmpeg import burn_subtitles
from editing.text_model import Document
from utils.exporters import export_document, export_srt_with_rules, SubtitleRules
from core.audio_io import prepare_audio


app = typer.Typer(help="Artemonim's Speech Kit CLI")


@app.command()
def transcribe(
    input_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Input media file or directory"
    ),
    output_dir: Path = typer.Option(
        Path("transcriptions"), "--output", "-o", help="Output directory"
    ),
    whisper_model: str = typer.Option(
        "large-v3",
        help="Whisper/WhisperX model name (default: large-v3)",
    ),
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
        "float16",
        help=(
            "Compute type for faster-whisper: float16|int8|int8_float16|auto. "
            "Default float16 (extreme profile) for best quality on 8+ GiB VRAM."
        ),
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


@app.command()
def subtitle(
    input_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Input media file or directory"
    ),
    output_dir: Path = typer.Option(
        Path("transcriptions"), "--output", "-o", help="Output directory"
    ),
    whisper_model: str = typer.Option(
        "large-v3",
        help="Whisper/WhisperX model name (default: large-v3)",
    ),
    language: Optional[str] = typer.Option(None, help="Language code (optional)"),
    device: str = typer.Option("auto", help="Device: auto|cuda|cpu"),
    compute_type: str = typer.Option(
        "float16",
        help="Compute type for faster-whisper (default float16; consider int8_float16 on 8–12 GiB if OOM)",
    ),
    diarization: bool = typer.Option(False, help="Enable speaker diarization"),
    burn_in: bool = typer.Option(True, help="Burn subtitles into the video"),
    save_srt: bool = typer.Option(True, help="Always save .srt sidecar"),
    format: str = typer.Option("srt", help="Subtitle format: srt|vtt|ass (srt only for burn)"),
    max_cps: float = typer.Option(18.0, help="Max characters per second"),
    max_line_chars: int = typer.Option(42, help="Max characters per line"),
    max_lines: int = typer.Option(2, help="Max lines per cue"),
    min_duration: float = typer.Option(1.2, help="Minimum cue duration (s)"),
    max_duration: float = typer.Option(6.0, help="Maximum cue duration (s)"),
) -> None:
    """Generate subtitles with readability rules and optionally burn them into the video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = LocalPipeline(
        whisper_model=whisper_model,
        engine="whisperx",
        enable_diarization=diarization,
        enable_dialog_blocks=False,
        language=language,
        device=device,
        compute_type=compute_type,
    )
    files = [input_path]
    if input_path.is_dir():
        files = [p for p in input_path.iterdir() if p.is_file()]

    rules = SubtitleRules(
        max_line_chars=max_line_chars,
        max_lines=max_lines,
        min_duration=min_duration,
        max_duration=max_duration,
        max_cps=max_cps,
    )

    for media in files:
        typer.echo(f"Processing {media}...")
        doc: Document = pipeline.process(media, output_dir)
        srt_path = output_dir / f"{media.stem}.srt"
        # Export srt with rules
        srt_text = export_srt_with_rules(doc, rules)
        srt_path.write_text(srt_text, encoding="utf-8")
        if save_srt:
            typer.echo(f"Saved SRT: {srt_path}")
        if burn_in and media.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
            out_video = output_dir / f"{media.stem}_subbed.mp4"
            burn_subtitles(media, srt_path, out_video)
            typer.echo(f"Burned-in video: {out_video}")




def main() -> None:
    app()


if __name__ == "__main__":
    main()


