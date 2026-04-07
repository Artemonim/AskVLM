from __future__ import annotations

import contextlib
import importlib
import tempfile
from pathlib import Path
from typing import Any, Optional

import typer

app = typer.Typer(
    help=(
        "AskVLM CLI for local transcription, subtitles, and machine-friendly "
        "external transcription."
    )
)


def _load_cli_runtime() -> dict[str, Any]:
    """Load heavy runtime modules lazily so `--help` stays lightweight."""
    return {
        "LocalPipeline": importlib.import_module("core.pipelines").LocalPipeline,
        "burn_subtitles": importlib.import_module("core.ffmpeg").burn_subtitles,
        "export_document": importlib.import_module("utils.exporters").export_document,
        "export_srt_with_rules": importlib.import_module(
            "utils.exporters"
        ).export_srt_with_rules,
        "SubtitleRules": importlib.import_module("utils.exporters").SubtitleRules,
    }


def _collect_files(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_dir():
        if recursive:
            return [path for path in input_path.rglob("*") if path.is_file()]
        return [path for path in input_path.iterdir() if path.is_file()]
    return [input_path]


def _create_local_pipeline(  # noqa: PLR0913
    *,
    whisper_model: str,
    engine: str,
    diarization: bool,
    dialog_blocks: bool,
    language: Optional[str],
    device: str,
    compute_type: str,
) -> Any:
    runtime = _load_cli_runtime()
    local_pipeline_cls = runtime["LocalPipeline"]
    return local_pipeline_cls(
        whisper_model=whisper_model,
        engine=engine,
        enable_diarization=diarization,
        enable_dialog_blocks=dialog_blocks,
        language=language,
        device=device,
        compute_type=compute_type,
    )


def _write_plain_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        help="Whisper model name for batch transcription (default: large-v3)",
    ),
    engine: str = typer.Option(
        "whisper",
        help=(
            "Backend compatibility hint: whisper | whisperx | auto. The current "
            "local batch pipeline uses the Whisper/Faster-Whisper path."
        ),
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
    runtime = _load_cli_runtime()
    export_document = runtime["export_document"]
    output_dir.mkdir(parents=True, exist_ok=True)
    # * Normalize engine when auto is requested
    chosen_engine = "whisperx" if engine == "auto" else engine
    pipeline = _create_local_pipeline(
        whisper_model=whisper_model,
        engine=chosen_engine,
        diarization=diarization,
        dialog_blocks=dialog_blocks,
        language=language,
        device=device,
        compute_type=compute_type,
    )

    try:
        for media in _collect_files(input_path, recursive=recursive):
            out_file = output_dir / f"{media.stem}.{export.lower()}"
            if out_file.exists() and not overwrite:
                typer.echo(f"Exists, skip: {out_file}")
                continue
            typer.echo(f"Processing {media}...")
            doc = pipeline.process(media, output_dir)
            export_document(doc, export, out_file)
            typer.echo(f"Saved to {out_file}")
    finally:
        with contextlib.suppress(Exception):
            pipeline.close(aggressive=True)


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
        help="Whisper model name for subtitle generation (default: large-v3)",
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
    runtime = _load_cli_runtime()
    export_srt_with_rules = runtime["export_srt_with_rules"]
    subtitle_rules_cls = runtime["SubtitleRules"]
    burn_subtitles = runtime["burn_subtitles"]
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _create_local_pipeline(
        whisper_model=whisper_model,
        engine="whisperx",
        diarization=diarization,
        dialog_blocks=False,
        language=language,
        device=device,
        compute_type=compute_type,
    )
    files = _collect_files(input_path, recursive=False)

    rules = subtitle_rules_cls(
        max_line_chars=max_line_chars,
        max_lines=max_lines,
        min_duration=min_duration,
        max_duration=max_duration,
        max_cps=max_cps,
    )

    try:
        for media in files:
            typer.echo(f"Processing {media}...")
            doc = pipeline.process(media, output_dir)
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
    finally:
        with contextlib.suppress(Exception):
            pipeline.close(aggressive=True)


@app.command("external-transcribe")
def external_transcribe(  # noqa: PLR0913
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        help="Single audio or video file to transcribe for an external caller.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Optional plain-text file to write alongside stdout output.",
    ),
    whisper_model: str = typer.Option(
        "small",
        help="Whisper model name for the external one-shot flow (default: small).",
    ),
    language: Optional[str] = typer.Option(
        None, help="Optional language code, for example: en, ru, de."
    ),
    device: str = typer.Option(
        "auto",
        help=(
            "Preferred device: auto|cuda|cpu. When CUDA memory is exhausted, "
            "the command retries on CPU automatically."
        ),
    ),
    compute_type: str = typer.Option(
        "auto",
        help=(
            "Compute type: auto|float16|int8|int8_float16. 'auto' uses float16 "
            "on CUDA and int8 on CPU."
        ),
    ),
    diarization: bool = typer.Option(
        False,
        "--diarization/--no-diarization",
        help=(
            "Enable speaker diarization. Disabled by default for external calls; "
            "this can require additional GPU memory."
        ),
    ),
    dialog_blocks: bool = typer.Option(
        False,
        "--dialog-blocks/--no-dialog-blocks",
        help="Enable LLM-based text formatting. Disabled by default.",
    ),
    stdout: bool = typer.Option(
        True,
        "--stdout/--no-stdout",
        help="Write only the final transcript text to stdout. Enabled by default.",
    ),
    work_dir: Optional[Path] = typer.Option(
        None,
        "--work-dir",
        help=(
            "Optional directory for intermediate files. When omitted, AskVLM uses "
            "a temporary directory and deletes it after completion."
        ),
    ),
) -> None:
    """Transcribe one media file and return plain text for external applications."""
    if not stdout and output_file is None:
        raise typer.BadParameter(
            "Either keep --stdout enabled or provide --output-file."
        )

    pipeline = _create_local_pipeline(
        whisper_model=whisper_model,
        engine="whisperx",
        diarization=diarization,
        dialog_blocks=dialog_blocks,
        language=language,
        device=device,
        compute_type=compute_type,
    )
    try:
        if work_dir is None:
            with tempfile.TemporaryDirectory(prefix="askvlm-cli-") as temp_dir:
                text = pipeline.process(input_path, Path(temp_dir)).get_full_text()
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            text = pipeline.process(input_path, work_dir).get_full_text()
    finally:
        with contextlib.suppress(Exception):
            pipeline.close(aggressive=True)

    if output_file is not None:
        _write_plain_text(output_file, text)
    if stdout:
        typer.echo(text)




def main() -> None:
    app()


if __name__ == "__main__":
    main()


