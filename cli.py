from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

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


@dataclass(frozen=True)
class _ExternalChildAttemptResult:
    success_text: Optional[str]
    return_code: int
    stdout: str
    stderr: str


def _normalize_transcript_text(text: Optional[str]) -> str:
    value = text or ""
    return value if value.strip() else ""


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
    ) as temp_file:
        json.dump(payload, temp_file)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_path = Path(temp_file.name)
    temp_path.replace(path)


def _read_child_success_result(path: Path) -> Optional[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("status") != "ok":
        return None
    text = data.get("text")
    if not isinstance(text, str):
        return None
    return _normalize_transcript_text(text)


def _run_external_transcribe_once(  # noqa: PLR0913
    *,
    input_path: Path,
    whisper_model: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    diarization: bool,
    dialog_blocks: bool,
    work_dir: Optional[Path],
    before_close: Optional[Callable[[str], None]] = None,
) -> str:
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
        normalized_text = _normalize_transcript_text(text)
        if before_close is not None:
            before_close(normalized_text)
        return normalized_text
    finally:
        pipeline.close(aggressive=False)


def _build_external_child_command(  # noqa: PLR0913
    *,
    input_path: Path,
    whisper_model: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    diarization: bool,
    dialog_blocks: bool,
    work_dir: Optional[Path],
    child_result_file: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "external-transcribe",
        str(input_path),
        "--whisper-model",
        whisper_model,
        "--device",
        device,
        "--compute-type",
        compute_type,
        "--_internal-child-mode",
        f"--_internal-result-file={child_result_file}",
    ]
    if language is not None:
        command.extend(["--language", language])
    if diarization:
        command.append("--diarization")
    if dialog_blocks:
        command.append("--dialog-blocks")
    if work_dir is not None:
        command.extend(["--work-dir", str(work_dir)])
    return command


def _run_external_transcribe_isolated_attempt(  # noqa: PLR0913
    *,
    input_path: Path,
    whisper_model: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    diarization: bool,
    dialog_blocks: bool,
    work_dir: Optional[Path],
) -> _ExternalChildAttemptResult:
    with tempfile.TemporaryDirectory(prefix="askvlm-ext-ipc-") as ipc_dir:
        child_result_file = Path(ipc_dir) / "child_result.json"
        command = _build_external_child_command(
            input_path=input_path,
            whisper_model=whisper_model,
            language=language,
            device=device,
            compute_type=compute_type,
            diarization=diarization,
            dialog_blocks=dialog_blocks,
            work_dir=work_dir,
            child_result_file=child_result_file,
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "_ASKVLM_CHILD_RESULT_FILE": str(child_result_file)},
        )
        return _ExternalChildAttemptResult(
            success_text=_read_child_success_result(child_result_file),
            return_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


def _is_windows_crash_like_return_code(return_code: int) -> bool:
    if return_code < 0:
        return True
    return (return_code & 0xFFFFFFFF) >= 0xC0000000


_INTERNAL_CHILD_IPC_ERROR = "Internal child mode requires"


def _is_internal_child_ipc_error(stderr: str) -> bool:
    """Return True when child stderr signals a missing IPC result-file path.

    Args:
        stderr: The captured stderr string from the child process, or None.

    Returns:
        True if the IPC setup error marker is present in stderr.
    """
    return _INTERNAL_CHILD_IPC_ERROR in (stderr or "")


def _raise_external_transcribe_error(attempt: _ExternalChildAttemptResult) -> None:
    details = (attempt.stderr or "").strip() or (attempt.stdout or "").strip()
    if details:
        typer.secho(details, err=True)
    raise typer.Exit(code=1)


def _emit_external_transcribe_outputs(
    *, text: str, output_file: Optional[Path], stdout: bool
) -> None:
    if output_file is not None:
        _write_plain_text(output_file, text)
    if stdout and text:
        typer.echo(text)


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
    _internal_child_mode: bool = typer.Option(
        False,
        "--_internal-child-mode",
        hidden=True,
    ),
    _internal_result_file: Optional[Path] = typer.Option(
        None,
        "--_internal-result-file",
        hidden=True,
    ),
) -> None:
    """Transcribe one media file and return plain text for external applications."""
    if not stdout and output_file is None:
        raise typer.BadParameter(
            "Either keep --stdout enabled or provide --output-file."
        )
    # * Env-var fallback: recover IPC path if CLI arg was not parsed
    if _internal_child_mode and _internal_result_file is None:
        _env_path = os.environ.get("_ASKVLM_CHILD_RESULT_FILE")
        if _env_path:
            _internal_result_file = Path(_env_path)

    if _internal_child_mode:
        if _internal_result_file is None:
            raise typer.BadParameter(
                "Internal child mode requires --_internal-result-file."
            )
        try:
            _run_external_transcribe_once(
                input_path=input_path,
                whisper_model=whisper_model,
                language=language,
                device=device,
                compute_type=compute_type,
                diarization=diarization,
                dialog_blocks=dialog_blocks,
                work_dir=work_dir,
                before_close=lambda text: _write_json_atomic(
                    _internal_result_file,
                    {"status": "ok", "text": text},
                ),
            )
            return
        except Exception as exc:
            if not _internal_result_file.exists():
                with contextlib.suppress(Exception):
                    _write_json_atomic(
                        _internal_result_file,
                        {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
            raise

    windows_non_cpu = sys.platform.startswith("win") and device.lower() != "cpu"
    if windows_non_cpu:
        first_attempt = _run_external_transcribe_isolated_attempt(
            input_path=input_path,
            whisper_model=whisper_model,
            language=language,
            device=device,
            compute_type=compute_type,
            diarization=diarization,
            dialog_blocks=dialog_blocks,
            work_dir=work_dir,
        )
        if first_attempt.success_text is not None:
            _emit_external_transcribe_outputs(
                text=first_attempt.success_text,
                output_file=output_file,
                stdout=stdout,
            )
            return

        if (
            not _is_windows_crash_like_return_code(first_attempt.return_code)
            and not _is_internal_child_ipc_error(first_attempt.stderr)
        ):
            _raise_external_transcribe_error(first_attempt)

        retry_attempt = _run_external_transcribe_isolated_attempt(
            input_path=input_path,
            whisper_model=whisper_model,
            language=language,
            device="cpu",
            compute_type=compute_type,
            diarization=diarization,
            dialog_blocks=dialog_blocks,
            work_dir=work_dir,
        )
        if retry_attempt.success_text is not None:
            _emit_external_transcribe_outputs(
                text=retry_attempt.success_text,
                output_file=output_file,
                stdout=stdout,
            )
            return
        _raise_external_transcribe_error(retry_attempt)

    _run_external_transcribe_once(
        input_path=input_path,
        whisper_model=whisper_model,
        language=language,
        device=device,
        compute_type=compute_type,
        diarization=diarization,
        dialog_blocks=dialog_blocks,
        work_dir=work_dir,
        before_close=lambda text: _emit_external_transcribe_outputs(
            text=text,
            output_file=output_file,
            stdout=stdout,
        ),
    )


@app.command("external-extract-frames")
def external_extract_frames(  # noqa: PLR0913
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        help="Video file to extract frames from.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory where extracted frame images are written.",
    ),
    fps: float = typer.Option(
        0.5,
        "--fps",
        help="Target sampling rate in frames per second (default: 0.5).",
        min=0.001,
    ),
    fps_fallback: float = typer.Option(
        0.2,
        "--fps-fallback",
        help="Fallback FPS used when frame-budget would be exceeded (default: 0.2).",
        min=0.001,
    ),
    frame_budget: int = typer.Option(
        20,
        "--frame-budget",
        help=(
            "Maximum number of frames to extract. "
            "When the target FPS would produce more frames, fps-fallback is used instead. "
            "0 disables the cap."
        ),
        min=0,
    ),
    as_json: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Output a JSON object instead of one path per line.",
    ),
) -> None:
    """Extract video frames at adaptive FPS for external vision pipelines.

    Writes frame images to OUTPUT_DIR. Prints extracted frame paths to stdout
    (one per line), or a JSON manifest when --json is used.

    Exit code 0 on success (even if the video has zero frames). Exit code 1 on
    any processing failure.
    """
    import json as _json

    from core.ffmpeg import extract_frames_for_span, get_media_duration_seconds

    output_dir.mkdir(parents=True, exist_ok=True)

    duration_s = get_media_duration_seconds(input_path)
    if duration_s <= 0.0:
        typer.secho(
            f"Warning: could not determine duration of {input_path}; defaulting to 0 frames.",
            err=True,
        )
        if as_json:
            typer.echo(
                _json.dumps({"frames": [], "fps_used": fps, "duration_s": 0.0})
            )
        return

    # * Select FPS: use target unless frame budget would be exceeded
    effective_fps = fps
    if frame_budget > 0:
        estimated = math.ceil(duration_s * fps)
        if estimated > frame_budget:
            effective_fps = fps_fallback

    frame_paths = extract_frames_for_span(
        video_file=input_path,
        start_s=0.0,
        end_s=duration_s,
        output_pattern=output_dir / "frame-%06d.png",
        fps=effective_fps,
    )

    if as_json:
        typer.echo(
            _json.dumps(
                {
                    "frames": [str(p) for p in frame_paths],
                    "fps_used": effective_fps,
                    "duration_s": duration_s,
                }
            )
        )
    else:
        for p in frame_paths:
            typer.echo(str(p))


def main() -> None:
    app()


if __name__ == "__main__":
    main()


