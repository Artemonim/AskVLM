# Placeholder for audio_io.py

import contextlib
import logging
import time
from collections.abc import Callable
from pathlib import Path

import ffmpeg

from .ffmpeg import ensure_window_has_audio, extract_audio, window_ffmpeg_kwargs

logger = logging.getLogger(__name__)


def _extract_blocking(
    input_path: Path,
    output_wav: Path,
    *,
    sample_rate: int,
    channels: int,
    start_s: float | None,
    duration_s: float | None,
) -> None:
    """Convert *input_path* to WAV in one blocking ffmpeg call.

    Without a window the call is exactly ``extract_audio(input, output,
    sample_rate=..., channels=...)``; the window arguments are passed on only
    when a window was requested.
    """
    if any(window_ffmpeg_kwargs(start_s, duration_s)):
        extract_audio(
            input_path,
            output_wav,
            sample_rate=sample_rate,
            channels=channels,
            start_s=start_s,
            duration_s=duration_s,
        )
        return
    extract_audio(input_path, output_wav, sample_rate=sample_rate, channels=channels)


# * Prepare audio for downstream processing
# * If input is not WAV or wrong sample rate/channels, extract and convert


def prepare_audio(
    input_path: Path,
    work_dir: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    should_cancel: Callable[[], bool] | None = None,
    *,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> Path:
    """Ensure input media is converted to a WAV file with correct specs.

    When ``should_cancel`` is provided, the conversion becomes cancellable and
    responds promptly to cancellation requests.

    Args:
        input_path: Media file to convert.
        work_dir: Folder that receives the ``_work`` subfolder with the WAV.
        sample_rate: Output sample rate in Hz.
        channels: Output channel count.
        should_cancel: Optional cancel poll; makes the conversion cancellable.
        start_s: Window start in seconds (input seek before ``-i``, applied
            only when positive); ``None`` starts at the beginning.
        duration_s: Window length in seconds; ``None`` reads to the end.

    Returns:
        Path of the prepared WAV; its timestamps are relative to the window
        start.

    Raises:
        RuntimeError: Cancellation was requested or ffmpeg failed.
        ValueError: A window was requested and no audio frames were decoded.

    """
    # Place intermediates into a hidden work subfolder to avoid cluttering outputs
    work_subdir = work_dir / "_work"
    work_subdir.mkdir(parents=True, exist_ok=True)
    output_wav = work_subdir / f"{input_path.stem}.wav"
    logger.info("Preparing audio: %s -> %s", input_path, output_wav)
    if should_cancel is None:
        _extract_blocking(
            input_path,
            output_wav,
            sample_rate=sample_rate,
            channels=channels,
            start_s=start_s,
            duration_s=duration_s,
        )
        return output_wav
    # Cancellable path: run ffmpeg asynchronously and poll for cancel
    if should_cancel is not None and should_cancel():
        msg = "Canceled"
        raise RuntimeError(msg)

    succeeded_local = False
    input_kwargs, output_kwargs = window_ffmpeg_kwargs(start_s, duration_s)
    stream_local = (
        ffmpeg.input(str(input_path), **input_kwargs)
        .output(
            str(output_wav),
            format="wav",
            acodec="pcm_s16le",
            ac=channels,
            ar=sample_rate,
            **output_kwargs,
        )
        .overwrite_output()
        .global_args("-progress", "pipe:2", "-nostats")
    )
    proc_local = stream_local.run_async(
        pipe_stdin=True, pipe_stdout=False, pipe_stderr=True
    )
    try:
        while True:
            if should_cancel is not None and should_cancel():
                with contextlib.suppress(Exception):
                    proc_local.terminate()
                with contextlib.suppress(Exception):
                    proc_local.kill()
                msg = "Canceled"
                raise RuntimeError(msg)
            with contextlib.suppress(Exception):
                if proc_local.stderr is not None:
                    _ = proc_local.stderr.readline()
            ret = proc_local.poll()
            if ret is not None:
                if ret != 0:
                    msg = f"ffmpeg exited with code {ret}"
                    raise RuntimeError(msg)
                break
            time.sleep(0.1)
        # * A window that decodes to zero frames is an error, not an empty transcript.
        ensure_window_has_audio(
            output_wav, input_path, start_s=start_s, duration_s=duration_s
        )
        succeeded_local = True
    finally:
        if not succeeded_local:
            with contextlib.suppress(Exception):
                if output_wav.exists():
                    output_wav.unlink()
    return output_wav


def cleanup_intermediate_audio(input_path: Path, work_dir: Path) -> None:
    """Remove intermediate WAV and empty work directory (best effort)."""
    try:
        work_subdir = work_dir / "_work"
        wav_path = work_subdir / f"{input_path.stem}.wav"
        if wav_path.exists():
            wav_path.unlink()
        if work_subdir.exists() and not any(work_subdir.iterdir()):
            work_subdir.rmdir()
    except OSError as exc:
        # * Best-effort cleanup: log and continue
        logger.debug("Intermediate cleanup failed: %s", exc)
