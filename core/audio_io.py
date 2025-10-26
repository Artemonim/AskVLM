# Placeholder for audio_io.py

import contextlib
import logging
import time
from collections.abc import Callable
from pathlib import Path

import ffmpeg

from .ffmpeg import extract_audio

logger = logging.getLogger(__name__)

# * Prepare audio for downstream processing
# * If input is not WAV or wrong sample rate/channels, extract and convert


def prepare_audio(
    input_path: Path,
    work_dir: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """Ensure input media is converted to a WAV file with correct specs.

    When ``should_cancel`` is provided, the conversion becomes cancellable and
    responds promptly to cancellation requests.
    """
    # Place intermediates into a hidden work subfolder to avoid cluttering outputs
    work_subdir = work_dir / "_work"
    work_subdir.mkdir(parents=True, exist_ok=True)
    output_wav = work_subdir / f"{input_path.stem}.wav"
    logger.info("Preparing audio: %s -> %s", input_path, output_wav)
    if should_cancel is None:
        extract_audio(
            input_path, output_wav, sample_rate=sample_rate, channels=channels
        )
        return output_wav
    # Cancellable path: run ffmpeg asynchronously and poll for cancel
    if should_cancel is not None and should_cancel():
        msg = "Canceled"
        raise RuntimeError(msg)

    succeeded_local = False
    stream_local = (
        ffmpeg.input(str(input_path))
        .output(
            str(output_wav),
            format="wav",
            acodec="pcm_s16le",
            ac=channels,
            ar=sample_rate,
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
                succeeded_local = True
                break
            time.sleep(0.1)
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
