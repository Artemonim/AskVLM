# Placeholder for audio_io.py

import contextlib
import errno
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Self

import ffmpeg

from .ffmpeg import ensure_window_has_audio, extract_audio, window_ffmpeg_kwargs

logger = logging.getLogger(__name__)

# * ffmpeg exit codes that mean "out of memory", as Windows reports them
# * (unsigned 32-bit): the loader could not commit memory for the process
# * (STATUS_COMMITMENT_LIMIT, for example under a Job Object cap), an
# * allocation failed with STATUS_NO_MEMORY, or ffmpeg itself returned
# * AVERROR(ENOMEM) = -12.
_FFMPEG_MEMORY_EXIT_CODES: dict[int, str] = {
    0xC000012D: "STATUS_COMMITMENT_LIMIT",
    0xC0000017: "STATUS_NO_MEMORY",
    0xFFFFFFF4: "ENOMEM",
}


class FfmpegOutOfMemoryError(MemoryError, RuntimeError):
    """ffmpeg exited with an exit code that means it ran out of memory.

    It is a :class:`MemoryError`, so the STT pool reports the job as
    ``memory_limit`` (retried on a fresh worker with a larger reservation),
    and a :class:`RuntimeError`, like every other ffmpeg failure of
    :func:`prepare_audio`, so existing handlers keep catching it.

    Attributes:
        exit_code: ffmpeg's exit code as the process reported it.
        status: Name of the memory status the exit code stands for.

    """

    def __init__(self, exit_code: int, status: str) -> None:
        """Describe the failed ffmpeg run.

        Args:
            exit_code: ffmpeg's exit code as the process reported it.
            status: Name of the memory status the exit code stands for.

        """
        self.exit_code = exit_code
        self.status = status
        super().__init__(
            f"ffmpeg exited with code {exit_code} (0x{exit_code & 0xFFFFFFFF:08X}, "
            f"{status}): out of memory"
        )

    def __reduce__(self) -> tuple[type[Self], tuple[int, str]]:
        """Pickle with the constructor arguments rather than the message."""
        return (type(self), (self.exit_code, self.status))


def _ffmpeg_exit_error(code: int) -> RuntimeError:
    """Return the exception for ffmpeg's non-zero exit *code*.

    Windows reports exit codes as unsigned 32-bit values. POSIX ``-ENOMEM``
    also denotes a memory failure by the external transcriber contract.
    SIGKILL alone supplies no OOM evidence: only the pool's worker cgroup
    counters can attribute such a failure to memory exhaustion.

    Args:
        code: Non-zero exit code of the ffmpeg process.

    Returns:
        :class:`FfmpegOutOfMemoryError` for a memory-exhaustion code, else a
        plain :class:`RuntimeError`.

    """
    lookup = code & 0xFFFFFFFF if sys.platform == "win32" else code
    status = (
        "ENOMEM"
        if sys.platform != "win32" and code == -errno.ENOMEM
        else _FFMPEG_MEMORY_EXIT_CODES.get(lookup)
    )
    if status is not None:
        return FfmpegOutOfMemoryError(code, status)
    return RuntimeError(f"ffmpeg exited with code {code}")


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
        FfmpegOutOfMemoryError: On the cancellable path, ffmpeg exited with an
            exit code that means it ran out of memory (also a
            :class:`RuntimeError`).
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
                    raise _ffmpeg_exit_error(ret)
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
