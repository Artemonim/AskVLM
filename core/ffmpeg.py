# Placeholder for ffmpeg.py

import contextlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import cast

import ffmpeg

# * Extract audio from media file and convert to WAV format
# * This function uses ffmpeg-python to produce a PCM 16-bit WAV file


def extract_audio(
    input_file: str | Path,
    output_file: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
    *,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> None:
    """Extract audio from input file and save as WAV with specified sample rate and channels.

    Args:
        input_file: Media file to read.
        output_file: WAV file to write.
        sample_rate: Output sample rate in Hz.
        channels: Output channel count.
        start_s: Window start in seconds; applied as an input seek (``-ss``
            before ``-i``) only when positive.
        duration_s: Window length in seconds (``-t``); ``None`` reads to the end.

    Raises:
        ValueError: A window was requested and the decoded WAV has no audio
            frames (for example the window starts after the end of the media).

    """
    input_kwargs, output_kwargs = window_ffmpeg_kwargs(start_s, duration_s)
    ffmpeg.input(str(input_file), **input_kwargs).output(
        str(output_file),
        format="wav",
        acodec="pcm_s16le",
        ac=channels,
        ar=sample_rate,
        **output_kwargs,
    ).overwrite_output().run(quiet=True)
    ensure_window_has_audio(
        output_file, input_file, start_s=start_s, duration_s=duration_s
    )


def window_ffmpeg_kwargs(
    start_s: float | None, duration_s: float | None
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the ffmpeg-python option kwargs of a media window.

    Times are passed as fixed-point strings (``"1.500"``), never as the float
    ``repr`` (``1e-05``, ``1e+16``) that the ffmpeg time parser may reject.

    Args:
        start_s: Window start in seconds; seeks only when positive.
        duration_s: Window length in seconds; limits the output when set.

    Returns:
        ``(input_kwargs, output_kwargs)``: ``ss`` is an input option (the seek
        happens before ``-i``) and ``t`` an output option (``-t`` after
        ``-i``); both dicts are empty when no window is requested.

    """
    input_kwargs: dict[str, str] = {}
    if start_s is not None and start_s > 0:
        input_kwargs["ss"] = f"{start_s:.3f}"
    output_kwargs: dict[str, str] = {}
    if duration_s is not None:
        output_kwargs["t"] = f"{duration_s:.3f}"
    return input_kwargs, output_kwargs


def ensure_window_has_audio(
    wav_path: str | Path,
    input_file: str | Path,
    *,
    start_s: float | None,
    duration_s: float | None,
) -> None:
    """Fail when a requested media window produced a WAV without audio frames.

    Does nothing when no window was requested (no positive ``start_s`` and no
    ``duration_s``).

    Args:
        wav_path: WAV file written by ffmpeg for the window.
        input_file: Media file the window was cut from (for the message).
        start_s: Requested window start in seconds.
        duration_s: Requested window length in seconds.

    Raises:
        ValueError: The WAV has zero frames (or is empty) or cannot be parsed;
            an empty file is removed on a best-effort basis.

    """
    if not any(window_ffmpeg_kwargs(start_s, duration_s)):
        return
    length = "to the end" if duration_s is None else f"{duration_s:g} s long"
    window = f"the window starting at {start_s or 0.0:g} s ({length})"
    try:
        with wave.open(str(wav_path), "rb") as handle:
            frames = handle.getnframes()
    except EOFError:
        frames = 0
    except wave.Error as exc:
        msg = f"cannot read the WAV decoded from {Path(input_file)} for {window}: {exc}"
        raise ValueError(msg) from exc
    if frames > 0:
        return
    with contextlib.suppress(OSError):
        Path(wav_path).unlink()
    msg = (
        f"no audio decoded from {Path(input_file)} for {window}; "
        "the window may lie beyond the end of the media"
    )
    raise ValueError(msg)


logger = logging.getLogger(__name__)


def get_media_duration_seconds(path: str | Path) -> float:
    """Return media duration in seconds using ffmpeg.probe; returns 0.0 on failure."""
    try:
        info = ffmpeg.probe(str(Path(path).resolve()))
        fmt = info.get("format", {})
        dur = fmt.get("duration")
        return float(dur) if dur is not None else 0.0
    except Exception as exc:  # noqa: BLE001
        logger.debug("ffprobe duration failed: %s", exc)
        return 0.0


class MediaProbeError(RuntimeError):
    """ffprobe could not report a valid media duration."""


# * Hide the console window of ffprobe children on Windows.
_NO_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
_PROBE_STDERR_TAIL_CHARS = 500


def _parse_probe_duration(stdout: bytes) -> float:
    """Return the ``format.duration`` of ffprobe JSON output.

    Raises:
        MediaProbeError: When the output is not JSON with a finite,
            non-negative ``format.duration``.

    """
    try:
        info = json.loads(stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        msg = "ffprobe returned output that is not JSON"
        raise MediaProbeError(msg) from exc
    fmt = info.get("format") if isinstance(info, dict) else None
    raw = fmt.get("duration") if isinstance(fmt, dict) else None
    if raw is None:
        msg = "ffprobe reported no format duration"
        raise MediaProbeError(msg)
    try:
        duration = float(raw)
    except (TypeError, ValueError) as exc:
        msg = f"ffprobe reported an unparsable duration {raw!r}"
        raise MediaProbeError(msg) from exc
    if not math.isfinite(duration) or duration < 0:
        msg = f"ffprobe reported an invalid duration {raw!r}"
        raise MediaProbeError(msg)
    return duration


def probe_media_duration(path: str | Path, *, timeout_s: float = 10.0) -> float:
    """Return the media duration in seconds reported by ffprobe.

    Runs ``ffprobe -v error -show_entries format=duration -of json <path>`` as
    a child process (no shell, no console window on Windows); the child is
    killed when it outlives *timeout_s*.

    Args:
        path: Media file; passed as an absolute path so it is never read as
            an ffprobe option.
        timeout_s: Wall-clock limit of the probe in seconds.

    Returns:
        The finite, non-negative duration in seconds.

    Raises:
        MediaProbeError: ffprobe is missing, cannot start, times out (including
            a non-positive *timeout_s*), exits non-zero, or reports no valid
            duration.

    """
    if not (math.isfinite(timeout_s) and timeout_s > 0):
        msg = f"no time left to probe the media duration (timeout_s={timeout_s!r})"
        raise MediaProbeError(msg)
    executable = shutil.which("ffprobe")
    if executable is None:
        msg = "ffprobe is not on PATH"
        raise MediaProbeError(msg)
    target = os.path.abspath(os.fspath(path))  # noqa: PTH100 - no filesystem access
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        target,
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - resolved ffprobe, fixed argv, no shell
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            creationflags=_NO_WINDOW_FLAGS,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"ffprobe timed out after {timeout_s:.1f} s on {target}"
        raise MediaProbeError(msg) from exc
    except OSError as exc:
        msg = f"ffprobe could not start: {exc}"
        raise MediaProbeError(msg) from exc
    if completed.returncode != 0:
        tail = completed.stderr.decode("utf-8", errors="replace").strip()
        msg = (
            f"ffprobe exited with code {completed.returncode} on {target}: "
            f"{tail[-_PROBE_STDERR_TAIL_CHARS:]}"
        )
        raise MediaProbeError(msg)
    return _parse_probe_duration(completed.stdout)


def burn_subtitles(
    video_file: str | Path,
    subtitle_file: str | Path,
    output_file: str | Path,
    force_style: str | None = None,
    *,
    autoscale: bool = True,
    normalize_audio: bool = True,
    font_name: str | None = None,
) -> None:
    """Burn subtitles into a video using ffmpeg's subtitles filter.

    Uses libass; ensure ffmpeg build supports it. The `force_style` string is passed
    directly to the subtitles filter. When `normalize_audio` is True, applies
    EBU R128 loudness normalization. When `font_name` is provided, forces libass
    to use that font family for subtitles rendering.
    """
    # * Build a safe subtitles filter using forward slashes to avoid Windows backslash escapes
    v = Path(video_file).resolve()
    s = Path(subtitle_file).resolve()
    o = Path(output_file).resolve()

    # Determine video height for auto font sizing
    auto_style: str | None = None
    if autoscale and not (force_style and "Fontsize=" in force_style):
        try:
            probe = ffmpeg.probe(str(v))
            vstreams = [
                st for st in probe.get("streams", []) if st.get("codec_type") == "video"
            ]
            if vstreams:
                height = int(vstreams[0].get("height", 0))
                if height > 0:
                    # ~3% of height, clamped to [20, 38]
                    size = max(20, min(38, round(height * 0.03)))
                    auto_style = f"Fontsize={size},Outline=2,Shadow=0"
        except (OSError, ffmpeg.Error, ValueError) as exc:
            # Fallback to conservative size if probe fails
            logger.debug("ffmpeg probe failed: %s", exc)
            auto_style = "Fontsize=28,Outline=2,Shadow=0"

    # Merge provided style with auto sizing and optional font family
    style_parts = []
    if force_style:
        style_parts.append(force_style)
    elif auto_style:
        style_parts.append(auto_style)
    else:
        style_parts.append("Fontsize=28,Outline=2,Shadow=0")
    if font_name:
        style_parts.append(f"FontName={font_name}")
    style = ",".join(style_parts)

    # Use POSIX-style path to avoid backslash escaping in filtergraph
    sub_path = str(s).replace("\\", "/")

    # Prepare streams
    inp = ffmpeg.input(str(v))
    video_f = inp.video.filter("subtitles", filename=sub_path, force_style=style)
    # Normalize audio to -16 LUFS, True Peak -3 dB (broadcast-ish target)
    if normalize_audio:
        audio_f = inp.audio.filter_(
            "loudnorm",
            I=-16,
            TP=-3.0,
            LRA=11,
            dual_mono="true",
            print_format="summary",
        )
    else:
        audio_f = inp.audio

    (
        ffmpeg.output(
            video_f,
            audio_f,
            str(o),
            vcodec="libx264",
            preset="veryfast",
            crf=18,
            acodec="aac",
            audio_bitrate="192k",
            movflags="+faststart",
        )
        .overwrite_output()
        .run(quiet=False)
    )


def start_burn_process(
    video_file: str | Path,
    subtitle_file: str | Path,
    output_file: str | Path,
    force_style: str | None = None,
    *,
    autoscale: bool = True,
    normalize_audio: bool = True,
    font_name: str | None = None,
    progress_path: Path | None = None,
) -> subprocess.Popen[bytes]:
    """Start a cancellable ffmpeg burn-in process and return the Popen handle.

    If `progress_path` is provided, ffmpeg writes structured progress to that file.
    """
    v = Path(video_file).resolve()
    s = Path(subtitle_file).resolve()
    o = Path(output_file).resolve()

    # Determine size & build style as in burn_subtitles
    auto_style: str | None = None
    if autoscale and not (force_style and "Fontsize=" in force_style):
        try:
            probe = ffmpeg.probe(str(v))
            vstreams = [
                st for st in probe.get("streams", []) if st.get("codec_type") == "video"
            ]
            if vstreams:
                height = int(vstreams[0].get("height", 0))
                if height > 0:
                    size = max(20, min(38, round(height * 0.03)))
                    auto_style = f"Fontsize={size},Outline=2,Shadow=0"
        except (OSError, ffmpeg.Error, ValueError) as exc:
            logger.debug("ffmpeg probe failed: %s", exc)
            auto_style = "Fontsize=28,Outline=2,Shadow=0"

    style_parts = []
    if force_style:
        style_parts.append(force_style)
    elif auto_style:
        style_parts.append(auto_style)
    else:
        style_parts.append("Fontsize=28,Outline=2,Shadow=0")
    if font_name:
        style_parts.append(f"FontName={font_name}")
    style = ",".join(style_parts)

    sub_path = str(s).replace("\\", "/")
    inp = ffmpeg.input(str(v))
    video_f = inp.video.filter("subtitles", filename=sub_path, force_style=style)
    audio_stream = (
        inp.audio.filter_(
            "loudnorm",
            I=-16,
            TP=-3.0,
            LRA=11,
            dual_mono="true",
            print_format="summary",
        )
        if normalize_audio
        else inp.audio
    )

    stream = ffmpeg.output(
        video_f,
        audio_stream,
        str(o),
        vcodec="libx264",
        preset="veryfast",
        crf=18,
        acodec="aac",
        audio_bitrate="192k",
        movflags="+faststart",
    ).overwrite_output()
    if progress_path is not None:
        # Write structured progress into file for external polling
        stream = stream.global_args("-progress", str(progress_path), "-nostats")
    proc = stream.run_async(pipe_stdin=True, pipe_stdout=False, pipe_stderr=False)
    return cast("subprocess.Popen[bytes]", proc)


def extract_frame_to_file(
    video_file: str | Path,
    timestamp_s: float,
    output_file: str | Path,
) -> Path:
    """Extract a single video frame at `timestamp_s` seconds into `output_file`.

    The output format is inferred from `output_file` extension (e.g., PNG/JPG).
    Returns the absolute `Path` to the written file.
    """
    v = Path(video_file).resolve()
    out = Path(output_file).resolve()
    # * Use accurate seek after input for better precision; fast seek is fine for preview
    (
        ffmpeg.input(str(v), ss=max(0.0, float(timestamp_s)))
        .output(str(out), vframes=1, q=2)
        .overwrite_output()
        .run(quiet=True)
    )
    return out


# * ffmpeg colorspace fallbacks for frame extraction. Some containers tag frames
# * with a color matrix that libswscale cannot map to RGB for the image encoder,
# * which fails with "Invalid color space" and wipes the extracted frames. The
# * plain attempt is tried first; on failure each strategy forces a known-good
# * colorspace / pixel format before the encoder's implicit RGB conversion.
# * Strategies are attempted in order until one yields frames.
_FRAME_EXTRACT_VF_FALLBACKS: tuple[str, ...] = (
    "",
    # * Reset a bogus/unspecified frame colorspace tag to BT.709 before conversion.
    "setparams=colorspace=bt709,format=yuv420p",
    # * Force the swscale matrices on both sides for SD-tagged content.
    "scale=w=iw:h=ih:in_color_matrix=bt709:out_color_matrix=bt709",
    # * Normalize the pixel format while preserving otherwise valid color metadata.
    "format=yuv420p",
    # * Override every invalid input color property only after conservative fixes fail.
    "colorspace=iall=bt709:all=bt709:format=yuv420p:fast=1",
)


def _frame_extract_filtergraph(fps: float, colorspace_fix: str) -> str:
    """Build the ``-vf`` filtergraph for one extraction attempt.

    Args:
        fps: Sampling rate for the span.
        colorspace_fix: Extra filter chain appended after ``fps``, or ``""``.

    Returns:
        The linear ffmpeg filtergraph string.

    """
    base = f"fps={max(0.001, float(fps))}"
    return f"{base},{colorspace_fix}" if colorspace_fix else base


def _clear_extracted_frames(out_dir: Path, prefix: str, suffix: str) -> None:
    """Remove any frames matching the numbered output pattern in *out_dir*."""
    for existing in sorted(out_dir.glob(f"{prefix}*{suffix}")):
        try:
            existing.unlink()
        except OSError:
            logger.debug("Could not remove stale extracted frame: %s", existing)


def _ffmpeg_error_text(exc: ffmpeg.Error) -> str:
    """Return a bounded, human-readable tail of an ffmpeg error's stderr."""
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", errors="replace")[-500:]
    return str(stderr or exc)[-500:]


def _run_frame_extraction_once(
    video_file: Path,
    start_s: float,
    duration_s: float,
    output_pattern: Path,
    filtergraph: str,
) -> None:
    """Invoke ffmpeg for a single frame-extraction attempt (raises on failure)."""
    (
        ffmpeg.input(str(video_file), ss=max(0.0, float(start_s)), t=duration_s)
        .output(str(output_pattern), start_number=1, vf=filtergraph)
        .overwrite_output()
        .run(quiet=True)
    )


def extract_frames_for_span(
    video_file: str | Path,
    start_s: float,
    end_s: float,
    output_pattern: str | Path,
    *,
    fps: float = 2.0,
) -> tuple[Path, ...]:
    """Extract multiple frames for a time span into numbered files.

    Frames are sampled with ffmpeg. When a container tags frames with a color
    matrix that libswscale cannot map to RGB (``Invalid color space``), the
    extraction is retried with progressively stronger colorspace-normalizing
    filter chains so a single problematic colorspace no longer wipes the whole
    span.

    Args:
        video_file: Input video path.
        start_s: Inclusive span start in seconds.
        end_s: Exclusive span end in seconds.
        output_pattern: Numbered output pattern such as ``chunk-%03d.png``.
        fps: Sampling rate for the span.

    Returns:
        Absolute output paths sorted by filename.

    Raises:
        ffmpeg.Error: When every colorspace strategy fails to run ffmpeg.

    """
    v = Path(video_file).resolve()
    out_pattern = Path(output_pattern).resolve()
    out_dir = out_pattern.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_pattern.stem
    if "%" not in stem:
        msg = "Output pattern must contain an ffmpeg image sequence placeholder."
        raise ValueError(msg)

    prefix = stem.split("%", 1)[0]
    suffix = out_pattern.suffix
    duration_s = max(1e-6, float(end_s) - float(start_s))

    last_error: ffmpeg.Error | None = None
    for colorspace_fix in _FRAME_EXTRACT_VF_FALLBACKS:
        # * Clear before each attempt so a successful run's glob holds only its frames.
        _clear_extracted_frames(out_dir, prefix, suffix)
        filtergraph = _frame_extract_filtergraph(fps, colorspace_fix)
        try:
            _run_frame_extraction_once(v, start_s, duration_s, out_pattern, filtergraph)
        except ffmpeg.Error as exc:
            last_error = exc
            logger.debug(
                "ffmpeg frame extract failed; trying next colorspace strategy: "
                "vf=%s stderr=%s",
                filtergraph,
                _ffmpeg_error_text(exc),
            )
            continue
        # * A clean exit is authoritative even with zero frames (not a colorspace fault).
        return tuple(
            path.resolve()
            for path in sorted(out_dir.glob(f"{prefix}*{suffix}"))
            if path.is_file()
        )

    if last_error is not None:
        raise last_error
    return ()
