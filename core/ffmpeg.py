# Placeholder for ffmpeg.py

import logging
import subprocess
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
) -> None:
    """Extract audio from input file and save as WAV with specified sample rate and channels."""
    ffmpeg.input(str(input_file)).output(
        str(output_file),
        format="wav",
        acodec="pcm_s16le",
        ac=channels,
        ar=sample_rate,
    ).overwrite_output().run(quiet=True)


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
