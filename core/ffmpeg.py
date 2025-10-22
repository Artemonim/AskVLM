# Placeholder for ffmpeg.py

from pathlib import Path

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


def burn_subtitles(
    video_file: str | Path,
    subtitle_file: str | Path,
    output_file: str | Path,
    force_style: str | None = None,
    *,
    autoscale: bool = True,
) -> None:
    """Burn subtitles into a video using ffmpeg's subtitles filter.

    Uses libass; ensure ffmpeg build supports it. The `force_style` string is passed
    directly to the subtitles filter.
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
        except Exception:
            # Fallback to conservative size if probe fails
            auto_style = "Fontsize=28,Outline=2,Shadow=0"

    style = force_style or auto_style or "Fontsize=28,Outline=2,Shadow=0"

    # Use POSIX-style path to avoid backslash escaping in filtergraph
    sub_path = str(s).replace("\\", "/")

    # Prepare streams
    inp = ffmpeg.input(str(v))
    video_f = inp.video.filter("subtitles", filename=sub_path, force_style=style)
    # Normalize audio to -16 LUFS, True Peak -3 dB (broadcast-ish target)
    audio_f = inp.audio.filter_(
        "loudnorm", I=-16, TP=-3.0, LRA=11, dual_mono="true", print_format="summary"
    )

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
