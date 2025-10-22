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
    force_style: str | None = "Fontsize=42,Outline=2,Shadow=0",
) -> None:
    """Burn subtitles into a video using ffmpeg's subtitles filter.

    Uses libass; ensure ffmpeg build supports it. The `force_style` string is passed
    directly to the subtitles filter.
    """
    # Use absolute paths and escape single quotes for filter string
    v = Path(video_file).resolve()
    s = Path(subtitle_file).resolve()
    o = Path(output_file).resolve()
    sub_escaped = str(s).replace("'", "\\'")
    vf = f"subtitles='{sub_escaped}'"
    if force_style:
        vf += f":force_style='{force_style}'"
    (
        ffmpeg.input(str(v))
        .output(
            str(o),
            vf=vf,
            vcodec="libx264",
            preset="veryfast",
            crf=18,
            acodec="copy",
        )
        .overwrite_output()
        .run(quiet=False)
    )
