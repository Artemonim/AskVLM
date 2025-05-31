# Placeholder for ffmpeg.py

from pathlib import Path
from typing import Union

import ffmpeg

# * Extract audio from media file and convert to WAV format
# * This function uses ffmpeg-python to produce a PCM 16-bit WAV file


def extract_audio(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
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
