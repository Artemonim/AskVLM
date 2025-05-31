# Placeholder for audio_io.py

import logging
from pathlib import Path

from .ffmpeg import extract_audio

# * Prepare audio for downstream processing
# * If input is not WAV or wrong sample rate/channels, extract and convert


def prepare_audio(
    input_path: Path,
    work_dir: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Ensure input media is converted to a WAV file with correct specs."""
    output_wav = work_dir / f"{input_path.stem}.wav"
    logging.info("Preparing audio: %s -> %s", input_path, output_wav)
    extract_audio(input_path, output_wav, sample_rate=sample_rate, channels=channels)
    return output_wav
