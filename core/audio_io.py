# Placeholder for audio_io.py

import logging
from pathlib import Path

from .ffmpeg import extract_audio

logger = logging.getLogger(__name__)

# * Prepare audio for downstream processing
# * If input is not WAV or wrong sample rate/channels, extract and convert


def prepare_audio(
    input_path: Path,
    work_dir: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Ensure input media is converted to a WAV file with correct specs."""
    # Place intermediates into a hidden work subfolder to avoid cluttering outputs
    work_subdir = work_dir / "_work"
    work_subdir.mkdir(parents=True, exist_ok=True)
    output_wav = work_subdir / f"{input_path.stem}.wav"
    logger.info("Preparing audio: %s -> %s", input_path, output_wav)
    extract_audio(input_path, output_wav, sample_rate=sample_rate, channels=channels)
    return output_wav
