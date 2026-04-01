import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from core.audio_io import prepare_audio

if TYPE_CHECKING:
    from core.diarization import DiarizationPipeline


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.heavy_ml
@pytest.mark.xdist_group(name="gpu")
def test_diarization_real_file(
    tmp_path: Path, short_audio_fixture: Path, shared_diarization_pipeline: object
) -> None:
    """Test that diarization works on a real file using pyannote 3.0.

    Uses short_audio_fixture (committed short clip) and shared pipeline to speed up test.
    """
    # * HF_TOKEN is loaded by conftest.py from .env

    if not shared_diarization_pipeline:
        token = os.getenv("HF_TOKEN") or os.getenv("PYANNOTE_AUTH_TOKEN")
        if not token:
            pytest.skip(
                "HF_TOKEN not set in .env or environment, skipping gated model test."
            )
        pytest.skip("Diarization pipeline failed to initialize (check logs/install)")

    pipeline = cast("DiarizationPipeline", shared_diarization_pipeline)

    # 1. Prepare audio
    wav_path = prepare_audio(short_audio_fixture, tmp_path)

    # 2. Run diarization (pipeline already init)
    try:
        segments = pipeline.diarize(str(wav_path))
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Diarization crashed: {e}")

    # 3. Assertions
    assert isinstance(segments, list)
    # Verify that we actually got segments (audio has speech)
    # This protects against silent inputs or broken models
    assert len(segments) > 0, (
        "Diarization returned empty list; input audio might be silent?"
    )
