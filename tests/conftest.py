"""Pytest configuration and shared fixtures.

Heavy ML / LM Studio / GPU integration tests declare
``@pytest.mark.xdist_group(name="ml_singleton")`` so that with
``addopts = -n auto --dist=loadgroup`` (see ``pyproject.toml``) pytest-xdist
assigns them to one worker and runs them sequentially. That keeps at most one
neural-network-heavy process path active at a time across the suite (policy:
one active NN stack).
"""

import shutil
import sys
from pathlib import Path

import pytest

# Add project root to sys.path to ensure imports work correctly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.env import load_env_file  # noqa: E402

# Load environment variables from .env file
load_env_file(project_root / ".env")


@pytest.fixture(scope="session")
def short_audio_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped copy of the committed short mp4 for fast, deterministic tests."""
    fixture_path = project_root / "tests/fixtures/test_video_short.mp4"

    if not fixture_path.is_file():
        pytest.skip(f"Fixture not found: {fixture_path}")

    out_dir = tmp_path_factory.mktemp("short_audio")
    output_path = out_dir / "short_test_video.mp4"
    shutil.copy2(fixture_path, output_path)
    return output_path


@pytest.fixture(scope="session")
def shared_diarization_pipeline() -> object:
    """Initialize DiarizationPipeline once per session to save load time.

    Returns:
        Initialized DiarizationPipeline instance or None if unavailable.

    """
    # Avoid top-level import to prevent issues if dependencies are missing
    try:
        from core.diarization import DiarizationPipeline  # noqa: PLC0415
    except ImportError:
        return None

    pipeline = DiarizationPipeline(device="auto")
    # If underlying pipeline failed to load (e.g. no token), return None
    # Accessing private member _pipeline to check status
    if getattr(pipeline, "_pipeline", None) is None:
        return None

    return pipeline
