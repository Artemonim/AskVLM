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
    """Create a 10s audio clip from test_video_first.mp4 for faster tests.

    Returns path to the shortened video file.
    """
    fixture_path = project_root / "tests/fixtures/test_video_first.mp4"

    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    # Create a session-scoped temp dir
    out_dir = tmp_path_factory.mktemp("short_audio")
    output_path = out_dir / "short_test_video.mp4"

    # Extract 10s clip using ffmpeg-python
    # We import locally to avoid top-level side effects
    import ffmpeg  # noqa: PLC0415

    try:
        (
            ffmpeg.input(str(fixture_path), t=10)
            .output(str(output_path), c="copy")
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Failed to create short audio fixture: {e}")

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
