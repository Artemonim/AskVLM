"""Shared helpers of the STT concurrency benchmark tests."""

from __future__ import annotations

import sys
import wave
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
FAKE_WORKER = Path(__file__).resolve().with_name("stt_concurrency_fake_worker.py")
PARENT_DRIVER = Path(__file__).resolve().with_name("stt_concurrency_parent_driver.py")
SAMPLE_RATE = 16000

# * The benchmark is a tool script, not an installed package: expose it like
# * `tools/benchmark_stt_concurrency.py` does at runtime.
for _path in (PROJECT_ROOT, TOOLS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


class JobApiFault:
    """Fail one Win32 API; all other job operations reach the real kernel."""

    def __init__(self, api: Any, failure: str) -> None:  # noqa: ANN401
        self._api = api
        self._failure = failure

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - foreign API proxy
        """Return the real API function or the injected Win32 failure."""
        if name == self._failure:
            return lambda *_args: 0
        return getattr(self._api, name)


def write_wav(path: Path, seconds: int) -> None:
    """Write *seconds* of 16 kHz mono silence to *path*."""
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(b"\x00\x00" * SAMPLE_RATE * seconds)


def make_clips(directory: Path, durations: list[int]) -> list[Path]:
    """Create ``clip00.wav``, ``clip01.wav``, ... with the given durations."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, seconds in enumerate(durations):
        path = directory / f"clip{index:02d}.wav"
        write_wav(path, seconds)
        paths.append(path)
    return paths


def make_run_options(tmp_path: Path, **overrides: Any) -> Any:  # noqa: ANN401
    """Build ``RunOptions`` with neutral defaults; *overrides* replace fields."""
    from stt_concurrency.configs import (  # noqa: PLC0415
        DEFAULT_ESTIMATES,
        ThreadDefaults,
    )
    from stt_concurrency.runner import RunOptions  # noqa: PLC0415

    fields: dict[str, Any] = {
        "clips_dir": tmp_path,
        "out_dir": tmp_path / "out",
        "max_clips": 0,
        "long_files": (),
        "language": "ru",
        "selection": "",
        "repeat": 1,
        "clear": False,
        "retry_failed": False,
        "max_config_seconds": 60.0,
        "min_free_commit_gb": 8.0,
        "commit_reserve_gb": 0.0,
        "min_free_vram_mb": 512.0,
        "estimates": dict(DEFAULT_ESTIMATES),
        "thread_defaults": ThreadDefaults(cpu_budget=16, cuda_threads=4),
        "whisper_model": "small",
        "whisper_options": {"language": "ru", "beam_size": 5},
        "whisper_chunk_s": 30.0,
        "warmup_clip": None,
        "sample_interval_s": 0.5,
        "cooldown_s": 0.0,
        "gigaam_cuda_streams": True,
        "dry_run": False,
    }
    fields.update(overrides)
    return RunOptions(**fields)


def default_estimates() -> dict[str, Any]:
    """Return a copy of the tool's default guard estimates."""
    from stt_concurrency.configs import DEFAULT_ESTIMATES  # noqa: PLC0415

    return dict(DEFAULT_ESTIMATES)


def make_audio_items(directory: Path, seconds: list[int]) -> list[Any]:
    """Create WAV clips and return their ``AudioItem`` list."""
    from stt_concurrency.workload import discover_clips  # noqa: PLC0415

    clips_dir = directory / f"clips-{'-'.join(map(str, seconds))}"
    make_clips(clips_dir, seconds)
    return discover_clips(clips_dir, 0)
