"""Shared helpers of the STT concurrency benchmark tests."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pytest
    from click.testing import Result

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


class GrowthTools:
    """Replace external audio programs and the torch probe, never the pool worker."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[list[str]] = []
        self.fail_cut = False
        self.after_cut: Callable[[Path], None] | None = None
        self.probe_allowed = True
        self.gpu_csv = "1024, 8192, 0\n"
        native_which = shutil.which

        def which(name: str, *args: Any, **kwargs: Any) -> str | None:  # noqa: ANN401 - external executable lookup
            if name in {"ffmpeg", "pwsh", "nvidia-smi"}:
                return name
            return native_which(name, *args, **kwargs)

        monkeypatch.setattr(shutil, "which", which)
        monkeypatch.setattr(subprocess, "run", self.run)

    def run(
        self,
        argv: Sequence[str],
        **_kwargs: Any,  # noqa: ANN401 - external process options
    ) -> subprocess.CompletedProcess[Any]:
        """Emulate only documented external commands and retain their argv contracts."""
        args = list(argv)
        self.calls.append(args)
        if args[0] == "ffmpeg":
            self._ffmpeg(args)
            return subprocess.CompletedProcess(args, 0, b"", b"")
        if args[0] == "pwsh":
            match = re.search(r"SetOutputToWaveFile\('((?:[^']|'')*)'", args[-1])
            assert match is not None
            write_wav(Path(match[1].replace("''", "'")), 2)
            return subprocess.CompletedProcess(args, 0, b"", b"")
        if args[0] == "nvidia-smi":
            return subprocess.CompletedProcess(args, 0, self.gpu_csv, "")
        assert args[1] == "-c", args
        assert self.probe_allowed, args
        return subprocess.CompletedProcess(args, 0, '{"torch":"fake-probe"}\n', "")

    def _ffmpeg(self, args: list[str]) -> None:
        """Honor the output media shape without an installed codec or voice."""
        target = Path(args[-1])
        if "-t" not in args:
            shutil.copyfile(args[args.index("-i") + 1], target)
            return
        if self.fail_cut:
            raise subprocess.CalledProcessError(1, args, stderr=b"encoder unavailable")
        seconds = float(args[args.index("-t") + 1])
        if target.suffix == ".wav":
            with wave.open(str(target), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(SAMPLE_RATE)
                writer.writeframes(b"\x00\x00" * round(seconds * SAMPLE_RATE))
        else:
            target.write_bytes(b"fake-opus-input")
        if self.after_cut is not None:
            self.after_cut(target)


class GrowthBench:
    """Drive the real growth CLI, worker process, Job Object, IPC and output files."""

    def __init__(
        self, root: Path, monkeypatch: pytest.MonkeyPatch, *, desktop: str | None
    ) -> None:
        from stt_concurrency import memory_growth  # noqa: PLC0415
        from stt_concurrency.sysmetrics import CommitInfo  # noqa: PLC0415

        self.root = root
        self.out = root / "growth"
        self.source = root / "source.wav"
        write_wav(self.source, 2)
        self.tools = GrowthTools(monkeypatch)
        self.free_commit = 64 * 1024**3
        self.commit_reading: Callable[[], int] = lambda: self.free_commit
        self.workers: list[subprocess.Popen[bytes]] = []
        self.worker_envs: list[dict[str, str]] = []
        self.worker_allowed = True
        self._native_popen = subprocess.Popen
        self._desktop = desktop
        monkeypatch.setattr(
            memory_growth,
            "query_commit",
            lambda: CommitInfo(self.commit_reading(), None, None, "test-commit"),
        )
        monkeypatch.setattr(subprocess, "Popen", self._popen)

    def _popen(
        self,
        argv: Sequence[str],
        **kwargs: Any,  # noqa: ANN401 - external process options
    ) -> subprocess.Popen[bytes]:
        """Observe real launches and contain hard errors on a hidden desktop."""
        assert self.worker_allowed, argv
        assert argv[-1] == "core.stt_pool.worker", argv
        if self._desktop is not None:
            startup = subprocess.STARTUPINFO()
            startup.lpDesktop = self._desktop
            kwargs["startupinfo"] = startup
        process = self._native_popen(argv, **kwargs)
        self.workers.append(process)
        self.worker_envs.append(dict(kwargs["env"]))
        return process

    def invoke(self, *extra: str) -> Result:
        """Enter the public CLI with a tiny workload and an injected fake ML engine."""
        from stt_concurrency.cli import app  # noqa: PLC0415
        from typer.testing import CliRunner  # noqa: PLC0415

        return CliRunner().invoke(
            app,
            [
                "memory-growth",
                "--out-dir",
                str(self.out),
                "--source",
                str(self.source),
                "--jobs",
                "4",
                "--durations",
                "fixed",
                "--fixed-duration-s",
                "1.25",
                "--engine-factory",
                "tests.stt_pool_worker_engine:create_engine",
                "--load-timeout-s",
                "20",
                "--job-timeout-s",
                "20",
                *extra,
            ],
        )

    def run_dir(self) -> Path:
        """Find the single run owned by this sandbox."""
        directories = list(self.out.iterdir())
        assert len(directories) == 1
        return directories[0]

    def document(self, name: str) -> dict[str, Any]:
        """Read a persisted artifact without using the production summary helpers."""
        return json.loads((self.run_dir() / name).read_text(encoding="utf-8"))

    def records(self) -> list[dict[str, Any]]:
        """Read only the durable per-job records."""
        lines = (self.run_dir() / "results.jsonl").read_text("utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def close(self) -> None:
        """Reap owned processes on failure and assert normal worker teardown."""
        alive = [process for process in self.workers if process.poll() is None]
        for process in alive:
            process.kill()
            process.wait(timeout=30)
        for process in self.workers:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        assert not alive, [process.pid for process in alive]
