"""Scenario tests of ``tools/benchmark_stt_concurrency.py`` with fake STT workers.

The CLI, the orchestrator, the Windows job object, the pipes, and the child event
loop run for real in separate processes (``tests/stt_concurrency_fake_worker.py``
swaps only the speech engine for a deterministic fake). Expected values come from
the requirement: equal workload per configuration, clock started after every
worker is loaded, one model per ``S`` process, batches of the requested size,
resume that skips finished configurations, no worker surviving a timeout or the
death of the parent, and failures recorded rather than reported as success.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil
import pytest
from typer.testing import CliRunner

# * The support module puts `tools/` on sys.path for the imports below.
from tests.stt_concurrency_support import (
    FAKE_WORKER,
    PARENT_DRIVER,
    PROJECT_ROOT,
    GrowthBench,
    JobApiFault,
    make_clips,
)
from tests.stt_pool_windows_support import _hidden_desktop

# isort: split

from stt_concurrency import cli, runner, sysmetrics
from stt_concurrency.combo import auto_combos
from stt_concurrency.configs import ThreadDefaults, parse_config_name
from stt_concurrency.protocol import (
    EXIT_PARENT_GONE,
    ChildSpec,
    decode_message,
    encode_message,
)
from stt_concurrency.sysmetrics import CommitInfo
from stt_concurrency.workload import AudioItem

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# * The scenarios spawn processes and measure time: keep them on one xdist worker.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.xdist_group(name="stt_concurrency_processes"),
]

_DURATIONS = [1, 2, 3, 1, 2, 3]
_WAIT_S = 90.0


class Bench:
    """One benchmark sandbox: clips, output directory, fake-worker markers."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.clips = root / "clips"
        self.out = root / "out"
        self.fake = root / "fake"
        self.fake.mkdir()
        make_clips(self.clips, _DURATIONS)
        self._monkeypatch = monkeypatch
        monkeypatch.setattr(runner, "ENTRY_SCRIPT", FAKE_WORKER)
        monkeypatch.setenv("FAKE_DIR", str(self.fake))

    def env(self, **values: str) -> None:
        """Set environment variables for the fake workers."""
        for key, value in values.items():
            self._monkeypatch.setenv(key, value)

    def args(self, configs: str, *extra: str) -> list[str]:
        """Build the ``run`` command line."""
        return [
            "run",
            "--clips-dir",
            str(self.clips),
            "--out-dir",
            str(self.out),
            "--configs",
            configs,
            "--min-free-commit-gb",
            "0",
            "--commit-reserve-gb",
            "0",
            "--cooldown",
            "0",
            "--sample-interval",
            "0.2",
            *extra,
        ]

    def run(self, configs: str, *extra: str) -> Any:  # noqa: ANN401
        """Invoke the CLI and require a zero exit code."""
        result = CliRunner().invoke(cli.app, self.args(configs, *extra))
        assert result.exit_code == 0, result.output
        return result

    def manifest(self) -> Path:
        """Write the reference manifest of the fake engine's transcripts."""
        path = self.root / "manifest.jsonl"
        lines = [
            json.dumps(
                {"wav": f"clip{i:02d}.wav", "ref_fleurs_norm": f"text clip{i:02d}"}
            )
            for i in range(len(_DURATIONS))
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def shared(self, *extra: str) -> list[str]:
        """Options shared by sweep and combo (fast, tiny, same fingerprint)."""
        return [
            "--clips-dir",
            str(self.clips),
            "--out-dir",
            str(self.out),
            "--manifest",
            str(self.manifest()),
            "--min-free-commit-gb",
            "0",
            "--commit-reserve-gb",
            "0",
            "--cooldown",
            "0",
            "--sample-interval",
            "0.2",
            "--cpu-budget",
            "4",
            "--clips-per-stream",
            "4",
            "--min-step-seconds",
            "0.3",
            *extra,
        ]

    def sweep(self, groups: str, *extra: str) -> Any:  # noqa: ANN401
        """Invoke ``sweep`` on a few groups and require exit code 0."""
        args = ["sweep", "--groups", groups, "--threads", "1", *self.shared(*extra)]
        result = CliRunner().invoke(cli.app, args)
        assert result.exit_code == 0, result.output
        return result

    def combo(self, pairs: str, *extra: str) -> Any:  # noqa: ANN401
        """Invoke ``combo`` and require exit code 0."""
        args = ["combo", "--pairs", pairs, *self.shared(*extra)]
        result = CliRunner().invoke(cli.app, args)
        assert result.exit_code == 0, result.output
        return result

    def group(self, name: str) -> dict[str, Any]:
        """Load a sweep group document."""
        path = self.out / "sweep" / "groups" / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def step(self, name: str) -> dict[str, Any] | None:
        """Load a sweep step result, if it exists."""
        path = self.out / "sweep" / "results" / f"{name}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def result(self, name: str, repeat: int = 0) -> dict[str, Any]:
        """Load the result document of a configuration."""
        path = self.out / "results" / f"{name}__r{repeat}.json"
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def pids(self, prefix: str) -> list[int]:
        """Return the pids recorded by the fake workers under *prefix*."""
        return [int(p.read_text()) for p in sorted(self.fake.glob(f"{prefix}-*.pid"))]

    def processes(self, prefix: str) -> list[psutil.Process]:
        """Return the running processes the fake workers recorded under *prefix*.

        A pid file outlives its process, and by then the pid may belong to a
        stranger. A process counts as the sandbox's only while its creation time
        equals the one the worker recorded about itself; a missing or unreadable
        record proves nothing and excludes the process.
        """
        confirmed: list[psutil.Process] = []
        for marker in sorted(self.fake.glob(f"{prefix}-*.pid")):
            try:
                recorded = float(marker.with_suffix(".created").read_text())
                process = psutil.Process(int(marker.read_text()))
                if process.create_time() == recorded:
                    confirmed.append(process)
            except (OSError, ValueError, psutil.Error):
                continue
        return confirmed

    def reap(self) -> None:
        """Kill the sandbox processes that are still running, and nothing else.

        The kill re-checks the identity: a pid taken over after the confirmation
        raises ``NoSuchProcess`` instead of terminating the newcomer.
        """
        for process in [*self.processes("worker"), *self.processes("grandchild")]:
            with contextlib.suppress(psutil.Error):
                process.kill()


@contextlib.contextmanager
def sandboxed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Bench]:
    """Provide a sandbox and kill any worker it left behind."""
    sandbox = Bench(tmp_path, monkeypatch)
    try:
        yield sandbox
    finally:
        # * Safety net: never leave a sandbox process behind, even if a test failed.
        sandbox.reap()


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Bench]:
    """Provide a sandbox and kill any worker it left behind."""
    with sandboxed(tmp_path, monkeypatch) as sandbox:
        yield sandbox


def _gone(processes: list[psutil.Process], timeout_s: float) -> list[psutil.Process]:
    """Wait until the processes end; return the ones still running."""
    deadline = time.monotonic() + timeout_s
    alive = [p for p in processes if p.is_running()]
    while alive and time.monotonic() < deadline:
        time.sleep(0.2)
        alive = [p for p in alive if p.is_running()]
    return alive


@pytest.fixture
def growth_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[GrowthBench]:
    """Keep growth runs in a temporary root with every native child contained."""
    desktop = _hidden_desktop() if sys.platform == "win32" else contextlib.nullcontext()
    with desktop as name:
        sandbox = GrowthBench(tmp_path, monkeypatch, desktop=name)
        try:
            yield sandbox
        finally:
            sandbox.close()


@pytest.mark.parametrize(
    "options",
    [
        ("--jobs", "0"),
        ("--durations", "unknown"),
        ("--duration-min-s", "3", "--duration-max-s", "1", "--durations", "varied"),
        ("--gpu-memory-bounds", "unknown"),
        ("--idle-gap-s", "-1"),
        ("--pool-set", "gpu.cache_typo=1"),
        ("--pool-set", "gpu.vram_cap=1"),
        ("--pool-set", "gpu.cufft_plan_cache_max_size=-1"),
        ("--pool-set", "gpu.batch_lanes.gigaam=0"),
        # * Non-finite floats must fail before resource probes or run creation,
        # * including duration options inactive in the selected fixed mode.
        *[
            (f"{option}={value}",)
            for option in (
                "--duration-min-s",
                "--duration-max-s",
                "--fixed-duration-s",
                "--reservation-commit-gib",
                "--reservation-vram-gib",
                "--min-free-commit-gib",
                "--min-free-vram-gib",
                "--commit-reserve-gib",
                "--plateau-tolerance-mib",
                "--idle-gap-s",
                "--job-timeout-s",
                "--load-timeout-s",
            )
            for value in ("nan", "inf", "-inf")
        ],
        ("--reservation-commit-gib", "1e308"),
        ("--reservation-vram-gib", "1e308"),
        ("--idle-gap-s", "1e12"),
        ("--job-timeout-s", "1e12"),
        ("--load-timeout-s", "1e12"),
    ],
)
def test_growth_cli_rejects_invalid_workloads_and_pool_overrides_before_side_effects(
    growth_bench: GrowthBench, options: tuple[str, ...]
) -> None:
    """Catch invalid plans, NaN/Inf or huge floats creating a run or crashing."""
    growth_bench.tools.probe_allowed = growth_bench.worker_allowed = False

    result = growth_bench.invoke(*options)

    assert result.exit_code == 2, (result.output, result.exception)
    assert not growth_bench.out.exists()
    assert growth_bench.tools.calls == []


@pytest.mark.parametrize("reason", ["commit", "vram", "missing_vram", "job_cap"])
def test_growth_preflight_records_refusal_before_audio_probe_or_worker_launch(
    growth_bench: GrowthBench, reason: str
) -> None:
    """Catch preflight running torch or allowing the Job cap to spend the reserve."""
    growth_bench.tools.probe_allowed = growth_bench.worker_allowed = False
    options: tuple[str, ...] = ()
    if reason == "commit":
        growth_bench.free_commit = 19 * 1024**3
    elif reason == "vram":
        growth_bench.tools.gpu_csv = "7168, 8192, 0\n"
    elif reason == "missing_vram":
        growth_bench.tools.gpu_csv = ""
    else:
        growth_bench.free_commit = 24 * 1024**3
        options = ("--reservation-commit-gib", "12")

    result = growth_bench.invoke(*options)

    assert result.exit_code == 3, (result.output, result.exception)
    plan = growth_bench.document("run.json")
    assert plan["stack"] is None
    assert plan["guard"]["refusals"]
    assert plan["guard"]["free_commit_bytes"] == growth_bench.free_commit
    if reason == "job_cap":
        assert plan["guard"]["job_limit_bytes"] == 13 * 1024**3
    assert {path.name for path in growth_bench.run_dir().iterdir()} == {"run.json"}
    assert all(args[0] == "nvidia-smi" for args in growth_bench.tools.calls)


@pytest.mark.parametrize("summary_blocked", [False, True])
def test_growth_input_failure_keeps_diagnostics_and_its_exit_code_when_report_fails(
    growth_bench: GrowthBench, monkeypatch: pytest.MonkeyPatch, *, summary_blocked: bool
) -> None:
    """Catch lost inputs_failed diagnostics, leaked inputs or a masked input error."""
    growth_bench.worker_allowed = False
    growth_bench.tools.fail_cut = True
    if summary_blocked:

        def block_report(
            argv: Sequence[str],
            **kwargs: Any,  # noqa: ANN401 - external command boundary
        ) -> subprocess.CompletedProcess[Any]:
            if argv[0] == "ffmpeg" and "-t" in argv:
                (growth_bench.run_dir() / "summary.json").mkdir(exist_ok=True)
            return native_run(argv, **kwargs)

        native_run = subprocess.run
        monkeypatch.setattr(subprocess, "run", block_report)

    result = growth_bench.invoke()

    assert result.exit_code == 2, (result.output, result.exception)
    assert growth_bench.document("run.json")["stack"]["torch"] == "fake-probe"
    assert growth_bench.records() == []
    if not summary_blocked:
        summary = growth_bench.document("summary.json")
        assert summary["status"] == "inputs_failed"
        assert summary["stats"]["throughput"]["jobs"] == 0
    assert not (growth_bench.run_dir() / "tmp").exists()
    assert not (growth_bench.run_dir() / "inputs").exists()


def test_growth_spawn_guard_rechecks_commit_after_input_preparation_and_cleans_up(
    growth_bench: GrowthBench,
) -> None:
    """Catch a stale preflight allowing a worker after input generation spent commit."""
    growth_bench.worker_allowed = False

    def spend_commit(_path: Path) -> None:
        growth_bench.free_commit = 12 * 1024**3

    growth_bench.tools.after_cut = spend_commit

    result = growth_bench.invoke()

    assert result.exit_code == 3, (result.output, result.exception)
    plan = growth_bench.document("run.json")
    assert not plan["guard"]["refusals"]
    assert plan["spawn_guard"]["refusals"]
    assert plan["spawn_guard"]["free_commit_bytes"] == 12 * 1024**3
    summary = growth_bench.document("summary.json")
    assert summary["status"] == "guard_refused"
    assert summary["stats"]["throughput"]["jobs"] == 0
    assert growth_bench.records() == []
    assert not (growth_bench.run_dir() / "tmp").exists()
    assert not (growth_bench.run_dir() / "inputs").exists()


def test_growth_commit_stop_persists_partial_results_and_reaps_the_worker(
    growth_bench: GrowthBench,
) -> None:
    """Catch continuing below the reserve or losing completed jobs on an early stop."""

    def commit() -> int:
        files = list(growth_bench.out.glob("*/results.jsonl"))
        done = len(files[0].read_text("utf-8").splitlines()) if files else 0
        return (11 if done >= 2 else 64) * 1024**3

    growth_bench.commit_reading = commit

    result = growth_bench.invoke()

    assert result.exit_code == 1, (result.output, result.exception)
    assert [record["index"] for record in growth_bench.records()] == [1, 2]
    summary = growth_bench.document("summary.json")
    assert summary["status"] == "system_commit_low"
    assert summary["stats"]["throughput"]["ok"] == 2
    assert len(growth_bench.workers) == 1
    assert growth_bench.workers[0].poll() == 0
    assert not (growth_bench.run_dir() / "inputs").exists()


def test_growth_completed_jobs_with_an_unwritable_summary_exit_four_and_clean_up(
    growth_bench: GrowthBench,
) -> None:
    """Catch successful exit codes and leaked audio when final artifact writes fail."""

    def block_summary(path: Path) -> None:
        (path.parent.parent / "summary.json").mkdir(exist_ok=True)

    growth_bench.tools.after_cut = block_summary

    result = growth_bench.invoke()

    assert result.exit_code == 4, (result.output, result.exception)
    assert [record["index"] for record in growth_bench.records()] == [1, 2, 3, 4]
    assert growth_bench.workers[0].poll() == 0
    assert growth_bench.document("run.json")["spawn_guard"]["refusals"] == []
    assert not (growth_bench.run_dir() / "tmp").exists()
    assert not (growth_bench.run_dir() / "inputs").exists()
    assert not list(growth_bench.run_dir().glob(".*.tmp"))


@pytest.mark.parametrize("bounds", ["pool", "library"])
def test_growth_cli_completes_real_worker_jobs_and_applies_the_requested_bounds(
    growth_bench: GrowthBench, monkeypatch: pytest.MonkeyPatch, bounds: str
) -> None:
    """Catch broken IPC wiring, inherited library-run bounds or lost artifacts."""
    cache_env = "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"
    monkeypatch.setenv(cache_env, "77")
    monkeypatch.setenv("ASKVLM_STT_POOL_CONFIG", str(growth_bench.root / "absent.json"))

    result = growth_bench.invoke(
        "--gpu-memory-bounds", bounds, "--pool-set", "gpu.host_threads.gigaam=2"
    )

    assert result.exit_code == 0, (result.output, result.exception)
    records = growth_bench.records()
    assert [record["index"] for record in records] == [1, 2, 3, 4]
    assert all(
        record["status"] == "ok" and record["audio_s"] == 1.25 for record in records
    )
    summary = growth_bench.document("summary.json")
    assert summary["status"] == "completed"
    assert summary["stats"]["throughput"]["ok"] == 4
    if sys.platform == "win32":
        assert all(
            record["private_bytes"] > 0 and record["job_peak_bytes"] > 0
            for record in records
        )
    assert len(growth_bench.workers) == 1
    assert summary["worker_pid"] == growth_bench.workers[0].pid
    assert growth_bench.workers[0].poll() == 0
    plan = growth_bench.document("run.json")
    assert plan["spec"]["options"]["batch_lanes"] == plan["spec"]["capacity"]
    env = growth_bench.worker_envs[0]
    assert [
        env[key]
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    ] == ["2", "2", "2"]
    assert env["TEMP"] == env["TMP"] == str(growth_bench.run_dir() / "tmp")
    if bounds == "library":
        assert cache_env not in env
        assert plan["spec"]["options"]["vram_cap_bytes"] is None
        assert plan["spec"]["options"]["cufft_plan_cache_max_size"] == 0
    else:
        assert env[cache_env] == "256"
        assert plan["spec"]["options"]["vram_cap_bytes"] > 0
    with (growth_bench.run_dir() / "results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert [int(row["index"]) for row in csv.DictReader(handle)] == [1, 2, 3, 4]
    assert (growth_bench.run_dir() / "summary.md").is_file()
    assert not (growth_bench.run_dir() / "tmp").exists()
    assert not (growth_bench.run_dir() / "inputs").exists()


def test_growth_dry_run_prints_the_requested_plan_without_writing_or_spawning(
    growth_bench: GrowthBench,
) -> None:
    """Catch dry-run side effects or omitted user-selected spec values."""
    growth_bench.worker_allowed = growth_bench.tools.probe_allowed = False

    result = growth_bench.invoke(
        "--dry-run",
        "--pool-set",
        "gpu.cufft_plan_cache_max_size=11",
        "--pool-set",
        "gpu.idle_cache_release_s=0.25",
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert "gigaam-e2e-ctc/ai-sage/GigaAM-v3@e2e_ctc/fp32/gpu" in result.output
    assert re.search(r"cuFFT\D+11\b", result.output)
    assert "0.25" in result.output
    assert not growth_bench.out.exists()
    assert all(args[0] == "nvidia-smi" for args in growth_bench.tools.calls)


def test_growth_metadata_replace_failure_preserves_the_previous_complete_plan(
    growth_bench: GrowthBench, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch truncated metadata or input leaks after a failed atomic rewrite."""
    growth_bench.worker_allowed = False
    native_replace = os.replace
    previous: list[bytes] = []

    def replace_at_filesystem(source: object, destination: object) -> None:
        target = Path(destination)
        if target.name == "run.json" and target.exists():
            previous.append(target.read_bytes())
            msg = "test filesystem refuses a metadata replacement"
            raise PermissionError(msg)
        native_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace_at_filesystem)

    result = growth_bench.invoke("--no-probe-stack")

    assert result.exit_code == 1
    assert isinstance(result.exception, PermissionError)
    assert len(previous) == 1
    assert (growth_bench.run_dir() / "run.json").read_bytes() == previous[0]
    assert growth_bench.document("summary.json")["status"] == "error"
    assert not (growth_bench.run_dir() / "inputs").exists()
    assert not (growth_bench.run_dir() / "tmp").exists()
    assert not list(growth_bench.run_dir().glob(".*.tmp"))


def test_growth_broken_console_does_not_turn_a_written_summary_into_exit_four(
    growth_bench: GrowthBench, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch console errors conflated with artifact write failures."""
    native_echo = cli.typer.echo

    def echo_at_console(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401 - external console boundary
        if list(growth_bench.out.glob("*/summary.md")):
            msg = "test console pipe is closed"
            raise BrokenPipeError(msg)
        native_echo(*args, **kwargs)

    monkeypatch.setattr(cli.typer, "echo", echo_at_console)

    result = growth_bench.invoke()

    assert result.exit_code == 0, (result.output, result.exception)
    summary = growth_bench.document("summary.json")
    assert summary["status"] == "completed"
    assert summary["stats"]["throughput"]["ok"] == 4
    assert growth_bench.workers[0].poll() == 0


@pytest.mark.parametrize("jobs", [0, 4])
def test_summarize_rebuilds_growth_artifacts_even_before_the_first_job(
    tmp_path: Path, jobs: int
) -> None:
    """Catch overwritten zero-job growth reports or CSV crashes on old keys."""
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "run_id": "growth-example",
                "label": "example",
                "key": "gpu-test",
                "recycle_threshold_bytes": 200,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"status": "system_commit_low", "stats": {}}), encoding="utf-8"
    )
    if jobs:
        (tmp_path / "results.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "index": index,
                        "elapsed_s": index * 30,
                        "private_bytes": 100 + index * 2,
                        "status": "ok",
                        "wall_s": 1,
                        "audio_s": 3,
                        "legacy_extra_key": "kept",
                    }
                )
                for index in range(1, jobs + 1)
            )
            + '\n{"index":',
            encoding="utf-8",
        )

    result = CliRunner().invoke(cli.app, ["summarize", "--out-dir", str(tmp_path)])

    assert result.exit_code == 0, (result.output, result.exception)
    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))
    assert summary["schema"] == "askvlm.stt_memory_growth.summary"
    assert summary["status"] == "system_commit_low"
    assert summary["stats"]["throughput"]["jobs"] == jobs
    if jobs:
        assert summary["stats"]["metrics"]["private_bytes"]["slope_per_job"] == 2
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == jobs
    assert (tmp_path / "summary.md").is_file()


def test_summarize_excludes_two_warmup_jobs_from_a_ten_job_growth_projection(
    tmp_path: Path,
) -> None:
    """Catch an 80% window rounded down to include warm-up and lose projection."""
    mib = 1024**2
    (tmp_path / "run.json").write_text(
        json.dumps({"run_id": "ten-jobs", "recycle_threshold_bytes": 200 * mib}),
        encoding="utf-8",
    )
    # * Jobs 3-10 add 2 MiB per 30 s and end at 120 MiB. The 200 MiB
    # * threshold is 80 MiB away, hence 40 more jobs at 240 MiB/hour.
    values = [180, 180, 106, 108, 110, 112, 114, 116, 118, 120]
    (tmp_path / "results.jsonl").write_text(
        "".join(
            json.dumps(
                {"index": index, "elapsed_s": 30 * index, "private_bytes": value * mib}
            )
            + "\n"
            for index, value in enumerate(values, start=1)
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli.app, ["summarize", "--out-dir", str(tmp_path)])

    assert result.exit_code == 0, (result.output, result.exception)
    stats = json.loads((tmp_path / "summary.json").read_text("utf-8"))["stats"]
    private = stats["metrics"]["private_bytes"]
    assert private["slope_per_job"] == pytest.approx(2 * mib)
    assert private["slope_per_hour"] == pytest.approx(240 * mib)
    assert stats["recycle"]["projected_jobs_to_threshold"] == pytest.approx(40)
    assert stats["recycle"]["first_job_at_threshold"] is None


def test_summarize_keeps_the_ordinary_benchmark_result_schema(bench: Bench) -> None:
    """Catch memory-growth routing taking over ordinary benchmark output folders."""
    bench.run("whisper-cpu-P1x1")
    (bench.out / "summary.json").write_text("[]", encoding="utf-8")

    result = CliRunner().invoke(cli.app, ["summarize", "--out-dir", str(bench.out)])

    assert result.exit_code == 0, (result.output, result.exception)
    rows = json.loads((bench.out / "summary.json").read_text("utf-8"))
    assert isinstance(rows, list)
    assert [row["name"] for row in rows] == ["whisper-cpu-P1x1"]


def test_p2_clock_starts_after_all_workers_loaded_and_workload_is_complete(
    bench: Bench,
) -> None:
    """Catches: load time leaking into throughput, lost or duplicated items."""
    bench.env(FAKE_LOAD_S="0.1,2.0", FAKE_ITEM_S="0.05")

    bench.run("whisper-cpu-P2x1")

    doc = bench.result("whisper-cpu-P2x1")
    assert doc["status"] == "ok"
    assert doc["error_count"] == 0
    assert [item["index"] for item in doc["items"]] == list(range(6))
    assert [item["text"] for item in doc["items"]] == [
        f"text clip{i:02d}" for i in range(6)
    ]
    timing = doc["timing"]
    assert timing["audio_s"] == sum(_DURATIONS)
    # * One worker loads for 2 s; the timed run (6 x 50 ms over 2 workers) is far
    # * shorter and must not include the load.
    assert timing["load_max_s"] >= 1.9
    assert 0 < timing["wall_s"] < 1.5
    assert timing["throughput_x"] == pytest.approx(
        sum(_DURATIONS) / timing["wall_s"], rel=0.01
    )
    assert {item["child"] for item in doc["items"]} == {0, 1}
    assert len(doc["children"]) == 2
    if sys.platform == "win32":
        # * The recorded child must be the venv interpreter inside the cap,
        # * rather than a launcher whose interpreter escapes the job.
        assert doc["job"]["cap_enforced"]
        for worker in doc["children"]:
            assert worker["pid"] == worker["info"]["worker_pid"]
            assert worker["info"]["python_prefix"] == sys.prefix
            assert worker["in_step_job"]
    assert (bench.out / "summary.md").is_file()
    assert (bench.out / "summary.json").is_file()


def test_s_mode_uses_one_model_and_overlapping_threads(bench: Bench) -> None:
    """Catches: S mode loading one model per thread or serializing the threads."""
    bench.env(FAKE_ITEM_S="0.3")

    bench.run("whisper-cpu-S3x1")

    doc = bench.result("whisper-cpu-S3x1")
    assert doc["status"] == "ok"
    assert len(doc["children"]) == 1
    summary = json.loads((bench.fake / "summary-0.json").read_text())
    assert summary["loads"] == 1
    assert summary["peak_overlap"] >= 2
    assert {item["slot"] for item in doc["items"]} >= {0, 1}


def test_b_mode_dispatches_full_batches(bench: Bench) -> None:
    """Catches: batching mode that dispatches items one by one."""
    bench.run("gigaam-cpu-B3x1")

    doc = bench.result("gigaam-cpu-B3x1")
    assert doc["status"] == "ok"
    assert [item["batch"] for item in doc["items"]] == [3] * 6


def test_item_failure_is_recorded_not_reported_as_success(bench: Bench) -> None:
    """Catches: failed items counted as processed audio or a status of ok."""
    bench.env(FAKE_FAIL_NAME="clip02")

    bench.run("whisper-cpu-P1x1")

    doc = bench.result("whisper-cpu-P1x1")
    assert doc["status"] == "partial"
    assert doc["error_count"] == 1
    assert [e["item"] for e in doc["errors"]] == [2]
    # * Clip 2 lasts 3 s and does not count toward processed audio.
    assert doc["timing"]["audio_s"] == sum(_DURATIONS) - 3


def test_load_failure_fails_the_configuration_and_leaves_no_worker(
    bench: Bench,
) -> None:
    """Catches: a failed model load reported as ok (e.g. silent device fallback)."""
    bench.env(FAKE_LOAD_FAIL="1")

    bench.run("gigaam-cpu-P2x1")

    doc = bench.result("gigaam-cpu-P2x1")
    assert doc["status"] == "error"
    assert "fake load failure" in doc["reason"]
    assert doc["timing"]["throughput_x"] is None
    workers = [
        psutil.Process(pid) for pid in bench.pids("worker") if psutil.pid_exists(pid)
    ]
    assert _gone(workers, 15.0) == []


def test_worker_crash_fails_the_configuration_without_waiting_for_the_timeout(
    bench: Bench,
) -> None:
    """Catches: a crashed worker (OOM kill, CUDA crash) hanging until the timeout."""
    bench.env(FAKE_DIE="1")
    started = time.monotonic()

    bench.run("gigaam-cpu-P2x1", "--max-config-seconds", "300")

    doc = bench.result("gigaam-cpu-P2x1")
    assert doc["status"] == "error"
    assert doc["error_count"] == 6
    assert doc["timing"]["throughput_x"] is None
    assert time.monotonic() - started < 60.0


def test_timeout_kills_the_whole_process_tree(bench: Bench) -> None:
    """Catches: orphan workers (and their helper processes) after a timeout."""
    bench.env(FAKE_HANG="1", FAKE_GRANDCHILD="1")

    bench.run("whisper-cpu-P2x1", "--max-config-seconds", "12")

    doc = bench.result("whisper-cpu-P2x1")
    assert doc["status"] == "timeout"
    assert doc["timing"]["throughput_x"] is None
    pids = [*bench.pids("worker"), *bench.pids("grandchild")]
    assert len(pids) == 4
    survivors = _gone([p for p in map(_proc, pids) if p is not None], 15.0)
    assert survivors == []


def _proc(pid: int) -> psutil.Process | None:
    try:
        return psutil.Process(pid)
    except psutil.Error:
        return None


@contextlib.contextmanager
def _own_process() -> Iterator[subprocess.Popen[str]]:
    """Run a process of the test's own that exits with 0 only when told to."""
    script = (
        "import sys; print('ready', flush=True); "
        "sys.exit(sys.stdin.readline().strip() != 'exit')"
    )
    child = subprocess.Popen(  # noqa: S603 - fixed one-line script, no shell
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        yield child
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=30)
        for stream in (child.stdin, child.stdout):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()


@pytest.mark.parametrize("prefix", ["worker", "grandchild"])
@pytest.mark.parametrize(
    ("record", "killed"),
    [
        ("own", True),
        ("earlier_owner", False),
        ("none", False),
        ("invalid", False),
        ("unreadable", False),
    ],
)
class TestBenchTeardown:
    """Catch the actual fixture killing strangers or leaving identified workers.

    The pid files outlive the fake workers. Once a worker is gone, its pid may
    belong to an unrelated process, and a bare ``psutil.Process(pid).kill()`` made
    at teardown terminates that process (a stranger's console died with exit code
    15). A pid file whose recorded creation time is another process's, or that has
    none, must be left alone, while a sandbox process still running with its
    recorded identity must still die (the net must not become a no-op).
    """

    @pytest.fixture(autouse=True)
    def observed_process(self, *, killed: bool) -> Iterator[subprocess.Popen[str]]:
        """Keep the child alive through pytest's actual ``bench`` finalizer.

        Autouse fixtures start before ordinary fixtures in the same scope and
        finish after them, so the exit-code assertion observes ``bench`` teardown.
        """
        with _own_process() as child:
            yield child
            assert child.stdin is not None
            with contextlib.suppress(OSError):
                child.stdin.write("exit\n")
                child.stdin.flush()
            returncode = child.wait(timeout=30)
            assert (returncode == 0) is not killed, (
                f"process ended with {returncode}; expected killed={killed}"
            )

    def test_bench_teardown_kills_only_a_process_it_can_identify(
        self,
        bench: Bench,
        observed_process: subprocess.Popen[str],
        prefix: str,
        record: str,
    ) -> None:
        """Plant owned child identities for the real fixture finalizer to inspect."""
        child = observed_process
        created = psutil.Process(child.pid).create_time()
        (bench.fake / f"{prefix}-0.pid").write_text(str(child.pid))
        identity = bench.fake / f"{prefix}-0.created"
        if record == "unreadable":
            identity.mkdir()
        elif record == "invalid":
            identity.write_text("incomplete-identity")
        elif record != "none":
            # * The earlier owner of the pid was created before this process.
            recorded = created if record == "own" else created - 1.0
            identity.write_text(repr(recorded))


@pytest.mark.skipif(
    sys.platform != "win32" or sys.prefix == sys.base_prefix,
    reason="Windows venv redirector launcher",
)
def test_bench_reap_ends_the_python_behind_a_recorded_venv_launcher(
    bench: Bench,
) -> None:
    """Catch killing only the recorded launcher and leaking its real interpreter."""
    with _own_process() as child:
        launcher = psutil.Process(child.pid)
        # * The child's ready pipe proves the interpreter behind the launcher exists.
        descendants = launcher.children(recursive=True)
        assert descendants, "the venv launcher did not expose its real interpreter"
        try:
            (bench.fake / "grandchild-0.created").write_text(
                repr(launcher.create_time())
            )
            (bench.fake / "grandchild-0.pid").write_text(str(launcher.pid))

            bench.reap()

            _terminated, survivors = psutil.wait_procs(
                [launcher, *descendants], timeout=30
            )
            assert survivors == [], [process.pid for process in survivors]
        finally:
            # * Retain live identities; fallback cleanup touches only this child.
            for process in descendants:
                with contextlib.suppress(psutil.Error):
                    process.kill()
            _terminated, remaining = psutil.wait_procs(descendants, timeout=30)
            assert remaining == []


def test_resume_skips_finished_configs_and_retry_failed_reruns_the_rest(
    bench: Bench,
) -> None:
    """Catches: re-measuring finished configs; --retry-failed ignoring failures."""
    bench.env(FAKE_FAIL_NAME="clip02")
    bench.run("whisper-cpu-P1x1,whisper-cpu-P2x1")
    first = {
        name: (bench.out / "results" / f"{name}__r0.json").read_bytes()
        for name in ("whisper-cpu-P1x1", "whisper-cpu-P2x1")
    }
    assert bench.result("whisper-cpu-P1x1")["status"] == "partial"

    for marker in bench.fake.glob("worker-*.pid"):
        marker.unlink()
    bench.run("whisper-cpu-P1x1,whisper-cpu-P2x1")
    assert bench.pids("worker") == []
    for name, content in first.items():
        assert (bench.out / "results" / f"{name}__r0.json").read_bytes() == content

    bench.env(FAKE_FAIL_NAME="no-such-clip")
    bench.run("whisper-cpu-P1x1,whisper-cpu-P2x1", "--retry-failed")
    assert bench.result("whisper-cpu-P1x1")["status"] == "ok"
    assert bench.result("whisper-cpu-P2x1")["status"] == "ok"


def test_repeats_alternate_the_configuration_order(bench: Bench) -> None:
    """Catches: a fixed order across repeats (drift bias in every comparison)."""
    bench.run("whisper-cpu-P1x1,gigaam-cpu-P1x1", "--repeat", "2")

    order = (bench.fake / "order.txt").read_text().split()
    assert order == [
        "whisper-cpu-P1x1",
        "gigaam-cpu-P1x1",
        "gigaam-cpu-P1x1",
        "whisper-cpu-P1x1",
    ]


@pytest.mark.parametrize("command", ["run", "combo"])
def test_guard_skips_a_config_that_does_not_fit_without_starting_workers(
    bench: Bench, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """Catches: unsafe admission or a skipped result reused after memory frees."""
    free_bytes = 2 * 1024**3
    monkeypatch.setattr(
        runner,
        "query_commit",
        lambda: CommitInfo(free_bytes, None, None, "test"),
    )

    extra = (
        "--min-free-commit-gb",
        "1",
        "--estimate",
        "gigaam-cpu.commit_gb=6",
    )
    pair = "gigaam-cpu-P1x1+gigaam-cpu-P1x1"

    def invoke() -> dict[str, Any]:
        if command == "run":
            bench.run("gigaam-cpu-P2x1", *extra)
            return bench.result("gigaam-cpu-P2x1")
        bench.combo(pair, *extra)
        return _combo_doc(bench, pair)

    doc = invoke()
    # * Two processes need 12 GB; 2 GB free cannot hold even one.
    assert doc["status"] == "skipped"
    assert doc["guard"]["need_commit_gb"] == 12.0
    assert bench.pids("worker") == []

    monkeypatch.setattr(
        runner,
        "query_commit",
        lambda: CommitInfo(64 * 1024**3, None, None, "test"),
    )
    resumed = invoke()
    assert resumed["status"] == "ok"
    assert resumed["fingerprint"] == doc["fingerprint"]
    assert bench.pids("worker")


def test_dry_run_does_not_delete_earlier_results(bench: Bench) -> None:
    """Catches: --dry-run (documented as 'without running') destroying results."""
    bench.run("whisper-cpu-P1x1")
    result_file = bench.out / "results" / "whisper-cpu-P1x1__r0.json"
    assert result_file.is_file()

    bench.run("whisper-cpu-P1x1", "--dry-run", "--clear")

    assert result_file.is_file()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
def test_workers_die_with_the_parent(bench: Bench) -> None:
    """Catches: orphan workers (and grandchildren) after the parent is killed."""
    env = {
        **os.environ,
        "FAKE_DIR": str(bench.fake),
        "FAKE_LOAD_S": "3600",
        "FAKE_GRANDCHILD": "1",
    }
    args = bench.args("whisper-cpu-P2x1")
    parent = subprocess.Popen(  # noqa: S603
        [sys.executable, str(PARENT_DRIVER), *args],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + _WAIT_S
        while time.monotonic() < deadline and not (
            len(bench.pids("worker")) == 2 and len(bench.pids("grandchild")) == 2
        ):
            assert parent.poll() is None, "driver exited early"
            time.sleep(0.2)
        pids = [*bench.pids("worker"), *bench.pids("grandchild")]
        assert len(pids) == 4, "workers did not start"
        tracked: dict[int, psutil.Process] = {}
        for pid in pids:
            proc = psutil.Process(pid)
            tracked[pid] = proc
            for descendant in proc.children(recursive=True):
                tracked[descendant.pid] = descendant
            # * The venv launcher sits between the parent and the interpreter.
            launcher = proc.parent()
            if launcher is not None and launcher.pid != parent.pid:
                tracked[launcher.pid] = launcher
        assert all(p.is_running() for p in tracked.values())
        parent.kill()
        parent.wait(timeout=30)
        survivors = _gone(list(tracked.values()), 20.0)
        assert survivors == [], [(p.pid, p.cmdline()) for p in survivors]
    finally:
        parent.kill()
        parent.wait(timeout=30)


def _start_child(bench: Bench) -> subprocess.Popen[str]:
    """Start one fake worker like the orchestrator does and send its spec."""
    clip = sorted(bench.clips.glob("*.wav"))[0]
    spec = ChildSpec(
        config=parse_config_name("whisper-cpu-P1x1", ThreadDefaults(2, 2)),
        child_index=0,
        items=(AudioItem(index=0, path=clip, duration_s=1.0),),
        warmup_path=clip,
        language="ru",
        whisper_model="small",
        whisper_options={},
        whisper_chunk_s=30.0,
        gigaam_cuda_streams=False,
    )
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-X", "utf8", str(FAKE_WORKER), "_child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        cwd=PROJECT_ROOT,
    )
    assert proc.stdin is not None
    proc.stdin.write(encode_message(spec.to_json()))
    proc.stdin.flush()
    return proc


def _read_until(proc: subprocess.Popen[str], kind: str) -> dict[str, Any]:
    """Read protocol messages until one of type *kind* arrives."""
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        message = decode_message(line)
        if message is not None and message["type"] == kind:
            return message
    msg = f"child ended before sending {kind!r}"
    raise AssertionError(msg)


def _close_child(proc: subprocess.Popen[str]) -> None:
    """Kill the child if needed and close its pipes."""
    proc.kill()
    proc.wait(timeout=30)
    for stream in (proc.stdin, proc.stdout):
        if stream is not None:
            stream.close()


def test_child_exits_when_the_parent_closes_stdin(bench: Bench) -> None:
    """Catches: a worker that lingers (holding a model) after the parent is gone."""
    proc = _start_child(bench)
    try:
        _read_until(proc, "ready")
        assert proc.stdin is not None
        proc.stdin.close()

        code = proc.wait(timeout=30)

        assert code == EXIT_PARENT_GONE
    finally:
        _close_child(proc)


def test_child_serves_work_and_stops_gracefully(bench: Bench) -> None:
    """Catches: protocol drift between parent and child (want/work/result/stop)."""
    proc = _start_child(bench)
    try:
        _read_until(proc, "ready")
        want = _read_until(proc, "want")
        assert proc.stdin is not None
        proc.stdin.write(
            encode_message({"type": "work", "slot": want["slot"], "items": [0]})
        )
        proc.stdin.flush()

        result = _read_until(proc, "result")
        _read_until(proc, "want")
        proc.stdin.write(encode_message({"type": "stop"}))
        proc.stdin.flush()
        done = _read_until(proc, "done")

        assert result["item"] == 0
        assert result["text"] == "text clip00"
        assert result["error"] is None
        assert done["cpu_s"] >= 0.0
        assert proc.wait(timeout=30) == 0
    finally:
        _close_child(proc)


def test_bench_teardown_kills_the_live_worker_and_helper_it_recorded(
    bench: Bench,
) -> None:
    """Catches: worker records the teardown cannot confirm, so the net kills nothing."""
    bench.env(FAKE_GRANDCHILD="1")
    proc = _start_child(bench)
    try:
        _read_until(proc, "ready")
        recorded = [*bench.pids("worker"), *bench.pids("grandchild")]
        live = [*bench.processes("worker"), *bench.processes("grandchild")]
        assert [p.pid for p in live] == recorded
        assert len(live) == 2

        bench.reap()

        _terminated, survivors = psutil.wait_procs(live, timeout=30)
        assert survivors == []
    finally:
        _close_child(proc)


# --- sweep, combo, quality ---------------------------------------------------

_GROUP = "whisper-cpu-S-T1"


def test_sweep_stops_on_quality_loss_and_never_recommends_the_degraded_step(
    bench: Bench,
) -> None:
    """Catches: a knee table that recommends a step the quality rule rejected."""
    bench.env(FAKE_ITEM_S="0.15", FAKE_DEGRADE_FROM="3")

    bench.sweep(_GROUP, "--max-steps", "4")

    group = bench.group(_GROUP)
    steps = {step["name"]: step for step in group["steps"]}
    assert list(steps) == ["whisper-cpu-P1x1", "whisper-cpu-S2x1", "whisper-cpu-S3x1"]
    assert group["stop_reason"].startswith("quality")
    assert bench.step("whisper-cpu-S4x1") is None
    # * Every transcript of the degraded step has one of two words wrong: 50 pp.
    assert steps["whisper-cpu-S3x1"]["wer_delta_pp"] == pytest.approx(50.0)
    assert steps["whisper-cpu-S2x1"]["wer_delta_pp"] == 0.0
    xrt = {name: step["throughput_x"] for name, step in steps.items()}
    assert xrt["whisper-cpu-S3x1"] > xrt["whisper-cpu-S2x1"] > xrt["whisper-cpu-P1x1"]
    assert group["knees"]["best"] == "whisper-cpu-S2x1"
    assert group["knees"]["efficient"] == "whisper-cpu-S2x1"


def test_sweep_cuda_oom_is_the_saturation_point_and_the_run_continues(
    bench: Bench,
) -> None:
    """Catches: OOM treated as a crash that aborts the sweep or as a good step."""
    bench.env(FAKE_ITEM_S="0.05", FAKE_OOM_FROM="3")

    bench.sweep("whisper-cpu-S-T1,whisper-cpu-P-T1", "--max-steps", "4")

    threads = bench.group("whisper-cpu-S-T1")
    procs = bench.group("whisper-cpu-P-T1")
    assert [s["name"] for s in threads["steps"]] == [
        "whisper-cpu-P1x1",
        "whisper-cpu-S2x1",
        "whisper-cpu-S3x1",
    ]
    assert [s["name"] for s in procs["steps"]] == [
        "whisper-cpu-P1x1",
        "whisper-cpu-P2x1",
        "whisper-cpu-P3x1",
    ]
    assert threads["stop_reason"].startswith("cuda_oom")
    assert procs["stop_reason"].startswith("cuda_oom")
    assert threads["knees"]["best"] == "whisper-cpu-S2x1"
    assert procs["knees"]["best"] == "whisper-cpu-P2x1"


def test_sweep_guard_refusal_stops_the_group_and_is_decided_again_on_resume(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches: a refused step stored as final (never measured once memory frees)."""
    bench.env(FAKE_ITEM_S="0.05")
    monkeypatch.setattr(
        runner, "query_commit", lambda: CommitInfo(3 * 1024**3, None, None, "test")
    )
    estimate = [
        "--estimate",
        "whisper-cpu.commit_gb=1",
        "--estimate",
        "whisper-cpu.extra_thread_gb=1",
    ]

    bench.sweep(_GROUP, "--max-steps", "4", *estimate)

    # * Needs: P1 1 GB, S2 2 GB, S3 3 GB (fits 3 GB free), S4 4 GB (refused).
    group = bench.group(_GROUP)
    assert [s["name"] for s in group["steps"]][-1] == "whisper-cpu-S4x1"
    assert group["stop_reason"].startswith("guard")
    skipped = bench.step("whisper-cpu-S4x1")
    assert skipped is not None
    assert skipped["status"] == "skipped"
    assert group["knees"]["best"] == "whisper-cpu-S3x1"

    monkeypatch.setattr(
        runner,
        "query_commit",
        lambda: CommitInfo(64 * 1024**3, None, None, "test"),
    )
    bench.sweep(_GROUP, "--max-steps", "4", *estimate)

    remeasured = bench.step("whisper-cpu-S4x1")
    assert remeasured is not None
    assert remeasured["status"] == "ok"


def test_sweep_resume_reuses_same_inputs_and_remeasures_changed_inputs(
    bench: Bench,
) -> None:
    """Catches: stale results reused after the workload changed (or discarded)."""
    bench.env(FAKE_ITEM_S="0.05")
    results = bench.out / "sweep" / "results"
    bench.sweep(_GROUP, "--max-steps", "2")
    first_fp = bench.group(_GROUP)["fingerprint"]
    first = {path.name: path.read_bytes() for path in results.glob("*.json")}
    assert sorted(first) == ["whisper-cpu-P1x1.json", "whisper-cpu-S2x1.json"]

    for marker in bench.fake.glob("worker-*.pid"):
        marker.unlink()
    bench.sweep(_GROUP, "--max-steps", "2")
    assert bench.pids("worker") == []
    assert {p.name: p.read_bytes() for p in results.glob("*.json")} == first

    bench.sweep(_GROUP, "--max-steps", "2", "--clips-per-stream", "5")
    second_fp = bench.group(_GROUP)["fingerprint"]
    assert second_fp != first_fp
    assert len(bench.pids("worker")) >= 1
    stale = sorted(path.name for path in (results / "stale").glob("*.json"))
    assert stale == [
        f"whisper-cpu-P1x1__{first_fp}.json",
        f"whisper-cpu-S2x1__{first_fp}.json",
    ]
    for name in ("whisper-cpu-P1x1", "whisper-cpu-S2x1"):
        current = bench.step(name)
        assert current is not None
        assert current["fingerprint"] == second_fp
    summary = json.loads((bench.out / "sweep" / "summary.json").read_text("utf-8"))
    assert summary["fingerprint"] == second_fp
    assert [g["fingerprint"] for g in summary["groups"]] == [second_fp]

    for marker in bench.fake.glob("worker-*.pid"):
        marker.unlink()
    bench.sweep(
        _GROUP, "--max-steps", "2", "--clips-per-stream", "5", "--max-clips", "4"
    )
    third = bench.step("whisper-cpu-P1x1")
    assert third is not None
    assert third["fingerprint"] not in {first_fp, second_fp}
    assert len(bench.pids("worker")) >= 1
    assert len({item["clip"] for item in third["items"]}) <= 4


def test_run_resume_remeasures_when_the_workload_changed(bench: Bench) -> None:
    """Catches: resume reusing results for another selection or replaced audio."""
    bench.run("whisper-cpu-P1x1")
    old = bench.result("whisper-cpu-P1x1")
    assert len(old["items"]) == len(_DURATIONS)

    bench.run("whisper-cpu-P1x1", "--max-clips", "3")

    new = bench.result("whisper-cpu-P1x1")
    assert len(new["items"]) == 3
    assert new["fingerprint"] != old["fingerprint"]
    stale = list((bench.out / "results" / "stale").glob("*.json"))
    assert [p.name for p in stale] == [
        f"whisper-cpu-P1x1__r0__{old['fingerprint']}.json"
    ]

    # * Replacing PCM samples preserves the WAV name, size, and duration.
    # * The previous measurement still belongs to different input audio.
    clip_path = bench.clips / "clip00.wav"
    with clip_path.open("r+b") as audio_file:
        audio_file.seek(-2, os.SEEK_END)
        audio_file.write(b"\x01\x00")
    loads_before = (bench.fake / "order.txt").read_bytes()

    bench.run("whisper-cpu-P1x1", "--max-clips", "3")

    replaced = bench.result("whisper-cpu-P1x1")
    assert replaced["fingerprint"] != new["fingerprint"]
    assert (bench.fake / "order.txt").read_bytes() != loads_before
    assert len(replaced["items"]) == 3
    archived = (
        bench.out
        / "results"
        / "stale"
        / f"whisper-cpu-P1x1__r0__{new['fingerprint']}.json"
    )
    assert json.loads(archived.read_text(encoding="utf-8")) == new


@pytest.mark.parametrize("command", ["run", "sweep", "combo"])
def test_dry_run_creates_nothing_in_a_new_out_dir(bench: Bench, command: str) -> None:
    """Catches: a dry run that creates directories or files in --out-dir."""
    if command == "run":
        args = bench.args("whisper-cpu-P1x1", "--dry-run")
    elif command == "sweep":
        args = [
            "sweep",
            "--groups",
            _GROUP,
            "--threads",
            "1",
            *bench.shared("--dry-run"),
        ]
    else:
        pair = "whisper-cpu-S2x1+gigaam-cpu-P1x1"
        args = ["combo", "--pairs", pair, *bench.shared("--dry-run")]

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    assert not bench.out.exists()
    assert bench.pids("worker") == []


def test_sweep_dry_run_leaves_results_of_other_inputs_in_place(bench: Bench) -> None:
    """Catches: a dry run moving or deleting results as 'stale'."""
    bench.env(FAKE_ITEM_S="0.05")
    bench.sweep(_GROUP, "--max-steps", "2")
    results = bench.out / "sweep" / "results"
    before = {path.name: path.read_bytes() for path in results.glob("*.json")}

    bench.sweep(_GROUP, "--max-steps", "2", "--clips-per-stream", "9", "--dry-run")

    assert {path.name: path.read_bytes() for path in results.glob("*.json")} == before
    assert not (results / "stale").exists()


def test_run_scores_wer_against_the_manifest(bench: Bench) -> None:
    """Catches: WER not wired from the manifest to the result and the summary."""
    bench.env(FAKE_DEGRADE_FROM="1")

    bench.run("whisper-cpu-P1x1", "--manifest", str(bench.manifest()))

    wer = bench.result("whisper-cpu-P1x1")["quality"]["wer"]
    # * Six clips, two reference words each; the fake corrupts one word per clip.
    assert wer["ref_words"] == 12
    assert wer["wer_percent"] == pytest.approx(50.0)
    rows = json.loads((bench.out / "summary.json").read_text("utf-8"))
    assert rows[0]["wer_percent"] == pytest.approx(50.0)


def _combo_doc(bench: Bench, key: str) -> dict[str, Any]:
    path = bench.out / "combos" / "results" / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_combo_runs_two_sides_on_disjoint_clips_under_one_clock(bench: Bench) -> None:
    """Catches: shared clips, sides that stop overlapping, or wrong pair accounting."""
    bench.env(FAKE_ITEM_S="0.1")
    pair = "whisper-cpu-S2x1+gigaam-cpu-P1x1"
    bench.sweep(_GROUP, "--max-steps", "2", "--min-step-seconds", "1.5")
    solo = bench.step("whisper-cpu-S2x1")
    assert solo is not None

    bench.combo(pair, "--min-step-seconds", "1.5")

    doc = _combo_doc(bench, pair)
    side_a, side_b = doc["sides"]
    assert doc["status"] == "ok"
    assert [side_a["label"], side_b["label"]] == ["a", "b"]
    assert {i["clip"] for i in side_a["items"]} <= {0, 2, 4}
    assert {i["clip"] for i in side_b["items"]} <= {1, 3, 5}
    assert all(i["index"] < 1_000_000 for i in side_a["items"])
    assert all(i["index"] >= 1_000_000 for i in side_b["items"])
    # * Both sides keep working for the whole shared measurement (>= 1.5 s).
    for side, minimum in ((side_a, 4 * 2), (side_b, 4 * 1)):
        assert side["timing"]["wall_s"] >= 1.4
        assert side["timing"]["items_processed"] > minimum
        assert all(i["error"] is None for i in side["items"])
    own = side_a["timing"]["throughput_x"]
    assert side_a["solo_throughput_x"] == solo["timing"]["throughput_x"]
    assert side_a["slowdown_percent"] == pytest.approx(
        100.0 * (1.0 - own / side_a["solo_throughput_x"])
    )
    assert side_b["solo_throughput_x"] is None
    assert side_b["slowdown_percent"] is None
    timing = doc["timing"]
    assert timing["audio_s"] == pytest.approx(
        side_a["timing"]["audio_s"] + side_b["timing"]["audio_s"], abs=0.01
    )
    walls = [side_a["timing"]["wall_s"], side_b["timing"]["wall_s"]]
    assert timing["wall_s"] == pytest.approx(max(walls), abs=0.01)
    assert (bench.out / "combos" / "summary.md").is_file()


def test_combo_reports_a_failing_side_without_hiding_the_other(bench: Bench) -> None:
    """Catches: a combo whose status ignores a failing side."""
    bench.env(FAKE_FAIL_NAME="clip01")
    pair = "whisper-cpu-P1x1+gigaam-cpu-P1x1"

    bench.combo(pair)

    doc = _combo_doc(bench, pair)
    side_a, side_b = doc["sides"]
    assert side_a["status"] == "ok"
    assert side_b["status"] == "partial"
    assert doc["status"] == "partial"


def test_combo_keeps_the_fast_side_busy_until_the_slow_side_met_its_minimum(
    bench: Bench,
) -> None:
    """Catches: a side that finishes its share alone, leaving the other one solo."""
    bench.env(FAKE_ITEM_S_WHISPER="0.05", FAKE_ITEM_S_GIGAAM="0.4")
    pair = "whisper-cpu-S2x1+gigaam-cpu-P1x1"

    bench.combo(pair)

    doc = _combo_doc(bench, pair)
    side_a, side_b = doc["sides"]
    assert doc["status"] == "ok"
    # * Side a needs 8 items (0.2 s of work) but is fed until side b dispatched
    # * its 4 items (about 1.2 s at 0.4 s each).
    assert side_a["timing"]["items_processed"] > 20
    assert side_a["timing"]["window"] == side_b["timing"]["window"] == "common"
    assert side_a["timing"]["wall_s"] == side_b["timing"]["wall_s"]
    assert doc["timing"]["wall_s"] == side_a["timing"]["wall_s"]
    # * The window ends when the first side finished; the other ran longer.
    assert side_b["timing"]["full_wall_s"] > side_b["timing"]["wall_s"] + 0.2
    assert side_a["timing"]["full_wall_s"] >= side_a["timing"]["wall_s"] - 0.01
    assert side_b["timing"]["audio_s"] <= side_b["timing"]["full_audio_s"] + 0.01


def test_combo_with_a_hanging_side_ends_by_timeout_without_orphans(
    bench: Bench,
) -> None:
    """Catches: a side held open forever by a side that never finishes."""
    bench.env(FAKE_HANG_GIGAAM="1")
    pair = "whisper-cpu-S2x1+gigaam-cpu-P1x1"
    started = time.monotonic()

    bench.combo(pair, "--max-config-seconds", "8")

    doc = _combo_doc(bench, pair)
    assert doc["status"] == "timeout"
    assert time.monotonic() - started < 60.0
    workers = [p for p in map(_proc, bench.pids("worker")) if p is not None]
    assert _gone(workers, 15.0) == []


def test_combo_with_a_failing_side_still_terminates(bench: Bench) -> None:
    """Catches: a side whose items all fail blocking or hanging the pair."""
    bench.env(FAKE_FAIL_NAME_GIGAAM="clip", FAKE_ITEM_S_WHISPER="0.05")
    pair = "whisper-cpu-S2x1+gigaam-cpu-P1x1"
    started = time.monotonic()

    bench.combo(pair)

    doc = _combo_doc(bench, pair)
    side_a, side_b = doc["sides"]
    assert side_b["status"] == "error"
    assert side_a["status"] == "ok"
    assert doc["status"] == "error"
    assert time.monotonic() - started < 60.0


def test_rejected_step_stays_rejected_on_resume_summarize_and_auto_combos(
    bench: Bench,
) -> None:
    """Catches: a rejected knee returning via resume, summarize, or auto combos."""
    bench.env(FAKE_ITEM_S="0.15", FAKE_DEGRADE_FROM="3")
    bench.sweep(_GROUP, "--max-steps", "4")
    for marker in bench.fake.glob("worker-*.pid"):
        marker.unlink()

    bench.sweep(_GROUP, "--max-steps", "4")

    assert bench.pids("worker") == []
    group = bench.group(_GROUP)
    steps = group["steps"]
    assert [bool(s.get("rejected")) for s in steps] == [False, False, True]
    assert group["knees"]["best"] == "whisper-cpu-S2x1"
    summarized = CliRunner().invoke(cli.app, ["summarize", "--out-dir", str(bench.out)])
    assert summarized.exit_code == 0, summarized.output
    summary = (bench.out / "sweep" / "summary.md").read_text(encoding="utf-8")
    assert "rejected" in summary
    assert "whisper-cpu-S3x1" in summary
    cuda = dict(group)
    cuda.update(
        group="gigaam-cuda-B",
        knees={"best": "gigaam-cuda-B8x4", "best_xrt": 90.0},
    )
    groups_dir = bench.out / "sweep" / "groups"
    (groups_dir / "gigaam-cuda-B.json").write_text(json.dumps(cuda), encoding="utf-8")
    for knee in ("best", "efficient"):
        cuda["knees"] = {
            "best": "gigaam-cuda-B8x4",
            "best_xrt": 90.0,
            "efficient": "gigaam-cuda-B8x4",
            "efficient_xrt": 90.0,
        }
        (groups_dir / "gigaam-cuda-B.json").write_text(
            json.dumps(cuda), encoding="utf-8"
        )
        pairs = auto_combos(
            bench.out, group["fingerprint"], ThreadDefaults(4, 4), knee=knee
        )
        assert [(a.name, b.name) for a, b in pairs] == [
            ("gigaam-cuda-B8x4", "whisper-cpu-S2x1")
        ]


def _safety_command(bench: Bench, command: str, *extra: str) -> dict[str, Any]:
    """Run a minimal measurement through the requested real CLI entry point."""
    if command == "run":
        bench.run("whisper-cpu-P1x1", *extra)
        return bench.result("whisper-cpu-P1x1")
    if command == "sweep":
        bench.sweep(_GROUP, "--max-steps", "1", *extra)
        document = bench.step("whisper-cpu-P1x1")
        assert document is not None
        return document
    pair = "whisper-cpu-P1x1+gigaam-cpu-P1x1"
    bench.combo(pair, *extra)
    return _combo_doc(bench, pair)


def _fail_job_api(monkeypatch: pytest.MonkeyPatch, api_name: str) -> None:
    """Inject a Win32 failure without replacing production job management."""
    api = sysmetrics._kernel32()  # noqa: SLF001 - external OS boundary
    proxy = JobApiFault(api, api_name)
    monkeypatch.setattr(sysmetrics, "_kernel32", lambda: proxy)


def _assert_no_workers(document: dict[str, Any], bench: Bench) -> None:
    """Require every recorded child and fake helper to have terminated."""
    sides = document.get("sides") or [document]
    pids = {worker["pid"] for side in sides for worker in side.get("children", [])}
    pids.update(bench.pids("worker"))
    pids.update(bench.pids("grandchild"))
    processes = [proc for pid in pids if (proc := _proc(pid)) is not None]
    assert _gone(processes, 10.0) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job commit cap")
def test_step_job_caps_the_sum_of_workers_and_kills_descendants(bench: Bench) -> None:
    """Catches: per-process caps, missing notifications, and surviving helpers."""
    bench.env(FAKE_COMMIT_MB="192", FAKE_GRANDCHILD="1")
    bench.run(
        "whisper-cpu-P1x1",
        "--job-commit-limit-gb",
        "0.75",
        "--max-config-seconds",
        "15",
    )
    calibration = bench.result("whisper-cpu-P1x1")
    assert calibration["status"] == "ok"
    assert calibration["job"]["cap_enforced"]
    peak_gb = calibration["job"]["peak_job_commit_gb"]
    assert peak_gb > 0
    _assert_no_workers(calibration, bench)

    # * A cap halfway between one and two measured worker trees leaves room
    # * for interpreter startup variation without letting two models fit.
    cap_gb = round(1.5 * peak_gb, 3)
    options = ("--job-commit-limit-gb", str(cap_gb), "--max-config-seconds", "15")

    bench.run("whisper-cpu-P1x1", *options, "--repeat", "2")
    single = bench.result("whisper-cpu-P1x1", repeat=1)
    assert single["status"] == "ok"
    assert single["job"]["cap_enforced"]
    assert single["job"]["limit_gb"] == pytest.approx(cap_gb)
    _assert_no_workers(single, bench)

    # * Each worker fits alone. The two worker trees exceed their shared cap;
    # * their requested buffers total only 384 MiB.
    bench.run("whisper-cpu-P2x1", *options)
    capped = bench.result("whisper-cpu-P2x1")
    assert capped["status"] == "memory_cap"
    assert capped["job"]["cap_enforced"]
    assert capped["job"]["cap_reason"] == "explicit"
    assert capped["job"]["limit_gb"] == pytest.approx(cap_gb)
    assert capped["job"]["cap_hit"]
    assert not capped["job"].get("cap_inferred_from_peak")
    assert capped["job"]["peak_job_commit_gb"] > 0
    _assert_no_workers(capped, bench)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job commit cap")
@pytest.mark.parametrize(
    "api_name",
    ["CreateJobObjectW", "SetInformationJobObject", "AssignProcessToJobObject"],
)
def test_missing_cap_refuses_work_and_allow_uncapped_is_explicit(
    bench: Bench, monkeypatch: pytest.MonkeyPatch, api_name: str
) -> None:
    """Catches: starting work after cap setup/assignment fails or hiding bypass."""
    _fail_job_api(monkeypatch, api_name)
    options = ("--job-commit-limit-gb", "0.5", "--max-config-seconds", "8")

    bench.run("whisper-cpu-P2x1", *options)
    refused = bench.result("whisper-cpu-P2x1")
    assert refused["status"] == "skipped"
    assert not refused["job"]["cap_enforced"]
    assert bench.pids("worker") == []
    _assert_no_workers(refused, bench)

    bench.run("whisper-cpu-P2x1", *options, "--allow-uncapped")
    uncapped = bench.result("whisper-cpu-P2x1")
    assert uncapped["status"] == "ok"
    assert not uncapped["job"]["cap_enforced"]
    assert uncapped["fingerprint"] == refused["fingerprint"]
    assert uncapped["timing"]["audio_s"] == sum(_DURATIONS)
    _assert_no_workers(uncapped, bench)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job commit cap")
@pytest.mark.parametrize(
    "commit_env", ["FAKE_COMMIT_MB", "FAKE_ITEM_COMMIT_MB"], ids=["load", "transcribe"]
)
def test_failed_completion_port_preserves_the_cap_and_records_saturation(
    bench: Bench, monkeypatch: pytest.MonkeyPatch, commit_env: str
) -> None:
    """Catches: a missed cap signal misclassifying load or item allocation failure."""
    _fail_job_api(monkeypatch, "CreateIoCompletionPort")
    bench.env(**{commit_env: "384"})

    bench.run(
        "whisper-cpu-P1x1",
        "--job-commit-limit-gb",
        "0.25",
        "--max-config-seconds",
        "15",
    )

    document = bench.result("whisper-cpu-P1x1")
    assert document["status"] == "memory_cap", (
        document["job"]["limit_gb"],
        document["job"]["peak_job_commit_gb"],
        document["errors"][:1],
    )
    assert document["job"]["cap_enforced"]
    assert document["job"]["cap_monitor_error"]
    assert document["job"]["cap_hit"]
    assert document["job"]["cap_inferred_from_peak"]
    _assert_no_workers(document, bench)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job commit cap")
@pytest.mark.parametrize(
    ("explicit", "limit", "reason"),
    [(None, 0.5, "headroom"), (1.0, 0.5, "headroom"), (0.25, 0.25, "explicit")],
)
def test_cli_cap_uses_headroom_and_an_explicit_limit_can_only_lower_it(
    bench: Bench,
    monkeypatch: pytest.MonkeyPatch,
    explicit: float | None,
    limit: float,
    reason: str,
) -> None:
    """Catches: estimate-based caps or an explicit cap consuming the reserve."""
    reading = CommitInfo(640 * 1024**2, None, None, "GetPerformanceInfo")
    monkeypatch.setattr(sysmetrics, "query_commit", lambda: reading)
    monkeypatch.setattr(runner, "query_commit", lambda: reading)
    options = [
        "--commit-reserve-gb",
        "0.125",
        "--estimate",
        "whisper-cpu.commit_gb=0.05",
    ]
    if explicit is not None:
        options.extend(("--job-commit-limit-gb", str(explicit)))

    bench.run("whisper-cpu-P1x1", *options)

    document = bench.result("whisper-cpu-P1x1")
    assert document["status"] == "ok"
    assert document["job"]["limit_gb"] == pytest.approx(limit)
    assert document["job"]["cap_reason"] == reason
    assert document["job"]["cap_enforced"]
    _assert_no_workers(document, bench)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job commit watchdog")
@pytest.mark.parametrize("command", ["run", "sweep", "combo"])
def test_watchdog_kills_the_step_and_resume_remeasures_it(
    bench: Bench, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """Catches: watchdog kills without a transient outcome, or orphan children."""
    armed = True

    def commit_reading() -> CommitInfo:
        helpers_started = bool(bench.pids("worker") and bench.pids("grandchild"))
        free_gb = 0 if armed and helpers_started else 16
        return CommitInfo(free_gb * 1024**3, None, None, "GetPerformanceInfo")

    monkeypatch.setattr(sysmetrics, "query_commit", commit_reading)
    monkeypatch.setattr(runner, "query_commit", commit_reading)
    bench.env(FAKE_HANG="1", FAKE_GRANDCHILD="1")
    options = ("--commit-reserve-gb", "1", "--max-config-seconds", "8")

    stopped = _safety_command(bench, command, *options)

    assert stopped["status"] == "system_commit_low"
    assert stopped["job"]["cap_enforced"]
    assert bench.pids("grandchild")
    _assert_no_workers(stopped, bench)
    if command == "sweep":
        group = bench.group(_GROUP)
        assert group["stop_reason"].startswith("system_commit_low:")
        assert group["knees"]["best"] is None

    armed = False
    bench.env(FAKE_HANG="", FAKE_GRANDCHILD="")
    resumed = _safety_command(bench, command, *options)
    assert resumed["status"] == "ok"
    assert resumed["fingerprint"] == stopped["fingerprint"]
    _assert_no_workers(resumed, bench)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job commit cap")
@pytest.mark.parametrize("command", ["run", "sweep", "combo"])
def test_memory_cap_resume_tolerance_and_more_headroom_trigger_remeasurement(
    bench: Bench, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """Catches: permanent saturation from an old lower cap, or needless retries."""
    reading = CommitInfo(16 * 1024**3, None, None, "GetPerformanceInfo")
    monkeypatch.setattr(sysmetrics, "query_commit", lambda: reading)
    monkeypatch.setattr(runner, "query_commit", lambda: reading)
    bench.env(FAKE_COMMIT_MB_WHISPER="384", FAKE_COMMIT_MB_GIGAAM="0")
    options = ("--max-config-seconds", "15")
    initial_cap = 0.25
    capped = _safety_command(
        bench, command, *options, "--job-commit-limit-gb", str(initial_cap)
    )
    assert capped["status"] == "memory_cap"
    assert capped["job"]["cap_enforced"]
    _assert_no_workers(capped, bench)
    if command == "sweep":
        group = bench.group(_GROUP)
        assert group["stop_reason"].startswith("memory_cap:")
        assert group["knees"]["best"] is None

    loads = (bench.fake / "order.txt").read_bytes()
    bench.env(FAKE_COMMIT_MB_WHISPER="0")
    boundary = initial_cap + runner.CAP_REUSE_TOLERANCE_GB
    for cap in (initial_cap, boundary - 0.001, boundary):
        reused = _safety_command(
            bench, command, *options, "--job-commit-limit-gb", str(cap)
        )
        assert reused == capped
        assert (bench.fake / "order.txt").read_bytes() == loads

    resume_cap = boundary + 0.001
    resumed = _safety_command(
        bench, command, *options, "--job-commit-limit-gb", str(resume_cap)
    )
    assert resumed["status"] == "ok"
    assert resumed["fingerprint"] == capped["fingerprint"]
    assert resumed["job"]["limit_gb"] == pytest.approx(resume_cap)
    assert (bench.fake / "order.txt").read_bytes() != loads
    _assert_no_workers(resumed, bench)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job commit watchdog")
def test_watchdog_protects_the_reserve_while_the_engine_closes(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches: disabling reserve protection while a worker still holds memory."""
    low_observed_with_live_worker = False

    def commit_reading() -> CommitInfo:
        nonlocal low_observed_with_live_worker
        closing = (bench.fake / "closing-0").exists()
        if closing and any(psutil.pid_exists(pid) for pid in bench.pids("worker")):
            low_observed_with_live_worker = True
            free_gb = 0
        else:
            free_gb = 16
        return CommitInfo(free_gb * 1024**3, None, None, "GetPerformanceInfo")

    monkeypatch.setattr(sysmetrics, "query_commit", commit_reading)
    monkeypatch.setattr(runner, "query_commit", commit_reading)
    bench.env(FAKE_CLOSE_S="5", FAKE_COMMIT_MB="16")

    document = _safety_command(
        bench, "run", "--commit-reserve-gb", "1", "--max-config-seconds", "8"
    )

    assert low_observed_with_live_worker
    assert document["status"] == "system_commit_low"
    _assert_no_workers(document, bench)
