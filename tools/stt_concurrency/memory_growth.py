"""``memory-growth``: memory of one long-lived GigaAM GPU pool worker per job.

The scenario answers whether a long-lived pool worker can grow without bound.
It starts ONE real pool worker for the GigaAM GPU key
(``gigaam-e2e-ctc/ai-sage/GigaAM-v3@e2e_ctc/fp32/gpu``) through the production
launch path: :func:`~core.stt_pool.procs.default_worker_launch` and
:class:`~core.stt_pool.procs.WorkerProcess` (own Job Object with the commit cap
of the reservation plus ``memory.child_headroom_gib``), the production engine
factory (``GigaamGpuBatchEngine``) and a ``spec`` and launch environment built
by the supervisor's own functions (:mod:`core.stt_pool.worker_setup`) from a
validated pool config, the seed estimates and the pool limits.
It then feeds N jobs strictly one after another (the next job is sent after
the previous result, optionally after an idle gap) and records per job the
memory snapshot the worker attaches to every result (private bytes, CUDA
caching-allocator figures, cuFFT plan cache size), the Job Object peak, the
worker's private bytes read before the job, whole-GPU used VRAM, and timings.

The supervisor is bypassed on purpose: it adds no memory to the worker, and
its recycle, live cap raise and uptime cap would hide the raw growth. The
summary instead reports where the pool's recycle threshold
(``memory.recycle_commit_fraction`` x commit reservation) would have retired
the worker.

Inputs are synthetic (:mod:`stt_concurrency.growth_inputs`). Everything is
written under ``<out-dir>/<UTC stamp>-<label>/``: ``run.json`` (plan and stack),
``results.jsonl`` (one record per job, written as the jobs finish),
``results.csv``, ``summary.json``, ``summary.md`` and ``worker.log``. The worker's
temp directories live in ``<run dir>/tmp``; no queue root is used. The inputs
are deleted at the end unless ``--keep-inputs`` is given.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import IO, TYPE_CHECKING, Any, cast

import psutil  # type: ignore[import-untyped]
import typer

from core.stt_pool import ipc
from core.stt_pool.config import (
    CUDNN_V8_LRU_CACHE_ENV,
    PoolConfigError,
    config_summary,
    parse_pool_config,
)
from core.stt_pool.engine_api import (
    DEFAULT_ENGINE_FACTORY,
    MEMORY_SNAPSHOT_KEYS,
    EngineResult,
    LoadInfo,
    WorkerSpec,
    WorkItem,
)
from core.stt_pool.estimates import seed_estimate
from core.stt_pool.keys import device_for_backend, make_key
from core.stt_pool.procs import WorkerProcess, WorkerSpawnError, default_worker_launch
from core.stt_pool.scheduler import (
    Limits,
    reservation_for,
    threads_for,
    worker_capacity,
)
from core.stt_pool.worker_setup import (
    THREAD_ENV_VARS,
    worker_environment,
    worker_spec_options,
)

from . import growth_stats
from .configs import ConfigError
from .growth_inputs import AUDIO_FORMATS, prepare_inputs
from .paths import PROJECT_ROOT, SUMMARY_JSON_NAME, SUMMARY_MD_NAME
from .store import read_json
from .sysmetrics import GIB, MIB, find_nvidia_smi, query_commit, query_gpu
from .workload import WorkloadError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from core.stt_pool.config import PoolConfig
    from core.stt_pool.keys import WorkerKey
    from core.stt_pool.procs import WorkerLaunch
    from core.stt_pool.scheduler import Reservation

    from .growth_inputs import DurationPlan, InputClip

JsonDict = dict[str, Any]

# * User-tunable defaults.
GROWTH_DIRNAME = "stt_memory_growth"
DEFAULT_OUT_DIR = PROJECT_ROOT / ".cache" / GROWTH_DIRNAME
DEFAULT_JOBS = 300
DEFAULT_MIN_FREE_COMMIT_GIB = 20.0
DEFAULT_MIN_FREE_VRAM_GIB = 3.0
DEFAULT_COMMIT_RESERVE_GIB = 12.0
DEFAULT_JOB_TIMEOUT_S = 600.0
DEFAULT_LOAD_TIMEOUT_S = 600.0
DEFAULT_MAX_ERRORS = 5
DEFAULT_PLATEAU_TOLERANCE_MIB = 16.0
DEFAULT_PROGRESS_EVERY = 25
# * Upper bound of the size and duration options (GiB, MiB, seconds of audio):
# * far beyond any real value, and small enough that byte counts never overflow.
MAX_SIZE_OPTION = 1_000_000.0
PROVIDER = "gigaam-e2e-ctc"
COMPUTE_REQUEST = "auto"
LANGUAGE = "ru"
WORKER_ID = "growth-worker"
CALL_ID = "memory-growth"

# * --gpu-memory-bounds: ``pool`` keeps the config; ``library`` switches the four
# * GPU memory bounds of the pool off (the behavior before they existed).
BOUNDS_POOL = "pool"
BOUNDS_LIBRARY = "library"
GPU_MEMORY_BOUNDS = (BOUNDS_POOL, BOUNDS_LIBRARY)
LIBRARY_GPU_BOUNDS: Mapping[str, object] = MappingProxyType(
    {
        "cufft_plan_cache_max_size": 0,
        "cudnn_v8_lru_cache_limit": 0,
        "idle_cache_release_s": 0,
        "vram_cap": False,
    }
)

STATUS_COMPLETED = "completed"
STATUS_LOAD_FAILED = "load_failed"
STATUS_WORKER_EXITED = "worker_exited"
STATUS_FATAL = "fatal"
STATUS_TIMEOUT = "timeout"
STATUS_MEMORY_CAP = "memory_cap"
STATUS_SYSTEM_COMMIT_LOW = "system_commit_low"
STATUS_TOO_MANY_ERRORS = "too_many_errors"
STATUS_INTERRUPTED = "interrupted"
STATUS_INPUTS_FAILED = "inputs_failed"
STATUS_ERROR = "error"
STATUS_GUARD_REFUSED = "guard_refused"
# * Exit codes. The CLI wrapper adds 2 (invalid input, or the inputs could not
# * be built: status ``inputs_failed``, the run folder keeps ``run.json`` and
# * the summary) and 130 (Ctrl+C).
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_GUARD = 3
EXIT_SUMMARY_NOT_WRITTEN = 4

RUN_JSON_NAME = "run.json"
RESULTS_JSONL_NAME = "results.jsonl"
RESULTS_CSV_NAME = "results.csv"
WORKER_LOG_NAME = "worker.log"
INPUTS_DIRNAME = "inputs"
TEMP_DIRNAME = "tmp"
SUMMARY_SCHEMA = "askvlm.stt_memory_growth.summary"
RUN_SCHEMA = "askvlm.stt_memory_growth.run"
SUMMARY_VERSION = 1
# * Run fields of ``summary.json`` next to the computed ``idle_releases``/``stats``.
SUMMARY_IDENTITY_KEYS = (
    "run_id",
    "label",
    "status",
    "detail",
    "started_at",
    "finished_at",
    "key",
    "worker_pid",
    "job_limit_bytes",
    "load",
    "baseline",
    "memory_limit_at_job",
)

# * Marker of the engine's idle CUDA cache release in the worker log.
IDLE_RELEASE_MARKER = "idle CUDA cache released"
SHUTDOWN_WAIT_S = 30.0
CLOSE_TIMEOUT_S = 10.0
STACK_PROBE_TIMEOUT_S = 180.0
_LABEL_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
_ERROR_DETAIL_CHARS = 300
_STACK_PACKAGES = (
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "huggingface_hub",
    "numpy",
)
# * Runs in the worker interpreter with the package names as arguments and
# * prints one JSON line.
_STACK_PROBE = """
import json, sys
from importlib import metadata
info = {}
for name in sys.argv[1:]:
    try:
        info[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        info[name] = None
try:
    import torch
    info["cuda_runtime"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        info["device"] = torch.cuda.get_device_name(0)
except Exception as exc:
    info["torch_error"] = repr(exc)
print(json.dumps(info))
"""

RECORD_FIELDS: tuple[str, ...] = (
    "index",
    "job_id",
    "input",
    "planned_duration_s",
    "offset_s",
    "audio_s",
    "status",
    "error_kind",
    "error_detail",
    "text_chars",
    "run_s",
    "wall_s",
    "gap_before_s",
    "elapsed_s",
    "uptime_s",
    "timestamp",
    *MEMORY_SNAPSHOT_KEYS,
    "pre_job_private_bytes",
    "job_peak_bytes",
    "gpu_used_bytes",
    "free_commit_bytes",
    "memory_limit_report",
)


@dataclass(frozen=True, slots=True)
class GrowthOptions:
    """Options of one ``memory-growth`` run (see the CLI help for each field)."""

    jobs: int
    durations: DurationPlan
    source: str
    audio_format: str
    out_dir: Path
    label: str
    pool_config: Path | None = None
    gpu_memory_bounds: str = BOUNDS_POOL
    pool_set: tuple[str, ...] = ()
    reservation_commit_gib: float | None = None
    reservation_vram_gib: float | None = None
    idle_gap_s: float = 0.0
    idle_gap_every: int = 1
    keep_inputs: bool = False
    min_free_commit_gib: float = DEFAULT_MIN_FREE_COMMIT_GIB
    min_free_vram_gib: float = DEFAULT_MIN_FREE_VRAM_GIB
    commit_reserve_gib: float = DEFAULT_COMMIT_RESERVE_GIB
    job_timeout_s: float = DEFAULT_JOB_TIMEOUT_S
    load_timeout_s: float = DEFAULT_LOAD_TIMEOUT_S
    max_errors: int = DEFAULT_MAX_ERRORS
    gpu_sample: bool = True
    probe_stack: bool = True
    plateau_tolerance_mib: float = DEFAULT_PLATEAU_TOLERANCE_MIB
    progress_every: int = DEFAULT_PROGRESS_EVERY
    dry_run: bool = False
    engine_factory: str = DEFAULT_ENGINE_FACTORY

    def validate(self) -> None:
        """Check the options that the CLI cannot bound by type.

        Raises:
            ConfigError: For an invalid value.

        """
        self._validate_float_ranges()
        checks: tuple[tuple[bool, str], ...] = (
            (self.jobs >= 1, "--jobs must be at least 1"),
            (self.idle_gap_s >= 0, "--idle-gap-s must not be negative"),
            (self.idle_gap_every >= 1, "--idle-gap-every must be at least 1"),
            (self.job_timeout_s > 0, "--job-timeout-s must be positive"),
            (self.load_timeout_s > 0, "--load-timeout-s must be positive"),
            (self.max_errors >= 0, "--max-errors must not be negative"),
            (self.min_free_commit_gib >= 0, "--min-free-commit-gib must be >= 0"),
            (self.min_free_vram_gib >= 0, "--min-free-vram-gib must be >= 0"),
            (self.commit_reserve_gib >= 0, "--commit-reserve-gib must be >= 0"),
            (self.plateau_tolerance_mib >= 0, "--plateau-tolerance-mib must be >= 0"),
            (self.progress_every >= 0, "--progress-every must not be negative"),
            (
                self.audio_format in AUDIO_FORMATS,
                f"--audio-format must be one of {', '.join(AUDIO_FORMATS)}",
            ),
            (
                self.gpu_memory_bounds in GPU_MEMORY_BOUNDS,
                f"--gpu-memory-bounds must be one of {', '.join(GPU_MEMORY_BOUNDS)}",
            ),
            (
                _LABEL_RE.fullmatch(self.label) is not None,
                "--label must be 1-64 characters of A-Z, a-z, 0-9, '.', '_', '-'",
            ),
        )
        for passed, message in checks:
            if not passed:
                raise ConfigError(message)
        for name, value in (
            ("--reservation-commit-gib", self.reservation_commit_gib),
            ("--reservation-vram-gib", self.reservation_vram_gib),
        ):
            if value is not None and value <= 0:
                msg = f"{name} must be positive"
                raise ConfigError(msg)

    def _validate_float_ranges(self) -> None:
        """Reject non-finite and overflowing float options.

        NaN passes every ``<``/``>`` comparison test, and an infinite or huge
        value overflows later (byte counts, thread wait timeouts) instead of
        failing as invalid input. Durations are bounded again by
        :meth:`DurationPlan.validate`.

        Raises:
            ConfigError: For a value that is not finite or exceeds its limit.

        """
        plan = self.durations
        limited: tuple[tuple[str, float | None, float], ...] = (
            ("--duration-min-s", plan.min_s, MAX_SIZE_OPTION),
            ("--duration-max-s", plan.max_s, MAX_SIZE_OPTION),
            ("--fixed-duration-s", plan.fixed_s, MAX_SIZE_OPTION),
            ("--reservation-commit-gib", self.reservation_commit_gib, MAX_SIZE_OPTION),
            ("--reservation-vram-gib", self.reservation_vram_gib, MAX_SIZE_OPTION),
            ("--min-free-commit-gib", self.min_free_commit_gib, MAX_SIZE_OPTION),
            ("--min-free-vram-gib", self.min_free_vram_gib, MAX_SIZE_OPTION),
            ("--commit-reserve-gib", self.commit_reserve_gib, MAX_SIZE_OPTION),
            ("--plateau-tolerance-mib", self.plateau_tolerance_mib, MAX_SIZE_OPTION),
            ("--idle-gap-s", self.idle_gap_s, threading.TIMEOUT_MAX),
            ("--job-timeout-s", self.job_timeout_s, threading.TIMEOUT_MAX),
            ("--load-timeout-s", self.load_timeout_s, threading.TIMEOUT_MAX),
        )
        for name, value, limit in limited:
            if value is None:
                continue
            if not math.isfinite(value):
                msg = f"{name} must be a finite number (got {value!r})"
                raise ConfigError(msg)
            if abs(value) > limit:
                msg = f"{name} must be at most {limit:g} in magnitude (got {value:g})"
                raise ConfigError(msg)


# * ---------------------------------------------------------------------------
# * Pool config and worker plan
# * ---------------------------------------------------------------------------


def _base_payload(path: Path | None) -> dict[str, object]:
    """Return the base config document (empty for the code defaults).

    Raises:
        ConfigError: When the file cannot be read or is not a JSON object.

    """
    if path is None:
        return {}
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"cannot read --pool-config {path}: {exc}"
        raise ConfigError(msg) from exc
    if not isinstance(data, dict):
        msg = f"--pool-config {path} is not a JSON object"
        raise ConfigError(msg)
    return cast("dict[str, object]", data)


def _child_object(node: dict[str, object], name: str, token: str) -> dict[str, object]:
    """Return the JSON object *node[name]*, creating it when absent.

    Raises:
        ConfigError: When *node[name]* exists but is not an object.

    """
    child = node.setdefault(name, {})
    if not isinstance(child, dict):
        msg = f"{token}: {name!r} is not a JSON object in the base config"
        raise ConfigError(msg)
    return cast("dict[str, object]", child)


def apply_pool_setting(payload: dict[str, object], token: str) -> None:
    """Apply one ``SECTION.KEY=VALUE`` override to a config document in place.

    VALUE is parsed as JSON when it parses, else kept as a string; the dotted
    path may be deeper than two levels (``gpu.batch_lanes.gigaam=8``).

    Raises:
        ConfigError: For a token without ``=`` or with an empty path segment.

    """
    path, sep, raw = token.partition("=")
    parts = [part.strip() for part in path.split(".")]
    if not sep or not all(parts):
        msg = f"invalid --pool-set {token!r}; expected SECTION.KEY=VALUE"
        raise ConfigError(msg)
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    node = payload
    for part in parts[:-1]:
        node = _child_object(node, part, f"--pool-set {token!r}")
    node[parts[-1]] = value


def build_pool_config(options: GrowthOptions) -> PoolConfig:
    """Validate the pool config of the run (base file, bounds preset, overrides).

    ``stt_pool.json`` and ``ASKVLM_STT_POOL_CONFIG`` are never read: the base
    is the code defaults or ``--pool-config``.

    Raises:
        ConfigError: For an unreadable base or an invalid key or value.

    """
    payload = _base_payload(options.pool_config)
    if options.gpu_memory_bounds == BOUNDS_LIBRARY:
        _child_object(payload, "gpu", "--gpu-memory-bounds library").update(
            LIBRARY_GPU_BOUNDS
        )
    for token in options.pool_set:
        apply_pool_setting(payload, token)
    base = str(options.pool_config) if options.pool_config is not None else "defaults"
    try:
        return parse_pool_config(payload, source=f"{base} + memory-growth overrides")
    except PoolConfigError as exc:
        raise ConfigError(str(exc)) from exc


def gpu_worker_env(
    config: PoolConfig,
    limits: Limits,
    key: WorkerKey,
    reservation: Reservation,
    temp_dir: Path,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return the worker environment additions and the variables to remove.

    The additions are the supervisor's (:func:`worker_environment`: thread
    caps and the cuDNN v8 cache bound) plus ``TEMP``/``TMP``, which point the
    job temp directories into the run directory. Unlike the supervisor,
    which leaves an inherited variable alone, ``gpu.cudnn_v8_lru_cache_limit``
    0 removes the variable, so torch's own default applies.
    """
    env = worker_environment(config, limits, key, reservation)
    env["TEMP"] = env["TMP"] = str(temp_dir)
    removed = (
        () if config.gpu.cudnn_v8_lru_cache_limit > 0 else (CUDNN_V8_LRU_CACHE_ENV,)
    )
    return env, removed


def _without(launch: WorkerLaunch, names: Sequence[str]) -> WorkerLaunch:
    """Return *launch* without the environment variables *names* (any case)."""
    if not names:
        return launch
    drop = {name.upper() for name in names}
    env = {key: value for key, value in launch.env.items() if key.upper() not in drop}
    return replace(launch, env=MappingProxyType(env))


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    """Everything needed to start the worker of a run.

    Attributes:
        spec: ``spec`` message content.
        reservation: Commit/VRAM/host-thread reservation (seed estimates or
            the explicit overrides).
        job_limit_bytes: Job Object commit cap (reservation plus
            ``memory.child_headroom_gib``; 0 = uncapped).
        recycle_threshold_bytes: Private bytes at which the pool would retire
            the worker, or ``None`` when the recycle is off.
        launch: Command line, environment and working directory.
        removed_env: Variables removed from the worker environment.

    """

    spec: WorkerSpec
    reservation: Reservation
    job_limit_bytes: int
    recycle_threshold_bytes: int | None
    launch: WorkerLaunch
    removed_env: tuple[str, ...]


def plan_worker(
    config: PoolConfig, options: GrowthOptions, temp_dir: Path
) -> WorkerPlan:
    """Build the worker plan of the GigaAM GPU key from a validated config.

    Raises:
        ConfigError: When the config gives the GPU key no lanes.

    """
    key = make_key(PROVIDER, "", COMPUTE_REQUEST, "gpu")
    limits = Limits.from_config(config)
    capacity = worker_capacity(key, limits)
    reservation = reservation_for(key, capacity, seed_estimate(key), limits)
    if options.reservation_commit_gib is not None:
        reservation = replace(
            reservation, commit_bytes=round(options.reservation_commit_gib * GIB)
        )
    if options.reservation_vram_gib is not None:
        reservation = replace(
            reservation, vram_bytes=round(options.reservation_vram_gib * GIB)
        )
    spec = WorkerSpec(
        worker_id=WORKER_ID,
        instance_id=uuid.uuid4().hex,
        key=key,
        capacity=capacity,
        threads_per_slot=threads_for(key, limits),
        host_threads=reservation.host_threads,
        device=device_for_backend(key.backend),
        options=worker_spec_options(config, limits, key, capacity, reservation),
        engine_factory=options.engine_factory,
    )
    env, removed = gpu_worker_env(config, limits, key, reservation, temp_dir)
    launch = _without(
        default_worker_launch(project_root=PROJECT_ROOT, extra_env=env), removed
    )
    commit = reservation.commit_bytes
    fraction = config.memory.recycle_commit_fraction
    return WorkerPlan(
        spec=spec,
        reservation=reservation,
        job_limit_bytes=commit + config.memory.child_headroom_bytes
        if commit > 0
        else 0,
        recycle_threshold_bytes=round(commit * fraction) if fraction > 0 else None,
        launch=launch,
        removed_env=removed,
    )


# * ---------------------------------------------------------------------------
# * Host readings
# * ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GuardReading:
    """Free system commit and VRAM before the run, and the refusals they cause.

    Attributes:
        free_commit_bytes: Free system commit at the reading.
        commit_source: Probe that measured it.
        gpu_used_bytes: Used VRAM of the whole GPU 0, if known.
        gpu_free_bytes: Free VRAM of GPU 0, if known.
        job_limit_bytes: Job Object commit cap the worker would get (``0``:
            uncapped).
        refusals: Reasons the run must not start (empty: it may start).

    """

    free_commit_bytes: int
    commit_source: str
    gpu_used_bytes: int | None
    gpu_free_bytes: int | None
    job_limit_bytes: int
    refusals: tuple[str, ...]

    def to_json(self) -> JsonDict:
        """Serialize for the run metadata."""
        return {
            "free_commit_bytes": self.free_commit_bytes,
            "commit_source": self.commit_source,
            "gpu_used_bytes": self.gpu_used_bytes,
            "gpu_free_bytes": self.gpu_free_bytes,
            "job_limit_bytes": self.job_limit_bytes,
            "refusals": list(self.refusals),
        }


def gpu_used_bytes(nvidia_smi: str | None) -> int | None:
    """Return the used VRAM of the whole GPU 0 in bytes, if known."""
    sample = query_gpu(nvidia_smi) if nvidia_smi else None
    return None if sample is None else round(sample.used_mb * MIB)


def _commit_refusals(
    options: GrowthOptions, free_bytes: int, job_limit_bytes: int
) -> list[str]:
    """Return the refusals caused by free system commit.

    The run needs ``--min-free-commit-gib`` of free commit, and the worker's
    Job Object cap must fit into the free commit minus ``--commit-reserve-gib``
    (the level at which the run would stop itself), so the worker can never
    take the commit the reserve keeps for the rest of the PC.
    """
    refusals: list[str] = []
    if free_bytes < round(options.min_free_commit_gib * GIB):
        refusals.append(
            f"free system commit {free_bytes / GIB:.1f} GiB is below "
            f"--min-free-commit-gib {options.min_free_commit_gib:g}"
        )
    room = free_bytes - round(options.commit_reserve_gib * GIB)
    if job_limit_bytes > 0 and job_limit_bytes > room:
        refusals.append(
            f"the worker's Job Object cap {job_limit_bytes / GIB:.2f} GiB does not "
            f"fit into free system commit {free_bytes / GIB:.1f} GiB minus "
            f"--commit-reserve-gib {options.commit_reserve_gib:g} "
            f"({max(room, 0) / GIB:.1f} GiB left)"
        )
    return refusals


def _vram_refusals(options: GrowthOptions, free_vram: int | None) -> list[str]:
    """Return the refusals caused by free VRAM (none with a 0 threshold)."""
    if options.min_free_vram_gib <= 0:
        return []
    if free_vram is None:
        return [
            "free VRAM is unknown (nvidia-smi missing or failed); pass "
            "--min-free-vram-gib 0 to skip the check"
        ]
    if free_vram < round(options.min_free_vram_gib * GIB):
        return [
            f"free VRAM {free_vram / GIB:.2f} GiB is below --min-free-vram-gib "
            f"{options.min_free_vram_gib:g}"
        ]
    return []


def read_guard(
    options: GrowthOptions, nvidia_smi: str | None, job_limit_bytes: int
) -> GuardReading:
    """Read free commit and VRAM and decide whether the run may start.

    Args:
        options: Run options (thresholds and the commit reserve).
        nvidia_smi: ``nvidia-smi`` path, if installed.
        job_limit_bytes: Job Object commit cap of the planned worker
            (:attr:`WorkerPlan.job_limit_bytes`; ``0``: uncapped).

    Returns:
        The readings and the refusals.

    """
    commit = query_commit()
    sample = query_gpu(nvidia_smi) if nvidia_smi else None
    free_vram = None if sample is None else round(sample.free_mb * MIB)
    refusals = [
        *_commit_refusals(options, commit.free_bytes, job_limit_bytes),
        *_vram_refusals(options, free_vram),
    ]
    return GuardReading(
        free_commit_bytes=commit.free_bytes,
        commit_source=commit.source,
        gpu_used_bytes=None if sample is None else round(sample.used_mb * MIB),
        gpu_free_bytes=free_vram,
        job_limit_bytes=job_limit_bytes,
        refusals=tuple(refusals),
    )


def probe_stack() -> JsonDict:
    """Return the ML stack versions of this interpreter (the worker's packages).

    Runs a short child process with the same interpreter and environment;
    a failure is reported in the result instead of raised.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - this interpreter, fixed code
            [sys.executable, "-c", _STACK_PROBE, *_STACK_PACKAGES],
            capture_output=True,
            text=True,
            timeout=STACK_PROBE_TIMEOUT_S,
            check=False,
            cwd=str(PROJECT_ROOT),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"probe_error": repr(exc)}
    lines = completed.stdout.strip().splitlines()
    try:
        data: object = json.loads(lines[-1]) if lines else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return {"probe_error": (completed.stderr or completed.stdout)[-500:]}
    return cast("JsonDict", data)


def process_private_bytes(pid: int | None) -> int | None:
    """Return the private bytes of process *pid* (``None`` when unknown)."""
    if pid is None:
        return None
    try:
        info = psutil.Process(pid).memory_info()
    except (OSError, psutil.Error):
        return None
    private = getattr(info, "private", None)
    return (
        private if isinstance(private, int) and not isinstance(private, bool) else None
    )


def now_iso() -> str:
    """Return the current UTC time in ISO format (seconds)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def echo(message: str) -> None:
    """Print a progress line."""
    typer.echo(message)


# * ---------------------------------------------------------------------------
# * Worker session
# * ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Event:
    """One event of the worker: a message, its EOF, or a commit-cap report."""

    kind: str
    message: Mapping[str, object] | None = None
    pid: int | None = None


@dataclass(slots=True)
class _Wait:
    """Outcome of waiting for one job result."""

    result: EngineResult | None = None
    stop_status: str | None = None
    detail: str | None = None
    memory_limit: bool = False


def _result_for(
    message: Mapping[str, object], attempt_id: str
) -> tuple[EngineResult | None, str | None]:
    """Match a worker message against the awaited attempt.

    Returns:
        ``(result, None)`` for the attempt's result, ``(None, detail)`` for a
        malformed result of the attempt, ``(None, None)`` for anything else.

    """
    if message.get("type") != ipc.MSG_RESULT:
        return None, None
    try:
        result = EngineResult.from_message(message)
    except ValueError as exc:
        if message.get("attempt_id") != attempt_id:
            return None, None
        return None, f"malformed result of {attempt_id}: {exc}"
    return (result, None) if result.attempt_id == attempt_id else (None, None)


class _Session:
    """One real pool worker process and the queue its threads feed."""

    def __init__(self, plan: WorkerPlan, log_path: Path) -> None:
        self._events: queue.SimpleQueue[_Event] = queue.SimpleQueue()
        self.proc = WorkerProcess(
            WORKER_ID,
            plan.launch,
            log_path=log_path,
            memory_limit_bytes=plan.job_limit_bytes or None,
            on_event=self._on_event,
            on_memory_limit=self._on_memory_limit,
        )

    def _on_event(self, _worker_id: str, message: dict[str, object] | None) -> None:
        self._events.put(_Event("eof" if message is None else "message", message))

    def _on_memory_limit(self, _worker_id: str, pid: int) -> None:
        self._events.put(_Event("memory_limit", pid=pid))

    def next_event(self, timeout_s: float) -> _Event | None:
        """Return the next event, or ``None`` after *timeout_s* without one."""
        try:
            return self._events.get(timeout=max(0.0, timeout_s))
        except queue.Empty:
            return None

    def exit_detail(self) -> str:
        """Describe the exit of the worker with the tail of its log."""
        return (
            f"exit code {self.proc.poll()}; log tail: "
            f"{self.proc.log_tail(_ERROR_DETAIL_CHARS)!r}"
        )

    def terminal(self, event: _Event) -> tuple[str, str] | None:
        """Return ``(status, detail)`` when *event* ends the run, else ``None``."""
        if event.kind == "eof":
            return STATUS_WORKER_EXITED, self.exit_detail()
        message = event.message or {}
        if message.get("type") == ipc.MSG_FATAL:
            return STATUS_FATAL, (
                f"stage {message.get('stage')}: {message.get('error_kind')}: "
                f"{str(message.get('error_detail'))[:_ERROR_DETAIL_CHARS]}"
            )
        return None

    def await_load(self, timeout_s: float) -> tuple[LoadInfo | None, str | None]:
        """Wait for ``loaded``; return the load info or the failure detail."""
        deadline = time.monotonic() + timeout_s
        while True:
            event = self.next_event(deadline - time.monotonic())
            if event is None:
                return None, f"no 'loaded' within {timeout_s:g} s"
            if event.kind == "memory_limit":
                return None, "the worker hit its commit cap while loading"
            stop = self.terminal(event)
            if stop is not None:
                return None, stop[1]
            message = event.message or {}
            if message.get("type") == ipc.MSG_LOADED:
                try:
                    return LoadInfo.from_message(message), None
                except ValueError as exc:
                    return None, f"malformed 'loaded' message: {exc}"

    def await_result(self, attempt_id: str, timeout_s: float) -> _Wait:
        """Wait for the result of *attempt_id* (or an event that ends the run)."""
        wait = _Wait()
        deadline = time.monotonic() + timeout_s
        while True:
            event = self.next_event(deadline - time.monotonic())
            if event is None:
                wait.stop_status = STATUS_TIMEOUT
                wait.detail = f"no result within {timeout_s:g} s"
                return wait
            if event.kind == "memory_limit":
                wait.memory_limit = True
                continue
            stop = self.terminal(event)
            if stop is not None:
                wait.stop_status, wait.detail = stop
                return wait
            result, problem = _result_for(event.message or {}, attempt_id)
            if problem is not None:
                wait.stop_status, wait.detail = STATUS_FATAL, problem
                return wait
            if result is not None:
                wait.result = result
                return wait

    def idle(self, seconds: float) -> tuple[str, str] | None:
        """Stay idle for *seconds*; return ``(status, detail)`` if the run ends."""
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            event = self.next_event(remaining)
            if event is None:
                return None
            if event.kind == "memory_limit":
                return (
                    STATUS_MEMORY_CAP,
                    "the Job Object reported its commit cap while idle",
                )
            stop = self.terminal(event)
            if stop is not None:
                return stop
        return None

    def shutdown(self) -> None:
        """Ask for a graceful shutdown, wait for the exit, then release everything."""
        if self.proc.pid is not None and self.proc.poll() is None:
            self.proc.request_shutdown()
            self.proc.wait(SHUTDOWN_WAIT_S)
        self.proc.close(timeout_s=CLOSE_TIMEOUT_S)


# * ---------------------------------------------------------------------------
# * Run
# * ---------------------------------------------------------------------------


@dataclass(slots=True)
class _RunState:
    """Mutable state of one run."""

    run_id: str
    run_dir: Path
    started_at: str
    status: str = STATUS_COMPLETED
    detail: str | None = None
    records: list[JsonDict] = field(default_factory=list)
    baseline: JsonDict = field(default_factory=dict)
    load: JsonDict = field(default_factory=dict)
    memory_limit_at_job: int | None = None
    worker_pid: int | None = None

    def stop(self, status: str, detail: str | None) -> None:
        """Record why the run ended early (the first reason wins)."""
        if self.status == STATUS_COMPLETED:
            self.status, self.detail = status, detail


def _record(  # noqa: PLR0913 - one flat record per job
    clip: InputClip,
    result: EngineResult,
    *,
    wall_s: float,
    gap_s: float,
    elapsed_s: float,
    uptime_s: float,
    pre_private: int | None,
    job_peak: int | None,
    gpu_used: int | None,
    free_commit: int,
    memory_limit: bool,
) -> JsonDict:
    """Build the record of one finished job."""
    memory = dict(result.memory) if result.memory is not None else {}
    detail = result.error_detail
    record: JsonDict = {
        "index": clip.index,
        "job_id": f"growth-{clip.index:04d}",
        "input": clip.path.name,
        "planned_duration_s": clip.duration_s,
        "offset_s": clip.offset_s,
        "audio_s": result.audio_s,
        "status": result.status,
        "error_kind": result.error_kind,
        "error_detail": detail[:_ERROR_DETAIL_CHARS] if detail else None,
        "text_chars": len(result.text),
        "run_s": round(result.run_s, 4),
        "wall_s": round(wall_s, 4),
        "gap_before_s": round(gap_s, 3),
        "elapsed_s": round(elapsed_s, 3),
        "uptime_s": round(uptime_s, 3),
        "timestamp": now_iso(),
        "pre_job_private_bytes": pre_private,
        "job_peak_bytes": job_peak,
        "gpu_used_bytes": gpu_used,
        "free_commit_bytes": free_commit,
        "memory_limit_report": memory_limit,
    }
    record.update({key: memory.get(key) for key in MEMORY_SNAPSHOT_KEYS})
    return {name: record.get(name) for name in RECORD_FIELDS}


class _JobFeeder:
    """Feeds the jobs of a run to a loaded worker, one at a time."""

    def __init__(
        self,
        options: GrowthOptions,
        session: _Session,
        state: _RunState,
        nvidia_smi: str | None,
        sink: IO[str],
        loaded_at: float,
    ) -> None:
        self._options = options
        self._session = session
        self._state = state
        self._nvidia_smi = nvidia_smi if options.gpu_sample else None
        self._sink = sink
        self._loaded_at = loaded_at
        self._started_at = time.perf_counter()
        self._errors = 0

    def feed(self, clips: Sequence[InputClip]) -> None:
        """Run every clip as one job until the run ends."""
        gap_s = 0.0
        for clip in clips:
            if not self._commit_ok():
                return
            if not self._run_job(clip, gap_s):
                return
            gap_s = self._gap_after(clip.index, len(clips))
            if gap_s > 0:
                stop = self._session.idle(gap_s)
                if stop is not None:
                    self._state.stop(*stop)
                    return

    def _gap_after(self, index: int, total: int) -> float:
        """Return the idle gap after job *index* (none after the last job)."""
        options = self._options
        if options.idle_gap_s <= 0 or index >= total:
            return 0.0
        return options.idle_gap_s if index % options.idle_gap_every == 0 else 0.0

    def _commit_ok(self) -> bool:
        """Stop the run when free system commit fell below the reserve."""
        free = query_commit().free_bytes
        reserve = round(self._options.commit_reserve_gib * GIB)
        if free >= reserve:
            return True
        self._state.stop(
            STATUS_SYSTEM_COMMIT_LOW,
            f"free system commit {free / GIB:.1f} GiB is below --commit-reserve-gib "
            f"{self._options.commit_reserve_gib:g}",
        )
        return False

    def _run_job(self, clip: InputClip, gap_s: float) -> bool:
        """Send one job, wait for its result and record it; ``False`` ends the run."""
        session = self._session
        item = WorkItem(
            job_id=f"growth-{clip.index:04d}",
            attempt_id=f"growth-{clip.index:04d}:1",
            input_path=clip.path,
            language=LANGUAGE,
            duration_s=clip.duration_s,
            deadline_at=None,
            call_id=CALL_ID,
            seq=clip.index,
        )
        pre_private = process_private_bytes(session.proc.pid)
        sent = time.perf_counter()
        if not session.proc.send(item.to_message()):
            self._state.stop(STATUS_WORKER_EXITED, session.exit_detail())
            return False
        wait = session.await_result(item.attempt_id, self._options.job_timeout_s)
        wall_s = time.perf_counter() - sent
        if wait.result is None:
            self._state.stop(wait.stop_status or STATUS_WORKER_EXITED, wait.detail)
            return False
        record = _record(
            clip,
            wait.result,
            wall_s=wall_s,
            gap_s=gap_s,
            elapsed_s=time.perf_counter() - self._started_at,
            uptime_s=time.perf_counter() - self._loaded_at,
            pre_private=pre_private,
            job_peak=session.proc.peak_memory_bytes(),
            gpu_used=gpu_used_bytes(self._nvidia_smi),
            free_commit=query_commit().free_bytes,
            memory_limit=wait.memory_limit,
        )
        self._state.records.append(record)
        self._sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._sink.flush()
        self._progress(record, len(self._state.records))
        return self._after_record(record, wait)

    def _after_record(self, record: Mapping[str, Any], wait: _Wait) -> bool:
        """Decide whether the run continues after a recorded job."""
        if wait.memory_limit:
            self._state.memory_limit_at_job = int(record["index"])
            self._state.stop(
                STATUS_MEMORY_CAP,
                "the worker's Job Object reported its commit cap at job "
                f"{record['index']}",
            )
            return False
        if record["status"] not in growth_stats.OK_STATUSES:
            self._errors += 1
            if self._errors > self._options.max_errors:
                self._state.stop(
                    STATUS_TOO_MANY_ERRORS,
                    f"{self._errors} failed jobs (> --max-errors "
                    f"{self._options.max_errors}); last: {record['error_kind']}: "
                    f"{record['error_detail']}",
                )
                return False
        return True

    def _progress(self, record: Mapping[str, Any], done: int) -> None:
        """Print a progress line every ``progress_every`` jobs."""
        every = self._options.progress_every
        if every <= 0 or (done % every and done != self._options.jobs):
            return

        def mib(name: str) -> str:
            value = record.get(name)
            return "-" if value is None else f"{value / MIB:.1f}"

        echo(
            f"  job {done}/{self._options.jobs}: status {record['status']}, "
            f"private {mib('private_bytes')} MiB, reserved "
            f"{mib('cuda_reserved_bytes')} MiB, allocated "
            f"{mib('cuda_allocated_bytes')} MiB, cuFFT "
            f"{record.get('cufft_plan_cache_size')}, job peak "
            f"{mib('job_peak_bytes')} MiB, wall {record['wall_s']:.2f} s"
        )


# * ---------------------------------------------------------------------------
# * Orchestration
# * ---------------------------------------------------------------------------


def _new_run_dir(out_dir: Path, label: str) -> tuple[str, Path]:
    """Create ``<out_dir>/<UTC stamp>-<label>`` (suffixed when it exists)."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(100):
        run_id = f"{stamp}-{label}" + (f"-{attempt}" if attempt else "")
        run_dir = out_dir / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_id, run_dir
    msg = f"cannot create a run directory under {out_dir}"
    raise ConfigError(msg)


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    """Write a JSON document atomically (UTF-8, indented).

    The text goes to a temp file in the same directory, which then replaces
    *path*, so a kill or a full disk never leaves a truncated file (``run.json``
    is rewritten several times per run); the temp file is removed when the
    write fails.
    """
    text = json.dumps(document, ensure_ascii=False, indent=2, default=str) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write the job records as CSV with :data:`RECORD_FIELDS` columns.

    Records of a run written by another version of the tool may carry keys
    outside the columns (ignored) or miss some of them (left empty), so
    ``summarize`` can rebuild the CSV of any earlier run.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(RECORD_FIELDS), restval="", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)


def _count_marker(path: Path, marker: str) -> int | None:
    """Count the log lines that contain *marker* (``None``: log unreadable)."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return sum(marker in line for line in handle)
    except OSError:
        return None


def _plan_document(
    options: GrowthOptions,
    config: PoolConfig,
    plan: WorkerPlan,
    guard: GuardReading,
    run_id: str,
) -> JsonDict:
    """Return the ``run.json`` document of a run as the start guard leaves it.

    ``stack`` is ``None`` until the stack probe ran (it runs only after the
    guard passed, so a refused run never probes); ``spawn_guard`` is added
    by the second commit check right before the worker spawn.
    """
    env = plan.launch.env
    return {
        "schema": RUN_SCHEMA,
        "version": SUMMARY_VERSION,
        "run_id": run_id,
        "label": options.label,
        "created_at": now_iso(),
        "jobs": options.jobs,
        "durations": options.durations.describe(),
        "source": options.source,
        "audio_format": options.audio_format,
        "idle_gap": {"seconds": options.idle_gap_s, "every": options.idle_gap_every},
        "gpu_memory_bounds": options.gpu_memory_bounds,
        "pool_set": list(options.pool_set),
        "pool_config": config_summary(config),
        "key": plan.spec.key.label,
        "spec": plan.spec.to_message(),
        "reservation": {
            "commit_bytes": plan.reservation.commit_bytes,
            "vram_bytes": plan.reservation.vram_bytes,
            "host_threads": plan.reservation.host_threads,
        },
        "job_limit_bytes": plan.job_limit_bytes,
        "recycle_threshold_bytes": plan.recycle_threshold_bytes,
        "worker_env": {
            name: env.get(name) for name in (*THREAD_ENV_VARS, CUDNN_V8_LRU_CACHE_ENV)
        },
        "removed_env": list(plan.removed_env),
        "interpreter": plan.launch.argv[0],
        "python_executable": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "platform": platform.platform(),
        "stack": None,
        "guard": guard.to_json(),
    }


def _print_plan(
    options: GrowthOptions,
    plan: WorkerPlan,
    guard: GuardReading,
    clips_s: Sequence[float],
) -> None:
    """Print the plan and the guard decision."""
    spec_options = dict(plan.spec.options)
    echo(f"Key {plan.spec.key.label}, capacity {plan.spec.capacity} lanes")
    echo(
        f"  reservation commit {plan.reservation.commit_bytes / GIB:.2f} GiB, "
        f"VRAM {plan.reservation.vram_bytes / GIB:.2f} GiB; job cap "
        f"{plan.job_limit_bytes / GIB:.2f} GiB; recycle threshold "
        + (
            "off"
            if plan.recycle_threshold_bytes is None
            else f"{plan.recycle_threshold_bytes / GIB:.2f} GiB"
        )
    )
    echo(
        f"  GPU bounds: cuFFT {spec_options.get('cufft_plan_cache_max_size')}, cuDNN "
        f"{plan.launch.env.get(CUDNN_V8_LRU_CACHE_ENV, 'torch default')}, idle release "
        f"{spec_options.get('idle_cache_release_s')} s, VRAM cap "
        f"{spec_options.get('vram_cap_bytes')}"
    )
    total = sum(clips_s)
    echo(
        f"  {options.jobs} jobs, {options.durations.describe()}, "
        f"{total / 60:.1f} min of audio, format {options.audio_format}, idle gap "
        f"{options.idle_gap_s:g} s every {options.idle_gap_every} job(s)"
    )
    echo(
        f"  free commit {guard.free_commit_bytes / GIB:.1f} GiB, free VRAM "
        + (
            "unknown"
            if guard.gpu_free_bytes is None
            else f"{guard.gpu_free_bytes / GIB:.2f} GiB"
        )
    )
    for refusal in guard.refusals:
        echo(f"  GUARD: {refusal}")


def _start_worker(
    session: _Session, plan: WorkerPlan, state: _RunState, options: GrowthOptions
) -> float | None:
    """Start the worker and wait for ``loaded``; return the load time or ``None``."""
    started = time.perf_counter()
    try:
        session.proc.start(plan.spec.to_message())
    except WorkerSpawnError as exc:
        state.stop(STATUS_LOAD_FAILED, f"the worker could not start: {exc}")
        return None
    state.worker_pid = session.proc.pid
    info, failure = session.await_load(options.load_timeout_s)
    if info is None:
        state.stop(STATUS_LOAD_FAILED, failure)
        return None
    state.load = {
        "device": info.device,
        "load_s": info.load_s,
        "bring_up_s": round(time.perf_counter() - started, 3),
        "details": dict(info.details),
    }
    return time.perf_counter()


def _execute(
    options: GrowthOptions,
    plan: WorkerPlan,
    state: _RunState,
    clips: Sequence[InputClip],
    nvidia_smi: str | None,
    gpu_before: int | None,
) -> None:
    """Start the worker, record the baseline, feed every job and stop the worker."""
    session = _Session(plan, state.run_dir / WORKER_LOG_NAME)
    try:
        loaded_at = _start_worker(session, plan, state, options)
        if loaded_at is None:
            return
        state.baseline = {
            "private_bytes": process_private_bytes(session.proc.pid),
            "job_peak_bytes": session.proc.peak_memory_bytes(),
            "gpu_used_bytes": gpu_used_bytes(nvidia_smi),
            "gpu_used_before_spawn_bytes": gpu_before,
        }
        echo(
            f"Worker {state.worker_pid} loaded in {state.load['bring_up_s']:.1f} s; "
            f"feeding {len(clips)} jobs"
        )
        with (state.run_dir / RESULTS_JSONL_NAME).open("a", encoding="utf-8") as sink:
            _JobFeeder(options, session, state, nvidia_smi, sink, loaded_at).feed(clips)
    finally:
        session.shutdown()


def write_summary(
    run_dir: Path,
    identity: Mapping[str, object],
    records: Sequence[Mapping[str, Any]],
    *,
    tolerance_bytes: float,
    recycle_threshold_bytes: int | None,
) -> list[str]:
    """Compute the statistics and write ``results.csv`` and the summaries.

    Args:
        run_dir: Run directory (holds ``worker.log``).
        identity: The run fields of the summary (:data:`SUMMARY_IDENTITY_KEYS`).
        records: Job records in job order.
        tolerance_bytes: Plateau tolerance of the byte figures.
        recycle_threshold_bytes: Private bytes at which the pool would retire
            the worker, or ``None``.

    Returns:
        The summary lines.

    """
    summary: JsonDict = {
        "schema": SUMMARY_SCHEMA,
        "version": SUMMARY_VERSION,
        **{key: identity.get(key) for key in SUMMARY_IDENTITY_KEYS},
        "idle_releases": _count_marker(run_dir / WORKER_LOG_NAME, IDLE_RELEASE_MARKER),
        "stats": growth_stats.summarize_records(
            records,
            tolerance_bytes=tolerance_bytes,
            recycle_threshold_bytes=recycle_threshold_bytes,
        ),
    }
    lines = growth_stats.render_summary(summary)
    lines.append(f"Idle CUDA cache releases (worker log): {summary['idle_releases']}")
    _write_csv(run_dir / RESULTS_CSV_NAME, records)
    _write_json(run_dir / SUMMARY_JSON_NAME, summary)
    (run_dir / SUMMARY_MD_NAME).write_text(
        f"# memory-growth {summary['run_id']}\n\n```text\n"
        + "\n".join(lines)
        + "\n```\n",
        encoding="utf-8",
    )
    return lines


def _clean_up(options: GrowthOptions, state: _RunState, inputs_dir: Path) -> None:
    """Remove the worker's temp directories and, unless kept, the inputs."""
    shutil.rmtree(state.run_dir / TEMP_DIRNAME, ignore_errors=True)
    if not options.keep_inputs:
        shutil.rmtree(inputs_dir, ignore_errors=True)


def _finish(
    options: GrowthOptions, plan: WorkerPlan, state: _RunState, inputs_dir: Path
) -> list[str]:
    """Write the CSV and the summaries, clean up, and return the summary lines.

    The cleanup runs whatever happens to the summary: a write that fails on a
    full disk must not also leave gigabytes of generated clips behind.
    """
    identity: JsonDict = {
        "run_id": state.run_id,
        "label": options.label,
        "status": state.status,
        "detail": state.detail,
        "started_at": state.started_at,
        "finished_at": now_iso(),
        "key": plan.spec.key.label,
        "worker_pid": state.worker_pid,
        "job_limit_bytes": plan.job_limit_bytes,
        "load": state.load,
        "baseline": state.baseline,
        "memory_limit_at_job": state.memory_limit_at_job,
    }
    try:
        return write_summary(
            state.run_dir,
            identity,
            state.records,
            tolerance_bytes=options.plateau_tolerance_mib * MIB,
            recycle_threshold_bytes=plan.recycle_threshold_bytes,
        )
    finally:
        _clean_up(options, state, inputs_dir)


def is_growth_run(path: Path) -> bool:
    """Return whether *path* is a ``memory-growth`` run directory.

    Every run writes ``run.json`` before anything else (a run that failed
    before its first job has no records); benchmark output directories of
    the other commands never contain one.
    """
    return (path / RUN_JSON_NAME).is_file()


def load_records(run_dir: Path) -> list[JsonDict]:
    """Read the job records of a run (none without ``results.jsonl``).

    Malformed lines (for example a line cut by a killed run) are skipped.
    """
    records: list[JsonDict] = []
    path = run_dir / RESULTS_JSONL_NAME
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record: object = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and "index" in record:
                records.append(cast("JsonDict", record))
    return records


def rebuild_summary(
    run_dir: Path, *, plateau_tolerance_mib: float = DEFAULT_PLATEAU_TOLERANCE_MIB
) -> list[str]:
    """Recompute the summary of an existing run from ``results.jsonl``.

    The run fields (status, baseline, load) come from the previous
    ``summary.json`` when there is one (status ``unknown`` otherwise, for
    example after a killed run), the recycle threshold from ``run.json``.

    Returns:
        The summary lines.

    """
    plan = read_json(run_dir / RUN_JSON_NAME) or {}
    previous = read_json(run_dir / SUMMARY_JSON_NAME) or {}
    identity: JsonDict = {key: previous.get(key) for key in SUMMARY_IDENTITY_KEYS}
    identity["run_id"] = plan.get("run_id", run_dir.name)
    identity["label"] = plan.get("label")
    identity["key"] = plan.get("key")
    identity["job_limit_bytes"] = plan.get("job_limit_bytes")
    identity["status"] = previous.get("status", "unknown")
    threshold = plan.get("recycle_threshold_bytes")
    return write_summary(
        run_dir,
        identity,
        load_records(run_dir),
        tolerance_bytes=plateau_tolerance_mib * MIB,
        recycle_threshold_bytes=threshold if isinstance(threshold, int) else None,
    )


def _print_error(message: str) -> None:
    """Print *message* to stderr; a broken console is ignored."""
    with contextlib.suppress(Exception):
        typer.echo(message, err=True)


def _report(
    options: GrowthOptions, plan: WorkerPlan, state: _RunState, inputs_dir: Path
) -> bool:
    """Write the summary, then print it; never raises.

    It runs in the ``finally`` of the run, so a failure here (a full disk, a
    broken console) must not replace the run's own exception: it is printed
    to stderr instead. Writing and printing fail separately: a console that
    breaks after a successful write does not count as a failed write.

    Returns:
        Whether ``results.csv``, ``summary.json`` and ``summary.md`` were
        written.

    """
    try:
        lines = _finish(options, plan, state, inputs_dir)
    except Exception as exc:  # noqa: BLE001 - must not mask the run's own outcome
        _print_error(
            f"ERROR: the summary of {state.run_dir} could not be written: "
            f"{type(exc).__name__}: {exc}"
        )
        return False
    try:
        for line in lines:
            echo(line)
    except Exception as exc:  # noqa: BLE001 - the summary is written; printing is optional
        _print_error(
            f"WARNING: the summary of {state.run_dir} was written but could not "
            f"be printed: {type(exc).__name__}: {exc}"
        )
    return True


def exit_code(status: str, *, summary_written: bool) -> int:
    """Return the exit code of a finished run.

    Args:
        status: Final run status.
        summary_written: Whether the summary files were written.

    Returns:
        ``3`` when the commit check right before the worker spawn refused the
        run, ``1`` for any other early stop (whatever happened to the
        summary), ``4`` when every job ran but the summary was not written,
        ``0`` otherwise.

    """
    if status == STATUS_GUARD_REFUSED:
        return EXIT_GUARD
    if status != STATUS_COMPLETED:
        return EXIT_FAILED
    return EXIT_OK if summary_written else EXIT_SUMMARY_NOT_WRITTEN


def _spawn_guard(
    options: GrowthOptions, plan: WorkerPlan, state: _RunState, document: JsonDict
) -> bool:
    """Check free commit again right before the worker spawn.

    The inputs can take minutes to cut while other processes take commit, so
    the start guard's commit rules (:func:`_commit_refusals`) are applied
    again. The reading and any refusal are recorded in ``run.json``; a refusal
    stops the run with status ``guard_refused``.

    Returns:
        Whether the worker may be spawned.

    """
    free = query_commit().free_bytes
    refusals = _commit_refusals(options, free, plan.job_limit_bytes)
    document["spawn_guard"] = {"free_commit_bytes": free, "refusals": refusals}
    _write_json(state.run_dir / RUN_JSON_NAME, document)
    if not refusals:
        return True
    for refusal in refusals:
        echo(f"  GUARD (before the worker spawn): {refusal}")
    state.stop(STATUS_GUARD_REFUSED, "; ".join(refusals))
    return False


def run_memory_growth(options: GrowthOptions) -> int:
    """Run the ``memory-growth`` scenario.

    Args:
        options: Run options.

    Returns:
        The code of :func:`exit_code`, or ``3`` when the start guard refused
        the run (nothing was probed, cut or spawned; ``run.json`` records the
        readings and the refusals).

    Raises:
        ConfigError: For invalid options or pool settings.
        WorkloadError: When the inputs cannot be built.

    """
    options.validate()
    options.durations.validate()
    config = build_pool_config(options)
    nvidia_smi = find_nvidia_smi()
    if options.dry_run:
        plan = plan_worker(config, options, options.out_dir / TEMP_DIRNAME)
        guard = read_guard(options, nvidia_smi, plan.job_limit_bytes)
        _print_plan(options, plan, guard, options.durations.durations(options.jobs))
        echo("Dry run: nothing was written.")
        return EXIT_GUARD if guard.refusals else EXIT_OK
    run_id, run_dir = _new_run_dir(options.out_dir, options.label)
    plan = plan_worker(config, options, run_dir / TEMP_DIRNAME)
    guard = read_guard(options, nvidia_smi, plan.job_limit_bytes)
    document = _plan_document(options, config, plan, guard, run_id)
    _write_json(run_dir / RUN_JSON_NAME, document)
    _print_plan(options, plan, guard, options.durations.durations(options.jobs))
    echo(f"Run directory: {run_dir}")
    if guard.refusals:
        # * run.json records the refused plan and the guard readings.
        return EXIT_GUARD
    if options.probe_stack:
        # * After the guard: the probe child imports torch and starts CUDA.
        document["stack"] = probe_stack()
        _write_json(run_dir / RUN_JSON_NAME, document)
    (run_dir / TEMP_DIRNAME).mkdir()
    (run_dir / RESULTS_JSONL_NAME).touch()
    inputs_dir = run_dir / INPUTS_DIRNAME
    state = _RunState(run_id=run_id, run_dir=run_dir, started_at=now_iso())
    try:
        clips = prepare_inputs(
            options.durations,
            options.jobs,
            source=options.source,
            work_dir=inputs_dir,
            audio_format=options.audio_format,
        )
        echo(f"Prepared {len(clips)} inputs from {options.source}")
        if _spawn_guard(options, plan, state, document):
            _execute(options, plan, state, clips, nvidia_smi, guard.gpu_used_bytes)
    except KeyboardInterrupt:
        state.stop(STATUS_INTERRUPTED, "Ctrl+C")
        raise
    except WorkloadError as exc:
        state.stop(STATUS_INPUTS_FAILED, str(exc))
        raise
    except Exception as exc:
        state.stop(STATUS_ERROR, repr(exc))
        raise
    finally:
        summary_written = _report(options, plan, state, inputs_dir)
    return exit_code(state.status, summary_written=summary_written)
