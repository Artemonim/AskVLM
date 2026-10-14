"""Parent orchestrator: guard, spawn, dispatch, measure, and record configurations.

A measurement runs one or more *sides* at once (one side for ``run`` and
``sweep``, two for ``combo``). For every measurement the parent

1. spawns ``config.processes`` children of this tool in the hidden child mode
   per side (each child is assigned to a kill-on-close job object);
2. waits until every child of every side has loaded its model and finished its
   untimed warm-up, then starts one shared clock and answers ``want`` requests
   from the side's :class:`~stt_concurrency.workload.WorkStream` (dynamic
   distribution, one dispatch per request);
3. stops a side's clock at its last result once its stream is closed, stops the
   children gracefully, and kills whatever is left on any other exit path
   (error, timeout, Ctrl+C).

``run`` measures the fixed matrix: one pass over the clips per configuration,
one result JSON per configuration and repeat, then the summary.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import psutil  # type: ignore[import-untyped]
import typer

from . import report, sysmetrics
from .configs import ResourceNeed, estimate_need, select_configs
from .engines import (
    DEVICE_MISMATCH_TAG,
    GIGAAM_WINDOW_S,
    WHISPER_CPU_COMPUTE_TYPE,
    WHISPER_CUDA_COMPUTE_TYPE,
)
from .paths import (
    CHILD_MODE_ARG,
    ENTRY_SCRIPT,
    LOGS_DIRNAME,
    PROJECT_ROOT,
    RESULTS_DIRNAME,
    SUMMARY_JSON_NAME,
    SUMMARY_MD_NAME,
)
from .protocol import MESSAGE_TYPES, ChildSpec, decode_message, encode_message
from .quality import corpus_wer, load_manifest, repeat_consistency
from .store import ResultStore, fingerprint_of
from .sysmetrics import (
    GIB,
    MIB,
    CommitWatchdog,
    GpuPoller,
    KillOnCloseJob,
    ProcessTreeSampler,
    cpu_busy_percent,
    cpu_snapshot,
    find_nvidia_smi,
    kill_process_tree,
    query_commit,
    query_gpu,
    query_gpu_name,
)
from .workload import (
    WorkloadError,
    WorkStream,
    discover_clips,
    load_long_files,
    percentile,
    pick_warmup,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .configs import BenchConfig, ResourceEstimate, ThreadDefaults
    from .quality import Manifest
    from .sysmetrics import CommitInfo, CpuSnapshot, GpuSample
    from .workload import AudioItem, Occurrence

JsonDict = dict[str, Any]

# * Bump when the meaning of a measurement changes: older results become stale.
RESULT_SCHEMA_VERSION = 2
# * The step's job hit its commit cap: a saturation point, never a success.
STATUS_MEMORY_CAP = "memory_cap"
# * The system free commit fell below --commit-reserve-gb during the step.
STATUS_SYSTEM_COMMIT_LOW = "system_commit_low"
ALL_STATUSES = frozenset(
    {
        "ok",
        "partial",
        "error",
        "timeout",
        "skipped",
        STATUS_MEMORY_CAP,
        STATUS_SYSTEM_COMMIT_LOW,
    },
)
# * Depends on what else ran on the PC at that moment: always measured again.
TRANSIENT_STATUSES = frozenset({STATUS_SYSTEM_COMMIT_LOW})
# * Results reused on resume by run, sweep and combo. "skipped" (guard or no
# * cap) is decided again, so a nightly rerun measures what the day skipped.
REUSABLE_STATUSES = ALL_STATUSES - TRANSIENT_STATUSES - {"skipped"}
DEFAULT_COMMIT_RESERVE_GB = 12.0
# * Why the step job got its cap: the real headroom or --job-commit-limit-gb.
CAP_REASON_HEADROOM = "headroom"
CAP_REASON_EXPLICIT = "explicit"
# * A stored memory_cap result stands when its cap was at least the current cap
# * minus this tolerance (free commit drifts a little between runs).
CAP_REUSE_TOLERANCE_GB = 0.5
# * Failed outcomes that may hide a cap hit when the cap monitor was broken.
_FAILED_STATUSES = frozenset({"error", "partial", "timeout"})
# * Characters of the revealing failure kept in a cap reason.
_REASON_DETAIL_LIMIT = 300
WATCHDOG_INTERVAL_S = 0.25
# * A failed step whose job peak came this close to the cap counts as capped.
_CAP_PEAK_MARGIN_BYTES = 64 * 1024**2
_WAKE_EVENT = -1
# * Children unload their model after the last item before they exit.
GRACEFUL_STOP_TIMEOUT_S = 120.0
PROCESS_EXIT_TIMEOUT_S = 30.0
READER_JOIN_TIMEOUT_S = 5.0
PROGRESS_INTERVAL_S = 15.0
EVENT_POLL_S = 1.0
LOG_TAIL_CHARS = 1500
MAX_RECORDED_ERRORS = 200
_MIN_SIDES_FOR_WINDOW = 2
_OOM_MARKERS = ("out of memory", "outofmemory", "cublas_status_alloc_failed")


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Resolved command-line options shared by ``run``, ``sweep`` and ``combo``."""

    clips_dir: Path
    out_dir: Path
    max_clips: int
    long_files: tuple[Path, ...]
    language: str
    selection: str
    repeat: int
    clear: bool
    retry_failed: bool
    max_config_seconds: float
    min_free_commit_gb: float
    min_free_vram_mb: float
    estimates: Mapping[str, ResourceEstimate]
    thread_defaults: ThreadDefaults
    whisper_model: str
    whisper_options: Mapping[str, object]
    whisper_chunk_s: float
    warmup_clip: Path | None
    sample_interval_s: float
    cooldown_s: float
    gigaam_cuda_streams: bool
    dry_run: bool
    manifest_path: Path | None = None
    job_commit_limit_gb: float | None = None
    commit_reserve_gb: float = DEFAULT_COMMIT_RESERVE_GB
    allow_uncapped: bool = False


@dataclass(frozen=True, slots=True)
class Workload:
    """Resolved inputs shared by every configuration of one invocation."""

    clips: tuple[AudioItem, ...]
    long_items: tuple[AudioItem, ...]
    warmup_path: Path
    manifest: Manifest | None


def load_workload(options: RunOptions) -> Workload:
    """Discover clips and long files, pick the warm-up clip, load the manifest.

    Args:
        options: Run options.

    Returns:
        The workload.

    """
    clips = discover_clips(options.clips_dir, options.max_clips)
    return Workload(
        clips=tuple(clips),
        long_items=tuple(load_long_files(options.long_files)),
        warmup_path=pick_warmup(clips, options.warmup_clip),
        manifest=(
            load_manifest(options.manifest_path) if options.manifest_path else None
        ),
    )


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file's bytes, read as a stream.

    Args:
        path: File to hash.

    Returns:
        Hex digest.

    Raises:
        WorkloadError: When the file cannot be read.

    """
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        msg = f"Cannot read {path} for the workload fingerprint: {exc}"
        raise WorkloadError(msg) from exc


def fingerprint_payload(options: RunOptions, workload: Workload) -> JsonDict:
    """Return the inputs that make two measurements comparable.

    Every selected clip, the warm-up clip and every long file contribute their
    name, duration and SHA-256 content digest (hashed once per invocation), so
    audio replaced under the same name and duration changes the fingerprint.
    Results recorded before the digests were part of the payload have another
    fingerprint and become stale (moved to ``stale/`` when measured again).

    Args:
        options: Run options.
        workload: Resolved workload.

    Returns:
        JSON-serializable payload for :func:`~stt_concurrency.store.fingerprint_of`.

    Raises:
        WorkloadError: When an audio file cannot be read.

    """
    return {
        "schema": RESULT_SCHEMA_VERSION,
        "clips_dir": str(options.clips_dir),
        "max_clips": options.max_clips,
        "clips": [
            [c.path.name, round(c.duration_s, 3), file_digest(c.path)]
            for c in workload.clips
        ],
        "long_files": [
            [str(item.path), round(item.duration_s, 3), file_digest(item.path)]
            for item in workload.long_items
        ],
        "warmup": [workload.warmup_path.name, file_digest(workload.warmup_path)],
        "language": options.language,
        "whisper_model": options.whisper_model,
        "whisper_options": dict(options.whisper_options),
        "whisper_chunk_s": options.whisper_chunk_s,
        "compute_types": {
            "cpu": WHISPER_CPU_COMPUTE_TYPE,
            "cuda": WHISPER_CUDA_COMPUTE_TYPE,
        },
        "gigaam": {
            "window_s": GIGAAM_WINDOW_S,
            "cuda_streams": options.gigaam_cuda_streams,
        },
        "cpu_budget": options.thread_defaults.cpu_budget,
        "cuda_threads": options.thread_defaults.cuda_threads,
        "manifest": workload.manifest.sha256 if workload.manifest else None,
    }


@dataclass(slots=True)
class SideSpec:
    """One configuration of a measurement and the stream it consumes.

    Attributes:
        config: Configuration.
        stream: Work stream (its clips are this side's clips).
        label: Short label used in log names (empty for single-side runs).

    """

    config: BenchConfig
    stream: WorkStream
    label: str = ""


@dataclass(slots=True)
class _Outcome:
    item_id: int
    text: str
    proc_s: float
    child: int
    slot: int
    batch: int
    error: str | None


@dataclass(slots=True)
class _Side:
    index: int
    spec: SideSpec
    children: list[int] = field(default_factory=list)
    outcomes: dict[int, _Outcome] = field(default_factory=dict)
    inflight: int = 0
    last_result_t: float | None = None
    t_end: float | None = None
    dispatched_at: dict[int, float] = field(default_factory=dict)
    finished_at: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class _Child:
    index: int
    side: int
    proc: subprocess.Popen[str]
    log_path: Path
    log_handle: IO[bytes]
    reader: threading.Thread
    in_job: bool
    in_step_job: bool = False
    load_s: float | None = None
    info: JsonDict | None = None
    warmup_s: float | None = None
    warmup: JsonDict | None = None
    cpu_ready_s: float | None = None
    cpu_done_s: float | None = None
    ready: bool = False
    done: bool = False
    eof: bool = False
    fatal: str | None = None


def now_iso() -> str:
    """Return the current UTC time in ISO format (seconds)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def planned_job_cap(options: RunOptions, free_commit_bytes: int) -> tuple[int, str]:
    """Return the step job's commit cap and why it has that size.

    The cap is the real headroom, ``free commit - --commit-reserve-gb``, so a
    step may use all memory the rest of the PC can spare. ``--job-commit-limit-gb``
    lowers it (``explicit``); it never raises it above the headroom. A cap at or
    below zero means the step must not start.

    Args:
        options: Run options.
        free_commit_bytes: System free commit at the decision time.

    Returns:
        ``(cap bytes, cap_reason)`` with ``cap_reason`` ``headroom`` or ``explicit``.

    """
    headroom = free_commit_bytes - int(options.commit_reserve_gb * GIB)
    explicit = options.job_commit_limit_gb
    if explicit is not None and int(explicit * GIB) < headroom:
        return int(explicit * GIB), CAP_REASON_EXPLICIT
    return headroom, CAP_REASON_HEADROOM


def reusable_result(
    store: ResultStore,
    key: str,
    *,
    statuses: frozenset[str],
    options: RunOptions,
) -> JsonDict | None:
    """Return a stored result that can stand for a new measurement, if any.

    On top of :meth:`ResultStore.reusable`, a ``memory_cap`` result is reused
    only when the cap it ran under was at least the cap a step would get now
    (minus :data:`CAP_REUSE_TOLERANCE_GB`); with more room now it is measured
    again, because the cap is not part of the fingerprint. A failed result
    recorded while the cap monitor was broken (``cap_monitor_error``) may hide
    an unreported cap hit, so it is treated as transient and measured again.

    Args:
        store: Result store.
        key: Result key.
        statuses: Final statuses.
        options: Run options (reserve and explicit cap).

    Returns:
        The reusable document, or ``None``.

    """
    document = store.reusable(key, statuses=statuses)
    if document is None:
        return None
    status = document.get("status")
    job = document.get("job") or {}
    if status in _FAILED_STATUSES and job.get("cap_monitor_error"):
        return None
    if status != STATUS_MEMORY_CAP:
        return document
    stored = job.get("limit_gb")
    reading = sysmetrics.query_commit()
    if stored is None or reading.source != sysmetrics.COMMIT_SOURCE_WINDOWS:
        return None
    current_gb = planned_job_cap(options, reading.free_bytes)[0] / GIB
    return document if float(stored) >= current_gb - CAP_REUSE_TOLERANCE_GB else None


def echo(message: str) -> None:
    """Print a progress line."""
    typer.echo(message)


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def is_cuda_oom(document: Mapping[str, Any]) -> bool:
    """Return whether a result failed because CUDA (or CTranslate2) ran out of memory.

    Args:
        document: Result document or side document.

    Returns:
        ``True`` when the reason or an item error carries an OOM marker; a
        device mismatch (model placed on another device) never counts.

    """
    texts = [str(document.get("reason") or "")]
    texts.extend(str(e.get("error") or "") for e in document.get("errors") or [])
    lowered = [text.lower() for text in texts if DEVICE_MISMATCH_TAG not in text]
    return any(marker in text for text in lowered for marker in _OOM_MARKERS)


def child_interpreter() -> tuple[str, dict[str, str]]:
    """Return the interpreter that starts the workers and its extra environment.

    In a Windows venv ``sys.executable`` is a launcher that starts the real
    interpreter as its own child, which could allocate before the launcher is
    put into the step job. The workers therefore start from the base
    interpreter with ``__PYVENV_LAUNCHER__`` set (as ``multiprocessing`` does),
    so the process assigned to the job is the interpreter itself.

    Returns:
        ``(executable, environment additions)``.

    """
    base = getattr(sys, "_base_executable", None) or sys.executable
    if (
        sys.platform == "win32"
        and Path(base) != Path(sys.executable)
        and Path(base).is_file()
    ):
        return str(base), {"__PYVENV_LAUNCHER__": sys.executable}
    return sys.executable, {}


class Measurement:
    """Runs one or more sides in child processes under one clock and measures them.

    Memory safety (Windows): every measurement puts all of its children (all
    sides) into one step job with a commit cap (see :meth:`_job_limit`). When
    the job reports the cap, the step is terminated and recorded as
    ``memory_cap``. A watchdog polls the system free commit every
    :data:`WATCHDOG_INTERVAL_S`; below ``--commit-reserve-gb`` it terminates the
    step and records ``system_commit_low``. Both use the real system probes,
    independent of the guard's estimates. Off Windows both are inactive.
    """

    def __init__(
        self,
        *,
        sides: Sequence[SideSpec],
        run_id: str,
        warmup_path: Path,
        options: RunOptions,
        job: KillOnCloseJob | None,
        logs_dir: Path,
        nvidia_smi: str | None,
    ) -> None:
        """Prepare a measurement (nothing starts before :meth:`execute`).

        Args:
            sides: Configurations and their streams.
            run_id: Log file prefix.
            warmup_path: Untimed warm-up clip.
            options: Run options.
            job: Run-level kill-on-close job (outer safety net), if any.
            logs_dir: Directory of the per-child logs.
            nvidia_smi: ``nvidia-smi`` path, if any.

        """
        self._sides = [_Side(index=i, spec=spec) for i, spec in enumerate(sides)]
        self._run_id = run_id
        self._warmup_path = warmup_path
        self._options = options
        self._job = job
        self._step_job: KillOnCloseJob | None = None
        self._logs_dir = logs_dir
        self._nvidia_smi = nvidia_smi
        self._events: queue.Queue[tuple[int, JsonDict | None]] = queue.Queue()
        self._children: list[_Child] = []
        self._held_wants: list[tuple[int, int, int]] = []
        self._t_spawn: float | None = None
        self._t_go: float | None = None
        self._t_end: float | None = None
        self._cpu_go: CpuSnapshot | None = None
        self._cpu_end: CpuSnapshot | None = None
        self._abort: tuple[str, str] | None = None
        self._abort_lock = threading.Lock()

    # * Process management.

    def _child_env(self, config: BenchConfig) -> dict[str, str]:
        """Environment of a child: thread pools, offline caches, UTF-8 I/O."""
        env = dict(os.environ)
        env.update(child_interpreter()[1])
        threads = str(config.threads)
        env.update(
            {
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "PYTHONIOENCODING": "utf-8",
            },
        )
        return env

    def _child_spec(self, side: _Side, index: int) -> ChildSpec:
        return ChildSpec(
            config=side.spec.config,
            child_index=index,
            items=side.spec.stream.clips,
            warmup_path=self._warmup_path,
            language=self._options.language,
            whisper_model=self._options.whisper_model,
            whisper_options=dict(self._options.whisper_options),
            whisper_chunk_s=self._options.whisper_chunk_s,
            gigaam_cuda_streams=self._options.gigaam_cuda_streams,
        )

    def _assign(self, pid: int) -> tuple[bool, bool]:
        """Put a child into the run job, then into the capped step job (nested).

        Returns:
            ``(in_run_job, in_step_job)``; only the step job enforces the cap.

        """
        in_run = self._job.assign(pid) if self._job is not None else False
        in_step = self._step_job.assign(pid) if self._step_job is not None else False
        return in_run, in_step

    def _spawn_children(self) -> None:
        """Start every child of every side, assign it to the jobs, send its spec."""
        argv = [
            child_interpreter()[0],
            "-X",
            "utf8",
            str(ENTRY_SCRIPT),
            CHILD_MODE_ARG,
        ]
        self._t_spawn = time.perf_counter()
        for side in self._sides:
            env = self._child_env(side.spec.config)
            for local in range(side.spec.config.processes):
                index = len(self._children)
                prefix = f"{side.spec.label}c" if side.spec.label else "c"
                log_path = self._logs_dir / f"{self._run_id}__{prefix}{local}.log"
                log_handle = log_path.open("wb")
                try:
                    proc = subprocess.Popen(  # noqa: S603 - this interpreter, this tool
                        argv,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=log_handle,
                        cwd=str(PROJECT_ROOT),
                        env=env,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                except OSError:
                    log_handle.close()
                    raise
                reader = threading.Thread(
                    target=self._read_child,
                    args=(index, proc.stdout),
                    name=f"child-reader-{index}",
                    daemon=True,
                )
                in_run, in_step = self._assign(proc.pid)
                child = _Child(
                    index=index,
                    side=side.index,
                    proc=proc,
                    log_path=log_path,
                    log_handle=log_handle,
                    reader=reader,
                    in_job=in_run,
                    in_step_job=in_step,
                )
                self._children.append(child)
                side.children.append(index)
                reader.start()
                if self._step_job is not None and not in_step:
                    if not self._options.allow_uncapped:
                        # ! Fail closed: never measure with a child outside the cap.
                        self._abort_step(
                            "skipped",
                            f"commit cap unavailable: child {index} could not join "
                            "the step job; pass --allow-uncapped to run without it",
                        )
                        return
                    echo(
                        f"  WARNING: child {index} runs OUTSIDE the commit cap "
                        "(--allow-uncapped)",
                    )
                self._send(child, self._child_spec(side, local).to_json())

    def _read_child(self, index: int, stream: IO[str] | None) -> None:
        """Forward protocol messages of child *index* to the event queue."""
        try:
            if stream is not None:
                for line in iter(stream.readline, ""):
                    message = decode_message(line)
                    if message is not None and message["type"] in MESSAGE_TYPES:
                        self._events.put((index, message))
        except (OSError, ValueError):
            pass
        finally:
            self._events.put((index, None))

    def _send(self, child: _Child, message: Mapping[str, object]) -> bool:
        """Write one message to a child's stdin; ``False`` if the pipe is gone."""
        stdin = child.proc.stdin
        if stdin is None or child.eof:
            return False
        try:
            stdin.write(encode_message(message))
            stdin.flush()
        except (OSError, ValueError):
            return False
        return True

    def _log_tail(self, child: _Child) -> str:
        with contextlib.suppress(OSError, ValueError):
            child.log_handle.flush()
        try:
            text = child.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return " | ".join(text[-LOG_TAIL_CHARS:].strip().splitlines()[-12:])

    def _exit_code(self, child: _Child) -> int | None:
        try:
            return child.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            return None

    def _terminate_children(self, *, graceful: bool) -> None:
        """Make sure no child survives: close stdin, wait, then kill the tree."""
        for child in self._children:
            with contextlib.suppress(OSError, ValueError):
                if child.proc.stdin is not None:
                    child.proc.stdin.close()
        for child in self._children:
            if child.proc.poll() is not None:
                continue
            if graceful:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    child.proc.wait(timeout=PROCESS_EXIT_TIMEOUT_S)
                    continue
            kill_process_tree(child.proc.pid)
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.proc.wait(timeout=PROCESS_EXIT_TIMEOUT_S)

    def _close_children(self) -> None:
        """Join reader threads and close every pipe and log handle."""
        for child in self._children:
            child.reader.join(timeout=READER_JOIN_TIMEOUT_S)
            with contextlib.suppress(OSError, ValueError):
                if child.proc.stdout is not None:
                    child.proc.stdout.close()
            with contextlib.suppress(OSError, ValueError):
                child.log_handle.close()

    # * Event loop.

    def _go(self) -> None:
        """Start the shared clock once every child is ready; release held requests."""
        self._t_go = time.perf_counter()
        self._cpu_go = cpu_snapshot()
        spawn = self._t_spawn or self._t_go
        echo(
            f"  all {len(self._children)} worker process(es) ready after "
            f"{self._t_go - spawn:.1f} s; clock started",
        )
        held, self._held_wants = self._held_wants, []
        for child_index, slot, count in held:
            self._dispatch(child_index, slot, count)

    def _dispatch(self, child_index: int, slot: int, count: int) -> None:
        """Answer one ``want`` from the side's stream (empty list = stop)."""
        child = self._children[child_index]
        side = self._sides[child.side]
        now = time.perf_counter()
        elapsed = now - (self._t_go or now)
        # * A combo keeps feeding every side until all sides met their minimum,
        # * so no side finishes its share alone.
        hold_open = any(
            not other.spec.stream.minimum_met(elapsed)
            for other in self._sides
            if other is not side
        )
        batch: list[Occurrence] = side.spec.stream.take(
            count,
            elapsed,
            hold_open=hold_open,
        )
        message = {
            "type": "work",
            "slot": slot,
            "items": [occurrence.item_id for occurrence in batch],
            "clips": [occurrence.clip.index for occurrence in batch],
        }
        if self._send(child, message):
            side.inflight += len(batch)
            for occurrence in batch:
                side.dispatched_at[occurrence.item_id] = now
        self._check_side_done(side)

    def _check_side_done(self, side: _Side) -> None:
        if side.t_end is not None or not side.spec.stream.closed or side.inflight > 0:
            return
        side.t_end = side.last_result_t or time.perf_counter()
        if all(s.t_end is not None for s in self._sides):
            self._t_end = max(s.t_end for s in self._sides if s.t_end is not None)
            self._cpu_end = cpu_snapshot()

    def _on_result(self, child_index: int, message: JsonDict) -> None:
        side = self._sides[self._children[child_index].side]
        item_id = int(message["item"])
        error = message.get("error")
        side.outcomes[item_id] = _Outcome(
            item_id=item_id,
            text=str(message.get("text") or ""),
            proc_s=float(message.get("proc_s") or 0.0),
            child=child_index,
            slot=int(message.get("slot") or 0),
            batch=int(message.get("batch") or 1),
            error=str(error) if error else None,
        )
        side.inflight -= 1
        side.last_result_t = time.perf_counter()
        side.finished_at[item_id] = side.last_result_t
        self._check_side_done(side)

    def _on_loaded(self, child_index: int, message: JsonDict) -> None:
        child = self._children[child_index]
        child.load_s = float(message["load_s"])
        child.info = dict(message.get("info") or {})
        # * The spawned PID may be a venv launcher; this is the interpreter itself.
        child.info["worker_pid"] = message.get("pid")
        echo(f"  child {child_index}: model loaded in {child.load_s:.1f} s")

    def _on_ready(self, child_index: int, message: JsonDict) -> None:
        child = self._children[child_index]
        child.ready = True
        child.warmup_s = float(message["warmup_s"])
        child.warmup = dict(message.get("warmup") or {})
        child.cpu_ready_s = float(message["cpu_s"])
        if all(c.ready for c in self._children):
            self._go()

    def _on_want(self, child_index: int, message: JsonDict) -> None:
        request = (child_index, int(message["slot"]), int(message["count"]))
        if self._t_go is None:
            self._held_wants.append(request)
        else:
            self._dispatch(*request)

    def _on_done(self, child_index: int, message: JsonDict) -> None:
        child = self._children[child_index]
        child.done = True
        child.cpu_done_s = float(message["cpu_s"])

    def _handle_event(self, child_index: int, message: JsonDict | None) -> str | None:
        """Apply one event; return a failure reason that aborts the measurement."""
        child = self._children[child_index]
        if message is None:
            child.eof = True
            if child.done:
                return None
            return (
                f"child {child_index} exited unexpectedly (exit code "
                f"{self._exit_code(child)}); log tail: {self._log_tail(child)}"
            )
        kind = message["type"]
        if kind == "fatal":
            child.fatal = str(message.get("error"))
            return (
                f"child {child_index} failed during {message.get('stage')}: "
                f"{child.fatal}"
            )
        handlers = {
            "loaded": self._on_loaded,
            "ready": self._on_ready,
            "want": self._on_want,
            "result": self._on_result,
            "done": self._on_done,
        }
        handler = handlers.get(kind)
        if handler is not None:
            handler(child_index, message)
        return None

    def _print_progress(self, now: float) -> None:
        if self._t_go is None:
            ready = sum(1 for c in self._children if c.ready)
            echo(f"  loading: {ready}/{len(self._children)} ready")
            return
        counts = ", ".join(
            f"{side.spec.config.name}: {len(side.outcomes)}" for side in self._sides
        )
        echo(f"  {now - self._t_go:.1f} s; items {counts}")

    def _event_loop(self) -> tuple[str, str | None]:
        """Run until every side is done, a child fails, or time runs out."""
        spawn = self._t_spawn if self._t_spawn is not None else time.perf_counter()
        limit_s = self._options.max_config_seconds
        deadline = spawn + limit_s
        next_progress = time.perf_counter() + PROGRESS_INTERVAL_S
        while self._t_end is None:
            if self._abort is not None:
                return self._abort
            now = time.perf_counter()
            if now >= deadline:
                return "timeout", f"exceeded --max-config-seconds={limit_s:g}"
            if now >= next_progress:
                self._print_progress(now)
                next_progress = now + PROGRESS_INTERVAL_S
            try:
                child_index, message = self._events.get(
                    timeout=min(EVENT_POLL_S, deadline - now),
                )
            except queue.Empty:
                continue
            if child_index == _WAKE_EVENT:
                continue
            failure = self._handle_event(child_index, message)
            # * A cap or watchdog kill also ends children: report its cause.
            aborted = self._aborted()
            if aborted is not None:
                return aborted
            if failure is not None:
                return "error", failure
        return "ok", None

    def _aborted(self) -> tuple[str, str] | None:
        """Return the abort cause set by the job monitor or the watchdog, if any."""
        with self._abort_lock:
            return self._abort

    def _abort_step(self, status: str, reason: str) -> None:
        """Terminate the whole step at once and record why (first cause wins).

        Called from the job monitor or the watchdog thread.
        """
        with self._abort_lock:
            if self._abort is not None:
                return
            self._abort = (status, reason)
        # * Read once: _close_safety_rails may clear the attribute concurrently.
        job = self._step_job
        terminated = job.terminate() if job is not None else False
        for child in list(self._children):
            # * The job kill covers only its members; kill the others directly.
            if not (terminated and child.in_step_job):
                kill_process_tree(child.proc.pid, timeout_s=0.0)
        self._events.put((_WAKE_EVENT, None))

    def _on_memory_cap(self, pid: int) -> None:
        job = self._step_job
        limit = job.memory_limit_bytes if job is not None else None
        limit_gb = (limit or 0) / GIB
        self._abort_step(
            STATUS_MEMORY_CAP,
            f"step job commit cap {limit_gb:.2f} GB reached (process {pid})",
        )

    def _on_commit_low(self, reading: CommitInfo) -> None:
        self._abort_step(
            STATUS_SYSTEM_COMMIT_LOW,
            f"system free commit {reading.free_bytes / GIB:.2f} GB < "
            f"--commit-reserve-gb {self._options.commit_reserve_gb:g}",
        )

    def _job_limit(self, free_commit_bytes: int) -> tuple[int, JsonDict]:
        """Return the step job's cap (:func:`planned_job_cap`) and its record.

        The cap never exceeds ``free - reserve``, so the step alone cannot push
        the system below the reserve; either way a ``memory_cap`` hit is a real
        limit of this run, i.e. a saturation point.
        """
        limit, cap_reason = planned_job_cap(self._options, free_commit_bytes)
        return limit, {
            "limit_gb": round(limit / GIB, 3),
            "cap_reason": cap_reason,
            "explicit_gb": self._options.job_commit_limit_gb,
            "free_commit_gb_at_start": round(free_commit_bytes / GIB, 3),
            "reserve_gb": self._options.commit_reserve_gb,
            "cap_enforced": False,
        }

    def _graceful_stop(self) -> None:
        """Send ``stop`` and collect ``done`` messages until every child exits."""
        for child in self._children:
            if not child.eof:
                self._send(child, {"type": "stop"})
        deadline = time.perf_counter() + GRACEFUL_STOP_TIMEOUT_S
        while not all(child.eof for child in self._children):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            try:
                child_index, message = self._events.get(
                    timeout=min(EVENT_POLL_S, remaining),
                )
            except queue.Empty:
                continue
            if child_index == _WAKE_EVENT:
                continue
            child = self._children[child_index]
            if message is None:
                child.eof = True
            elif message["type"] == "done":
                self._on_done(child_index, message)
            elif message["type"] == "want":
                self._send(
                    child, {"type": "work", "slot": message["slot"], "items": []}
                )

    # * Measurement.

    def _open_safety_rails(self) -> JsonDict:
        """Create the capped step job; return its description for the result.

        ``cap_enforced`` stays ``False`` when the job cannot be created; the
        caller then refuses to start the step unless ``--allow-uncapped``.
        """
        reading = sysmetrics.query_commit()
        if reading.source != sysmetrics.COMMIT_SOURCE_WINDOWS:
            echo("  WARNING: step job commit cap inactive (Windows-only)")
            return {
                "limit_gb": None,
                "cap_reason": None,
                "cap_enforced": False,
                "rule": "inactive: Windows-only",
            }
        limit, info = self._job_limit(reading.free_bytes)
        if limit <= 0:
            return info
        job = KillOnCloseJob(
            memory_limit_bytes=limit,
            on_memory_limit=self._on_memory_cap,
        )
        if not job.active:
            info["job_error"] = job.error
            job.close()
            return info
        self._step_job = job
        info["cap_enforced"] = True
        if job.monitor_error is not None:
            # * The cap still holds; only its immediate report is missing.
            info["cap_monitor_error"] = job.monitor_error
            echo(f"  WARNING: step job: {job.monitor_error}")
        return info

    def _refuse_to_start(self, job_info: Mapping[str, Any]) -> tuple[str, str] | None:
        """Return ``(status, reason)`` when the step must not start, else ``None``."""
        limit_gb = job_info.get("limit_gb")
        if limit_gb is not None and limit_gb <= 0:
            return (
                STATUS_SYSTEM_COMMIT_LOW,
                f"free commit {job_info['free_commit_gb_at_start']:.2f} GB leaves no "
                f"room above --commit-reserve-gb {self._options.commit_reserve_gb:g}; "
                "the step did not start",
            )
        uncapped = sys.platform == "win32" and not job_info.get("cap_enforced")
        if uncapped and not self._options.allow_uncapped:
            # ! Fail closed: without the cap a step could exhaust the commit limit.
            return (
                "skipped",
                f"commit cap unavailable: {job_info.get('job_error') or 'unknown'}; "
                "pass --allow-uncapped to run without it",
            )
        return None

    def _close_safety_rails(
        self,
        status: str,
        reason: str | None,
        info: JsonDict,
    ) -> tuple[str, str | None]:
        """Record the job peak, classify a missed cap, and close the step job.

        The completion-port message is the primary cap signal. As a fallback, a
        step that failed - a child died (``error``) or a living child reported
        failed items (for example ``MemoryError`` while decoding) - while its
        job peak came within :data:`_CAP_PEAK_MARGIN_BYTES` of the cap is
        classified as capped too, whether or not the port works.
        """
        job, self._step_job = self._step_job, None
        if job is None:
            info.setdefault("cap_hit", False)
            return status, reason
        try:
            if self._children:
                # * Enforced only when every child actually joined the step job.
                info["cap_enforced"] = all(c.in_step_job for c in self._children)
            peak = job.peak_memory_bytes()
            limit = job.memory_limit_bytes
            info["peak_job_commit_gb"] = (
                round(peak / GIB, 3) if peak is not None else None
            )
            near_cap = (
                limit is not None
                and peak is not None
                and peak >= limit - _CAP_PEAK_MARGIN_BYTES
            )
            item_errors = any(
                outcome.error
                for side in self._sides
                for outcome in side.outcomes.values()
            )
            failed = status == "error" or (status in {"ok", "partial"} and item_errors)
            info["cap_hit"] = job.memory_limit_hit or (failed and near_cap)
            if failed and (job.memory_limit_hit or near_cap):
                inferred = not job.memory_limit_hit
                info["cap_inferred_from_peak"] = inferred
                status = STATUS_MEMORY_CAP
                reason = self._cap_reason(limit, peak, reason, inferred=inferred)
        finally:
            job.close()
        return status, reason

    def _cap_reason(
        self,
        limit: int | None,
        peak: int | None,
        reason: str | None,
        *,
        inferred: bool,
    ) -> str:
        """Describe a cap outcome with the failure that revealed it.

        Args:
            limit: Job commit cap in bytes.
            peak: Job peak commit in bytes.
            reason: Earlier step reason, if any.
            inferred: The cap was deduced from the peak (no port message).

        Returns:
            Reason text naming the cap, the peak, and the first failure.

        """
        how = "reached, inferred from the job peak" if inferred else "reached"
        text = (
            f"step job commit cap {(limit or 0) / GIB:.2f} GB {how} "
            f"(peak {(peak or 0) / GIB:.2f} GB)"
        )
        detail = reason or next(
            (
                outcome.error
                for side in self._sides
                for outcome in side.outcomes.values()
                if outcome.error
            ),
            None,
        )
        return f"{text}; {detail[:_REASON_DETAIL_LIMIT]}" if detail else text

    def _run_children(
        self,
        job_info: Mapping[str, Any],
        watchdog: CommitWatchdog,
    ) -> tuple[str, str | None]:
        """Spawn, measure, and stop the children; return the step outcome."""
        # * Only Windows requires the cap; there it is missing only with
        # * --allow-uncapped (otherwise the step was refused).
        if sys.platform == "win32" and not job_info["cap_enforced"]:
            echo("  !!! WARNING: this step runs WITHOUT a commit cap !!!")
        # ! The watchdog and the cap monitor stay active until every child has
        # ! exited (graceful stop and termination included); _clean_up stops them.
        watchdog.start()
        self._spawn_children()
        status, reason = self._event_loop()
        if status == "ok":
            self._graceful_stop()
        return self._late_abort(status, reason)

    def _late_abort(self, status: str, reason: str | None) -> tuple[str, str | None]:
        """Let a cap or watchdog kill after the last result invalidate the step.

        Workers still hold their models while they shut down, so an abort in
        that window replaces a finished outcome (``ok``/``partial``); an earlier
        failure keeps its own status.
        """
        aborted = self._aborted()
        if aborted is not None and status in {"ok", "partial"}:
            return aborted
        return status, reason

    def _clean_up(
        self,
        status: str,
        reason: str | None,
        job_info: JsonDict,
        watchdog: CommitWatchdog | None,
    ) -> tuple[str, str | None]:
        """End the children, then stop the watchdog, close pipes and the step job.

        The watchdog runs until the children are gone, and an abort it (or the
        cap monitor) caused in the meantime is applied to the status. Every stage
        runs even when an earlier one raises, so the step job and its monitor
        thread are always closed.
        """
        try:
            self._terminate_children(graceful=status == "ok")
        finally:
            try:
                if watchdog is not None:
                    watchdog.stop()
                status, reason = self._late_abort(status, reason)
            finally:
                try:
                    self._close_children()
                finally:
                    status, reason = self._close_safety_rails(
                        status,
                        reason,
                        job_info,
                    )
        return status, reason

    def execute(self) -> JsonDict:
        """Run every side and return the measurement.

        Returns:
            ``status``, ``reason``, shared ``cpu``/``memory``/``vram``, the
            combined ``timing`` and one document per side under ``sides``.

        """
        memory = ProcessTreeSampler(self._options.sample_interval_s)
        gpu: GpuPoller | None = None
        vram_baseline: GpuSample | None = None
        uses_cuda = any(side.spec.config.device == "cuda" for side in self._sides)
        if uses_cuda and self._nvidia_smi is not None:
            vram_baseline = query_gpu(self._nvidia_smi)
            gpu = GpuPoller(self._nvidia_smi, self._options.sample_interval_s)
        status = "error"
        reason: str | None = "configuration did not start"
        job_info: JsonDict = {"limit_gb": None, "cap_enforced": False}
        watchdog: CommitWatchdog | None = None
        try:
            # * Everything that opens a resource sits inside the try, so the
            # * finally below always releases the step job and its threads.
            job_info = self._open_safety_rails()
            watchdog = CommitWatchdog(
                int(self._options.commit_reserve_gb * GIB),
                WATCHDOG_INTERVAL_S,
                self._on_commit_low,
            )
            job_info["watchdog"] = watchdog.error or "active"
            if watchdog.error is not None:
                echo(f"  WARNING: commit watchdog inactive ({watchdog.error})")
            memory.start()
            if gpu is not None:
                gpu.start()
            refusal = self._refuse_to_start(job_info)
            if refusal is not None:
                status, reason = refusal
            else:
                status, reason = self._run_children(job_info, watchdog)
        finally:
            try:
                status, reason = self._clean_up(status, reason, job_info, watchdog)
            finally:
                memory.stop()
                if gpu is not None:
                    gpu.stop()
        mem = memory.stats(self._t_go, self._t_end)
        gpu_stats = gpu.stats(vram_baseline, self._t_go, self._t_end) if gpu else None
        sides = [self._side_document(side, status, reason) for side in self._sides]
        return {
            "status": status,
            "reason": reason,
            "job": job_info,
            "timing": self._combined_timing(sides),
            "cpu": {
                "system_busy_percent": _round(
                    cpu_busy_percent(self._cpu_go, self._cpu_end),
                    1,
                ),
                "logical_cpus": os.cpu_count(),
            },
            "memory": {
                "peak_private_bytes": mem.peak_private_bytes,
                "peak_private_run_bytes": mem.peak_private_run_bytes,
                "peak_working_set_bytes": mem.peak_working_set_bytes,
                "peak_working_set_run_bytes": mem.peak_working_set_run_bytes,
                "min_free_commit_bytes": mem.min_free_commit_bytes,
                "max_processes": mem.max_processes,
                "samples": mem.samples,
            },
            "vram": (
                {
                    "baseline_mb": gpu_stats.baseline_mb,
                    "peak_mb": gpu_stats.peak_mb,
                    "peak_run_mb": gpu_stats.peak_run_mb,
                    "delta_mb": gpu_stats.delta_mb,
                    "delta_run_mb": gpu_stats.delta_run_mb,
                    "total_mb": gpu_stats.total_mb,
                    "util_avg_run_percent": _round(gpu_stats.util_avg_run_percent, 1),
                    "samples": gpu_stats.samples,
                }
                if gpu_stats is not None
                else None
            ),
            "sides": sides,
        }

    def _common_window_end(self) -> float | None:
        """End of the common concurrent window of a multi-side measurement.

        The window runs from the shared clock start until the first side
        finished its last item; ``None`` for one side or an unfinished side.
        """
        ends = [side.t_end for side in self._sides]
        if len(ends) < _MIN_SIDES_FOR_WINDOW or any(end is None for end in ends):
            return None
        return min(end for end in ends if end is not None)

    def _window_audio(self, side: _Side, end: float) -> float:
        """Audio seconds processed inside ``[t_go, end]``.

        An item counts with the share of its processing interval (dispatch to
        result) that lies inside the window, so items crossing the window end
        are neither dropped nor counted in full.
        """
        start = self._t_go or 0.0
        produced = {o.item_id: o for o in side.spec.stream.produced}
        total = 0.0
        for outcome in side.outcomes.values():
            began = side.dispatched_at.get(outcome.item_id)
            ended = side.finished_at.get(outcome.item_id)
            if outcome.error is not None or began is None or ended is None:
                continue
            inside = max(0.0, min(ended, end) - max(began, start))
            share = inside / (ended - began) if ended > began else float(ended <= end)
            total += produced[outcome.item_id].clip.duration_s * share
        return total

    def _combined_timing(self, sides: Sequence[JsonDict]) -> JsonDict:
        window_end = self._common_window_end()
        end = window_end if window_end is not None else self._t_end
        wall_s = (
            end - self._t_go if end is not None and self._t_go is not None else None
        )
        audio_s = sum(float(side["timing"]["audio_s"] or 0.0) for side in sides)
        spawn_to_go = (
            self._t_go - self._t_spawn
            if self._t_go is not None and self._t_spawn is not None
            else None
        )
        timing: JsonDict = {
            "wall_s": _round(wall_s),
            "audio_s": _round(audio_s),
            "throughput_x": _round(audio_s / wall_s) if wall_s else None,
            "spawn_to_ready_s": _round(spawn_to_go),
        }
        if window_end is not None:
            full_wall = (
                self._t_end - self._t_go
                if self._t_end is not None and self._t_go is not None
                else None
            )
            full_audio = sum(float(s["timing"]["full_audio_s"] or 0.0) for s in sides)
            timing.update(
                window="common",
                full_wall_s=_round(full_wall),
                full_audio_s=_round(full_audio),
                full_throughput_x=_round(full_audio / full_wall) if full_wall else None,
            )
        return timing

    def _side_timing(self, side: _Side) -> JsonDict:
        produced = {o.item_id: o for o in side.spec.stream.produced}
        ok = [o for o in side.outcomes.values() if o.error is None]
        audio_s = sum(produced[o.item_id].clip.duration_s for o in ok)
        wall_s = (
            side.t_end - self._t_go
            if side.t_end is not None and self._t_go is not None
            else None
        )
        window_end = self._common_window_end()
        window: JsonDict = {}
        if window_end is not None and self._t_go is not None:
            # * Combo: the headline numbers cover only the common concurrent
            # * window; the whole run of the side stays available as full_*.
            window = {
                "window": "common",
                "full_wall_s": _round(wall_s),
                "full_audio_s": _round(audio_s),
                "full_throughput_x": _round(audio_s / wall_s) if wall_s else None,
            }
            wall_s = window_end - self._t_go
            audio_s = self._window_audio(side, window_end)
        latencies = [o.proc_s for o in ok]
        children = [self._children[i] for i in side.children]
        loads = [c.load_s for c in children if c.load_s is not None]
        warmups = [c.warmup_s for c in children if c.warmup_s is not None]
        spawn_to_go = (
            self._t_go - self._t_spawn
            if self._t_go is not None and self._t_spawn is not None
            else None
        )
        return {
            "wall_s": _round(wall_s),
            "audio_s": _round(audio_s),
            "throughput_x": _round(audio_s / wall_s) if wall_s else None,
            "latency_p50_s": _round(percentile(latencies, 50.0)),
            "latency_p95_s": _round(percentile(latencies, 95.0)),
            "latency_mean_s": (
                _round(statistics.fmean(latencies)) if latencies else None
            ),
            "load_s": [_round(value) for value in loads],
            "load_max_s": _round(max(loads)) if loads else None,
            "load_mean_s": _round(statistics.fmean(loads)) if loads else None,
            "warmup_s": [_round(value) for value in warmups],
            "spawn_to_ready_s": _round(spawn_to_go),
            "items_processed": len(side.outcomes),
            **window,
        }

    def _side_document(self, side: _Side, status: str, reason: str | None) -> JsonDict:
        stream = side.spec.stream
        planned = stream.planned()
        errors = [o for o in side.outcomes.values() if o.error]
        missing = sum(1 for o in planned if o.item_id not in side.outcomes)
        side_status, side_reason = status, reason
        if status == "ok" and errors:
            side_status = "partial" if len(errors) < len(planned) else "error"
            side_reason = f"{len(errors)} item(s) failed; first: {errors[0].error}"
        timing = self._side_timing(side)
        children = [self._children[i] for i in side.children]
        cpu_spans = [
            c.cpu_done_s - c.cpu_ready_s
            for c in children
            if c.cpu_done_s is not None and c.cpu_ready_s is not None
        ]
        # * CPU time spans the whole run of the side, so divide by its full wall.
        wall_s = timing.get("full_wall_s", timing["wall_s"])
        parity = [
            c.warmup.get("api_parity")
            for c in children
            if c.warmup is not None and "api_parity" in c.warmup
        ]
        config = side.spec.config
        return {
            "label": side.spec.label,
            "config": config.to_json(),
            "status": side_status,
            "reason": side_reason,
            "timing": timing,
            "cpu": {
                "worker_cores_avg": (
                    _round(sum(cpu_spans) / wall_s, 2)
                    if len(cpu_spans) == len(children) and wall_s
                    else None
                ),
                "requested_threads": config.total_threads,
            },
            "stream": {
                "clips": len(stream.clips),
                "min_items": stream.min_items,
                "min_wall_s": stream.min_wall_s,
                "produced": len(stream.produced),
            },
            "api_parity": all(p is True for p in parity) if parity else None,
            "children": [
                {
                    "index": c.index,
                    "pid": c.proc.pid,
                    "exit_code": c.proc.returncode,
                    "in_job": c.in_job,
                    "in_step_job": c.in_step_job,
                    "load_s": _round(c.load_s),
                    "warmup_s": _round(c.warmup_s),
                    "info": c.info,
                    "warmup": c.warmup,
                    "fatal": c.fatal,
                    "log": str(c.log_path),
                }
                for c in children
            ],
            "error_count": len(errors) + missing,
            "errors": [
                {"item": o.item_id, "error": o.error}
                for o in errors[:MAX_RECORDED_ERRORS]
            ],
            "items": [self._item_record(side, o) for o in planned],
        }

    def _item_record(self, side: _Side, occurrence: Occurrence) -> JsonDict:
        clip = occurrence.clip
        record: JsonDict = {
            "index": occurrence.item_id,
            "clip": clip.index,
            "cycle": occurrence.cycle,
            "name": clip.path.name,
            "duration_s": _round(clip.duration_s),
        }
        outcome = side.outcomes.get(occurrence.item_id)
        if outcome is None:
            record.update(text="", proc_s=None, error="not processed")
            return record
        go = self._t_go or 0.0
        began = side.dispatched_at.get(occurrence.item_id)
        ended = side.finished_at.get(occurrence.item_id)
        record.update(
            text=outcome.text,
            proc_s=_round(outcome.proc_s, 4),
            child=outcome.child,
            slot=outcome.slot,
            batch=outcome.batch,
            error=outcome.error,
            dispatched_s=_round(began - go) if began is not None else None,
            finished_s=_round(ended - go) if ended is not None else None,
        )
        return record


def single_side_document(measured: JsonDict) -> JsonDict:
    """Flatten a one-side measurement into the result layout of one configuration.

    Args:
        measured: Output of :meth:`Measurement.execute` with one side.

    Returns:
        Status, timing, cpu, memory, vram, children, errors and items.

    """
    side = measured["sides"][0]
    return {
        "status": side["status"],
        "reason": side["reason"],
        "timing": side["timing"],
        "cpu": {
            "system_busy_percent": measured["cpu"]["system_busy_percent"],
            "worker_cores_avg": side["cpu"]["worker_cores_avg"],
            "requested_threads": side["cpu"]["requested_threads"],
            "logical_cpus": measured["cpu"]["logical_cpus"],
        },
        "memory": measured["memory"],
        "vram": measured["vram"],
        "job": measured.get("job"),
        "stream": side["stream"],
        "api_parity": side["api_parity"],
        "children": side["children"],
        "error_count": side["error_count"],
        "errors": side["errors"],
        "items": side["items"],
    }


def quality_block(
    items: Sequence[Mapping[str, Any]], manifest: Manifest | None
) -> JsonDict:
    """Return WER (with a manifest) and the repeat consistency of result items.

    Args:
        items: Result items.
        manifest: Reference manifest, if any.

    Returns:
        ``wer`` (``None`` without a manifest) and ``repeat_consistency_percent``.

    """
    return {
        "wer": corpus_wer(items, manifest) if manifest is not None else None,
        "repeat_consistency_percent": repeat_consistency(items),
    }


# * Guard and orchestration.


def _need_sum(configs: Sequence[BenchConfig], options: RunOptions) -> ResourceNeed:
    needs = [estimate_need(config, options.estimates) for config in configs]
    return ResourceNeed(
        commit_gb=sum(n.commit_gb for n in needs),
        vram_gb=sum(n.vram_gb for n in needs),
    )


def evaluate_guard(
    config: BenchConfig,
    options: RunOptions,
    nvidia_smi: str | None,
) -> tuple[bool, str | None, JsonDict]:
    """Check free commit and free VRAM against the configuration estimate.

    Args:
        config: Configuration.
        options: Run options (estimates and thresholds).
        nvidia_smi: ``nvidia-smi`` path, if any.

    Returns:
        ``(allowed, skip_reason, details)``.

    """
    return evaluate_guard_configs([config], options, nvidia_smi)


def evaluate_guard_configs(
    configs: Sequence[BenchConfig],
    options: RunOptions,
    nvidia_smi: str | None,
) -> tuple[bool, str | None, JsonDict]:
    """Check free commit and free VRAM against the summed estimate of *configs*.

    Args:
        configs: Configurations that run at the same time.
        options: Run options (estimates and thresholds).
        nvidia_smi: ``nvidia-smi`` path, if any.

    Returns:
        ``(allowed, skip_reason, details)``.

    """
    need = _need_sum(configs, options)
    commit = query_commit()
    free_commit_gb = commit.free_bytes / GIB
    details: JsonDict = {
        "need_commit_gb": round(need.commit_gb, 2),
        "need_vram_gb": round(need.vram_gb, 2),
        "free_commit_gb": round(free_commit_gb, 2),
        "commit_limit_gb": (
            round(commit.limit_bytes / GIB, 2) if commit.limit_bytes else None
        ),
        "commit_source": commit.source,
        "min_free_commit_gb": options.min_free_commit_gb,
        "commit_reserve_gb": options.commit_reserve_gb,
        "min_free_vram_mb": options.min_free_vram_mb,
    }
    headroom_gb = free_commit_gb - need.commit_gb
    # * The step job keeps --commit-reserve-gb free, so a step that would need
    # * more than the headroom above it is refused here instead of capped later.
    min_commit_gb = max(options.min_free_commit_gb, options.commit_reserve_gb)
    if headroom_gb < min_commit_gb:
        return (
            False,
            f"free commit {free_commit_gb:.1f} GB - need {need.commit_gb:.1f} GB = "
            f"{headroom_gb:.1f} GB < max(--min-free-commit-gb "
            f"{options.min_free_commit_gb:g}, --commit-reserve-gb "
            f"{options.commit_reserve_gb:g})",
            details,
        )
    if not any(config.device == "cuda" for config in configs):
        return True, None, details
    sample = query_gpu(nvidia_smi) if nvidia_smi is not None else None
    if sample is None:
        return False, "nvidia-smi is unavailable; free VRAM cannot be verified", details
    need_vram_mb = need.vram_gb * GIB / MIB
    headroom_mb = sample.free_mb - need_vram_mb
    details.update(
        free_vram_mb=sample.free_mb,
        total_vram_mb=sample.total_mb,
        need_vram_mb=round(need_vram_mb),
    )
    if headroom_mb < options.min_free_vram_mb:
        return (
            False,
            f"free VRAM {sample.free_mb:.0f} MB - need {need_vram_mb:.0f} MB = "
            f"{headroom_mb:.0f} MB < --min-free-vram-mb {options.min_free_vram_mb:g}",
            details,
        )
    return True, None, details


def collect_host_info(options: RunOptions, nvidia_smi: str | None) -> JsonDict:
    """Describe the host for the result files.

    Args:
        options: Run options.
        nvidia_smi: ``nvidia-smi`` path, if any.

    Returns:
        Host description.

    """
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_logical": os.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "cpu_budget": options.thread_defaults.cpu_budget,
        "cuda_threads": options.thread_defaults.cuda_threads,
        "ram_total_gb": round(float(psutil.virtual_memory().total) / GIB, 1),
        "gpu": query_gpu_name(nvidia_smi) if nvidia_smi else None,
    }


def decode_info(config: BenchConfig, options: RunOptions) -> JsonDict:
    """Describe the decode settings of a configuration for its result file.

    Args:
        config: Configuration.
        options: Run options.

    Returns:
        Model and decode description.

    """
    if config.engine == "whisper":
        batched = config.mode == "B"
        options_used = dict(options.whisper_options)
        if batched:
            options_used.setdefault("without_timestamps", False)
        return {
            "model": options.whisper_model,
            "options": options_used,
            "batching": (
                ("chunks of one file" if config.long else "across files")
                if batched
                else None
            ),
            "batched_chunk_s": options.whisper_chunk_s if batched else None,
        }
    model = (
        "ai-sage/GigaAM-Multilingual@ctc"
        if config.engine == "gigaam-ctc"
        else "ai-sage/GigaAM-v3@e2e_ctc"
    )
    return {
        "model": model,
        "window_s": GIGAAM_WINDOW_S,
        "batch_size": config.batch_size,
        "cuda_thread_streams": (
            options.gigaam_cuda_streams and config.device == "cuda" and config.slots > 1
        ),
    }


def document_base(
    config: BenchConfig,
    *,
    repeat: int,
    workload: Workload,
    options: RunOptions,
    host: JsonDict,
    stream_info: JsonDict,
) -> JsonDict:
    """Return the fields every result document of one configuration starts with.

    Args:
        config: Configuration.
        repeat: Repeat index.
        workload: Resolved workload.
        options: Run options.
        host: Host description.
        stream_info: Workload sizing of this measurement.

    Returns:
        Base result document.

    """
    items = workload.long_items if config.long else workload.clips
    return {
        "schema": RESULT_SCHEMA_VERSION,
        "config": config.to_json(),
        "repeat": repeat,
        "started_at": now_iso(),
        "host": host,
        "workload": {
            "kind": config.workload_kind,
            "items": len(items),
            "audio_s": round(sum(item.duration_s for item in items), 3),
            "source": (
                [str(item.path) for item in items]
                if config.long
                else str(options.clips_dir)
            ),
            "max_clips": None if config.long else options.max_clips,
            "warmup": str(workload.warmup_path),
            **stream_info,
        },
        "decode": decode_info(config, options),
    }


def _run_config(
    config: BenchConfig,
    repeat: int,
    *,
    workload: Workload,
    options: RunOptions,
    job: KillOnCloseJob,
    logs_dir: Path,
    nvidia_smi: str | None,
    host: JsonDict,
) -> JsonDict:
    """Guard, execute, and describe one configuration repeat of ``run``."""
    items = workload.long_items if config.long else workload.clips
    document = document_base(
        config,
        repeat=repeat,
        workload=workload,
        options=options,
        host=host,
        stream_info={"passes": 1},
    )
    if config.long and not items:
        document.update(
            status="skipped",
            reason="no --long-files given",
            finished_at=now_iso(),
        )
        return document
    allowed, reason, guard = evaluate_guard(config, options, nvidia_smi)
    document["guard"] = guard
    if not allowed:
        document.update(status="skipped", reason=reason, finished_at=now_iso())
        return document
    stream = WorkStream(
        items, min_items=len(items), min_wall_s=0.0, max_items=len(items)
    )
    measurement = Measurement(
        sides=[SideSpec(config=config, stream=stream)],
        run_id=f"{config.name}__r{repeat}",
        warmup_path=workload.warmup_path,
        options=options,
        job=job,
        logs_dir=logs_dir,
        nvidia_smi=nvidia_smi,
    )
    document.update(single_side_document(measurement.execute()))
    document["quality"] = quality_block(document["items"], workload.manifest)
    document["finished_at"] = now_iso()
    return document


def _clear_out_dir(out_dir: Path) -> None:
    """Delete only the files ``run`` writes (results, logs, summary)."""
    for directory, pattern in (
        (out_dir / RESULTS_DIRNAME, "*.json"),
        (out_dir / LOGS_DIRNAME, "*.log"),
    ):
        if directory.is_dir():
            for path in directory.glob(pattern):
                path.unlink()
    for name in (SUMMARY_MD_NAME, SUMMARY_JSON_NAME):
        (out_dir / name).unlink(missing_ok=True)


def result_line(result: Mapping[str, Any]) -> str:
    """Return a one-line description of a result for the console.

    Args:
        result: Result document.

    Returns:
        Status, wall time, throughput, latency, WER, and reason.

    """
    timing = result.get("timing") or {}
    parts = [f"  -> {result.get('status')}"]
    if timing.get("wall_s") is not None:
        throughput = timing.get("throughput_x") or 0.0
        p50 = timing.get("latency_p50_s") or 0.0
        parts.append(
            f"wall {timing['wall_s']:.1f} s, {throughput:.1f}x RT, p50 {p50:.2f} s",
        )
    wer = ((result.get("quality") or {}).get("wer") or {}).get("wer_percent")
    if wer is not None:
        parts.append(f"WER {wer:.2f}%")
    if result.get("reason"):
        parts.append(str(result["reason"]))
    return "; ".join(parts)


def _print_run_plan(
    configs: Sequence[BenchConfig],
    options: RunOptions,
    store: ResultStore,
    nvidia_smi: str | None,
    reuse: frozenset[str],
) -> None:
    """Print what ``run`` would do; read-only (no file is created or deleted)."""
    if options.clear:
        echo("Dry run: --clear would delete results, logs and the summary first.")
    for repeat in range(options.repeat):
        for config in configs:
            key = f"{config.name}__r{repeat}"
            reusable = reusable_result(store, key, statuses=reuse, options=options)
            if not options.clear and reusable is not None:
                echo(f"[{key}] {store.describe(key)}; would be skipped")
                continue
            allowed, reason, guard = evaluate_guard(config, options, nvidia_smi)
            verdict = "run" if allowed else f"skip: {reason}"
            echo(f"[{key}] {verdict}; guard {guard}")


def run_benchmark(options: RunOptions) -> int:
    """Run (or dry-run) every selected configuration and write the summary.

    A dry run neither creates nor deletes anything in ``--out-dir``.

    Args:
        options: Resolved options.

    Returns:
        Process exit code.

    """
    workload = load_workload(options)
    configs = select_configs(
        options.selection,
        options.thread_defaults,
        has_long=bool(workload.long_items),
    )
    fingerprint = fingerprint_of(fingerprint_payload(options, workload))
    store = ResultStore(options.out_dir / RESULTS_DIRNAME, fingerprint)
    nvidia_smi = find_nvidia_smi()
    total_audio = sum(item.duration_s for item in workload.clips)
    echo(
        f"Workload: {len(workload.clips)} clips ({total_audio:.0f} s audio), "
        f"{len(workload.long_items)} long file(s); "
        f"warm-up {workload.warmup_path.name}; fingerprint {fingerprint}",
    )
    echo(f"Configurations ({len(configs)} x {options.repeat} repeat(s)):")
    for config in configs:
        echo(f"  {config.name}")
    reuse = REUSABLE_STATUSES if not options.retry_failed else frozenset({"ok"})
    if options.dry_run:
        _print_run_plan(configs, options, store, nvidia_smi, reuse)
        return 0
    if options.clear:
        _clear_out_dir(options.out_dir)
    logs_dir = options.out_dir / LOGS_DIRNAME
    store.directory.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    host = collect_host_info(options, nvidia_smi)
    job = KillOnCloseJob()
    if job.error:
        echo(f"WARNING: kill-on-close job unavailable ({job.error})")
    executed = 0
    try:
        for repeat in range(options.repeat):
            # * Alternate the order across repeats to spread drift bias.
            ordered = configs if repeat % 2 == 0 else list(reversed(configs))
            for config in ordered:
                key = f"{config.name}__r{repeat}"
                if (
                    reusable_result(store, key, statuses=reuse, options=options)
                    is not None
                ):
                    echo(f"[{key}] {store.describe(key)}; skipped")
                    continue
                if executed and options.cooldown_s > 0:
                    time.sleep(options.cooldown_s)
                echo(f"[{key}] {now_iso()} ({store.describe(key)})")
                result = _run_config(
                    config,
                    repeat,
                    workload=workload,
                    options=options,
                    job=job,
                    logs_dir=logs_dir,
                    nvidia_smi=nvidia_smi,
                    host=host,
                )
                executed += 1
                store.save(key, result)
                echo(result_line(result))
                report.write_summary(options.out_dir, fingerprint=fingerprint)
    finally:
        job.close()
    md_path, table = report.write_summary(options.out_dir, fingerprint=fingerprint)
    echo("")
    echo(table)
    echo(f"\nSummary: {md_path}")
    return 0
