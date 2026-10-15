"""STT pool configuration: defaults, JSON loading, validation and overrides.

Resolution order for the config file: an explicit path, then the
``ASKVLM_STT_POOL_CONFIG`` environment variable, then ``<project>/stt_pool.json``
when it exists, then the conservative code defaults. Unknown keys and invalid
values fail fast with a :class:`PoolConfigError` that names the JSON path.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.stt_pool.keys import ProviderFamily

ENV_POOL_CONFIG = "ASKVLM_STT_POOL_CONFIG"
CONFIG_FILENAME = "stt_pool.json"
CONFIG_SCHEMA = "askvlm.stt_pool.config"
CONFIG_VERSION = 1
GIB = 1024**3

# * Code defaults for an unknown machine; deliberately conservative.
DEFAULT_CPU_MAX_SLOTS = 2
DEFAULT_CPU_THREADS_PER_SLOT = 2
DEFAULT_GPU_MAX_WORKERS = 1
DEFAULT_BATCH_LANES: dict[ProviderFamily, int] = {"gigaam": 8, "whisper": 4}
DEFAULT_HOST_THREADS: dict[ProviderFamily, int] = {"gigaam": 3, "whisper": 2}
DEFAULT_SCALE_OUT_BACKLOG_S = 30.0
DEFAULT_MIN_THREAD_BUDGET = 2
DEFAULT_COMMIT_BUDGET_GIB = 12.0
DEFAULT_FREE_COMMIT_GUARD_GIB = 8.0
DEFAULT_CRITICAL_FREE_COMMIT_GIB = 4.0
DEFAULT_VRAM_BUDGET_GIB = 4.0
DEFAULT_VRAM_FREE_GUARD_GIB = 0.5
DEFAULT_WORKER_IDLE_TTL_S = 600.0
DEFAULT_SUPERVISOR_LINGER_S = 30.0
DEFAULT_STOP_GRACE_S = 20.0
DEFAULT_CANCEL_GRACE_S = 30.0
DEFAULT_SPILL_RATIO = 0.8
DEFAULT_STARVATION_AFTER_S = 60.0
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_GPU_CPU_CONTENTION = 0.5
DEFAULT_PRIORITY_AGING_S = 300.0
DEFAULT_LOAD_FAILURE_COOLDOWN_S = 300.0
DEFAULT_CRASH_COOLDOWN_S = 60.0

# * CPU threads of one running speaker diarization job (the sherpa-onnx
# * networks). Without an explicit setting the value is clamped to the thread
# * budget, so the conservative default budget still runs diarization.
DEFAULT_DIARIZATION_THREADS_PER_SLOT = 4

# * Largest accepted per-worker batch lane count.
MAX_BATCH_LANES = 64

_FAMILIES: tuple[ProviderFamily, ...] = ("gigaam", "whisper")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "version",
        "cpu",
        "gpu",
        "threads",
        "memory",
        "vram",
        "lifecycle",
        "scheduling",
        "diarization",
    }
)
_CPU_KEYS = frozenset({"max_slots", "threads_per_slot", "max_slots_per_worker"})
_DIARIZATION_KEYS = frozenset({"threads_per_slot"})
_DIARIZATION_THREADS_PATH = "diarization.threads_per_slot"
_GPU_KEYS = frozenset(
    {"max_workers", "batch_lanes", "host_threads", "scale_out_backlog_s"}
)
_THREADS_KEYS = frozenset({"budget"})
_MEMORY_KEYS = frozenset(
    {"commit_budget_gib", "free_commit_guard_gib", "critical_free_commit_gib"}
)
_VRAM_KEYS = frozenset({"budget_gib", "free_guard_gib"})
_LIFECYCLE_KEYS = frozenset(
    {"worker_idle_ttl_s", "supervisor_linger_s", "stop_grace_s", "cancel_grace_s"}
)
_SCHEDULING_KEYS = frozenset(
    {
        "spill_ratio",
        "starvation_after_s",
        "max_attempts",
        "gpu_cpu_contention",
        "priority_aging_s",
        "load_failure_cooldown_s",
        "crash_cooldown_s",
    }
)


class PoolConfigError(ValueError):
    """Raised for an unreadable config file, an unknown key or an invalid value.

    Attributes:
        json_path: Dotted JSON path of the offending key, when known.

    """

    def __init__(self, message: str, *, json_path: str | None = None) -> None:
        """Create the error.

        Args:
            message: Human-readable description naming the JSON path.
            json_path: Dotted JSON path of the offending key, when known.

        """
        super().__init__(message)
        self.json_path = json_path


@dataclass(frozen=True, slots=True)
class CpuConfig:
    """CPU slot limits.

    Attributes:
        max_slots: Concurrent CPU jobs across all CPU workers (speech-to-text
            and speaker diarization).
        threads_per_slot: CPU threads used by one running speech-to-text CPU
            job (diarization jobs use :class:`DiarizationConfig`).
        max_slots_per_worker: Largest slot capacity of one CPU worker process.

    """

    max_slots: int
    threads_per_slot: int
    max_slots_per_worker: int


@dataclass(frozen=True, slots=True)
class GpuConfig:
    """GPU worker limits.

    The per-family mappings are stored as read-only views over private copies,
    so the frozen configuration cannot be mutated through them.

    Attributes:
        max_workers: Concurrent GPU worker processes.
        batch_lanes: Batch size of one GPU worker per provider family.
        host_threads: CPU threads reserved for one GPU worker per family.
        scale_out_backlog_s: Predicted GPU backlog that justifies one more
            GPU worker of the same configuration.

    """

    max_workers: int
    batch_lanes: Mapping[ProviderFamily, int]
    host_threads: Mapping[ProviderFamily, int]
    scale_out_backlog_s: float

    def __post_init__(self) -> None:
        """Replace the per-family mappings with read-only views of copies."""
        object.__setattr__(
            self, "batch_lanes", MappingProxyType(dict(self.batch_lanes))
        )
        object.__setattr__(
            self, "host_threads", MappingProxyType(dict(self.host_threads))
        )


@dataclass(frozen=True, slots=True)
class ThreadsConfig:
    """Supervisor-wide CPU thread budget.

    Attributes:
        budget: CPU threads shared by CPU jobs and GPU worker host threads.

    """

    budget: int


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """System memory (Windows commit) limits in bytes.

    Attributes:
        commit_budget_bytes: Total commit the pool may reserve.
        free_commit_guard_bytes: Free system commit that must remain after a
            worker spawn.
        critical_free_commit_bytes: Free system commit below which idle
            workers are stopped and no worker is spawned.

    """

    commit_budget_bytes: int
    free_commit_guard_bytes: int
    critical_free_commit_bytes: int


@dataclass(frozen=True, slots=True)
class VramConfig:
    """GPU memory limits in bytes.

    Attributes:
        budget_bytes: Total VRAM the pool may reserve.
        free_guard_bytes: Free VRAM that must remain after a GPU worker spawn.

    """

    budget_bytes: int
    free_guard_bytes: int


@dataclass(frozen=True, slots=True)
class LifecycleConfig:
    """Worker and supervisor lifetime timings in seconds.

    Attributes:
        worker_idle_ttl_s: Idle time after which a worker is stopped.
        supervisor_linger_s: Time the supervisor stays up with no work.
        stop_grace_s: Wait for a graceful worker shutdown before termination.
        cancel_grace_s: Wait for a cooperative cancel before the worker is killed.

    """

    worker_idle_ttl_s: float
    supervisor_linger_s: float
    stop_grace_s: float
    cancel_grace_s: float


@dataclass(frozen=True, slots=True)
class SchedulingConfig:
    """Scheduler tuning.

    Attributes:
        spill_ratio: An ``auto`` job spills to CPU when its CPU ETA is below
            this fraction of its GPU ETA.
        starvation_after_s: Wait after which the oldest feasible job is protected.
        max_attempts: Executions allowed per job (including the first).
        gpu_cpu_contention: Slowdown of GPU service per unit of CPU thread use.
        load_failure_cooldown_s: Cooldown of a configuration after a load failure.
        crash_cooldown_s: Base cooldown of a configuration after repeated crashes.
        priority_aging_s: Waiting time that lifts a pending job by one priority
            class (anti-starvation aging).

    """

    spill_ratio: float
    starvation_after_s: float
    max_attempts: int
    gpu_cpu_contention: float
    load_failure_cooldown_s: float
    crash_cooldown_s: float
    priority_aging_s: float = DEFAULT_PRIORITY_AGING_S


@dataclass(frozen=True, slots=True)
class DiarizationConfig:
    """Speaker diarization worker limits.

    A diarization worker runs one job at a time (one CPU slot per process), so
    only the threads of that job are configurable.

    Attributes:
        threads_per_slot: CPU threads of one running diarization job; the
            effective value (a default clamped to the thread budget when the
            config does not set it).
        threads_per_slot_explicit: Whether the config set ``threads_per_slot``
            explicitly; an explicit value is never clamped and must fit the
            thread budget.

    """

    threads_per_slot: int = DEFAULT_DIARIZATION_THREADS_PER_SLOT
    threads_per_slot_explicit: bool = False


@dataclass(frozen=True, slots=True)
class PoolConfig:
    """Complete validated pool configuration.

    Attributes:
        cpu: CPU slot limits.
        gpu: GPU worker limits.
        threads: CPU thread budget.
        memory: System memory limits.
        vram: GPU memory limits.
        lifecycle: Worker and supervisor lifetime timings.
        scheduling: Scheduler tuning.
        source: ``defaults`` or the path of the file the values came from.
        diarization: Speaker diarization worker limits.

    """

    cpu: CpuConfig
    gpu: GpuConfig
    threads: ThreadsConfig
    memory: MemoryConfig
    vram: VramConfig
    lifecycle: LifecycleConfig
    scheduling: SchedulingConfig
    source: str
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)


# * Field validators: every one names the JSON path in its error.


def _fail(path: str, message: str) -> PoolConfigError:
    """Build a config error for *path*.

    Args:
        path: Dotted JSON path of the offending key.
        message: What is wrong with the value.

    Returns:
        The error to raise.

    """
    return PoolConfigError(f"{path}: {message}", json_path=path)


def _as_int(
    value: object, path: str, *, minimum: int, maximum: int | None = None
) -> int:
    """Validate an integer within bounds (booleans are rejected).

    Args:
        value: Candidate value.
        path: Dotted JSON path for error messages.
        minimum: Smallest accepted value.
        maximum: Largest accepted value, or ``None`` for unbounded.

    Returns:
        The validated integer.

    Raises:
        PoolConfigError: When the value is not an integer or is out of bounds.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(path, f"expected an integer, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bound = (
            f">= {minimum}"
            if maximum is None
            else f"between {minimum} and {maximum} inclusive"
        )
        raise _fail(path, f"must be {bound}, got {value}")
    return value


def _as_float(
    value: object,
    path: str,
    *,
    minimum: float = 0.0,
    exclusive_minimum: bool = False,
    maximum: float | None = None,
) -> float:
    """Validate a finite number within bounds (booleans are rejected).

    Args:
        value: Candidate value.
        path: Dotted JSON path for error messages.
        minimum: Lower bound.
        exclusive_minimum: Whether *minimum* itself is rejected.
        maximum: Inclusive upper bound, or ``None`` for unbounded.

    Returns:
        The validated number as a float.

    Raises:
        PoolConfigError: When the value is not a finite number or is out of bounds.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(path, f"expected a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise _fail(path, "number is too large") from None
    if not math.isfinite(number):
        raise _fail(path, f"expected a finite number, got {value!r}")
    too_small = number <= minimum if exclusive_minimum else number < minimum
    if too_small or (maximum is not None and number > maximum):
        low = f"> {minimum}" if exclusive_minimum else f">= {minimum}"
        bound = low if maximum is None else f"{low} and <= {maximum}"
        raise _fail(path, f"must be {bound}, got {value}")
    return number


def _as_gib_bytes(value: object, path: str, *, allow_zero: bool) -> int:
    """Validate a size in GiB and convert it to bytes.

    Args:
        value: Candidate size in GiB.
        path: Dotted JSON path for error messages.
        allow_zero: Whether ``0`` is accepted (guards) or must be positive.

    Returns:
        The size in bytes.

    Raises:
        PoolConfigError: When the value is not a finite number or is out of bounds.

    """
    gib = _as_float(value, path, minimum=0.0, exclusive_minimum=not allow_zero)
    try:
        size = round(gib * GIB)
    except OverflowError:
        raise _fail(path, f"size is too large, got {value}") from None
    if size <= 0 and not allow_zero:
        raise _fail(path, f"must be greater than 0 bytes, got {value}")
    return size


# * Section parsers.


def _section(
    payload: Mapping[str, object], name: str, allowed: frozenset[str]
) -> Mapping[str, object]:
    """Return the object stored under *name* after rejecting unknown keys.

    Args:
        payload: Parent mapping.
        name: Section key.
        allowed: Keys the section may contain.

    Returns:
        The section mapping, or an empty mapping when the section is absent.

    Raises:
        PoolConfigError: When the section is not an object or has unknown keys.

    """
    if name not in payload:
        return {}
    raw = payload[name]
    if not isinstance(raw, Mapping):
        raise _fail(name, f"expected an object, got {raw!r}")
    _reject_unknown(raw, allowed, name)
    return raw


def _reject_unknown(
    section: Mapping[str, object], allowed: frozenset[str], parent: str
) -> None:
    """Reject keys outside *allowed*.

    Args:
        section: Mapping to inspect.
        allowed: Accepted keys.
        parent: Dotted JSON path of *section* (empty for the document root).

    Raises:
        PoolConfigError: Naming the first unknown key.

    """
    unknown = sorted(str(key) for key in section if key not in allowed)
    if unknown:
        path = f"{parent}.{unknown[0]}" if parent else unknown[0]
        allowed_list = ", ".join(sorted(allowed))
        raise _fail(path, f"unknown key; allowed keys: {allowed_list}")


def _family_ints(
    section: Mapping[str, object],
    name: str,
    parent: str,
    defaults: Mapping[ProviderFamily, int],
    *,
    minimum: int,
    maximum: int | None = None,
) -> dict[ProviderFamily, int]:
    """Parse a per-provider-family integer mapping.

    Args:
        section: Mapping that may contain *name*.
        name: Key of the family mapping.
        parent: Dotted JSON path of *section*.
        defaults: Values for absent families.
        minimum: Smallest accepted value.
        maximum: Largest accepted value, or ``None`` for unbounded.

    Returns:
        A mapping with exactly the ``gigaam`` and ``whisper`` families.

    Raises:
        PoolConfigError: When the mapping or a value is invalid.

    """
    path = f"{parent}.{name}"
    if name not in section:
        return dict(defaults)
    raw = section[name]
    if not isinstance(raw, Mapping):
        raise _fail(path, f"expected an object, got {raw!r}")
    _reject_unknown(raw, frozenset(_FAMILIES), path)
    return {
        family: _as_int(
            raw.get(family, defaults[family]),
            f"{path}.{family}",
            minimum=minimum,
            maximum=maximum,
        )
        for family in _FAMILIES
    }


def _parse_cpu(payload: Mapping[str, object]) -> CpuConfig:
    """Parse the ``cpu`` section.

    Args:
        payload: Document root.

    Returns:
        The CPU limits.

    """
    section = _section(payload, "cpu", _CPU_KEYS)
    max_slots = _as_int(
        section.get("max_slots", DEFAULT_CPU_MAX_SLOTS), "cpu.max_slots", minimum=0
    )
    return CpuConfig(
        max_slots=max_slots,
        threads_per_slot=_as_int(
            section.get("threads_per_slot", DEFAULT_CPU_THREADS_PER_SLOT),
            "cpu.threads_per_slot",
            minimum=1,
        ),
        max_slots_per_worker=_as_int(
            section.get("max_slots_per_worker", max(1, max_slots)),
            "cpu.max_slots_per_worker",
            minimum=1,
        ),
    )


def _parse_gpu(payload: Mapping[str, object]) -> GpuConfig:
    """Parse the ``gpu`` section.

    Args:
        payload: Document root.

    Returns:
        The GPU limits.

    """
    section = _section(payload, "gpu", _GPU_KEYS)
    return GpuConfig(
        max_workers=_as_int(
            section.get("max_workers", DEFAULT_GPU_MAX_WORKERS),
            "gpu.max_workers",
            minimum=0,
        ),
        batch_lanes=_family_ints(
            section,
            "batch_lanes",
            "gpu",
            DEFAULT_BATCH_LANES,
            minimum=1,
            maximum=MAX_BATCH_LANES,
        ),
        host_threads=_family_ints(
            section, "host_threads", "gpu", DEFAULT_HOST_THREADS, minimum=0
        ),
        scale_out_backlog_s=_as_float(
            section.get("scale_out_backlog_s", DEFAULT_SCALE_OUT_BACKLOG_S),
            "gpu.scale_out_backlog_s",
            exclusive_minimum=True,
        ),
    )


def _parse_threads(payload: Mapping[str, object], logical_cpus: int) -> ThreadsConfig:
    """Parse the ``threads`` section.

    Args:
        payload: Document root.
        logical_cpus: Logical CPU count that sizes the default budget.

    Returns:
        The thread budget.

    """
    section = _section(payload, "threads", _THREADS_KEYS)
    default_budget = max(DEFAULT_MIN_THREAD_BUDGET, logical_cpus // 2)
    return ThreadsConfig(
        budget=_as_int(
            section.get("budget", default_budget), "threads.budget", minimum=1
        )
    )


def _parse_memory(payload: Mapping[str, object]) -> MemoryConfig:
    """Parse the ``memory`` section.

    Args:
        payload: Document root.

    Returns:
        The commit limits in bytes.

    """
    section = _section(payload, "memory", _MEMORY_KEYS)
    return MemoryConfig(
        commit_budget_bytes=_as_gib_bytes(
            section.get("commit_budget_gib", DEFAULT_COMMIT_BUDGET_GIB),
            "memory.commit_budget_gib",
            allow_zero=False,
        ),
        free_commit_guard_bytes=_as_gib_bytes(
            section.get("free_commit_guard_gib", DEFAULT_FREE_COMMIT_GUARD_GIB),
            "memory.free_commit_guard_gib",
            allow_zero=True,
        ),
        critical_free_commit_bytes=_as_gib_bytes(
            section.get("critical_free_commit_gib", DEFAULT_CRITICAL_FREE_COMMIT_GIB),
            "memory.critical_free_commit_gib",
            allow_zero=True,
        ),
    )


def _parse_vram(payload: Mapping[str, object]) -> VramConfig:
    """Parse the ``vram`` section.

    Args:
        payload: Document root.

    Returns:
        The VRAM limits in bytes.

    """
    section = _section(payload, "vram", _VRAM_KEYS)
    return VramConfig(
        budget_bytes=_as_gib_bytes(
            section.get("budget_gib", DEFAULT_VRAM_BUDGET_GIB),
            "vram.budget_gib",
            allow_zero=False,
        ),
        free_guard_bytes=_as_gib_bytes(
            section.get("free_guard_gib", DEFAULT_VRAM_FREE_GUARD_GIB),
            "vram.free_guard_gib",
            allow_zero=True,
        ),
    )


def _positive_seconds(
    section: Mapping[str, object], name: str, parent: str, default: float
) -> float:
    """Parse a strictly positive duration in seconds.

    Args:
        section: Mapping that may contain *name*.
        name: Key of the duration.
        parent: Dotted JSON path of *section*.
        default: Value when the key is absent.

    Returns:
        The duration in seconds.

    """
    return _as_float(
        section.get(name, default), f"{parent}.{name}", exclusive_minimum=True
    )


def _parse_lifecycle(payload: Mapping[str, object]) -> LifecycleConfig:
    """Parse the ``lifecycle`` section.

    Args:
        payload: Document root.

    Returns:
        The lifetime timings.

    """
    section = _section(payload, "lifecycle", _LIFECYCLE_KEYS)
    return LifecycleConfig(
        worker_idle_ttl_s=_positive_seconds(
            section, "worker_idle_ttl_s", "lifecycle", DEFAULT_WORKER_IDLE_TTL_S
        ),
        supervisor_linger_s=_positive_seconds(
            section, "supervisor_linger_s", "lifecycle", DEFAULT_SUPERVISOR_LINGER_S
        ),
        stop_grace_s=_positive_seconds(
            section, "stop_grace_s", "lifecycle", DEFAULT_STOP_GRACE_S
        ),
        cancel_grace_s=_positive_seconds(
            section, "cancel_grace_s", "lifecycle", DEFAULT_CANCEL_GRACE_S
        ),
    )


def _parse_scheduling(payload: Mapping[str, object]) -> SchedulingConfig:
    """Parse the ``scheduling`` section.

    Args:
        payload: Document root.

    Returns:
        The scheduler tuning.

    """
    section = _section(payload, "scheduling", _SCHEDULING_KEYS)
    return SchedulingConfig(
        spill_ratio=_as_float(
            section.get("spill_ratio", DEFAULT_SPILL_RATIO),
            "scheduling.spill_ratio",
            exclusive_minimum=True,
            maximum=1.0,
        ),
        starvation_after_s=_positive_seconds(
            section, "starvation_after_s", "scheduling", DEFAULT_STARVATION_AFTER_S
        ),
        max_attempts=_as_int(
            section.get("max_attempts", DEFAULT_MAX_ATTEMPTS),
            "scheduling.max_attempts",
            minimum=1,
        ),
        gpu_cpu_contention=_as_float(
            section.get("gpu_cpu_contention", DEFAULT_GPU_CPU_CONTENTION),
            "scheduling.gpu_cpu_contention",
        ),
        priority_aging_s=_positive_seconds(
            section, "priority_aging_s", "scheduling", DEFAULT_PRIORITY_AGING_S
        ),
        load_failure_cooldown_s=_positive_seconds(
            section,
            "load_failure_cooldown_s",
            "scheduling",
            DEFAULT_LOAD_FAILURE_COOLDOWN_S,
        ),
        crash_cooldown_s=_positive_seconds(
            section, "crash_cooldown_s", "scheduling", DEFAULT_CRASH_COOLDOWN_S
        ),
    )


def _fit_diarization(
    threads_per_slot: int | None, budget: int, *, budget_origin: str
) -> DiarizationConfig:
    """Return the diarization limits for a thread budget.

    Args:
        threads_per_slot: Explicitly configured threads of one diarization
            job (already validated as an integer >= 1), or ``None`` when the
            config does not set them.
        budget: Effective ``threads.budget``.
        budget_origin: Where the budget comes from, for the error message.

    Returns:
        The explicit value, or the default clamped to *budget*.

    Raises:
        PoolConfigError: When an explicit value exceeds *budget*: no
            diarization job could ever start.

    """
    if threads_per_slot is None:
        return DiarizationConfig(
            threads_per_slot=max(1, min(DEFAULT_DIARIZATION_THREADS_PER_SLOT, budget)),
            threads_per_slot_explicit=False,
        )
    if threads_per_slot > budget:
        raise _fail(
            _DIARIZATION_THREADS_PATH,
            f"must not exceed the thread budget of {budget} ({budget_origin}), "
            f"got {threads_per_slot}: a diarization job needs all its threads "
            "within threads.budget, so no diarization job could ever start; "
            "lower diarization.threads_per_slot or raise threads.budget",
        )
    return DiarizationConfig(
        threads_per_slot=threads_per_slot, threads_per_slot_explicit=True
    )


def _parse_diarization(
    payload: Mapping[str, object], threads: ThreadsConfig
) -> DiarizationConfig:
    """Parse the ``diarization`` section.

    Args:
        payload: Document root.
        threads: Parsed thread budget the diarization threads must fit.

    Returns:
        The diarization limits; an absent ``threads_per_slot`` is the default
        clamped to the thread budget.

    """
    section = _section(payload, "diarization", _DIARIZATION_KEYS)
    explicit = (
        _as_int(section["threads_per_slot"], _DIARIZATION_THREADS_PATH, minimum=1)
        if "threads_per_slot" in section
        else None
    )
    return _fit_diarization(explicit, threads.budget, budget_origin="threads.budget")


def _check_header(payload: Mapping[str, object]) -> None:
    """Validate the optional ``schema`` and ``version`` markers.

    Args:
        payload: Document root.

    Raises:
        PoolConfigError: When a marker is present but not the expected value.

    """
    if "schema" in payload and payload["schema"] != CONFIG_SCHEMA:
        path = "schema"
        raise _fail(path, f"must be {CONFIG_SCHEMA!r}, got {payload[path]!r}")
    if "version" in payload:
        path = "version"
        version = payload[path]
        if isinstance(version, bool) or version != CONFIG_VERSION:
            raise _fail(path, f"must be {CONFIG_VERSION}, got {version!r}")


def _logical_cpu_count(logical_cpus: int | None) -> int:
    """Return the logical CPU count used to size defaults.

    Args:
        logical_cpus: Explicit count, or ``None`` to ask the OS.

    Returns:
        A positive logical CPU count.

    """
    if logical_cpus is not None:
        return max(1, logical_cpus)
    return max(1, os.cpu_count() or 1)


def parse_pool_config(
    payload: Mapping[str, object],
    *,
    source: str,
    logical_cpus: int | None = None,
) -> PoolConfig:
    """Validate a config document and fill missing keys with defaults.

    Args:
        payload: Parsed JSON document.
        source: ``defaults`` or the file path, recorded in the result and in
            error messages.
        logical_cpus: Logical CPU count that sizes the default thread budget;
            defaults to :func:`os.cpu_count`.

    Returns:
        The validated configuration.

    Raises:
        PoolConfigError: For an unknown key or an invalid value, naming its
            JSON path.

    """
    try:
        _reject_unknown(payload, _TOP_LEVEL_KEYS, "")
        _check_header(payload)
        cpu = _parse_cpu(payload)
        gpu = _parse_gpu(payload)
        threads = _parse_threads(payload, _logical_cpu_count(logical_cpus))
        return PoolConfig(
            cpu=cpu,
            gpu=gpu,
            threads=threads,
            memory=_parse_memory(payload),
            vram=_parse_vram(payload),
            lifecycle=_parse_lifecycle(payload),
            scheduling=_parse_scheduling(payload),
            source=source,
            diarization=_parse_diarization(payload, threads),
        )
    except PoolConfigError as exc:
        msg = f"Invalid STT pool config ({source}): {exc}"
        raise PoolConfigError(msg, json_path=exc.json_path) from exc


def default_pool_config(*, logical_cpus: int | None = None) -> PoolConfig:
    """Return the conservative code-default configuration.

    Args:
        logical_cpus: Logical CPU count that sizes the default thread budget;
            defaults to :func:`os.cpu_count`.

    Returns:
        The default configuration with ``source == "defaults"``.

    """
    return parse_pool_config({}, source="defaults", logical_cpus=logical_cpus)


def project_root() -> Path:
    """Return the repository root (the directory that contains ``cli.py``).

    Returns:
        The absolute project root derived from this file's location.

    """
    return Path(__file__).resolve().parents[2]


def resolve_config_path(
    explicit: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    root: Path | None = None,
) -> Path | None:
    """Select the config file to load.

    Args:
        explicit: Path given on the command line; wins over everything else.
        env: Environment mapping; defaults to :data:`os.environ`.
        root: Project root holding the default file; defaults to
            :func:`project_root`.

    Returns:
        The explicit path, the path in ``ASKVLM_STT_POOL_CONFIG`` (a blank value
        is ignored), ``<root>/stt_pool.json`` when that file exists, or ``None``
        when the code defaults apply. Explicit and environment paths are
        returned without an existence check.

    """
    if explicit is not None:
        return explicit
    environment = os.environ if env is None else env
    env_value = environment.get(ENV_POOL_CONFIG, "").strip()
    if env_value:
        return Path(env_value)
    candidate = (root if root is not None else project_root()) / CONFIG_FILENAME
    return candidate if candidate.is_file() else None


def load_pool_config(
    explicit: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    root: Path | None = None,
    logical_cpus: int | None = None,
) -> PoolConfig:
    """Resolve, read and validate the pool configuration.

    Args:
        explicit: Path given on the command line.
        env: Environment mapping; defaults to :data:`os.environ`.
        root: Project root holding the default file.
        logical_cpus: Logical CPU count that sizes the default thread budget.

    Returns:
        The validated configuration; the code defaults when no file applies.

    Raises:
        PoolConfigError: When the selected file is missing, unreadable, not
            valid JSON, not a JSON object, or fails validation.

    """
    path = resolve_config_path(explicit, env=env, root=root)
    if path is None:
        return default_pool_config(logical_cpus=logical_cpus)
    if not path.is_file():
        msg = f"STT pool config file not found: {path}"
        raise PoolConfigError(msg)
    try:
        # * utf-8-sig tolerates the BOM that Windows editors add.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        msg = f"Cannot read STT pool config {path}: {exc}"
        raise PoolConfigError(msg) from exc
    try:
        payload = json.loads(text)
    except (ValueError, RecursionError) as exc:
        # * RecursionError covers documents nested deeper than the parser allows.
        msg = f"STT pool config {path} is not valid JSON: {exc}"
        raise PoolConfigError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"STT pool config {path} must contain a JSON object"
        raise PoolConfigError(msg)
    return parse_pool_config(payload, source=str(path), logical_cpus=logical_cpus)


def apply_overrides(
    config: PoolConfig,
    *,
    cpu_slots: int | None = None,
    gpu_workers: int | None = None,
    thread_budget: int | None = None,
    commit_budget_gib: float | None = None,
    vram_budget_gib: float | None = None,
    worker_idle_ttl_s: float | None = None,
) -> PoolConfig:
    """Return *config* with command-line overrides applied and validated.

    Overrides are checked exactly like the matching file keys. A per-worker slot
    cap that equals the old slot limit follows ``cpu_slots``; an explicitly
    smaller cap is kept. A thread budget override re-fits the diarization
    threads: a default value is clamped to the new budget again, an explicit
    value must fit it.

    Args:
        config: Base configuration.
        cpu_slots: Replacement for ``cpu.max_slots``.
        gpu_workers: Replacement for ``gpu.max_workers``.
        thread_budget: Replacement for ``threads.budget``.
        commit_budget_gib: Replacement for ``memory.commit_budget_gib``.
        vram_budget_gib: Replacement for ``vram.budget_gib``.
        worker_idle_ttl_s: Replacement for ``lifecycle.worker_idle_ttl_s``.

    Returns:
        The updated configuration; unchanged sections are shared with *config*.

    Raises:
        PoolConfigError: When an override value is invalid, or a thread budget
            override is below an explicit ``diarization.threads_per_slot``.

    """
    cpu = config.cpu
    if cpu_slots is not None:
        max_slots = _as_int(cpu_slots, "cpu.max_slots", minimum=0)
        follows_slots = cpu.max_slots_per_worker == max(1, cpu.max_slots)
        cpu = replace(
            cpu,
            max_slots=max_slots,
            max_slots_per_worker=(
                max(1, max_slots) if follows_slots else cpu.max_slots_per_worker
            ),
        )
    gpu = config.gpu
    if gpu_workers is not None:
        gpu = replace(
            gpu, max_workers=_as_int(gpu_workers, "gpu.max_workers", minimum=0)
        )
    threads = config.threads
    diarization = config.diarization
    if thread_budget is not None:
        threads = replace(
            threads, budget=_as_int(thread_budget, "threads.budget", minimum=1)
        )
        diarization = _fit_diarization(
            (
                diarization.threads_per_slot
                if diarization.threads_per_slot_explicit
                else None
            ),
            threads.budget,
            budget_origin="the thread budget override",
        )
    memory = config.memory
    if commit_budget_gib is not None:
        memory = replace(
            memory,
            commit_budget_bytes=_as_gib_bytes(
                commit_budget_gib, "memory.commit_budget_gib", allow_zero=False
            ),
        )
    vram = config.vram
    if vram_budget_gib is not None:
        vram = replace(
            vram,
            budget_bytes=_as_gib_bytes(
                vram_budget_gib, "vram.budget_gib", allow_zero=False
            ),
        )
    lifecycle = config.lifecycle
    if worker_idle_ttl_s is not None:
        lifecycle = replace(
            lifecycle,
            worker_idle_ttl_s=_as_float(
                worker_idle_ttl_s,
                "lifecycle.worker_idle_ttl_s",
                exclusive_minimum=True,
            ),
        )
    return replace(
        config,
        cpu=cpu,
        gpu=gpu,
        threads=threads,
        memory=memory,
        vram=vram,
        lifecycle=lifecycle,
        diarization=diarization,
    )


def config_summary(config: PoolConfig) -> dict[str, object]:
    """Return the JSON-serializable limits published in the heartbeat.

    Args:
        config: Validated configuration.

    Returns:
        A nested mapping of plain numbers, strings and dictionaries.

    """
    return {
        "source": config.source,
        "cpu": {
            "max_slots": config.cpu.max_slots,
            "threads_per_slot": config.cpu.threads_per_slot,
            "max_slots_per_worker": config.cpu.max_slots_per_worker,
        },
        "gpu": {
            "max_workers": config.gpu.max_workers,
            "batch_lanes": dict(config.gpu.batch_lanes),
            "host_threads": dict(config.gpu.host_threads),
            "scale_out_backlog_s": config.gpu.scale_out_backlog_s,
        },
        "threads": {"budget": config.threads.budget},
        "diarization": {"threads_per_slot": config.diarization.threads_per_slot},
        "memory": {
            "commit_budget_bytes": config.memory.commit_budget_bytes,
            "free_commit_guard_bytes": config.memory.free_commit_guard_bytes,
            "critical_free_commit_bytes": config.memory.critical_free_commit_bytes,
        },
        "vram": {
            "budget_bytes": config.vram.budget_bytes,
            "free_guard_bytes": config.vram.free_guard_bytes,
        },
        "lifecycle": {
            "worker_idle_ttl_s": config.lifecycle.worker_idle_ttl_s,
            "supervisor_linger_s": config.lifecycle.supervisor_linger_s,
            "stop_grace_s": config.lifecycle.stop_grace_s,
            "cancel_grace_s": config.lifecycle.cancel_grace_s,
        },
        "scheduling": {
            "spill_ratio": config.scheduling.spill_ratio,
            "starvation_after_s": config.scheduling.starvation_after_s,
            "max_attempts": config.scheduling.max_attempts,
            "gpu_cpu_contention": config.scheduling.gpu_cpu_contention,
            "priority_aging_s": config.scheduling.priority_aging_s,
            "load_failure_cooldown_s": config.scheduling.load_failure_cooldown_s,
            "crash_cooldown_s": config.scheduling.crash_cooldown_s,
        },
    }
