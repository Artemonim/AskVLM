"""Seed and learned estimates of STT worker configurations (spec section 7.7).

The seed table comes from the committed concurrency benchmark
(``doc/STT_CONCURRENCY_BENCHMARK.en.md``). :class:`EstimateStore` refines it per
worker key with what the supervisor observes:

* time: ``service_correction`` is an exponential moving average (alpha 0.2) of
  ``run_s / base_service_s``, each sample clamped to ``[0.25, 4.0]``;
* memory: only ever grows. A worker exit with a ``PeakJobMemoryUsed`` reading
  raises the commit floor of the key; a Job Object memory-limit hit (or a CUDA
  OOM at load, through :meth:`EstimateStore.bump_memory`) multiplies the next
  reservation by 1.25. Every growth chain is capped at four times the seed.

Learned data persists as ``{"schema", "version", "fingerprint", "keys"}``
through atomic writes of :mod:`core.stt_pool.fsio`; a file written for another
hardware fingerprint or another version is discarded.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from core.stt_pool import fsio
from core.stt_pool.keys import WorkerKey
from core.stt_pool.scheduler import KeyEstimate

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)

ESTIMATES_SCHEMA = "askvlm.stt.estimates"
ESTIMATES_VERSION = 1

# * Learning and persistence constants (spec 7.7).
DEFAULT_SAVE_INTERVAL_S = 30.0
CORRECTION_ALPHA = 0.2
CORRECTION_MIN = 0.25
CORRECTION_MAX = 4.0
MEMORY_BUMP_FACTOR = 1.25
# * Largest growth of a memory estimate over its seed (bump chain cap).
MAX_MEMORY_FACTOR = 4.0

_GIB = 1024**3
_WHISPER = "whisper"
_GIGAAM_CTC = "gigaam-ctc"
_GIGAAM_E2E_CTC = "gigaam-e2e-ctc"


@dataclass(frozen=True, slots=True)
class _Seed:
    """One row of the spec 7.7 seed table (sizes in GiB)."""

    load_s: float
    a_s: float
    xrt_single: float
    xrt_aggregate: float
    commit_base_gib: float
    commit_per_slot_gib: float
    vram_base_gib: float = 0.0
    vram_per_lane_gib: float = 0.0


# * Seed table of spec 7.7; Whisper rows describe ``small`` (int8 CPU, fp16 GPU).
_SEEDS: dict[tuple[str, str], _Seed] = {
    (_GIGAAM_E2E_CTC, "cpu"): _Seed(10.0, 0.13, 16.7, 59.9, 4.5, 0.15),
    (_GIGAAM_CTC, "cpu"): _Seed(10.0, 0.13, 16.7, 60.3, 4.0, 0.15),
    (_WHISPER, "cpu"): _Seed(2.0, 0.9, 4.5, 11.0, 1.8, 0.25),
    (_GIGAAM_E2E_CTC, "gpu"): _Seed(8.0, 0.23, 68.0, 137.5, 6.6, 0.05, 1.8, 0.05),
    (_WHISPER, "gpu"): _Seed(2.0, 0.07, 20.0, 35.4, 3.2, 0.1, 1.2, 0.2),
}


@dataclass(frozen=True, slots=True)
class _SizeClass:
    """Scaling of a Whisper model relative to ``small``."""

    memory: float
    speed: float


_TINY_CLASS = _SizeClass(memory=0.5, speed=3.0)
_BASE_CLASS = _SizeClass(memory=0.6, speed=2.0)
_SMALL_CLASS = _SizeClass(memory=1.0, speed=1.0)
_MEDIUM_CLASS = _SizeClass(memory=2.0, speed=0.45)
_LARGE_CLASS = _SizeClass(memory=3.0, speed=0.2)

# * Exact Whisper model names (lowercase) and their size classes.
_WHISPER_SIZE_NAMES: dict[str, _SizeClass] = {
    "tiny": _TINY_CLASS,
    "tiny.en": _TINY_CLASS,
    "base": _BASE_CLASS,
    "base.en": _BASE_CLASS,
    "small": _SMALL_CLASS,
    "small.en": _SMALL_CLASS,
    "medium": _MEDIUM_CLASS,
    "medium.en": _MEDIUM_CLASS,
    "large": _LARGE_CLASS,
    "large-v1": _LARGE_CLASS,
    "large-v2": _LARGE_CLASS,
    "large-v3": _LARGE_CLASS,
    "large-v3-turbo": _LARGE_CLASS,
    "turbo": _LARGE_CLASS,
}

# * Name prefixes followed by an exact size name: distilled models
# * (``distil-medium.en``) and the faster-whisper CTranslate2 repositories
# * (``Systran/faster-whisper-small``, ``Systran/faster-distil-whisper-large-v3``).
_WHISPER_NAME_PREFIXES: tuple[str, ...] = (
    "distil-",
    "faster-whisper-",
    "faster-distil-whisper-",
)

_PATH_SEPARATORS_RE = re.compile(r"[/\\]")

# * Whisper compute types that hold full-precision weights, against the int8
# * (CPU) and float16 (GPU) seeds.
_FULL_PRECISION_COMPUTE = frozenset({"float32", "fp32"})
_FULL_PRECISION_MEMORY_FACTOR = 2.0


def _size_of_name(name: str) -> _SizeClass | None:
    """Return the size class of an exact (lowercase) Whisper model name."""
    size = _WHISPER_SIZE_NAMES.get(name)
    if size is not None:
        return size
    for prefix in _WHISPER_NAME_PREFIXES:
        if name.startswith(prefix):
            size = _WHISPER_SIZE_NAMES.get(name.removeprefix(prefix))
            if size is not None:
                return size
    return None


def whisper_size_class(model: str) -> tuple[float, float]:
    """Return the ``(memory, speed)`` factors of a Whisper model name.

    The lowercased name, and then the last component of a repo id or local
    path, must equal a known size name exactly (``tiny[.en]``, ``base[.en]``,
    ``small[.en]``, ``medium[.en]``, ``large``, ``large-v1/v2/v3``,
    ``large-v3-turbo``, ``turbo``), optionally behind a ``distil-``,
    ``faster-whisper-`` or ``faster-distil-whisper-`` prefix. Anything else
    (custom local paths, unknown repos) gets the conservative large class,
    so a size word inside a directory name never shrinks a reservation.

    Args:
        model: Whisper model identity of a worker key.

    Returns:
        Memory and speed factors relative to ``small``.

    """
    name = model.strip().lower().rstrip("/\\")
    last = _PATH_SEPARATORS_RE.split(name)[-1]
    for candidate in (name, last):
        size = _size_of_name(candidate)
        if size is not None:
            return size.memory, size.speed
    return _LARGE_CLASS.memory, _LARGE_CLASS.speed


def _bytes(gib: float) -> int:
    """Convert GiB to whole bytes, rounding up."""
    return math.ceil(gib * _GIB)


def seed_estimate(key: WorkerKey) -> KeyEstimate:
    """Return the seed estimate of a worker key (spec 7.7).

    GigaAM keys use their table rows. Whisper keys scale the ``small`` rows by
    the model's size class: memory and load time by the memory factor, both
    throughputs by the speed factor and the per-job overhead by its inverse.
    Full-precision Whisper compute types (``float32``) double the memory. An
    unknown provider is treated as a Whisper model of the large class.

    Args:
        key: Normalized worker key.

    Returns:
        The seed estimate with ``service_correction == 1.0``.

    """
    backend = key.backend
    if key.provider in {_GIGAAM_E2E_CTC, _GIGAAM_CTC}:
        seed = _SEEDS.get((key.provider, backend), _SEEDS[(_GIGAAM_E2E_CTC, backend)])
        memory, speed = 1.0, 1.0
    else:
        seed = _SEEDS[(_WHISPER, backend)]
        if key.provider == _WHISPER:
            memory, speed = whisper_size_class(key.model)
        else:
            memory, speed = _LARGE_CLASS.memory, _LARGE_CLASS.speed
        if key.compute_type.lower() in _FULL_PRECISION_COMPUTE:
            memory *= _FULL_PRECISION_MEMORY_FACTOR
    return KeyEstimate(
        load_s=seed.load_s * max(1.0, memory),
        a_s=seed.a_s / speed,
        xrt_single=seed.xrt_single * speed,
        xrt_aggregate=seed.xrt_aggregate * speed,
        commit_base_bytes=_bytes(seed.commit_base_gib * memory),
        commit_per_slot_bytes=_bytes(seed.commit_per_slot_gib * memory),
        vram_base_bytes=_bytes(seed.vram_base_gib * memory),
        vram_per_lane_bytes=_bytes(seed.vram_per_lane_gib * memory),
    )


@dataclass(slots=True)
class _Learned:
    """Learned statistics of one worker key.

    Attributes:
        commit_floor_bytes: Lower bound of ``commit_base`` from observed peaks
            (and memory-limit bumps); never decreases.
        commit_factor: Bump factor of the seed commit; never decreases.
        vram_factor: Bump factor of the seed VRAM; never decreases.
        service_correction: EMA of observed over predicted service time.
        jobs_observed: Jobs that updated the correction.
        exits_observed: Worker exits that were observed.
        memory_limit_hits: Exits caused by the Job Object memory limit.
        updated_at: Clock value of the last update.

    """

    commit_floor_bytes: int = 0
    commit_factor: float = 1.0
    vram_factor: float = 1.0
    service_correction: float = 1.0
    jobs_observed: int = 0
    exits_observed: int = 0
    memory_limit_hits: int = 0
    updated_at: float = 0.0

    def to_dict(self, key: WorkerKey) -> dict[str, object]:
        """Return the persisted form of the entry."""
        return {
            "key": key.to_dict(),
            "commit_floor_bytes": self.commit_floor_bytes,
            "commit_factor": self.commit_factor,
            "vram_factor": self.vram_factor,
            "service_correction": self.service_correction,
            "jobs_observed": self.jobs_observed,
            "exits_observed": self.exits_observed,
            "memory_limit_hits": self.memory_limit_hits,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Parse a persisted entry, clamping numbers into their valid ranges.

        Args:
            payload: Persisted entry without its ``key`` field checked.

        Returns:
            The learned statistics.

        Raises:
            TypeError: When a field has the wrong type.
            ValueError: When a number is not finite or a count is negative.

        """
        return cls(
            commit_floor_bytes=_count(payload, "commit_floor_bytes"),
            commit_factor=_clamp(
                _number(payload, "commit_factor", 1.0), 1.0, MAX_MEMORY_FACTOR
            ),
            vram_factor=_clamp(
                _number(payload, "vram_factor", 1.0), 1.0, MAX_MEMORY_FACTOR
            ),
            service_correction=_clamp(
                _number(payload, "service_correction", 1.0),
                CORRECTION_MIN,
                CORRECTION_MAX,
            ),
            jobs_observed=_count(payload, "jobs_observed"),
            exits_observed=_count(payload, "exits_observed"),
            memory_limit_hits=_count(payload, "memory_limit_hits"),
            updated_at=_number(payload, "updated_at", 0.0),
        )


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* into ``[low, high]``."""
    return min(high, max(low, value))


def _number(payload: Mapping[str, object], name: str, default: float) -> float:
    """Read a finite number field (booleans are rejected).

    Raises:
        TypeError: When the field is present but not a number.
        ValueError: When the field is a number but not finite, including an
            integer beyond the float range.

    """
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a number, got {value!r}"
        raise TypeError(msg)
    try:
        number = float(value)
    except OverflowError:
        msg = f"{name} must be finite, got an integer beyond the float range"
        raise ValueError(msg) from None
    if not math.isfinite(number):
        msg = f"{name} must be finite, got {value!r}"
        raise ValueError(msg)
    return number


def _count(payload: Mapping[str, object], name: str) -> int:
    """Read a non-negative integer field (absent means 0).

    Raises:
        TypeError: When the field is present but not an integer.
        ValueError: When the integer is negative.

    """
    value = payload.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer, got {value!r}"
        raise TypeError(msg)
    if value < 0:
        msg = f"{name} must not be negative, got {value}"
        raise ValueError(msg)
    return value


def _normalize_fingerprint(fingerprint: Mapping[str, object]) -> dict[str, object]:
    """Return *fingerprint* as it reads back from JSON, for stable comparison."""
    try:
        parsed = json.loads(fsio.dumps_json(dict(fingerprint)))
    except (TypeError, ValueError):
        return {str(name): str(value) for name, value in fingerprint.items()}
    return parsed if isinstance(parsed, dict) else {}


def _positive_finite(value: float) -> bool:
    """Return whether *value* is a finite number greater than zero."""
    return math.isfinite(value) and value > 0


class EstimateStore:
    """Seed estimates refined by observations, persisted per hardware fingerprint.

    Implements the scheduler's ``EstimateSource`` protocol. All methods are
    thread-safe. The clock is injectable; it stamps updates and paces saves.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        fingerprint: Mapping[str, object],
        clock: Callable[[], float] = time.time,
        save_interval_s: float = DEFAULT_SAVE_INTERVAL_S,
    ) -> None:
        """Create an empty store.

        Args:
            path: Persistence file, or ``None`` for an in-memory store.
            fingerprint: Hardware fingerprint (``sysprobe.hardware_fingerprint``).
            clock: Wall clock used for ``updated_at`` and save pacing.
            save_interval_s: Minimum time between two unforced saves.

        """
        self._path = path
        self._fingerprint = _normalize_fingerprint(fingerprint)
        self._clock = clock
        self._save_interval_s = max(0.0, save_interval_s)
        self._learned: dict[WorkerKey, _Learned] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._last_attempt: float | None = None

    @classmethod
    def load(
        cls,
        path: Path | None,
        *,
        fingerprint: Mapping[str, object],
        clock: Callable[[], float] = time.time,
        save_interval_s: float = DEFAULT_SAVE_INTERVAL_S,
    ) -> Self:
        """Load a store from *path*, or start fresh.

        A missing file, a file of another schema, version or fingerprint, and a
        corrupt file (logged as a warning) all yield an empty store. Invalid
        entries are skipped with a warning; the others are kept.

        Args:
            path: Persistence file, or ``None`` for an in-memory store.
            fingerprint: Current hardware fingerprint.
            clock: Wall clock used for ``updated_at`` and save pacing.
            save_interval_s: Minimum time between two unforced saves.

        Returns:
            The store.

        """
        store = cls(
            path, fingerprint=fingerprint, clock=clock, save_interval_s=save_interval_s
        )
        if path is None or not path.exists():
            return store
        payload = fsio.read_json(path)
        if payload is None:
            logger.warning("estimates file is corrupt or unreadable; ignored: %s", path)
            return store
        store._restore(payload)
        return store

    def _restore(self, payload: Mapping[str, object]) -> None:
        """Adopt the entries of a persisted document when it matches this host."""
        version = payload.get("version")
        if (
            payload.get("schema") != ESTIMATES_SCHEMA
            or isinstance(version, bool)
            or version != ESTIMATES_VERSION
        ):
            logger.info("estimates file has another schema or version; discarded")
            return
        stored = payload.get("fingerprint")
        if not isinstance(stored, dict) or (
            _normalize_fingerprint(stored) != self._fingerprint
        ):
            logger.info("estimates file belongs to other hardware; discarded")
            return
        entries = payload.get("keys")
        if not isinstance(entries, dict):
            logger.warning("estimates file has no valid 'keys' object; ignored")
            return
        with self._lock:
            for label, entry in entries.items():
                parsed = _parse_entry(label, entry)
                if parsed is None:
                    logger.warning("ignoring invalid estimates entry %r", label)
                    continue
                key, learned = parsed
                self._learned[key] = learned

    # * EstimateSource.

    def estimate(self, key: WorkerKey) -> KeyEstimate:
        """Return the seed of *key* refined by learned statistics.

        ``commit_base = min(4 * seed, max(seed * commit_factor, commit_floor))``,
        ``commit_per_slot = seed * commit_factor``, VRAM fields ``seed *
        vram_factor``; timings come from the seed and the correction is learned.

        Args:
            key: Normalized worker key.

        Returns:
            The current estimate.

        """
        with self._lock:
            return self._estimate_locked(key)

    def _estimate_locked(self, key: WorkerKey) -> KeyEstimate:
        """Return the estimate of *key*; the caller holds the lock."""
        seed = seed_estimate(key)
        learned = self._learned.get(key)
        if learned is None:
            return seed
        base_cap = seed.commit_base_bytes * MAX_MEMORY_FACTOR
        commit_base = min(
            base_cap,
            max(
                seed.commit_base_bytes * learned.commit_factor,
                learned.commit_floor_bytes,
            ),
        )
        return KeyEstimate(
            load_s=seed.load_s,
            a_s=seed.a_s,
            xrt_single=seed.xrt_single,
            xrt_aggregate=seed.xrt_aggregate,
            commit_base_bytes=math.ceil(commit_base),
            commit_per_slot_bytes=math.ceil(
                seed.commit_per_slot_bytes * learned.commit_factor
            ),
            vram_base_bytes=math.ceil(seed.vram_base_bytes * learned.vram_factor),
            vram_per_lane_bytes=math.ceil(
                seed.vram_per_lane_bytes * learned.vram_factor
            ),
            service_correction=learned.service_correction,
        )

    # * Observations.

    def observe_job(
        self, key: WorkerKey, *, run_s: float, base_service_s: float
    ) -> None:
        """Update the service correction of *key* with one finished job.

        Args:
            key: Worker key that ran the job.
            run_s: Measured run time.
            base_service_s: Predicted run time without the correction (for the
                measured audio length).

        """
        if not (_positive_finite(run_s) and _positive_finite(base_service_s)):
            return
        sample = _clamp(run_s / base_service_s, CORRECTION_MIN, CORRECTION_MAX)
        with self._lock:
            learned = self._learned.setdefault(key, _Learned())
            corrected = (
                1.0 - CORRECTION_ALPHA
            ) * learned.service_correction + CORRECTION_ALPHA * sample
            learned.service_correction = _clamp(
                corrected, CORRECTION_MIN, CORRECTION_MAX
            )
            learned.jobs_observed += 1
            self._touch_locked(learned)

    def observe_worker_exit(
        self,
        key: WorkerKey,
        *,
        capacity: int,
        peak_commit_bytes: int | None,
        memory_limit_hit: bool,
    ) -> None:
        """Learn from a worker exit.

        A known peak raises the commit floor to ``peak - capacity *
        commit_per_slot`` (current per-slot estimate). A memory-limit hit then
        multiplies the next reservation of *key* by 1.25.

        Args:
            key: Worker key of the exited worker.
            capacity: Slots or lanes of the worker.
            peak_commit_bytes: ``PeakJobMemoryUsed`` of its Job Object, if known.
            memory_limit_hit: Whether the Job Object memory limit was reached.

        """
        with self._lock:
            learned = self._learned.setdefault(key, _Learned())
            if peak_commit_bytes is not None and peak_commit_bytes > 0:
                current = self._estimate_locked(key)
                floor = (
                    peak_commit_bytes - max(0, capacity) * current.commit_per_slot_bytes
                )
                cap = math.floor(
                    seed_estimate(key).commit_base_bytes * MAX_MEMORY_FACTOR
                )
                learned.commit_floor_bytes = min(
                    cap, max(learned.commit_floor_bytes, floor)
                )
            if memory_limit_hit:
                self._bump_locked(
                    key, learned, commit=True, vram=False, factor=MEMORY_BUMP_FACTOR
                )
                learned.memory_limit_hits += 1
            learned.exits_observed += 1
            self._touch_locked(learned)

    def bump_memory(
        self,
        key: WorkerKey,
        *,
        commit: bool = True,
        vram: bool = False,
        factor: float = MEMORY_BUMP_FACTOR,
    ) -> None:
        """Multiply the memory estimates of *key* (for example after a CUDA OOM).

        Args:
            key: Worker key to bump.
            commit: Whether to bump the commit reservation.
            vram: Whether to bump the VRAM reservation.
            factor: Multiplier; values that are not finite or not above 1 are
                ignored, so memory estimates never shrink.

        """
        if not math.isfinite(factor) or factor <= 1.0:
            return
        with self._lock:
            learned = self._learned.setdefault(key, _Learned())
            self._bump_locked(key, learned, commit=commit, vram=vram, factor=factor)
            self._touch_locked(learned)

    @staticmethod
    def _bump_locked(
        key: WorkerKey,
        learned: _Learned,
        *,
        commit: bool,
        vram: bool,
        factor: float,
    ) -> None:
        """Apply a memory bump; the caller holds the lock.

        The commit bump scales both the seed factor and the learned floor, so
        the whole reservation grows by *factor* whichever of them dominates.
        """
        if commit:
            learned.commit_factor = min(
                MAX_MEMORY_FACTOR, learned.commit_factor * factor
            )
            cap = math.floor(seed_estimate(key).commit_base_bytes * MAX_MEMORY_FACTOR)
            learned.commit_floor_bytes = min(
                cap, math.ceil(learned.commit_floor_bytes * factor)
            )
        if vram:
            learned.vram_factor = min(MAX_MEMORY_FACTOR, learned.vram_factor * factor)

    def _touch_locked(self, learned: _Learned) -> None:
        """Stamp an update and mark the store dirty; the caller holds the lock."""
        learned.updated_at = self._clock()
        self._dirty = True

    # * Persistence.

    def save_if_due(self, *, force: bool = False) -> bool:
        """Write the store atomically when it changed and the interval elapsed.

        The interval runs from the last write attempt, successful or not, so a
        failing disk (for example a read-only one) is retried, and logged, at
        most once per interval.

        Args:
            force: Write regardless of the interval (shutdown); an unchanged
                store is still not written.

        Returns:
            ``True`` when the file was written. A failed write is logged, keeps
            the changes pending and returns ``False``.

        """
        path = self._path
        if path is None:
            return False
        with self._lock:
            if not self._dirty:
                return False
            now = self._clock()
            last = self._last_attempt
            if not force and last is not None and now - last < self._save_interval_s:
                return False
            self._last_attempt = now
            payload = self._to_dict_locked()
            try:
                fsio.atomic_write_json(path, payload)
            except OSError as exc:
                logger.warning("cannot save estimates to %s: %s", path, exc)
                return False
            self._dirty = False
            return True

    def to_dict(self) -> dict[str, object]:
        """Return the persisted document of the store."""
        with self._lock:
            return self._to_dict_locked()

    def _to_dict_locked(self) -> dict[str, object]:
        """Return the persisted document; the caller holds the lock."""
        keys = sorted(self._learned, key=lambda key: key.label)
        return {
            "schema": ESTIMATES_SCHEMA,
            "version": ESTIMATES_VERSION,
            "fingerprint": dict(self._fingerprint),
            "keys": {key.label: self._learned[key].to_dict(key) for key in keys},
        }


def _parse_entry(label: object, entry: object) -> tuple[WorkerKey, _Learned] | None:
    """Parse one persisted entry; ``None`` when it is invalid.

    The commit floor is clamped to the growth cap of the key (four times the
    seed commit base), the bound every learning path keeps, so a huge
    persisted integer never reaches float arithmetic.

    Args:
        label: Mapping key of the entry (must equal the key's label).
        entry: Persisted entry.

    Returns:
        The worker key and its statistics, or ``None``.

    """
    if not isinstance(label, str) or not isinstance(entry, dict):
        return None
    raw_key = entry.get("key")
    if not isinstance(raw_key, dict):
        return None
    try:
        key = WorkerKey.from_dict(raw_key)
        learned = _Learned.from_dict(entry)
    except (TypeError, ValueError):
        return None
    if key.label != label:
        return None
    cap = math.floor(seed_estimate(key).commit_base_bytes * MAX_MEMORY_FACTOR)
    learned.commit_floor_bytes = min(cap, learned.commit_floor_bytes)
    return key, learned
