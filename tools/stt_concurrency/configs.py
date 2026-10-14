"""Benchmark configurations: name grammar, default matrix, and resource estimates.

A configuration name is self-describing::

    <engine>-<device>-<mode><width>[x<threads>][-long]

* ``engine``: ``whisper`` (faster-whisper), ``gigaam`` (GigaAM-v3 e2e_ctc), or
  ``gigaam-ctc`` (GigaAM-Multilingual ctc, CPU only).
* ``device``: ``cpu`` or ``cuda``.
* ``mode`` and ``width``:
  ``P<N>`` - N worker processes, each holding its own model;
  ``S<N>`` - one process, one model, N concurrent Python threads;
  ``B<N>`` - one process, one model, N items per forward pass.
* ``threads``: CPU (intra-op) threads per process. When omitted it is derived:
  CPU ``P``/``S`` -> ``cpu_budget // N``, CPU ``B`` -> ``cpu_budget``,
  CUDA -> ``cuda_threads``.
* ``-long``: the workload is the ``--long-files`` list instead of the clips.

Whisper ``B`` batches across files on the clips (one clip = one chunk of at
most 30 s, N clips per forward pass) and batches 30 s chunks of one file with
``-long``; both use faster-whisper ``BatchedInferencePipeline``.

Examples: ``whisper-cpu-P2x8``, ``gigaam-cpu-S4x4``, ``gigaam-cuda-B8``,
``whisper-cuda-B8``, ``whisper-cuda-B8-long``, ``gigaam-ctc-cpu-S4x4``.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Engine = Literal["whisper", "gigaam", "gigaam-ctc"]
Device = Literal["cpu", "cuda"]
Mode = Literal["P", "S", "B"]

ENGINES: tuple[Engine, ...] = ("whisper", "gigaam", "gigaam-ctc")
# * GigaAM-Multilingual ctc runs on CPU only (project provider contract).
CPU_ONLY_ENGINES: frozenset[str] = frozenset({"gigaam-ctc"})
MODE_ORDER: dict[str, int] = {"P": 0, "S": 1, "B": 2}

# * Default matrix shape. CPU widths keep at least two threads per worker.
CPU_WIDTHS: tuple[int, ...] = (1, 2, 4, 8)
GPU_PROCESS_WIDTHS: tuple[int, ...] = (1, 2)
GPU_THREAD_WIDTHS: tuple[int, ...] = (2, 4)
GIGAAM_GPU_BATCH_SIZES: tuple[int, ...] = (4, 8, 16)
WHISPER_LONG_BATCH_SIZES: tuple[int, ...] = (4, 8, 16)
_MIN_DEFAULT_THREADS_PER_WORKER = 2

_NAME_RE = re.compile(
    r"(?P<engine>whisper|gigaam-ctc|gigaam)-(?P<device>cpu|cuda)-(?P<mode>[PSB])"
    r"(?P<width>[1-9]\d*)(?:x(?P<threads>[1-9]\d*))?(?P<long>-long)?",
)
_GLOB_CHARS = frozenset("*?[")
_NAME_HINT = (
    "expected <engine>-<device>-<P|S|B><width>[x<threads>][-long], "
    "e.g. whisper-cpu-P2x8 or gigaam-cuda-B8"
)


class ConfigError(ValueError):
    """Raised for an invalid configuration name, selection, or estimate override."""


@dataclass(frozen=True, slots=True)
class ThreadDefaults:
    """Thread counts used when a configuration name omits ``x<threads>``.

    Attributes:
        cpu_budget: Total CPU threads shared by the workers of a CPU config.
        cuda_threads: CPU threads per process of a CUDA config.

    """

    cpu_budget: int
    cuda_threads: int


@dataclass(frozen=True, slots=True)
class BenchConfig:
    """One benchmark configuration.

    Attributes:
        engine: STT engine id.
        device: Inference device.
        mode: ``P`` (processes), ``S`` (threads), or ``B`` (batching).
        width: Processes for ``P``, threads for ``S``, batch size for ``B``.
        threads: CPU (intra-op) threads per process.
        long: Whether the workload is the long-file list.

    """

    engine: Engine
    device: Device
    mode: Mode
    width: int
    threads: int
    long: bool = False

    def __post_init__(self) -> None:
        """Validate cross-field constraints.

        Raises:
            ConfigError: For non-positive sizes or a CPU-only engine on CUDA.

        """
        if self.width < 1 or self.threads < 1:
            msg = f"width and threads must be >= 1 (got {self.width}, {self.threads})"
            raise ConfigError(msg)
        if self.engine in CPU_ONLY_ENGINES and self.device != "cpu":
            msg = f"{self.engine} runs on CPU only"
            raise ConfigError(msg)

    @property
    def name(self) -> str:
        """Canonical configuration name (always includes the thread count)."""
        suffix = "-long" if self.long else ""
        return (
            f"{self.engine}-{self.device}-{self.mode}{self.width}"
            f"x{self.threads}{suffix}"
        )

    @property
    def processes(self) -> int:
        """Number of worker processes (model instances)."""
        return self.width if self.mode == "P" else 1

    @property
    def slots(self) -> int:
        """Concurrent worker threads per process."""
        return self.width if self.mode == "S" else 1

    @property
    def batch_size(self) -> int:
        """Items per forward pass."""
        return self.width if self.mode == "B" else 1

    @property
    def workload_kind(self) -> str:
        """``long`` for the long-file workload, ``clips`` otherwise."""
        return "long" if self.long else "clips"

    @property
    def group(self) -> str:
        """Correctness comparison group: engine, device, and workload."""
        return f"{self.engine}-{self.device}-{self.workload_kind}"

    @property
    def total_threads(self) -> int:
        """CPU threads requested by all workers together."""
        return self.processes * self.slots * self.threads

    @property
    def sort_key(self) -> tuple[int, int, int, int, int, int]:
        """Stable ordering for reports: engine, device, workload, mode, width."""
        return (
            ENGINES.index(self.engine),
            0 if self.device == "cpu" else 1,
            1 if self.long else 0,
            MODE_ORDER[self.mode],
            self.width,
            self.threads,
        )

    def to_json(self) -> dict[str, object]:
        """Serialize the configuration with its derived shape."""
        return {
            "name": self.name,
            "engine": self.engine,
            "device": self.device,
            "mode": self.mode,
            "width": self.width,
            "threads": self.threads,
            "long": self.long,
            "processes": self.processes,
            "slots": self.slots,
            "batch_size": self.batch_size,
            "total_threads": self.total_threads,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> BenchConfig:
        """Rebuild a configuration serialized by :meth:`to_json`.

        Args:
            data: Mapping with at least the dataclass fields.

        Returns:
            The configuration.

        """
        return cls(
            engine=cast("Engine", str(data["engine"])),
            device=cast("Device", str(data["device"])),
            mode=cast("Mode", str(data["mode"])),
            width=int(data["width"]),
            threads=int(data["threads"]),
            long=bool(data.get("long", False)),
        )


def default_threads(
    device: Device,
    mode: Mode,
    width: int,
    defaults: ThreadDefaults,
) -> int:
    """Return the thread count used when a name omits ``x<threads>``.

    Args:
        device: Inference device.
        mode: Concurrency mode.
        width: Mode width.
        defaults: Thread defaults.

    Returns:
        Threads per process.

    """
    if device == "cuda":
        return defaults.cuda_threads
    if mode == "B":
        return defaults.cpu_budget
    return max(1, defaults.cpu_budget // width)


def parse_config_name(token: str, defaults: ThreadDefaults) -> BenchConfig:
    """Parse a configuration name.

    Args:
        token: Name such as ``whisper-cpu-P2x8`` or ``gigaam-cuda-B8``.
        defaults: Thread defaults for names without ``x<threads>``.

    Returns:
        The parsed configuration.

    Raises:
        ConfigError: When *token* does not follow the grammar.

    """
    match = _NAME_RE.fullmatch(token.strip())
    if match is None:
        msg = f"Invalid configuration name {token!r}; {_NAME_HINT}"
        raise ConfigError(msg)
    device = cast("Device", match["device"])
    mode = cast("Mode", match["mode"])
    width = int(match["width"])
    threads_raw = match["threads"]
    threads = (
        int(threads_raw)
        if threads_raw
        else default_threads(device, mode, width, defaults)
    )
    return BenchConfig(
        engine=cast("Engine", match["engine"]),
        device=device,
        mode=mode,
        width=width,
        threads=threads,
        long=match["long"] is not None,
    )


def default_matrix(defaults: ThreadDefaults, *, has_long: bool) -> list[BenchConfig]:
    """Build the default configuration matrix.

    CPU, per engine: ``P`` for every width of :data:`CPU_WIDTHS` and ``S`` for
    every width above one, threads ``cpu_budget // width``. CUDA, per engine
    except the CPU-only ones: ``P1``, ``P2``, ``S2``, ``S4``; GigaAM adds
    ``B4``/``B8``/``B16`` on the clips. With long files, Whisper CUDA adds the
    sequential ``P1-long`` reference and ``B4``/``B8``/``B16`` batched long-file
    configs. Whisper cross-file batching on clips is covered by ``sweep``.

    Args:
        defaults: Thread defaults.
        has_long: Whether ``--long-files`` were given.

    Returns:
        Configurations in execution order (each group's reference first).

    """
    budget = defaults.cpu_budget
    cpu_widths = [
        width
        for width in CPU_WIDTHS
        if width == 1 or budget // width >= _MIN_DEFAULT_THREADS_PER_WORKER
    ]
    configs: list[BenchConfig] = []
    for engine in ENGINES:
        configs.extend(
            BenchConfig(engine, "cpu", "P", width, max(1, budget // width))
            for width in cpu_widths
        )
        configs.extend(
            BenchConfig(engine, "cpu", "S", width, max(1, budget // width))
            for width in cpu_widths
            if width > 1
        )
    cuda_threads = defaults.cuda_threads
    for engine in ENGINES:
        if engine in CPU_ONLY_ENGINES:
            continue
        configs.extend(
            BenchConfig(engine, "cuda", "P", width, cuda_threads)
            for width in GPU_PROCESS_WIDTHS
        )
        configs.extend(
            BenchConfig(engine, "cuda", "S", width, cuda_threads)
            for width in GPU_THREAD_WIDTHS
        )
        if engine == "gigaam":
            configs.extend(
                BenchConfig(engine, "cuda", "B", size, cuda_threads)
                for size in GIGAAM_GPU_BATCH_SIZES
            )
    if has_long:
        configs.append(BenchConfig("whisper", "cuda", "P", 1, cuda_threads, long=True))
        configs.extend(
            BenchConfig("whisper", "cuda", "B", size, cuda_threads, long=True)
            for size in WHISPER_LONG_BATCH_SIZES
        )
    return configs


def select_configs(
    selection: str,
    defaults: ThreadDefaults,
    *,
    has_long: bool,
) -> list[BenchConfig]:
    """Resolve a comma-separated selection into configurations.

    Each token is either a glob (``*``, ``?``, ``[``) matched against the
    canonical names of the default matrix, or an exact configuration name
    parsed by :func:`parse_config_name` (it may lie outside the matrix).

    Args:
        selection: Comma-separated tokens; empty selects the default matrix.
        defaults: Thread defaults.
        has_long: Whether ``--long-files`` were given.

    Returns:
        Unique configurations in selection order.

    Raises:
        ConfigError: When a glob matches nothing or a name is invalid.

    """
    matrix = default_matrix(defaults, has_long=has_long)
    tokens = [token.strip() for token in selection.split(",") if token.strip()]
    if not tokens:
        return matrix
    selected: dict[str, BenchConfig] = {}
    for token in tokens:
        if _GLOB_CHARS.intersection(token):
            matches = [cfg for cfg in matrix if fnmatch.fnmatchcase(cfg.name, token)]
            if not matches:
                msg = f"Selection {token!r} matches no default configuration"
                raise ConfigError(msg)
            for cfg in matches:
                selected.setdefault(cfg.name, cfg)
            continue
        cfg = parse_config_name(token, defaults)
        selected.setdefault(cfg.name, cfg)
    return list(selected.values())


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Per-instance resource estimate used by the safety guard.

    Attributes:
        commit_gb: Commit charge (private bytes) per model instance.
        vram_gb: VRAM per model instance (CUDA only).
        extra_thread_gb: Added per extra ``S`` thread (commit; also VRAM on CUDA).
        batch_item_gb: Added per extra ``B`` batch item (commit; also VRAM on CUDA).

    """

    commit_gb: float
    vram_gb: float
    extra_thread_gb: float
    batch_item_gb: float


# * Defaults measured on the reference PC (RTX 3070 8 GB, 64 GB RAM).
DEFAULT_ESTIMATES: dict[str, ResourceEstimate] = {
    "whisper-cpu": ResourceEstimate(1.5, 0.0, 0.2, 0.1),
    "whisper-cuda": ResourceEstimate(3.0, 0.9, 0.2, 0.1),
    "gigaam-cpu": ResourceEstimate(6.0, 0.0, 0.5, 0.05),
    "gigaam-cuda": ResourceEstimate(8.0, 1.7, 0.5, 0.05),
    "gigaam-ctc-cpu": ResourceEstimate(6.0, 0.0, 0.5, 0.05),
}
ESTIMATE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(ResourceEstimate))


def apply_estimate_overrides(
    base: Mapping[str, ResourceEstimate],
    overrides: Sequence[str],
) -> dict[str, ResourceEstimate]:
    """Apply ``<engine>-<device>.<field>=<value>`` overrides.

    Args:
        base: Estimates to start from.
        overrides: Tokens such as ``gigaam-cuda.vram_gb=1.7``.

    Returns:
        A new estimate mapping.

    Raises:
        ConfigError: For an unknown key, field, or a non-numeric value.

    """
    result = dict(base)
    for token in overrides:
        key, sep, raw_value = token.partition("=")
        target, dot, field_name = key.strip().partition(".")
        if (
            not sep
            or not dot
            or target not in result
            or field_name not in ESTIMATE_FIELDS
        ):
            msg = (
                f"Invalid estimate override {token!r}; expected "
                f"<{'|'.join(sorted(result))}>.<{'|'.join(ESTIMATE_FIELDS)}>=<number>"
            )
            raise ConfigError(msg)
        try:
            value = float(raw_value)
        except ValueError as exc:
            msg = f"Estimate override {token!r} has a non-numeric value"
            raise ConfigError(msg) from exc
        if value < 0:
            msg = f"Estimate override {token!r} must not be negative"
            raise ConfigError(msg)
        result[target] = replace(result[target], **{field_name: value})
    return result


@dataclass(frozen=True, slots=True)
class ResourceNeed:
    """Estimated commit charge and VRAM of one configuration.

    Attributes:
        commit_gb: Estimated commit charge.
        vram_gb: Estimated VRAM (zero for CPU configs).

    """

    commit_gb: float
    vram_gb: float


def estimate_need(
    config: BenchConfig,
    estimates: Mapping[str, ResourceEstimate],
) -> ResourceNeed:
    """Estimate the commit charge and VRAM a configuration needs.

    Args:
        config: Configuration.
        estimates: Per ``<engine>-<device>`` estimates.

    Returns:
        The estimated need.

    """
    est = estimates[f"{config.engine}-{config.device}"]
    extra_gb = (config.slots - 1) * est.extra_thread_gb + (
        config.batch_size - 1
    ) * est.batch_item_gb
    commit_gb = config.processes * est.commit_gb + extra_gb
    vram_gb = (
        config.processes * est.vram_gb + extra_gb if config.device == "cuda" else 0.0
    )
    return ResourceNeed(commit_gb=commit_gb, vram_gb=vram_gb)
