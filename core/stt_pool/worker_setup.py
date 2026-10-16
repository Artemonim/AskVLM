"""What a new pool worker gets: its ``spec`` engine options and extra environment.

The supervisor builds every worker ``spec`` and launch environment with these
pure functions, and measurement tools (``tools/stt_concurrency`` ``memory-growth``)
start workers with exactly the same values. The module imports no ML packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.stt_pool.config import CUDNN_V8_LRU_CACHE_ENV
from core.stt_pool.keys import provider_family, worker_kind
from core.stt_pool.scheduler import threads_for

if TYPE_CHECKING:
    from core.stt_pool.config import PoolConfig
    from core.stt_pool.keys import WorkerKey
    from core.stt_pool.scheduler import Limits, Reservation

# * Worker spec option (spec 6.3).
WHISPER_BATCH_MAX_CLIP_S = 30.0
# * BLAS/OpenMP thread caps set in every worker's environment.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def worker_spec_options(
    config: PoolConfig,
    limits: Limits,
    key: WorkerKey,
    capacity: int,
    reservation: Reservation,
) -> dict[str, object]:
    """Return the engine options of a worker ``spec`` (spec 6.3).

    A GPU worker batches as many lanes as the capacity it was granted: its
    VRAM and commit reservation is sized for that capacity, which can be
    below the configured lanes when the budgets are tight. CPU engines run
    ``capacity`` slots and take no lane count. A speaker diarization worker
    gets no speech-to-text options: only its compute type, the threads of its
    slot (``diarization.threads_per_slot``, the same value as the spec's
    ``threads_per_slot``) and the priority aging.

    ``vram_cap_bytes`` (GPU only, ``None`` when ``gpu.vram_cap`` is off or
    nothing is reserved) is the VRAM reservation plus
    ``gpu.vram_cap_margin_gib``; a torch engine caps its CUDA caching
    allocator there before the model loads.

    Args:
        config: Validated pool configuration.
        limits: Pool limits derived from *config*.
        key: Worker configuration.
        capacity: Slots or lanes granted to the worker.
        reservation: Resources reserved for the worker.

    Returns:
        The JSON-serializable ``options`` of the ``spec`` message.

    """
    if worker_kind(key.provider) == "diarize":
        return {
            "compute_type": key.compute_type,
            "threads_per_slot": threads_for(key, limits),
            "priority_aging_s": config.scheduling.priority_aging_s,
        }
    family = provider_family(key.provider)
    gpu = key.backend == "gpu"
    vram_cap: int | None = None
    if gpu and config.gpu.vram_cap and reservation.vram_bytes > 0:
        vram_cap = reservation.vram_bytes + config.gpu.vram_cap_margin_bytes
    return {
        "whisper_model": key.model if family == "whisper" else None,
        "compute_type": key.compute_type,
        "batch_lanes": capacity if gpu else None,
        "whisper_batch_max_clip_s": WHISPER_BATCH_MAX_CLIP_S,
        "priority_aging_s": config.scheduling.priority_aging_s,
        "vram_cap_bytes": vram_cap,
        "cufft_plan_cache_max_size": (
            config.gpu.cufft_plan_cache_max_size if gpu else None
        ),
        "idle_cache_release_s": config.gpu.idle_cache_release_s if gpu else None,
    }


def worker_thread_env(
    limits: Limits, key: WorkerKey, reservation: Reservation
) -> dict[str, str]:
    """Return the BLAS/OpenMP thread caps of a worker process.

    GPU workers get their reserved host threads; CPU workers the threads of
    one slot of their key (:func:`~core.stt_pool.scheduler.threads_for`).

    Args:
        limits: Pool limits.
        key: Worker configuration.
        reservation: Resources reserved for the worker.

    Returns:
        Every variable of :data:`THREAD_ENV_VARS` set to the thread count.

    """
    if key.backend == "gpu":
        threads = max(1, reservation.host_threads)
    else:
        threads = max(1, threads_for(key, limits))
    return dict.fromkeys(THREAD_ENV_VARS, str(threads))


def worker_environment(
    config: PoolConfig, limits: Limits, key: WorkerKey, reservation: Reservation
) -> dict[str, str]:
    """Return the environment a worker process gets on top of the supervisor's.

    The thread caps of :func:`worker_thread_env`, plus, for a GPU worker, the
    cuDNN v8 execution-plan cache bound (``gpu.cudnn_v8_lru_cache_limit``;
    ``0`` leaves the variable as the supervisor inherited it).

    Args:
        config: Validated pool configuration.
        limits: Pool limits derived from *config*.
        key: Worker configuration.
        reservation: Resources reserved for the worker.

    Returns:
        The variables added to the inherited environment.

    """
    env = worker_thread_env(limits, key, reservation)
    limit = config.gpu.cudnn_v8_lru_cache_limit
    if key.backend == "gpu" and limit > 0:
        env[CUDNN_V8_LRU_CACHE_ENV] = str(limit)
    return env
