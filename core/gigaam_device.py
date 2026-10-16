"""CUDA placement policy for the CUDA-capable GigaAM head (``gigaam-e2e-ctc``).

``torch`` is never imported here: callers pass the module they already loaded,
so this policy stays importable without the ML extra and can be exercised with
a stand-in ``torch`` object.

Policy:

* ``cpu`` never touches CUDA.
* ``cuda`` uses CUDA whenever it is available (CPU with a warning otherwise);
  an out-of-memory error while moving or running the model falls back to CPU.
* ``auto`` uses CUDA only when it is available and the free VRAM reported by
  ``torch.cuda.mem_get_info`` is at least
  :data:`GIGAAM_AUTO_CUDA_MIN_FREE_VRAM_BYTES`; otherwise CPU with a warning.

Strict policy (``strict=True``, used by the STT worker pool):

* only an explicit ``cpu`` or ``cuda`` request is accepted;
* ``cuda`` without a usable CUDA runtime raises
  :class:`~core.stt_errors.DeviceUnavailableError` instead of degrading to CPU;
* no free-VRAM threshold is applied.

The module also holds the torch-free helpers of the pool's CUDA memory
hygiene: the cuFFT plan cache bound (:func:`limit_cufft_plan_cache`), the
release of cached CUDA blocks (:func:`release_cuda_cache`) and the per-job
memory telemetry (:func:`cuda_memory_stats`), none of which initializes CUDA,
and the caching-allocator cap of a GPU worker (:func:`limit_cuda_allocator`),
which does, because it runs before the model loads on CUDA.
"""

from __future__ import annotations

import contextlib
import gc
import logging
from typing import TYPE_CHECKING, Any

from core.stt_errors import DeviceUnavailableError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# * Device tokens accepted by the strict policy (no ``auto``).
STRICT_DEVICE_REQUESTS: frozenset[str] = frozenset({"cpu", "cuda"})

_GIB = 1024**3

# * Free VRAM required for ``--device auto`` to pick CUDA. A GigaAM-v3 process
# * peaks at about 1.5-1.65 GiB of VRAM on an RTX 3070 with torch 2.10/cu128
# * (nvidia-smi delta: CUDA context, fp32 weights ~0.85 GiB, fp16-autocast
# * activations of one 20 s window, cuBLAS/cuFFT workspaces). ``mem_get_info``
# * already runs inside this process's context, so the threshold covers the
# * model and activations with roughly 1 GiB of headroom left for another GPU
# * tenant such as a resident Whisper daemon.
GIGAAM_AUTO_CUDA_MIN_FREE_VRAM_BYTES = int(2.5 * _GIB)

# * Lower-cased message fragments of CUDA allocation failures. CPU RAM
# * exhaustion ("DefaultCPUAllocator: not enough memory") is deliberately absent:
# * moving the model to CPU cannot cure it.
_CUDA_OOM_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cufft_alloc_failed",
)
_MAX_EXCEPTION_CHAIN = 8


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Return whether *exc* (or an exception it wraps) is a CUDA allocation failure.

    Args:
        exc: Exception raised while moving or running the model.

    Returns:
        ``True`` for ``torch.OutOfMemoryError`` and CUDA/cuBLAS/cuDNN/cuFFT
        allocation failures reported as ``RuntimeError``.

    """
    current: BaseException | None = exc
    for _ in range(_MAX_EXCEPTION_CHAIN):
        if current is None:
            return False
        if "outofmemory" in type(current).__name__.lower():
            return True
        message = str(current).lower()
        if any(marker in message for marker in _CUDA_OOM_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _cuda_is_available(torch_mod: object) -> bool:
    """Return whether *torch_mod* reports a usable CUDA runtime."""
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken driver probe means "no CUDA"
        logger.warning("CUDA availability probe failed", exc_info=True)
        return False


def _free_vram_bytes(torch_mod: object) -> int | None:
    """Return free VRAM of the current CUDA device, or ``None`` when unknown.

    ``cudaMemGetInfo`` runs inside this process's CUDA context, so the context
    itself is already subtracted from the reported free memory.
    """
    # ! The query creates the CUDA context; when ``auto`` then stays on CPU,
    # ! that context (a few hundred MiB) remains allocated until process exit.
    cuda: Any = getattr(torch_mod, "cuda", None)
    try:
        free_bytes, _total_bytes = cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - an unknown budget falls back to CPU
        logger.warning("CUDA free-memory query failed", exc_info=True)
        return None
    return int(free_bytes)


def _select_strict_device(requested: str, torch_mod: object) -> str:
    """Resolve an explicit ``cpu``/``cuda`` request without any fallback.

    Args:
        requested: Canonical request; must be ``cpu`` or ``cuda``.
        torch_mod: Loaded ``torch`` module (unused for ``cpu``).

    Returns:
        *requested* unchanged.

    Raises:
        ValueError: When *requested* is not ``cpu`` or ``cuda``.
        DeviceUnavailableError: When ``cuda`` is requested and CUDA is not usable.

    """
    if requested not in STRICT_DEVICE_REQUESTS:
        msg = (
            "Strict GigaAM device mode requires an explicit device 'cpu' or "
            f"'cuda' (got {requested!r})"
        )
        raise ValueError(msg)
    if requested == "cuda" and not _cuda_is_available(torch_mod):
        msg = (
            "GigaAM: CUDA is unavailable (strict device=cuda); the model is not "
            "loaded on CPU in strict mode"
        )
        raise DeviceUnavailableError(msg)
    return requested


def select_gigaam_device(
    requested: str,
    torch_mod: object,
    *,
    min_free_bytes: int = GIGAAM_AUTO_CUDA_MIN_FREE_VRAM_BYTES,
    strict: bool = False,
) -> str:
    """Resolve a GigaAM device request against the live CUDA state.

    Args:
        requested: Canonical request: ``auto``, ``cpu``, or ``cuda``.
        torch_mod: Loaded ``torch`` module (unused for ``cpu``).
        min_free_bytes: Free VRAM required for ``auto`` to choose CUDA
            (ignored in strict mode).
        strict: Apply the strict policy: only ``cpu``/``cuda`` are accepted
            and an unusable CUDA runtime raises instead of selecting CPU.

    Returns:
        ``"cuda"`` or ``"cpu"``.

    Raises:
        ValueError: In strict mode, when *requested* is not ``cpu``/``cuda``.
        DeviceUnavailableError: In strict mode, when ``cuda`` is requested and
            CUDA is not usable.

    """
    if strict:
        return _select_strict_device(requested, torch_mod)
    return _select_lenient_device(requested, torch_mod, min_free_bytes=min_free_bytes)


def _select_lenient_device(
    requested: str, torch_mod: object, *, min_free_bytes: int
) -> str:
    """Resolve a request with the default policy (CPU fallback with a warning).

    Args:
        requested: Canonical request: ``auto``, ``cpu``, or ``cuda``.
        torch_mod: Loaded ``torch`` module (unused for ``cpu``).
        min_free_bytes: Free VRAM required for ``auto`` to choose CUDA.

    Returns:
        ``"cuda"`` or ``"cpu"``.

    """
    if requested == "cpu":
        return "cpu"
    if not _cuda_is_available(torch_mod):
        logger.warning(
            "GigaAM: CUDA is unavailable (requested device=%s); loading on CPU.",
            requested,
        )
        return "cpu"
    if requested == "cuda":
        return "cuda"
    free_bytes = _free_vram_bytes(torch_mod)
    if free_bytes is None:
        logger.warning(
            "GigaAM: free VRAM is unknown (requested device=auto); loading on CPU."
        )
        return "cpu"
    if free_bytes < min_free_bytes:
        logger.warning(
            "GigaAM: free VRAM %.2f GiB is below the %.2f GiB auto threshold; "
            "loading on CPU.",
            free_bytes / _GIB,
            min_free_bytes / _GIB,
        )
        return "cpu"
    return "cuda"


def release_cuda_cache(torch_mod: object) -> None:
    """Collect garbage and return cached CUDA blocks to the driver (best effort).

    Args:
        torch_mod: Loaded ``torch`` module, or ``None`` when nothing was loaded.

    """
    gc.collect()
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None:
        return
    with contextlib.suppress(Exception):
        cuda.empty_cache()


def limit_cufft_plan_cache(torch_mod: object, max_size: int) -> int | None:
    """Bound the cuFFT plan cache of the current CUDA device (best effort).

    ``torch.stft`` keeps one cuFFT plan per distinct input shape in a per-device
    LRU cache (4096 plans by default), so variable-length audio grows it with
    every new length. A smaller bound evicts the least recently used plans.

    Args:
        torch_mod: Loaded ``torch`` module whose CUDA context is initialized.
        max_size: Most cached plans; must be positive (``0`` would disable the
            cache, which callers express by not calling this function).

    Returns:
        The bound now in effect, or ``None`` when it could not be applied (the
        failure is logged; the library default stays in effect).

    Raises:
        ValueError: When *max_size* is not a positive integer.

    """
    if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
        msg = f"max_size must be a positive integer (got {max_size!r})"
        raise ValueError(msg)
    try:
        module: Any = torch_mod
        index = module.cuda.current_device()
        cache = module.backends.cuda.cufft_plan_cache[index]
        cache.max_size = max_size
        return int(cache.max_size)
    except Exception:  # noqa: BLE001 - an unbounded cache must not fail the model load
        logger.warning("cannot bound the cuFFT plan cache", exc_info=True)
        return None


def limit_cuda_allocator(torch_mod: object, cap_bytes: int) -> int | None:
    """Cap the CUDA caching allocator of this process on the current device.

    ``torch.cuda.set_per_process_memory_fraction`` limits what the caching
    allocator may hold (cached blocks included) to ``fraction * total device
    memory``; an allocation beyond it first frees cached blocks and then
    fails with ``torch.OutOfMemoryError``. The CUDA context and loaded
    kernels are outside the allocator and outside the cap. Called before the
    model's first CUDA allocation, so the weights count against the cap too.
    Best effort: a failure is logged and leaves the allocator uncapped.

    Args:
        torch_mod: Loaded ``torch`` module; the call initializes its CUDA
            context.
        cap_bytes: Most bytes the allocator may hold (a positive integer);
            a cap at or above the device memory leaves the fraction at 1.

    Returns:
        The cap now in effect in bytes (the fraction times the device
        memory), or ``None`` when CUDA is unavailable or the cap could not be
        set.

    Raises:
        ValueError: When *cap_bytes* is not a positive integer.

    """
    if isinstance(cap_bytes, bool) or not isinstance(cap_bytes, int) or cap_bytes < 1:
        msg = f"cap_bytes must be a positive integer (got {cap_bytes!r})"
        raise ValueError(msg)
    if not _cuda_is_available(torch_mod):
        return None
    try:
        cuda: Any = getattr(torch_mod, "cuda", None)
        index = int(cuda.current_device())
        total = int(cuda.get_device_properties(index).total_memory)
        fraction = min(1.0, cap_bytes / total)
        cuda.set_per_process_memory_fraction(fraction, index)
    except Exception:  # noqa: BLE001 - an uncapped allocator must not fail the model load
        logger.warning("cannot cap the CUDA caching allocator", exc_info=True)
        return None
    return int(fraction * total)


# * Keys of :func:`cuda_memory_stats`, in log order.
CUDA_MEMORY_STAT_KEYS: tuple[str, ...] = (
    "cuda_reserved_bytes",
    "cuda_allocated_bytes",
    "cuda_max_reserved_bytes",
    "cufft_plan_cache_size",
)


def cuda_memory_stats(torch_mod: object) -> dict[str, int | None]:
    """Return the CUDA caching-allocator and cuFFT plan cache figures of this process.

    Nothing is read (every value is ``None``) when *torch_mod* is ``None`` or
    its CUDA context is not initialized, so a CPU process never creates a
    CUDA context for telemetry. Each figure is read independently; one that
    fails stays ``None``.

    Args:
        torch_mod: The already imported ``torch`` module (for example
            ``sys.modules.get("torch")``), or ``None``.

    Returns:
        A mapping with every key of :data:`CUDA_MEMORY_STAT_KEYS`:
        ``torch.cuda.memory_reserved()``, ``memory_allocated()`` and
        ``max_memory_reserved()`` of the current device in bytes, and the
        number of cached cuFFT plans of that device.

    """
    stats: dict[str, int | None] = dict.fromkeys(CUDA_MEMORY_STAT_KEYS)
    cuda: Any = getattr(torch_mod, "cuda", None)
    if cuda is None:
        return stats
    try:
        if not cuda.is_initialized():
            return stats
        index = int(cuda.current_device())
    except Exception:  # noqa: BLE001 - telemetry never fails its caller
        logger.debug("CUDA state query failed", exc_info=True)
        return stats
    readers: dict[str, Callable[[], object]] = {
        "cuda_reserved_bytes": lambda: cuda.memory_reserved(index),
        "cuda_allocated_bytes": lambda: cuda.memory_allocated(index),
        "cuda_max_reserved_bytes": lambda: cuda.max_memory_reserved(index),
        "cufft_plan_cache_size": lambda: _cufft_plan_cache_size(torch_mod, index),
    }
    for key, read in readers.items():
        try:
            value = read()
        except Exception:  # noqa: BLE001 - telemetry never fails its caller
            logger.debug("CUDA memory figure %s unavailable", key, exc_info=True)
            continue
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            stats[key] = value
    return stats


def _cufft_plan_cache_size(torch_mod: object, index: int) -> object:
    """Return the number of cached cuFFT plans of CUDA device *index*."""
    module: Any = torch_mod
    return module.backends.cuda.cufft_plan_cache[index].size


def module_device_types(model: object) -> set[str] | None:
    """Return the device types holding *model* parameters and buffers.

    Args:
        model: A ``torch.nn.Module`` (or a stand-in without tensors).

    Returns:
        Set of device type names such as ``{"cpu"}``, or ``None`` when *model*
        does not expose ``parameters()``/``buffers()``.

    """
    parameters = getattr(model, "parameters", None)
    buffers = getattr(model, "buffers", None)
    if not callable(parameters) or not callable(buffers):
        return None
    tensors = [*parameters(), *buffers()]
    return {str(tensor.device.type) for tensor in tensors}
