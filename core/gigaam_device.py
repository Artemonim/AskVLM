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
"""

from __future__ import annotations

import contextlib
import gc
import logging
from typing import Any

logger = logging.getLogger(__name__)

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


def select_gigaam_device(
    requested: str,
    torch_mod: object,
    *,
    min_free_bytes: int = GIGAAM_AUTO_CUDA_MIN_FREE_VRAM_BYTES,
) -> str:
    """Resolve a GigaAM device request against the live CUDA state.

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
