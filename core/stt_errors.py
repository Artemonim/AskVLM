"""Typed STT device errors shared by the model wrappers and the worker pool.

The module depends on the standard library only, so the thin client, the
supervisor and the worker processes can import it without loading any ML
package. Wrappers raise these errors; the pool maps them to result
``error_kind`` values (see :mod:`core.stt_pool.engine_api`).
"""

from __future__ import annotations


class SttDeviceError(RuntimeError):
    """Base class of failures caused by the compute device of an STT model."""


class CudaOutOfMemoryError(SttDeviceError):
    """CUDA ran out of memory while an STT model was loading or running."""


class DeviceUnavailableError(SttDeviceError):
    """The requested compute device cannot be used (for example no CUDA device)."""
