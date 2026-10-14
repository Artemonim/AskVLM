"""Stable STT provider identifiers for the external transcription contract.

Everything here is torch-free: the thin ``external-transcribe`` client validates
provider and device tokens through this module without importing ML packages.
"""

from __future__ import annotations

from typing import Literal

SttProvider = Literal["whisper", "gigaam-ctc", "gigaam-e2e-ctc"]
GigaAMDeviceRequest = Literal["auto", "cpu", "cuda"]

STT_PROVIDER_WHISPER: SttProvider = "whisper"
STT_PROVIDER_GIGAAM_CTC: SttProvider = "gigaam-ctc"
STT_PROVIDER_GIGAAM_E2E_CTC: SttProvider = "gigaam-e2e-ctc"

STT_PROVIDER_CHOICES: tuple[SttProvider, ...] = (
    STT_PROVIDER_WHISPER,
    STT_PROVIDER_GIGAAM_CTC,
    STT_PROVIDER_GIGAAM_E2E_CTC,
)

# * Both GigaAM heads share the windowed wrapper, pipeline dispatch, and the
# * GigaAM-specific device validation below.
GIGAAM_PROVIDERS: frozenset[SttProvider] = frozenset(
    {STT_PROVIDER_GIGAAM_CTC, STT_PROVIDER_GIGAAM_E2E_CTC}
)

# * GigaAM heads whose wrapper may place the resident model on CUDA.
GIGAAM_CUDA_CAPABLE_PROVIDERS: frozenset[SttProvider] = frozenset(
    {STT_PROVIDER_GIGAAM_E2E_CTC}
)

# * GigaAM heads that always run on CPU (``cuda`` is rejected before load).
GIGAAM_CPU_ONLY_PROVIDERS: frozenset[SttProvider] = (
    GIGAAM_PROVIDERS - GIGAAM_CUDA_CAPABLE_PROVIDERS
)

_GIGAAM_DEVICE_REQUESTS: frozenset[GigaAMDeviceRequest] = frozenset(
    {"auto", "cpu", "cuda"}
)

# * Fixed Hugging Face identity for the optional GigaAM Multilingual CTC path.
GIGAAM_HF_REPO_ID = "ai-sage/GigaAM-Multilingual"
GIGAAM_HF_REVISION = "ctc"

# * Fixed Hugging Face identity for GigaAM-v3 end-to-end CTC (punct + caps).
GIGAAM_V3_HF_REPO_ID = "ai-sage/GigaAM-v3"
GIGAAM_V3_HF_REVISION = "e2e_ctc"

_DEFAULT_PROVIDER: SttProvider = STT_PROVIDER_WHISPER


def is_gigaam_provider(provider: str) -> bool:
    """Return whether *provider* is one of the GigaAM STT backends.

    Args:
        provider: Canonical or raw STT provider id.

    Returns:
        ``True`` for ``gigaam-ctc`` and ``gigaam-e2e-ctc``.

    """
    return provider in GIGAAM_PROVIDERS


def is_cpu_only_gigaam_provider(provider: str) -> bool:
    """Return whether *provider* is a GigaAM backend restricted to CPU.

    Args:
        provider: Canonical or raw STT provider id.

    Returns:
        ``True`` only for ``gigaam-ctc``; ``gigaam-e2e-ctc`` may use CUDA.

    """
    return provider in GIGAAM_CPU_ONLY_PROVIDERS


def provider_may_use_cuda(provider: str) -> bool:
    """Return whether *provider* can place its model on CUDA.

    Gates the Windows child isolation (with its single CPU retry) of the legacy
    ``external-transcribe --no-daemon`` path per provider.

    Args:
        provider: Canonical or raw STT provider id.

    Returns:
        ``True`` for Whisper and ``gigaam-e2e-ctc``; ``False`` for
        ``gigaam-ctc``.

    """
    return provider == STT_PROVIDER_WHISPER or provider in GIGAAM_CUDA_CAPABLE_PROVIDERS


def normalize_stt_provider(value: str | None) -> SttProvider:
    """Normalize a provider token to a stable STT provider id.

    Args:
        value: Raw provider string, or ``None`` for the Whisper default.

    Returns:
        A canonical :data:`SttProvider` value.

    Raises:
        ValueError: When *value* is not a supported provider id.

    """
    if value is None or not str(value).strip():
        return _DEFAULT_PROVIDER
    normalized = str(value).strip().lower()
    if normalized in STT_PROVIDER_CHOICES:
        return normalized  # type: ignore[return-value]
    allowed = ", ".join(STT_PROVIDER_CHOICES)
    msg = f"Unsupported STT provider {value!r}; expected one of: {allowed}"
    raise ValueError(msg)


def resolve_gigaam_device(device: str) -> str:
    r"""Resolve a CLI/device token for the CPU-only GigaAM head (``gigaam-ctc``).

    Args:
        device: Requested device token (``auto``/``cpu``/``cuda``/...).

    Returns:
        Always ``\"cpu\"`` when the request is CPU-compatible.

    Raises:
        ValueError: When *device* would require CUDA or another non-CPU target.

    """
    token = str(device).strip().lower()
    if token in {"", "auto", "cpu"}:
        return "cpu"
    msg = (
        "GigaAM Multilingual CTC (gigaam-ctc) supports CPU only "
        f"(got device={device!r}); use --device cpu or --device auto"
    )
    raise ValueError(msg)


def normalize_gigaam_cuda_device(device: str) -> GigaAMDeviceRequest:
    """Validate a device token for a CUDA-capable GigaAM head without torch.

    The token is only validated here; ``auto`` and ``cuda`` are resolved
    against the live CUDA state when the model loads.

    Args:
        device: Requested device token; empty means ``auto``.

    Returns:
        The canonical request: ``auto``, ``cpu``, or ``cuda``.

    Raises:
        ValueError: When *device* is not one of ``auto``/``cpu``/``cuda``.

    """
    token = str(device).strip().lower() or "auto"
    if token in _GIGAAM_DEVICE_REQUESTS:
        return token  # type: ignore[return-value]
    msg = (
        "GigaAM-v3 e2e CTC (gigaam-e2e-ctc) supports --device auto|cpu|cuda "
        f"(got device={device!r})"
    )
    raise ValueError(msg)


def resolve_gigaam_requested_device(provider: str, device: str) -> str:
    """Validate *device* for a GigaAM *provider* before any model import.

    Args:
        provider: Canonical GigaAM provider id.
        device: Raw ``--device`` token.

    Returns:
        ``cpu`` for the CPU-only head; the canonical ``auto``/``cpu``/``cuda``
        request for a CUDA-capable head.

    Raises:
        ValueError: When the token is invalid for *provider*.

    """
    if provider in GIGAAM_CUDA_CAPABLE_PROVIDERS:
        return normalize_gigaam_cuda_device(device)
    return resolve_gigaam_device(device)
