"""Stable STT provider identifiers for the external transcription contract."""

from __future__ import annotations

from typing import Literal

SttProvider = Literal["whisper", "gigaam-ctc", "gigaam-e2e-ctc"]

STT_PROVIDER_WHISPER: SttProvider = "whisper"
STT_PROVIDER_GIGAAM_CTC: SttProvider = "gigaam-ctc"
STT_PROVIDER_GIGAAM_E2E_CTC: SttProvider = "gigaam-e2e-ctc"

STT_PROVIDER_CHOICES: tuple[SttProvider, ...] = (
    STT_PROVIDER_WHISPER,
    STT_PROVIDER_GIGAAM_CTC,
    STT_PROVIDER_GIGAAM_E2E_CTC,
)

# * CPU-only GigaAM heads share device resolution and CLI/pipeline gating.
GIGAAM_CPU_ONLY_PROVIDERS: frozenset[SttProvider] = frozenset(
    {STT_PROVIDER_GIGAAM_CTC, STT_PROVIDER_GIGAAM_E2E_CTC}
)

# * Fixed Hugging Face identity for the optional GigaAM Multilingual CTC path.
GIGAAM_HF_REPO_ID = "ai-sage/GigaAM-Multilingual"
GIGAAM_HF_REVISION = "ctc"

# * Fixed Hugging Face identity for GigaAM-v3 end-to-end CTC (punct + caps).
GIGAAM_V3_HF_REPO_ID = "ai-sage/GigaAM-v3"
GIGAAM_V3_HF_REVISION = "e2e_ctc"

_DEFAULT_PROVIDER: SttProvider = STT_PROVIDER_WHISPER


def is_cpu_only_gigaam_provider(provider: str) -> bool:
    """Return whether *provider* is a CPU-only GigaAM STT backend.

    Args:
        provider: Canonical or raw STT provider id.

    Returns:
        ``True`` for ``gigaam-ctc`` and ``gigaam-e2e-ctc``.

    """
    return provider in GIGAAM_CPU_ONLY_PROVIDERS


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
    r"""Resolve a CLI/device token to the only device GigaAM CTC providers may use.

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
        "GigaAM CTC providers support CPU only "
        f"(got device={device!r}); use --device cpu or --device auto"
    )
    raise ValueError(msg)
