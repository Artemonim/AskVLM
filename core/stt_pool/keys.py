"""Worker keys and request normalization for the STT worker pool.

A :class:`WorkerKey` names one worker configuration: the provider, the model
identity, the compute type and the backend class (CPU or GPU). Everything here
is torch-free so the thin client and the supervisor can import it directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Self

from core.stt_providers import (
    GIGAAM_CPU_ONLY_PROVIDERS,
    GIGAAM_HF_REPO_ID,
    GIGAAM_HF_REVISION,
    GIGAAM_V3_HF_REPO_ID,
    GIGAAM_V3_HF_REVISION,
    STT_PROVIDER_CHOICES,
    STT_PROVIDER_GIGAAM_CTC,
    STT_PROVIDER_GIGAAM_E2E_CTC,
    is_gigaam_provider,
    normalize_stt_provider,
    resolve_gigaam_requested_device,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

Backend = Literal["cpu", "gpu"]
DeviceRequest = Literal["auto", "cpu", "cuda"]
ProviderFamily = Literal["whisper", "gigaam"]
Priority = Literal["high", "normal", "low"]

# * Call priority classes, most urgent first, and the class used when none is given.
PRIORITIES: tuple[Priority, ...] = ("high", "normal", "low")
DEFAULT_PRIORITY: Priority = "normal"

# * Admission rank of each class: a lower rank is admitted first.
PRIORITY_RANK: Mapping[Priority, int] = MappingProxyType(
    {"high": 0, "normal": 1, "low": 2}
)

# * Whisper model used when the request names none.
DEFAULT_WHISPER_MODEL = "small"

# * ``auto`` is the pipeline's alias for the large Whisper model (mirrors
# * ``core/pipelines.py``: ``large-v3 if whisper_model == "auto"``).
WHISPER_AUTO_MODEL = "auto"
WHISPER_AUTO_RESOLVED_MODEL = "large-v3"

# * GigaAM wrappers always run in fp32, whatever compute type was requested.
GIGAAM_COMPUTE_TYPE = "fp32"

# * Compute types that a CPU Whisper load resolves to int8 (mirrors
# * ``WhisperXWrapper._normalize_compute_type`` on ``device == "cpu"``).
_WHISPER_CPU_INT8_ALIASES: frozenset[str] = frozenset(
    {"auto", "float16", "int8_float16", "int8"}
)

_ALL_BACKENDS: frozenset[Backend] = frozenset({"cpu", "gpu"})
_CPU_ONLY_BACKENDS: frozenset[Backend] = frozenset({"cpu"})

_DEVICE_REQUESTS: dict[str, DeviceRequest] = {
    "auto": "auto",
    "cpu": "cpu",
    "cuda": "cuda",
}

_LANGUAGE_RE = re.compile(r"[a-z]{2,8}(-[a-z0-9]{1,8})?")


def _require_backend(value: str) -> Backend:
    """Validate a backend token.

    Args:
        value: Candidate backend token.

    Returns:
        The typed backend.

    Raises:
        ValueError: When *value* is neither ``cpu`` nor ``gpu``.

    """
    if value == "cpu":
        return "cpu"
    if value == "gpu":
        return "gpu"
    msg = f"Unsupported backend {value!r}; expected 'cpu' or 'gpu'"
    raise ValueError(msg)


def provider_family(provider: str) -> ProviderFamily:
    """Return the provider family that decides batching and host-thread limits.

    Args:
        provider: Raw or canonical STT provider id.

    Returns:
        ``gigaam`` for both GigaAM heads, otherwise ``whisper``.

    Raises:
        ValueError: When *provider* is not a supported provider id.

    """
    canonical = normalize_stt_provider(provider)
    return "gigaam" if is_gigaam_provider(canonical) else "whisper"


def model_identity(provider: str, whisper_model: str) -> str:
    """Return the model identity that is part of a worker key.

    Args:
        provider: Raw or canonical STT provider id.
        whisper_model: Requested Whisper model name; ignored for GigaAM.

    Returns:
        The stripped Whisper model name in the caller's spelling (model names,
        Hugging Face repo ids and local paths are case-sensitive), ``large-v3``
        for ``auto`` (compared case-insensitively, as the pipeline resolves it),
        ``small`` when blank, or the fixed ``<repo>@<revision>`` identity of
        the GigaAM head.

    Raises:
        ValueError: When *provider* is not a supported provider id.

    """
    canonical = normalize_stt_provider(provider)
    if canonical == STT_PROVIDER_GIGAAM_CTC:
        return f"{GIGAAM_HF_REPO_ID}@{GIGAAM_HF_REVISION}"
    if canonical == STT_PROVIDER_GIGAAM_E2E_CTC:
        return f"{GIGAAM_V3_HF_REPO_ID}@{GIGAAM_V3_HF_REVISION}"
    model = str(whisper_model).strip()
    if model.lower() == WHISPER_AUTO_MODEL:
        return WHISPER_AUTO_RESOLVED_MODEL
    return model or DEFAULT_WHISPER_MODEL


def normalize_compute_type(provider: str, requested: str, backend: Backend) -> str:
    """Resolve the requested compute type to the one the backend actually uses.

    Args:
        provider: Raw or canonical STT provider id.
        requested: Requested compute type (``auto`` when blank).
        backend: Backend class the worker runs on.

    Returns:
        ``fp32`` for GigaAM. For Whisper on CPU, ``auto``/``float16``/
        ``int8_float16``/``int8`` resolve to ``int8``; on GPU ``auto`` resolves
        to ``float16``. Any other token passes through lowercased.

    Raises:
        ValueError: When the provider or the backend is invalid.

    """
    canonical = normalize_stt_provider(provider)
    resolved_backend = _require_backend(backend)
    if is_gigaam_provider(canonical):
        return GIGAAM_COMPUTE_TYPE
    compute = str(requested).strip().lower() or "auto"
    if resolved_backend == "gpu":
        return "float16" if compute == "auto" else compute
    return "int8" if compute in _WHISPER_CPU_INT8_ALIASES else compute


def normalize_device_request(provider: str, raw: str) -> DeviceRequest:
    """Validate and normalize a ``--device`` token for *provider*.

    Args:
        provider: Raw or canonical STT provider id.
        raw: Requested device token; blank means ``auto``.

    Returns:
        ``auto``, ``cpu`` or ``cuda``. The CPU-only GigaAM head maps ``auto``
        and ``cpu`` to ``cpu``.

    Raises:
        ValueError: When the token is invalid for the provider (including
            ``cuda`` for the CPU-only GigaAM head).

    """
    canonical = normalize_stt_provider(provider)
    token = str(raw).strip().lower()
    if is_gigaam_provider(canonical):
        token = resolve_gigaam_requested_device(canonical, token)
    else:
        token = token or "auto"
    device = _DEVICE_REQUESTS.get(token)
    if device is None:
        msg = (
            f"Unsupported device {raw!r} for STT provider {canonical}; "
            "expected auto, cpu or cuda"
        )
        raise ValueError(msg)
    return device


def allowed_backends(provider: str) -> frozenset[Backend]:
    """Return the backend classes a provider can run on.

    Args:
        provider: Raw or canonical STT provider id.

    Returns:
        Only ``cpu`` for the CPU-only GigaAM head, otherwise ``cpu`` and ``gpu``.

    Raises:
        ValueError: When *provider* is not a supported provider id.

    """
    canonical = normalize_stt_provider(provider)
    if canonical in GIGAAM_CPU_ONLY_PROVIDERS:
        return _CPU_ONLY_BACKENDS
    return _ALL_BACKENDS


def device_for_backend(backend: Backend) -> Literal["cpu", "cuda"]:
    """Return the strict device token that a backend class loads on.

    Args:
        backend: Backend class.

    Returns:
        ``cuda`` for ``gpu``, ``cpu`` for ``cpu``.

    Raises:
        ValueError: When *backend* is invalid.

    """
    return "cuda" if _require_backend(backend) == "gpu" else "cpu"


def normalize_language(raw: str | None) -> str | None:
    """Normalize a per-job language hint.

    Args:
        raw: Raw language token, or ``None``.

    Returns:
        ``None`` for ``None``/blank input, otherwise the lowercased token.

    Raises:
        ValueError: When the token is not ``[a-z]{2,8}`` with an optional
            ``-[a-z0-9]{1,8}`` subtag.

    """
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if not token:
        return None
    if _LANGUAGE_RE.fullmatch(token) is None:
        msg = f"Invalid language {raw!r}; expected a code such as 'ru' or 'en-us'"
        raise ValueError(msg)
    return token


def normalize_priority(raw: object) -> Priority:
    """Normalize a call priority token.

    Args:
        raw: Raw priority token, or ``None``.

    Returns:
        ``normal`` for ``None``/blank input, otherwise the matching class
        (compared case-insensitively after stripping).

    Raises:
        ValueError: When *raw* is not a string (other than ``None``) or names no
            priority class; the message lists the allowed values.

    """
    allowed = ", ".join(PRIORITIES)
    if raw is None:
        return DEFAULT_PRIORITY
    if not isinstance(raw, str):
        msg = f"Invalid priority {raw!r}; expected one of: {allowed}"
        raise ValueError(msg)  # noqa: TRY004 - payload values are untyped JSON
    token = raw.strip().lower()
    if not token:
        return DEFAULT_PRIORITY
    for priority in PRIORITIES:
        if token == priority:
            return priority
    msg = f"Invalid priority {raw!r}; expected one of: {allowed}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class WorkerKey:
    """Identity of one worker configuration.

    Attributes:
        provider: Canonical STT provider id.
        model: Whisper model name or the fixed GigaAM ``<repo>@<revision>``.
        compute_type: Compute type already resolved for the backend.
        backend: Backend class the worker runs on.

    """

    provider: str
    model: str
    compute_type: str
    backend: Backend

    @property
    def label(self) -> str:
        """Return the stable ``provider/model/compute_type/backend`` label."""
        return f"{self.provider}/{self.model}/{self.compute_type}/{self.backend}"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable mapping of the key.

        Returns:
            A mapping with the four key fields.

        """
        return {
            "provider": self.provider,
            "model": self.model,
            "compute_type": self.compute_type,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Rebuild a key from :meth:`to_dict` output.

        Args:
            payload: Mapping with ``provider``, ``model``, ``compute_type`` and
                ``backend`` string fields.

        Returns:
            The validated key.

        Raises:
            ValueError: When a field is missing or not a non-empty string, the
                provider is unsupported, or the backend is invalid or not
                allowed for the provider.

        """
        fields: dict[str, str] = {}
        for name in ("provider", "model", "compute_type", "backend"):
            value = payload.get(name)
            if not isinstance(value, str) or not value.strip():
                msg = f"Worker key field {name!r} must be a non-empty string"
                raise ValueError(msg)
            fields[name] = value.strip()
        provider = fields["provider"]
        if provider not in STT_PROVIDER_CHOICES:
            msg = f"Worker key has unsupported provider {provider!r}"
            raise ValueError(msg)
        backend = _require_backend(fields["backend"])
        if backend not in allowed_backends(provider):
            msg = f"Provider {provider} cannot run on backend {backend!r}"
            raise ValueError(msg)
        return cls(
            provider=provider,
            model=fields["model"],
            compute_type=fields["compute_type"],
            backend=backend,
        )


def make_key(
    provider: str,
    whisper_model: str,
    compute_request: str,
    backend: Backend,
) -> WorkerKey:
    """Build the normalized worker key for a request on a backend.

    Args:
        provider: Raw or canonical STT provider id.
        whisper_model: Requested Whisper model name; ignored for GigaAM.
        compute_request: Requested compute type.
        backend: Backend class the worker would run on.

    Returns:
        The normalized :class:`WorkerKey`.

    Raises:
        ValueError: When the provider is unsupported or cannot run on *backend*.

    """
    canonical = normalize_stt_provider(provider)
    resolved_backend = _require_backend(backend)
    if resolved_backend not in allowed_backends(canonical):
        msg = f"STT provider {canonical} cannot run on backend {resolved_backend!r}"
        raise ValueError(msg)
    return WorkerKey(
        provider=canonical,
        model=model_identity(canonical, whisper_model),
        compute_type=normalize_compute_type(
            canonical, compute_request, resolved_backend
        ),
        backend=resolved_backend,
    )
