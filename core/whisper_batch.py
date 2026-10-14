"""Whisper batch primitives for the STT worker pool (faster-whisper, lazy imports).

* :func:`speech_span` — first speech start and last speech end of a clip by
  faster-whisper's Silero VAD with default :class:`VadOptions` (the options
  ``WhisperModel.transcribe(vad_filter=True)`` uses).
* :func:`transcribe_clips_batched` — one ``BatchedInferencePipeline`` call over
  several short clips (for example of different files) concatenated with
  explicit ``clip_timestamps``, mapped back to per-clip texts.
* :func:`load_audio_16k` — decode a media file to 16 kHz mono float32.
* :func:`is_whisper_cuda_memory_error` — the CUDA memory error classifier of
  :class:`~core.whisperx_wrapper.WhisperXWrapper`.

Importing this module loads neither numpy nor faster-whisper.

Batched decoding differs from the sequential ``WhisperModel.transcribe`` path:
faster-whisper detects the language once per call (so callers pass an explicit
language and batch only clips of that language), ignores ``vad_filter`` when
``clip_timestamps`` are given, never conditions on previous text, and uses
only the first temperature (no temperature fallback). Timestamps are predicted
(``without_timestamps=False``) like the sequential path, so a clip can produce
several segments, which are joined with newlines like the pipeline's document.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from core.stt_errors import CudaOutOfMemoryError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

# * Whisper models work on 16 kHz mono audio.
WHISPER_SAMPLE_RATE = 16000
# * Longest clip one batched decoder window covers.
MAX_CLIP_S = 30.0
# * Zero frames between concatenated clips, in Whisper feature frames: keeps the
# * per-clip ``seek`` values (clip offset in frames) at least two frames apart.
_CLIP_GAP_FRAMES = 2

# * Lower-cased message fragments of CUDA memory failures (CTranslate2/torch).
# ! "not enough memory" also matches CPU RAM exhaustion: callers apply this
# ! classifier only to a model that runs on CUDA.
_CUDA_MEMORY_TOKENS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
    "cuda failed with error out of memory",
    "not enough memory",
)


def is_whisper_cuda_memory_error(exc: BaseException) -> bool:
    """Return whether *exc* looks like a CUDA memory failure of faster-whisper.

    Args:
        exc: Exception raised while loading or running a Whisper model.

    Returns:
        ``True`` for ``*OutOfMemory*`` exception types and CUDA/cuDNN memory
        failure messages.

    """
    if "outofmemory" in type(exc).__name__.lower():
        return True
    message = str(exc).lower()
    return any(token in message for token in _CUDA_MEMORY_TOKENS)


def _numpy() -> Any:  # noqa: ANN401 - the numpy module, imported lazily
    """Import numpy on first use so importing this module stays light."""
    return importlib.import_module("numpy")


def load_audio_16k(path: str | Path) -> NDArray[np.float32]:
    """Decode *path* to 16 kHz mono float32 samples (faster-whisper ``decode_audio``).

    Args:
        path: Audio or video file readable by PyAV.

    Returns:
        1-D float32 samples in ``[-1, 1)``.

    """
    audio_mod = importlib.import_module("faster_whisper.audio")
    audio = audio_mod.decode_audio(str(path), sampling_rate=WHISPER_SAMPLE_RATE)
    return cast("NDArray[np.float32]", audio)


def _as_mono_float32(audio: object, *, name: str) -> NDArray[np.float32]:
    """Return *audio* as a 1-D float32 array.

    Raises:
        ValueError: When *audio* is not one-dimensional.

    """
    np_mod = _numpy()
    array = np_mod.asarray(audio, dtype=np_mod.float32)
    if array.ndim != 1:
        msg = f"{name} must be a 1-D mono sample array (got shape {array.shape})"
        raise ValueError(msg)
    return cast("NDArray[np.float32]", array)


def speech_span(
    audio: NDArray[Any], *, sample_rate: int = WHISPER_SAMPLE_RATE
) -> tuple[float, float] | None:
    """Return ``(first speech start, last speech end)`` of *audio* in seconds.

    Uses faster-whisper's Silero VAD with default :class:`VadOptions`, the
    options of ``WhisperModel.transcribe(vad_filter=True)``; the span includes
    the VAD's speech padding.

    Args:
        audio: 1-D mono samples.
        sample_rate: Sample rate of *audio* (the Silero VAD model expects 16 kHz).

    Returns:
        The speech span, or ``None`` when the VAD finds no speech.

    Raises:
        ValueError: When *audio* is not 1-D or *sample_rate* is not positive.

    """
    if sample_rate <= 0:
        msg = f"sample_rate must be positive (got {sample_rate!r})"
        raise ValueError(msg)
    samples = _as_mono_float32(audio, name="audio")
    if samples.shape[0] == 0:
        return None
    vad = importlib.import_module("faster_whisper.vad")
    speeches = vad.get_speech_timestamps(
        samples, vad_options=vad.VadOptions(), sampling_rate=sample_rate
    )
    if not speeches:
        return None
    return (
        float(speeches[0]["start"]) / sample_rate,
        float(speeches[-1]["end"]) / sample_rate,
    )


@dataclass(slots=True)
class _BatchPlan:
    """Concatenated audio, its ``clip_timestamps``, and the seek → clip map."""

    pieces: list[NDArray[np.float32]] = field(default_factory=list)
    clip_timestamps: list[dict[str, float]] = field(default_factory=list)
    owner_by_seek: dict[int, int] = field(default_factory=dict)


def _plan_batch(
    clips: Sequence[NDArray[np.float32]],
    *,
    sample_rate: int,
    frames_per_second: int,
) -> _BatchPlan:
    """Lay the clips out on one timeline and precompute each clip's ``seek``.

    Clips are separated by ``_CLIP_GAP_FRAMES`` frames of zeros (never decoded:
    only the clip ranges are). Timestamps sit half a sample past the clip
    borders so the pipeline's ``int(seconds * rate)`` lands exactly on them,
    and ``seek`` is computed with the pipeline's own arithmetic
    (``int(start / rate * frames_per_second)``). Clips shorter than one frame
    are left out (their text is empty).

    Raises:
        RuntimeError: When two clips would share a ``seek`` (cannot happen with
            the gap; guards the mapping).

    """
    np_mod = _numpy()
    frame_samples = max(1, sample_rate // frames_per_second)
    gap = np_mod.zeros(_CLIP_GAP_FRAMES * frame_samples, dtype=np_mod.float32)
    plan = _BatchPlan()
    cursor = 0
    for owner, clip in enumerate(clips):
        size = int(clip.shape[0])
        if size < frame_samples:
            continue
        if plan.pieces:
            plan.pieces.append(gap)
            cursor += int(gap.shape[0])
        start, end = cursor, cursor + size
        plan.pieces.append(clip)
        cursor = end
        start_s = (start + 0.5) / sample_rate
        plan.clip_timestamps.append(
            {"start": start_s, "end": (end + 0.5) / sample_rate}
        )
        clip_start = int(start_s * sample_rate)
        seek = int(clip_start / sample_rate * frames_per_second)
        if seek in plan.owner_by_seek:
            msg = f"Whisper batch clips {plan.owner_by_seek[seek]} and {owner} share seek {seek}"
            raise RuntimeError(msg)
        plan.owner_by_seek[seek] = owner
    return plan


def _model_on_cuda(model: object) -> bool:
    """Return whether the CTranslate2 model behind *model* runs on CUDA."""
    device = getattr(getattr(model, "model", None), "device", "")
    return str(device).lower() == "cuda"


def _clip_owner(plan: _BatchPlan, seek: int) -> int:
    """Return the clip index of a decoded segment's ``seek``.

    Raises:
        RuntimeError: When *seek* matches no clip of *plan*.

    """
    owner = plan.owner_by_seek.get(seek)
    if owner is None:
        msg = f"Whisper batch segment seek {seek} matches no clip"
        raise RuntimeError(msg)
    return owner


def _validated_clips(
    clips: Sequence[NDArray[Any]], *, sample_rate: int
) -> list[NDArray[np.float32]]:
    """Return the clips as 1-D float32 arrays of at most ``MAX_CLIP_S``.

    Raises:
        ValueError: When a clip is not 1-D or is longer than ``MAX_CLIP_S``.

    """
    max_samples = int(MAX_CLIP_S * sample_rate)
    arrays: list[NDArray[np.float32]] = []
    for index, clip in enumerate(clips):
        array = _as_mono_float32(clip, name=f"clip {index}")
        if int(array.shape[0]) > max_samples:
            msg = (
                f"clip {index} is {array.shape[0] / sample_rate:.2f} s long; "
                f"batched clips are limited to {MAX_CLIP_S:.0f} s"
            )
            raise ValueError(msg)
        arrays.append(array)
    return arrays


def transcribe_clips_batched(
    model: Any,  # noqa: ANN401 - untyped faster-whisper WhisperModel
    clips: Sequence[NDArray[Any]],
    *,
    language: str,
    beam_size: int = 10,
    batch_size: int | None = None,
) -> list[str]:
    r"""Decode several short clips in shared batched forward passes.

    The clips are concatenated and decoded by one ``BatchedInferencePipeline``
    call with explicit ``clip_timestamps`` (one decoder window per clip,
    ``beam_size`` beams, predicted timestamps); every segment returns to its
    clip through its ``seek``.

    Args:
        model: Loaded faster-whisper ``WhisperModel`` (for example
            :meth:`~core.whisperx_wrapper.WhisperXWrapper.loaded_model`).
        clips: 1-D 16 kHz float32 clips, each at most ``MAX_CLIP_S`` (30 s).
        language: Explicit language code shared by all clips (the batched
            pipeline detects a language once per call, so it is required).
        beam_size: Beam size of the decoder.
        batch_size: Clips per forward pass; ``None`` decodes all clips in one.

    Returns:
        One text per clip, in input order: the clip's stripped segment texts
        joined with ``"\n"``, stripped; ``""`` for an empty clip or a clip
        shorter than one Whisper frame (10 ms).

    Raises:
        ValueError: Missing language, a clip that is not 1-D or longer than
            30 s, or a non-positive beam/batch size.
        CudaOutOfMemoryError: The model runs on CUDA and ran out of memory
            (never retried on CPU).
        RuntimeError: A decoded segment cannot be mapped back to a clip.

    """
    language_code = str(language or "").strip()
    if not language_code:
        msg = "transcribe_clips_batched requires an explicit language code"
        raise ValueError(msg)
    if beam_size < 1:
        msg = f"beam_size must be >= 1 (got {beam_size!r})"
        raise ValueError(msg)
    if batch_size is not None and batch_size < 1:
        msg = f"batch_size must be >= 1 or None (got {batch_size!r})"
        raise ValueError(msg)
    sample_rate = int(model.feature_extractor.sampling_rate)
    arrays = _validated_clips(clips, sample_rate=sample_rate)
    plan = _plan_batch(
        arrays,
        sample_rate=sample_rate,
        frames_per_second=int(model.frames_per_second),
    )
    parts: list[list[str]] = [[] for _ in arrays]
    if not plan.clip_timestamps:
        return ["" for _ in arrays]
    np_mod = _numpy()
    pipeline = importlib.import_module("faster_whisper").BatchedInferencePipeline(model)
    try:
        segments, _info = pipeline.transcribe(
            np_mod.concatenate(plan.pieces),
            language=language_code,
            beam_size=beam_size,
            batch_size=batch_size or len(plan.clip_timestamps),
            clip_timestamps=plan.clip_timestamps,
            without_timestamps=False,
        )
        # * The segment generator is lazy: decoding (and any OOM) happens here.
        for segment in segments:
            owner = _clip_owner(plan, int(segment.seek))
            parts[owner].append(str(segment.text or "").strip())
    except Exception as exc:
        if not (_model_on_cuda(model) and is_whisper_cuda_memory_error(exc)):
            raise
        msg = (
            f"Whisper batch of {len(plan.clip_timestamps)} clip(s) ran out of CUDA "
            f"memory: {type(exc).__name__}: {exc}"
        )
        raise CudaOutOfMemoryError(msg) from exc
    return ["\n".join(texts).strip() for texts in parts]
