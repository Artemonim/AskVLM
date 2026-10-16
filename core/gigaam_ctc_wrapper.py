r"""Lazy wrapper for GigaAM Multilingual CTC speech-to-text (CPU-only head).

Uses the official Transformers API::

    AutoModel.from_pretrained(..., revision="ctc", trust_remote_code=True)
    model.transcribe(path)

Results are normalized to the pipeline contract ``{"text", "segments"}``.
Upstream ``.transcribe`` rejects more than 400_000 samples (~25 s at 16 kHz).
This wrapper streams the WAV in 20 s windows so files of any length stay
under that limit without native ``transcribe_longform``.

The device machinery (CUDA placement after a CPU load, CUDA OOM fallback to
CPU per window, VRAM release on unload) is shared with the CUDA-capable
subclass; for this head the device is always ``cpu`` and CUDA is never used.

Opt-in modes used by the STT worker pool (defaults keep the behavior above):

* ``strict_device`` — only an explicit ``cpu``/``cuda`` device; no CUDA→CPU
  fallback anywhere: an unusable CUDA runtime raises
  :class:`~core.stt_errors.DeviceUnavailableError` before weights load, and a
  CUDA out-of-memory error raises :class:`~core.stt_errors.CudaOutOfMemoryError`.
* ``threadsafe`` — :meth:`GigaAMCtcWrapper.transcribe` reads the 20 s windows
  straight from the WAV and runs each through a lock-free per-call
  ``forward`` + CTC ``decode`` of the remote-code ASR module, so several Python
  threads can share one resident CPU model (see :meth:`ensure_loaded`).
* :func:`iter_wav_windows` and :meth:`GigaAMCtcWrapper.transcribe_window_batch`
  — the window stream and the padded batch inference used by GPU batching.
"""

from __future__ import annotations

import contextlib
import gc as _gc
import importlib
import logging
import math
import os
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, NoReturn, cast

from core.gigaam_device import (
    STRICT_DEVICE_REQUESTS,
    is_cuda_oom_error,
    limit_cufft_plan_cache,
    module_device_types,
    release_cuda_cache,
    select_gigaam_device,
)
from core.stt_errors import CudaOutOfMemoryError, DeviceUnavailableError
from core.stt_providers import (
    GIGAAM_HF_REPO_ID,
    GIGAAM_HF_REVISION,
    STT_PROVIDER_GIGAAM_CTC,
    resolve_gigaam_requested_device,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# * Window length in seconds of the source file's sample rate. After the model
# * resamples to 16 kHz, 20 s is 320_000 samples — below the 400_000 hard cap.
_WINDOW_S = 20.0
# * GigaAM remote-code preprocessor uses STFT center=False, n_fft=320 at 16 kHz,
# * so resampled PCM must be strictly longer than 320 samples (~20.06 ms).
# * Threshold is duration (rate-invariant), not source-frame count: 8/22/48 kHz
# * files change sample count after GigaAM's ffmpeg resample. 40 ms is 640
# * samples @16 kHz — headroom for integer resample rounding above 321 samples.
_MIN_AUDIO_DURATION_S = 0.04
_CHUNK_LOG_INTERVAL = 10

# * Sample rate of the remote code (its ``SAMPLE_RATE``) and of the 16-bit mono
# * PCM WAVs that the lock-free path reads directly without ffmpeg.
GIGAAM_SAMPLE_RATE = 16000
_PCM16_SAMPLE_WIDTH = 2
# * Scale of the remote ``load_audio``: int16 PCM -> float32 in [-1, 1).
_PCM16_SCALE = 32768.0
# * Sample bounds of one window accepted by the lock-free inference path:
# * the STFT duration floor and the 20 s window of the warm-up.
_MIN_WINDOW_SAMPLES = math.ceil(_MIN_AUDIO_DURATION_S * GIGAAM_SAMPLE_RATE)
_MAX_WINDOW_SAMPLES = int(_WINDOW_S * GIGAAM_SAMPLE_RATE)
# * Remote-code ASR attributes used by the lock-free path (``GigaAMASR``).
_ASR_ATTRS = ("forward", "head", "decoding", "parameters")

_MISSING_EXTRA_MSG = (
    "GigaAM CTC providers (Multilingual ctc, v3 e2e_ctc) require the AskVLM ML extra "
    '`pip install -e ".[ml]"` '
    "(torch/torchaudio 2.10 via run.ps1/build.ps1 CUDA ensure, transformers 5, "
    "hydra-core, omegaconf, sentencepiece, pyannote.audio)."
)


class _UnparseableWavError(Exception):
    """Stdlib ``wave`` cannot stream this file (bad rate or header fields)."""


class _WavFormat(NamedTuple):
    """PCM stream parameters used by the 20 s window loop."""

    rate: int
    sample_width: int
    channels: int
    window_frames: int


class UnsupportedWavError(RuntimeError):
    """The file cannot be streamed as 16 kHz mono 16-bit PCM WAV windows."""


@dataclass(frozen=True, slots=True)
class WavWindow:
    """One window of a 16 kHz mono 16-bit PCM WAV, ready for GigaAM inference.

    Attributes:
        index: Zero-based window index.
        start_s: Absolute window start in seconds.
        end_s: Absolute end of the window's audio in seconds (before padding).
        samples: 1-D float32 samples scaled like the remote ``load_audio``
            (``int16 / 32768``), zero-padded to the STFT duration floor
            (``_MIN_AUDIO_DURATION_S``) when the window is shorter.

    """

    index: int
    start_s: float
    end_s: float
    samples: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _LockFreeModel:
    """Resolved remote-code ASR module used by the lock-free inference path."""

    asr: Any
    torch: Any
    device: Any
    dtype: Any
    cuda: bool


@dataclass(slots=True)
class _TranscriptAccumulator:
    """Collects window transcripts, segment callbacks, and optional progress."""

    on_segment: Callable[[dict[str, Any]], None] | None
    progress: Callable[[float, str], None] | None
    duration_s: float | None
    segments_out: list[dict[str, Any]] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)

    def emit(self, start: float, end: float, text: str) -> None:
        """Append one window segment and notify optional callers.

        Args:
            start: Absolute window start in seconds.
            end: Absolute window end in seconds.
            text: Transcript text for this window.

        """
        seg = {"start": start, "end": end, "text": text}
        self.segments_out.append(seg)
        if text:
            self.text_parts.append(text)
        if self.on_segment is not None:
            self.on_segment(seg)
        duration_s = self.duration_s
        if self.progress is not None and duration_s is not None and duration_s > 0:
            self.progress(max(0.0, min(1.0, end / duration_s)), "transcribe")

    def as_result(self) -> dict[str, Any]:
        """Return the pipeline ``{text, segments}`` mapping."""
        return {
            "text": " ".join(self.text_parts).strip(),
            "segments": self.segments_out,
        }


def _raise_missing_extra(exc: BaseException) -> NoReturn:
    """Raise a friendly missing-extra RuntimeError, preserving import context.

    Args:
        exc: Original import failure (typically ``ImportError`` /
            ``ModuleNotFoundError`` from torch/transformers or remote code).

    Raises:
        RuntimeError: Always, with ``pip install -e ".[ml]"`` guidance and
            the missing module name when available.

    """
    missing_name = getattr(exc, "name", None)
    if isinstance(missing_name, str) and missing_name.strip():
        detail = f" Missing module: {missing_name.strip()}."
    else:
        detail = ""
    msg = f"{_MISSING_EXTRA_MSG}{detail}"
    raise RuntimeError(msg) from exc


def _pcm_frame_count(pcm: bytes, *, sample_width: int, channels: int) -> int:
    """Return the number of complete PCM frames in *pcm*."""
    frame_size = sample_width * channels
    if frame_size <= 0:
        return 0
    return len(pcm) // frame_size


def _header_duration_seconds(reader: wave.Wave_read) -> float | None:
    """Best-effort duration from the WAV header of an already open reader."""
    try:
        n_frames = reader.getnframes()
        rate = reader.getframerate()
    except (wave.Error, EOFError, OSError):
        return None
    if rate <= 0 or n_frames < 0:
        return None
    return float(n_frames) / float(rate)


def _format_from_reader(reader: wave.Wave_read) -> _WavFormat:
    """Read stream format or mark the WAV as unparseable.

    Raises:
        _UnparseableWavError: When sample rate, width, or channel count is invalid.

    """
    rate = reader.getframerate()
    sample_width = reader.getsampwidth()
    channels = reader.getnchannels()
    if rate <= 0 or sample_width < 1 or channels < 1:
        msg = (
            f"WAV header is not streamable (rate={rate}, "
            f"sample_width={sample_width}, channels={channels})"
        )
        raise _UnparseableWavError(msg)
    return _WavFormat(
        rate=rate,
        sample_width=sample_width,
        channels=channels,
        window_frames=int(_WINDOW_S * rate),
    )


def _write_pcm_wav(
    path: Path,
    pcm: bytes,
    *,
    rate: int,
    sample_width: int,
    channels: int,
) -> None:
    """Write a PCM WAV with the same format as the source stream."""
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(rate)
        writer.writeframes(pcm)


def _unlink_padded_wav(path: Path) -> None:
    """Best-effort unlink of a silence-padded temp WAV.

    Windows may transiently lock the file after ``model.transcribe``; a failed
    unlink is logged at debug and does not fail transcription.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug(
            "Could not unlink padded GigaAM WAV %s",
            path,
            exc_info=True,
        )


def _maybe_write_min_duration_wav(audio_path: Path) -> Path | None:
    """Write a silence-padded copy when WAV duration is below the STFT floor.

    Pads with zero PCM frames at the *source* sample rate until duration is at
    least ``_MIN_AUDIO_DURATION_S``. Unparseable headers skip padding so the
    caller can still hand the original path to the model.

    Args:
        audio_path: Source WAV path (original file or a 20 s chunk).

    Returns:
        Path of the padded WAV next to *audio_path*, or ``None`` when the
        original is long enough or ``wave.open`` cannot read the header.

    """
    try:
        with wave.open(str(audio_path), "rb") as reader:
            rate = reader.getframerate()
            n_frames = reader.getnframes()
            sample_width = reader.getsampwidth()
            channels = reader.getnchannels()
            if rate <= 0 or sample_width < 1 or channels < 1 or n_frames < 0:
                logger.debug(
                    "GigaAM min-duration check skipped for %s: invalid WAV fields",
                    audio_path,
                )
                return None
            duration_s = n_frames / float(rate)
            if duration_s >= _MIN_AUDIO_DURATION_S:
                return None
            pcm = reader.readframes(n_frames)
            target_frames = math.ceil(_MIN_AUDIO_DURATION_S * rate)
            pad_frames = max(0, target_frames - n_frames)
            zero_frame = b"\x00" * (sample_width * channels)
            padded_pcm = pcm + zero_frame * pad_frames
    except (OSError, EOFError, wave.Error) as exc:
        logger.debug(
            "GigaAM min-duration check skipped for %s: %s",
            audio_path,
            exc,
        )
        return None

    fd, name = tempfile.mkstemp(
        prefix="pad_",
        suffix=".wav",
        dir=str(audio_path.parent),
    )
    padded_path = Path(name)
    try:
        os.close(fd)
        _write_pcm_wav(
            padded_path,
            padded_pcm,
            rate=rate,
            sample_width=sample_width,
            channels=channels,
        )
    except OSError:
        _unlink_padded_wav(padded_path)
        raise
    return padded_path


def _call_model_transcribe(model: object, path: str) -> object:
    """Run ``model.transcribe`` and map lazy remote-code import failures.

    Args:
        model: Loaded GigaAM model.
        path: Audio path handed to the model.

    Returns:
        Raw ``.transcribe`` result.

    """
    # * First .transcribe() can still ImportError on lazy remote-code deps.
    try:
        return cast("Any", model).transcribe(path)
    except ImportError as exc:
        _raise_missing_extra(exc)


def _log_chunk_progress(
    *,
    idx: int,
    offset_s: float,
    started_at: float,
    processed_s: float,
) -> None:
    """Log window progress: info on the first and every 10th window, debug otherwise."""
    elapsed_s = time.perf_counter() - started_at
    rtf = elapsed_s / processed_s if processed_s > 0 else 0.0
    log = logger.info if idx % _CHUNK_LOG_INTERVAL == 0 else logger.debug
    log(
        "GigaAM CTC window idx=%s offset_s=%.2f elapsed_s=%.2f rtf=%.2f",
        idx,
        offset_s,
        elapsed_s,
        rtf,
    )


def _numpy() -> Any:  # noqa: ANN401 - the numpy module, imported lazily
    """Import numpy on first use so importing this module stays light."""
    return importlib.import_module("numpy")


def _decoded_text(entry: object) -> str:
    """Return the text of one remote-code CTC decoder entry.

    v3 ``e2e_ctc`` decodes to plain strings; the Multilingual ``ctc`` revision
    decodes to ``(text, token_ids, token_frames)`` tuples.
    """
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (tuple, list)) and entry:
        return str(entry[0])
    text = getattr(entry, "text", None)
    return str(text) if text is not None else str(entry)


def _canonical_format(reader: wave.Wave_read, *, window_s: float) -> _WavFormat:
    """Return the stream format of a 16 kHz mono 16-bit PCM WAV.

    Args:
        reader: Open WAV reader.
        window_s: Window length in seconds.

    Returns:
        Stream format with ``window_frames`` for *window_s*.

    Raises:
        _UnparseableWavError: When the header is not streamable or the format
            is not 16 kHz mono 16-bit PCM (which only ffmpeg can resample).

    """
    fmt = _format_from_reader(reader)
    if (fmt.rate, fmt.channels, fmt.sample_width) != (
        GIGAAM_SAMPLE_RATE,
        1,
        _PCM16_SAMPLE_WIDTH,
    ):
        msg = (
            f"WAV is not {GIGAAM_SAMPLE_RATE} Hz mono 16-bit PCM "
            f"(rate={fmt.rate}, channels={fmt.channels}, "
            f"sample_width={fmt.sample_width})"
        )
        raise _UnparseableWavError(msg)
    return fmt._replace(window_frames=int(window_s * fmt.rate))


def _make_window(
    index: int,
    *,
    start_frame: int,
    pcm: bytes,
    n_frames: int,
    rate: int,
) -> WavWindow:
    """Convert one PCM16 mono window to float32 samples with the STFT padding.

    Mirrors the default path: a window shorter than ``_MIN_AUDIO_DURATION_S``
    gets trailing zero frames up to ``ceil(_MIN_AUDIO_DURATION_S * rate)``.
    """
    np_mod = _numpy()
    pcm16 = np_mod.frombuffer(pcm[: n_frames * _PCM16_SAMPLE_WIDTH], dtype="<i2")
    samples = pcm16.astype(np_mod.float32) / np_mod.float32(_PCM16_SCALE)
    if n_frames / float(rate) < _MIN_AUDIO_DURATION_S:
        target_frames = math.ceil(_MIN_AUDIO_DURATION_S * rate)
        pad_frames = max(0, target_frames - n_frames)
        samples = np_mod.concatenate(
            [samples, np_mod.zeros(pad_frames, dtype=np_mod.float32)]
        )
    return WavWindow(
        index=index,
        start_s=start_frame / float(rate),
        end_s=(start_frame + n_frames) / float(rate),
        samples=cast("NDArray[np.float32]", samples),
    )


def _iter_reader_windows(
    reader: wave.Wave_read, fmt: _WavFormat
) -> Iterator[WavWindow]:
    """Yield the windows of an open canonical WAV with the default-path boundaries.

    A file shorter than one window is a single window (the default path's
    original-file fast path); longer files are consecutive full windows and a
    shorter tail, exactly like the default path's chunk files.
    """
    pcm = reader.readframes(fmt.window_frames)
    n_frames = _pcm_frame_count(
        pcm, sample_width=fmt.sample_width, channels=fmt.channels
    )
    if n_frames < fmt.window_frames:
        yield _make_window(0, start_frame=0, pcm=pcm, n_frames=n_frames, rate=fmt.rate)
        return
    start_frame = 0
    index = 0
    while pcm:
        yield _make_window(
            index, start_frame=start_frame, pcm=pcm, n_frames=n_frames, rate=fmt.rate
        )
        start_frame += n_frames
        index += 1
        pcm = reader.readframes(fmt.window_frames)
        n_frames = _pcm_frame_count(
            pcm, sample_width=fmt.sample_width, channels=fmt.channels
        )


def _iter_wav_windows_from(path: Path, window_s: float) -> Iterator[WavWindow]:
    """Stream windows of *path*, mapping parse failures to ``UnsupportedWavError``."""
    try:
        with wave.open(str(path), "rb") as reader:
            fmt = _canonical_format(reader, window_s=window_s)
            yield from _iter_reader_windows(reader, fmt)
    except (wave.Error, EOFError, _UnparseableWavError) as exc:
        msg = f"Cannot stream {path} as {GIGAAM_SAMPLE_RATE} Hz mono 16-bit PCM WAV"
        raise UnsupportedWavError(msg) from exc


def iter_wav_windows(path: Path, *, window_s: float = _WINDOW_S) -> Iterator[WavWindow]:
    """Stream a prepared 16 kHz mono 16-bit PCM WAV as GigaAM windows.

    The windows have the same boundaries, timestamps, and tail padding as the
    wrapper's default :meth:`GigaAMCtcWrapper.transcribe` path, and their
    samples equal what the remote ``load_audio`` decodes for each window. The
    file is read lazily, one window at a time; close the iterator (or exhaust
    it) to release the file handle early.

    Args:
        path: Prepared WAV (for example the output of ``prepare_audio``).
        window_s: Window length in seconds (the wrapper uses 20 s;
            :meth:`GigaAMCtcWrapper.transcribe_window_batch` accepts at most
            20 s per window).

    Returns:
        Iterator over :class:`WavWindow` items; an empty WAV yields one padded
        window ending at 0.0 s.

    Raises:
        ValueError: When *window_s* is not a positive finite number.

    Note:
        Iteration raises :class:`UnsupportedWavError` when the file is not a
        parseable 16 kHz mono 16-bit PCM WAV, and ``OSError`` when it cannot
        be opened.

    """
    if not (math.isfinite(window_s) and window_s > 0):
        msg = f"window_s must be a positive finite number (got {window_s!r})"
        raise ValueError(msg)
    return _iter_wav_windows_from(Path(path), float(window_s))


def _validated_window(window: object, position: int) -> NDArray[np.float32]:
    """Return *window* as a 1-D float32 array within the lock-free sample bounds.

    Raises:
        ValueError: When the window is not 1-D or its length is outside
            ``[_MIN_WINDOW_SAMPLES, _MAX_WINDOW_SAMPLES]``.

    """
    np_mod = _numpy()
    array = np_mod.asarray(window, dtype=np_mod.float32)
    if array.ndim != 1:
        msg = f"window {position} must be 1-D (got shape {array.shape})"
        raise ValueError(msg)
    size = int(array.shape[0])
    if not _MIN_WINDOW_SAMPLES <= size <= _MAX_WINDOW_SAMPLES:
        msg = (
            f"window {position} has {size} samples; expected "
            f"{_MIN_WINDOW_SAMPLES}..{_MAX_WINDOW_SAMPLES} at {GIGAAM_SAMPLE_RATE} Hz"
        )
        raise ValueError(msg)
    return cast("NDArray[np.float32]", array)


def _optional_positive_int(name: str, value: int | None) -> int | None:
    """Validate an optional positive integer option.

    Raises:
        ValueError: When *value* is not ``None`` and not an integer >= 1.

    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer or None (got {value!r})"
        raise ValueError(msg)
    return value


def _require_strict_device_token(device: str) -> None:
    """Reject a non-explicit device for strict device mode.

    Raises:
        ValueError: When *device* is not ``cpu`` or ``cuda``.

    """
    token = str(device).strip().lower()
    if token not in STRICT_DEVICE_REQUESTS:
        msg = (
            "Strict device mode requires an explicit device 'cpu' or 'cuda' "
            f"(got device={device!r})"
        )
        raise ValueError(msg)


class GigaAMCtcWrapper:
    """Load GigaAM CTC on CPU, transcribe, and release resources on close.

    Concurrency contract: with ``threadsafe=True`` and after
    :meth:`ensure_loaded`, :meth:`transcribe` and
    :meth:`transcribe_window_batch` may be called from several threads on one
    CPU-resident model. CUDA inference calls are serialized per wrapper (the
    project never runs concurrent GigaAM CUDA inference). :meth:`unload` must
    not overlap inference calls.
    """

    # * Provider served by this wrapper; selects device validation (CPU-only here).
    PROVIDER: ClassVar[str] = STT_PROVIDER_GIGAAM_CTC

    def __init__(
        self,
        *,
        device: str = "cpu",
        cache_dir: Path | None = None,
        strict_device: bool = False,
        torch_threads: int | None = None,
        threadsafe: bool = False,
        warmup_batch: int = 1,
    ) -> None:
        """Initialize the wrapper without loading model weights.

        Args:
            device: Requested device. The CPU-only head accepts ``cpu`` /
                ``auto`` (both mean CPU); a CUDA-capable subclass accepts
                ``auto`` / ``cpu`` / ``cuda`` and resolves it at model load.
            cache_dir: Optional Hugging Face cache directory override.
            strict_device: Require an explicit ``cpu``/``cuda`` device and
                never fall back to CPU: an unusable CUDA runtime raises
                :class:`~core.stt_errors.DeviceUnavailableError` before the
                weights load, and every CUDA out-of-memory error (model move,
                window, batch) raises
                :class:`~core.stt_errors.CudaOutOfMemoryError`.
            torch_threads: When set, ``torch.set_num_threads`` is applied once
                per model load (process-wide setting).
            threadsafe: Route :meth:`transcribe` through the lock-free
                per-window ``forward``/``decode`` path and warm the model up
                in :meth:`ensure_loaded`.
            warmup_batch: Batch size of the extra CUDA warm-up batch run by
                :meth:`ensure_loaded` in thread-safe mode (the maximum number
                of windows a caller passes to :meth:`transcribe_window_batch`).

        Raises:
            ValueError: When *device* is not valid for :attr:`PROVIDER` (in
                strict mode: not an explicit ``cpu``/``cuda`` accepted by the
                provider) or an option is out of range.

        """
        # * Reject invalid devices before any model download or import.
        if strict_device:
            _require_strict_device_token(device)
        self.device = resolve_gigaam_requested_device(self.PROVIDER, device)
        self.cache_dir = cache_dir
        self.strict_device = bool(strict_device)
        self.torch_threads = _optional_positive_int("torch_threads", torch_threads)
        self.threadsafe = bool(threadsafe)
        checked_batch = _optional_positive_int("warmup_batch", warmup_batch)
        self.warmup_batch = 1 if checked_batch is None else checked_batch
        self._model: Any | None = None
        self._torch: Any | None = None
        self._active_device: str | None = None
        self._cuda_used = False
        self._lockfree: _LockFreeModel | None = None
        # * Serializes lazy loading (and the thread-safe warm-up) per wrapper.
        self._load_lock = threading.Lock()
        # * Serializes CUDA inference of the lock-free path per wrapper.
        self._cuda_infer_lock = threading.Lock()

    @property
    def active_device(self) -> str | None:
        """Device of the resident model, or ``None`` when no model is loaded."""
        return self._active_device

    def ensure_loaded(self) -> None:
        """Load the model once; in thread-safe mode also warm it up.

        The thread-safe warm-up runs single-threaded under the load lock before
        any caller can reach the lock-free path: one full-length 20 s zero
        window and, on CUDA, one zero batch of ``warmup_batch`` windows. The
        first encoder forward creates the remote code's lazy rotary
        ``pos_enc.pe`` buffer (sized by ``pos_emb_max_len``, far above the 20 s
        window); later forwards only read it.

        Raises:
            DeviceUnavailableError: Strict ``cuda`` without a usable CUDA runtime.
            CudaOutOfMemoryError: Strict CUDA load/move, or any CUDA warm-up,
                ran out of memory (the model is discarded).

        """
        if self.threadsafe:
            self._lockfree_model()
            return
        self._load_model_locked()

    def _load_model_locked(self) -> None:
        """Run :meth:`_load_model` under the per-wrapper load lock."""
        if self._model is not None:
            return
        with self._load_lock:
            self._load_model()

    def _prepare_load(self, torch_mod: object) -> None:
        """Apply the strict device check and torch threads before weights load.

        Args:
            torch_mod: The ``torch`` module returned by :meth:`_ensure_deps`.

        """
        if self.strict_device:
            # * Raises DeviceUnavailableError before any CPU weight load.
            select_gigaam_device(self.device, torch_mod, strict=True)
        if self.torch_threads is not None:
            set_num_threads = getattr(torch_mod, "set_num_threads", None)
            if callable(set_num_threads):
                set_num_threads(self.torch_threads)

    def _ensure_deps(self) -> tuple[Any, Any]:
        """Import Transformers / torch or raise a useful missing-extra error.

        Returns:
            ``(AutoModel, torch)`` modules/classes.

        Raises:
            RuntimeError: When the optional ``gigaam`` extra is not installed.

        """
        try:
            torch = importlib.import_module("torch")
            transformers_mod = importlib.import_module("transformers")
            auto_model_cls = getattr(transformers_mod, "AutoModel", None)
        except ImportError as exc:  # pragma: no cover - exercised via unit mock
            _raise_missing_extra(exc)
        if auto_model_cls is None:
            raise RuntimeError(_MISSING_EXTRA_MSG)
        return auto_model_cls, torch

    def _load_model(self) -> None:
        """Load the pinned GigaAM CTC revision onto CPU when not already loaded."""
        if self._model is not None:
            return
        auto_model_cls, torch_mod = self._ensure_deps()
        self._prepare_load(torch_mod)
        kwargs: dict[str, Any] = {
            "revision": GIGAAM_HF_REVISION,
            "trust_remote_code": True,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        logger.info(
            "Loading GigaAM CTC: repo=%s revision=%s device=cpu",
            GIGAAM_HF_REPO_ID,
            GIGAAM_HF_REVISION,
        )
        # * Remote-code modeling may ImportError on sentencepiece/pyannote even
        # * after torch/transformers import; map that to the same install hint.
        try:
            model = auto_model_cls.from_pretrained(GIGAAM_HF_REPO_ID, **kwargs)
        except ImportError as exc:
            _raise_missing_extra(exc)
        if hasattr(model, "eval"):
            model.eval()
        self._install_loaded_model(model, torch_mod)

    def _install_loaded_model(self, model: object, torch_mod: object) -> None:
        """Make a freshly loaded CPU model resident and apply the device request.

        Args:
            model: Model loaded onto CPU.
            torch_mod: The ``torch`` module returned by :meth:`_ensure_deps`.

        """
        self._model = model
        self._torch = torch_mod
        self._active_device = "cpu"
        self._place_on_requested_device()

    def _place_on_requested_device(self) -> None:
        """Move the resident CPU model to CUDA when the device request allows it.

        A CUDA out-of-memory error during the move returns the whole model to
        CPU, so parameters never stay split across devices. Any other move
        failure drops the model and propagates. In strict device mode the
        out-of-memory error drops the model too and raises
        :class:`~core.stt_errors.CudaOutOfMemoryError`.

        Raises:
            Exception: Non-OOM failure of ``model.to("cuda")``.
            CudaOutOfMemoryError: Strict mode, CUDA out of memory on the move.
            DeviceUnavailableError: Strict ``cuda`` without a usable CUDA runtime.

        """
        model = self._model
        if model is None:
            return
        try:
            target = select_gigaam_device(
                self.device, self._torch, strict=self.strict_device
            )
        except DeviceUnavailableError:
            self._discard_model()
            raise
        if target != "cuda":
            return
        self._cuda_used = True
        try:
            moved = model.to("cuda")
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                # ! Never keep a half-moved model resident.
                self._discard_model()
                raise
            if self.strict_device:
                # * Strict mode never falls back to CPU: drop the half-moved
                # * model; the caller decides what to do with the device.
                del model
                self._discard_model()
                msg = (
                    "GigaAM: CUDA out of memory while moving the model to CUDA "
                    f"(strict device=cuda): {type(exc).__name__}: {exc}"
                )
                raise CudaOutOfMemoryError(msg) from exc
            reason = f"{type(exc).__name__}: {exc}"
        else:
            self._model = moved
            self._active_device = "cuda"
            logger.info("GigaAM model moved to CUDA (requested device=%s)", self.device)
            return
        # * Outside the except block the OOM traceback no longer pins CUDA tensors.
        logger.warning(
            "GigaAM: CUDA out of memory while moving the model (%s); using CPU.",
            reason[:300],
        )
        self._move_resident_model_to_cpu()

    def _move_resident_model_to_cpu(self) -> object:
        """Move the resident model (fully) to CPU and release cached VRAM.

        Returns:
            The resident CPU model.

        Raises:
            RuntimeError: When no model is loaded, or when tensors remain on a
                non-CPU device after the move.

        """
        model = self._model
        if model is None:
            msg = "GigaAM model is not loaded"
            raise RuntimeError(msg)
        model = model.to("cpu")
        self._model = model
        self._active_device = "cpu"
        self._lockfree = None
        release_cuda_cache(self._torch)
        device_types = module_device_types(model)
        if device_types is not None and device_types != {"cpu"}:
            self._discard_model()
            msg = f"GigaAM model still holds tensors on {sorted(device_types)}"
            raise RuntimeError(msg)
        return model

    def _discard_model(self) -> None:
        """Drop the resident model reference and release cached VRAM."""
        self._model = None
        self._lockfree = None
        self._active_device = None
        release_cuda_cache(self._torch)

    def unload(self, *, safe: bool = True) -> None:  # noqa: ARG002 - API parity
        """Release the resident model, cached VRAM when CUDA was used, and GC.

        Args:
            safe: Retained for WhisperXWrapper API parity; unused for GigaAM.

        """
        used_cuda = self._cuda_used
        self._lockfree = None
        if self._model is not None:
            with contextlib.suppress(Exception):
                del self._model
            self._model = None
        self._active_device = None
        self._cuda_used = False
        if used_cuda:
            release_cuda_cache(self._torch)
            return
        with contextlib.suppress(Exception):
            _gc.collect()

    def close(self) -> None:
        """Unload resources held by this wrapper."""
        self.unload(safe=False)

    def bound_cufft_plan_cache(self, max_size: int) -> int | None:
        """Bound the cuFFT plan cache of the resident CUDA model's device.

        The mel front-end runs ``torch.stft`` on CUDA, and every distinct
        input length (batch rows times frames) adds one cached cuFFT plan;
        the bound keeps a long-lived process from accumulating plans.

        Args:
            max_size: Most cached plans (a positive integer).

        Returns:
            The bound in effect, or ``None`` when the model is not resident on
            CUDA or the bound could not be applied.

        Raises:
            ValueError: When *max_size* is not a positive integer.

        """
        if self._active_device != "cuda" or self._torch is None:
            return None
        return limit_cufft_plan_cache(self._torch, max_size)

    def release_cached_cuda_memory(self) -> bool:
        """Return the cached CUDA blocks of an idle resident model to the driver.

        Runs ``gc.collect()`` and ``torch.cuda.empty_cache()`` while holding the
        per-wrapper CUDA inference lock, so it never overlaps an inference
        call of the lock-free path; the caller still calls it only while no
        work is in flight.

        Returns:
            ``True`` when the release ran, ``False`` when the model never used
            CUDA (nothing is cached on a device).

        """
        if not self._cuda_used or self._torch is None:
            return False
        with self._cuda_infer_lock:
            release_cuda_cache(self._torch)
        return True

    @staticmethod
    def _coerce_text(raw: object) -> str:
        """Normalize a GigaAM ``.transcribe`` return value to plain text.

        Args:
            raw: Value returned by the model.

        Returns:
            Stripped transcript text.

        """
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, dict):
            text = raw.get("text", raw)
            return str(text).strip()
        return str(raw).strip()

    def _transcribe_file(self, audio_path: Path) -> str:
        """Run one window-sized ``model.transcribe`` call.

        Files shorter than ``_MIN_AUDIO_DURATION_S`` are copied with trailing
        silence so GigaAM's ``center=False`` STFT (n_fft=320 @16 kHz) has
        enough samples after resample.

        Args:
            audio_path: Audio file path accepted by GigaAM.

        Returns:
            Transcript text for that file.

        """
        self._load_model_locked()
        model = self._model
        if model is None:
            msg = "GigaAM CTC model failed to load"
            raise RuntimeError(msg)
        padded_path = _maybe_write_min_duration_wav(audio_path)
        transcribe_path = padded_path if padded_path is not None else audio_path
        try:
            raw = self._transcribe_window(model, str(transcribe_path))
            return self._coerce_text(raw)
        finally:
            if padded_path is not None:
                _unlink_padded_wav(padded_path)

    def _transcribe_window(self, model: object, path: str) -> object:
        """Transcribe one window, retrying it on CPU after a CUDA OOM.

        After the fallback the model stays on CPU, so later windows (and later
        files of a resident pipeline) run on CPU too. In strict device mode the
        window raises :class:`~core.stt_errors.CudaOutOfMemoryError` instead
        and the model stays resident on CUDA.

        Args:
            model: Resident model.
            path: Window audio path.

        Returns:
            Raw ``.transcribe`` result.

        Raises:
            CudaOutOfMemoryError: Strict mode, CUDA out of memory on the window.

        """
        try:
            return _call_model_transcribe(model, path)
        except Exception as exc:
            if self._active_device != "cuda" or not is_cuda_oom_error(exc):
                raise
            if self.strict_device:
                msg = (
                    "GigaAM: CUDA out of memory on a window (strict device=cuda): "
                    f"{type(exc).__name__}: {exc}"
                )
                raise CudaOutOfMemoryError(msg) from exc
            reason = f"{type(exc).__name__}: {exc}"
        # * Retry outside the except block so the OOM traceback no longer pins
        # * the window's CUDA activations before the cache is released.
        logger.warning(
            "GigaAM: CUDA out of memory on a window (%s); moving the model to "
            "CPU and retrying; later windows stay on CPU.",
            reason[:300],
        )
        return _call_model_transcribe(self._move_resident_model_to_cpu(), path)

    def _fast_path_original(
        self,
        audio_path: Path,
        *,
        n_frames: int,
        rate: int,
        acc: _TranscriptAccumulator,
    ) -> dict[str, Any]:
        """Transcribe a file shorter than one window via the original path."""
        started_at = time.perf_counter()
        text = self._transcribe_file(audio_path)
        actual_s = n_frames / float(rate)
        acc.emit(0.0, actual_s, text)
        _log_chunk_progress(
            idx=0,
            offset_s=0.0,
            started_at=started_at,
            processed_s=actual_s,
        )
        return acc.as_result()

    def _transcribe_chunked_windows(
        self,
        reader: wave.Wave_read,
        *,
        first_pcm: bytes,
        first_frames: int,
        fmt: _WavFormat,
        acc: _TranscriptAccumulator,
    ) -> dict[str, Any]:
        """Transcribe successive 20 s windows from an already open reader."""
        started_at = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="askvlm-gigaam-") as tmp:
            work = Path(tmp)
            pcm = first_pcm
            n_frames = first_frames
            start_frame = 0
            idx = 0
            while pcm:
                chunk_path = work / f"chunk_{idx:03d}.wav"
                _write_pcm_wav(
                    chunk_path,
                    pcm,
                    rate=fmt.rate,
                    sample_width=fmt.sample_width,
                    channels=fmt.channels,
                )
                piece = self._transcribe_file(chunk_path)
                start_s = start_frame / float(fmt.rate)
                end_s = (start_frame + n_frames) / float(fmt.rate)
                acc.emit(start_s, end_s, piece)
                _log_chunk_progress(
                    idx=idx,
                    offset_s=start_s,
                    started_at=started_at,
                    processed_s=end_s,
                )
                start_frame += n_frames
                idx += 1
                pcm = reader.readframes(fmt.window_frames)
                n_frames = _pcm_frame_count(
                    pcm,
                    sample_width=fmt.sample_width,
                    channels=fmt.channels,
                )
            return acc.as_result()

    def _stream_transcribe(
        self,
        audio_path: Path,
        *,
        on_segment: Callable[[dict[str, Any]], None] | None,
        progress: Callable[[float, str], None] | None,
    ) -> dict[str, Any]:
        """Run the unified 20 s window loop on *audio_path*."""
        with wave.open(str(audio_path), "rb") as reader:
            fmt = _format_from_reader(reader)
            acc = _TranscriptAccumulator(
                on_segment=on_segment,
                progress=progress,
                duration_s=_header_duration_seconds(reader),
            )
            first_pcm = reader.readframes(fmt.window_frames)
            first_frames = _pcm_frame_count(
                first_pcm,
                sample_width=fmt.sample_width,
                channels=fmt.channels,
            )
            # * First window hitting EOF means the whole file is shorter than 20 s.
            if first_frames < fmt.window_frames:
                return self._fast_path_original(
                    audio_path,
                    n_frames=first_frames,
                    rate=fmt.rate,
                    acc=acc,
                )
            return self._transcribe_chunked_windows(
                reader,
                first_pcm=first_pcm,
                first_frames=first_frames,
                fmt=fmt,
                acc=acc,
            )

    def _stream_after_reencode(
        self,
        audio_path: Path,
        *,
        on_segment: Callable[[dict[str, Any]], None] | None,
        progress: Callable[[float, str], None] | None,
    ) -> dict[str, Any]:
        """Re-encode *audio_path* once to PCM WAV and retry the window loop."""
        # * Lazy import so `import core.gigaam_ctc_wrapper` does not require ffmpeg-python.
        from core.ffmpeg import extract_audio  # noqa: PLC0415

        logger.warning(
            "GigaAM CTC could not stream %s as WAV; re-encoding to PCM 16 kHz mono",
            audio_path,
        )
        with tempfile.TemporaryDirectory(prefix="askvlm-gigaam-reencode-") as tmp:
            dst = Path(tmp) / "reencoded.wav"
            extract_audio(audio_path, dst, sample_rate=16000, channels=1)
            try:
                return self._stream_transcribe(
                    dst,
                    on_segment=on_segment,
                    progress=progress,
                )
            except (wave.Error, EOFError, _UnparseableWavError) as exc:
                msg = (
                    "GigaAM CTC could not parse WAV after re-encoding to "
                    "PCM 16-bit 16 kHz mono"
                )
                raise RuntimeError(msg) from exc

    # * ----------------------------------------------------------------------
    # * Lock-free inference path (thread-safe mode and window batches)
    # * ----------------------------------------------------------------------

    def _lockfree_model(self) -> _LockFreeModel:
        """Return the resolved ASR module, loading and warming it up once.

        Raises:
            CudaOutOfMemoryError: The CUDA warm-up ran out of memory (the model
                is discarded).

        """
        state = self._lockfree
        if state is not None:
            return state
        with self._load_lock:
            state = self._lockfree
            if state is None:
                self._load_model()
                state = self._resolve_lockfree_model()
                if self.threadsafe:
                    try:
                        self._warm_up(state)
                    except CudaOutOfMemoryError:
                        self._discard_model()
                        raise
                self._lockfree = state
        return state

    def _resolve_lockfree_model(self) -> _LockFreeModel:
        """Resolve the remote-code ``GigaAMASR`` module of the resident model.

        Raises:
            RuntimeError: When no model is loaded or the remote-code layout or
                sample rate differs from what the lock-free path relies on.

        """
        model = self._model
        if model is None:
            msg = "GigaAM CTC model failed to load"
            raise RuntimeError(msg)
        asr = getattr(model, "model", None)
        missing = [name for name in _ASR_ATTRS if not hasattr(asr, name)]
        if asr is None or missing:
            msg = (
                f"Unexpected GigaAM remote-code layout; missing {missing or ['model']}"
            )
            raise RuntimeError(msg)
        remote = sys.modules.get(type(asr).__module__)
        sample_rate = int(getattr(remote, "SAMPLE_RATE", GIGAAM_SAMPLE_RATE))
        if sample_rate != GIGAAM_SAMPLE_RATE:
            msg = (
                f"GigaAM remote code expects {sample_rate} Hz audio; the lock-free "
                f"path reads {GIGAAM_SAMPLE_RATE} Hz WAVs"
            )
            raise RuntimeError(msg)
        first_param = next(iter(asr.parameters()), None)
        if first_param is None:
            msg = "GigaAM remote-code ASR module has no parameters"
            raise RuntimeError(msg)
        return _LockFreeModel(
            asr=asr,
            torch=self._torch,
            device=first_param.device,
            dtype=first_param.dtype,
            cuda=str(first_param.device.type) == "cuda",
        )

    def _warm_up(self, state: _LockFreeModel) -> None:
        """Run the single-threaded warm-up forwards of the thread-safe mode."""
        np_mod = _numpy()
        window = np_mod.zeros(_MAX_WINDOW_SAMPLES, dtype=np_mod.float32)
        started_at = time.perf_counter()
        self._infer_windows(state, [window])
        if state.cuda and self.warmup_batch > 1:
            self._infer_windows(state, [window] * self.warmup_batch)
        if state.cuda:
            synchronize = getattr(
                getattr(state.torch, "cuda", None), "synchronize", None
            )
            if callable(synchronize):
                synchronize()
        logger.info(
            "GigaAM lock-free path warmed up (device=%s, batch=%s, %.2f s)",
            "cuda" if state.cuda else "cpu",
            self.warmup_batch if state.cuda else 1,
            time.perf_counter() - started_at,
        )

    def _infer_windows(
        self, state: _LockFreeModel, windows: Sequence[NDArray[np.float32]]
    ) -> list[str]:
        """Run one zero-padded batch through ``forward`` and CTC ``decode``.

        Mirrors the remote ``GigaAMASR.transcribe`` (``prepare_wav`` →
        ``forward`` → ``decoding.decode``) for every row, with per-call tensors
        only: the batch of one is exactly a single-window transcription. CUDA
        calls are serialized per wrapper.

        Args:
            state: Resolved ASR module.
            windows: 1-D float32 windows (validated by the caller).

        Returns:
            One stripped text per window.

        Raises:
            CudaOutOfMemoryError: CUDA ran out of memory (no fallback, no retry;
                the model stays resident).

        """
        np_mod = _numpy()
        torch = state.torch
        lengths = [int(window.shape[0]) for window in windows]
        batch = np_mod.zeros((len(windows), max(lengths)), dtype=np_mod.float32)
        for row, window in enumerate(windows):
            batch[row, : lengths[row]] = window
        guard: contextlib.AbstractContextManager[object] = (
            self._cuda_infer_lock if state.cuda else contextlib.nullcontext()
        )
        try:
            with guard, torch.inference_mode():
                wav = torch.from_numpy(batch).to(state.device).to(state.dtype)
                wav_lengths = torch.tensor(lengths, dtype=torch.long).to(state.device)
                encoded, encoded_len = state.asr.forward(wav, wav_lengths)
                decoded = state.asr.decoding.decode(
                    state.asr.head, encoded, encoded_len
                )
        except Exception as exc:
            if not (state.cuda and is_cuda_oom_error(exc)):
                raise
            msg = (
                f"GigaAM: CUDA out of memory on a batch of {len(windows)} "
                f"window(s): {type(exc).__name__}: {exc}"
            )
            raise CudaOutOfMemoryError(msg) from exc
        return [self._coerce_text(_decoded_text(entry)) for entry in decoded]

    def transcribe_window_batch(self, windows: Sequence[NDArray[Any]]) -> list[str]:
        """Transcribe several windows in one zero-padded forward pass.

        The batch is zero-padded to its longest window and carries the real
        lengths, like the remote code's own collate. A batch of one equals the
        thread-safe single-window result bit for bit; longer batches can
        differ slightly at the right edge of shorter windows (the striding
        sub-sampling convolutions see the zero padding). Loads the model on
        first use (see :meth:`ensure_loaded`).

        Args:
            windows: 1-D 16 kHz float32 windows (for example
                :attr:`WavWindow.samples`), each between the STFT duration
                floor (640 samples) and 20 s.

        Returns:
            One stripped text per window, in input order.

        Raises:
            ValueError: When a window is not 1-D or its length is out of bounds.
            CudaOutOfMemoryError: CUDA ran out of memory (never retried on CPU).

        """
        if len(windows) == 0:
            return []
        arrays = [
            _validated_window(window, position)
            for position, window in enumerate(windows)
        ]
        return self._infer_windows(self._lockfree_model(), arrays)

    def _stream_lockfree(
        self,
        audio_path: Path,
        state: _LockFreeModel,
        *,
        on_segment: Callable[[dict[str, Any]], None] | None,
        progress: Callable[[float, str], None] | None,
    ) -> dict[str, Any]:
        """Run the 20 s window loop with lock-free inference on a canonical WAV."""
        with wave.open(str(audio_path), "rb") as reader:
            fmt = _canonical_format(reader, window_s=_WINDOW_S)
            acc = _TranscriptAccumulator(
                on_segment=on_segment,
                progress=progress,
                duration_s=_header_duration_seconds(reader),
            )
            started_at = time.perf_counter()
            for window in _iter_reader_windows(reader, fmt):
                text = self._infer_windows(state, [window.samples])[0]
                # * on_segment runs between windows, so a pipeline cancel stops here.
                acc.emit(window.start_s, window.end_s, text)
                _log_chunk_progress(
                    idx=window.index,
                    offset_s=window.start_s,
                    started_at=started_at,
                    processed_s=window.end_s,
                )
            return acc.as_result()

    def _transcribe_lockfree(
        self,
        audio_path: Path,
        *,
        on_segment: Callable[[dict[str, Any]], None] | None,
        progress: Callable[[float, str], None] | None,
    ) -> dict[str, Any]:
        """Transcribe through the lock-free path (thread-safe mode).

        A WAV that the stdlib reader cannot stream as 16 kHz mono 16-bit PCM is
        re-encoded once with ffmpeg, like the default path's fallback.
        """
        state = self._lockfree_model()
        try:
            return self._stream_lockfree(
                audio_path, state, on_segment=on_segment, progress=progress
            )
        except (wave.Error, EOFError, _UnparseableWavError) as exc:
            logger.debug("GigaAM lock-free WAV read failed for %s: %s", audio_path, exc)
        # * Lazy import so `import core.gigaam_ctc_wrapper` does not require ffmpeg-python.
        from core.ffmpeg import extract_audio  # noqa: PLC0415

        logger.warning(
            "GigaAM CTC could not stream %s as 16 kHz mono PCM WAV; re-encoding",
            audio_path,
        )
        with tempfile.TemporaryDirectory(prefix="askvlm-gigaam-reencode-") as tmp:
            dst = Path(tmp) / "reencoded.wav"
            extract_audio(audio_path, dst, sample_rate=GIGAAM_SAMPLE_RATE, channels=1)
            try:
                return self._stream_lockfree(
                    dst, state, on_segment=on_segment, progress=progress
                )
            except (wave.Error, EOFError, _UnparseableWavError) as exc:
                msg = (
                    "GigaAM CTC could not parse WAV after re-encoding to "
                    "PCM 16-bit 16 kHz mono"
                )
                raise RuntimeError(msg) from exc

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,  # noqa: ARG002 - API parity with Whisper
        *,
        on_segment: Callable[[dict[str, Any]], None] | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """Transcribe audio and return the pipeline ``{text, segments}`` contract.

        Whisper-only parameters (beam size, VAD, compute type, etc.) are not
        accepted here. *language* is ignored: GigaAM Multilingual CTC does not
        take a language hint on the official short-form API.

        The WAV is streamed in 20 s windows. A file shorter than one window is
        sent to the model on the original path, except clips below the STFT
        duration floor which use a silence-padded ``pad_*.wav`` copy; longer
        files are split into temporary window WAVs. Stdlib ``wave`` failures
        trigger one ffmpeg re-encode and a single retry.

        In thread-safe mode the same windows (boundaries, timestamps, tail
        padding, per-window stripped text, empty windows kept) are read
        straight from a 16 kHz mono 16-bit PCM WAV and run through the
        lock-free ``forward``/``decode`` path; any other WAV is re-encoded once
        to that format first. ``on_segment`` runs between windows, so its
        cancellation semantics are unchanged.

        Args:
            audio_path: Prepared WAV (or other path accepted by GigaAM).
            language: Unused; retained for call-site compatibility.
            on_segment: Optional callback invoked per emitted segment.
            progress: Optional coarse progress callback ``(fraction, stage)``.

        Returns:
            Mapping with ``text`` and ``segments`` (list of start/end/text).

        """
        if self.threadsafe:
            return self._transcribe_lockfree(
                audio_path, on_segment=on_segment, progress=progress
            )
        try:
            return self._stream_transcribe(
                audio_path,
                on_segment=on_segment,
                progress=progress,
            )
        except (wave.Error, EOFError, _UnparseableWavError):
            return self._stream_after_reencode(
                audio_path,
                on_segment=on_segment,
                progress=progress,
            )
