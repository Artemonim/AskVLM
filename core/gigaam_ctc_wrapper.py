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
"""

from __future__ import annotations

import contextlib
import gc as _gc
import importlib
import logging
import math
import os
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, NoReturn, cast

from core.gigaam_device import (
    is_cuda_oom_error,
    module_device_types,
    release_cuda_cache,
    select_gigaam_device,
)
from core.stt_providers import (
    GIGAAM_HF_REPO_ID,
    GIGAAM_HF_REVISION,
    STT_PROVIDER_GIGAAM_CTC,
    resolve_gigaam_requested_device,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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


class GigaAMCtcWrapper:
    """Load GigaAM CTC on CPU, transcribe, and release resources on close."""

    # * Provider served by this wrapper; selects device validation (CPU-only here).
    PROVIDER: ClassVar[str] = STT_PROVIDER_GIGAAM_CTC

    def __init__(
        self,
        *,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the wrapper without loading model weights.

        Args:
            device: Requested device. The CPU-only head accepts ``cpu`` /
                ``auto`` (both mean CPU); a CUDA-capable subclass accepts
                ``auto`` / ``cpu`` / ``cuda`` and resolves it at model load.
            cache_dir: Optional Hugging Face cache directory override.

        Raises:
            ValueError: When *device* is not valid for :attr:`PROVIDER`.

        """
        # * Reject invalid devices before any model download or import.
        self.device = resolve_gigaam_requested_device(self.PROVIDER, device)
        self.cache_dir = cache_dir
        self._model: Any | None = None
        self._torch: Any | None = None
        self._active_device: str | None = None
        self._cuda_used = False

    @property
    def active_device(self) -> str | None:
        """Device of the resident model, or ``None`` when no model is loaded."""
        return self._active_device

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
        failure drops the model and propagates.

        Raises:
            Exception: Non-OOM failure of ``model.to("cuda")``.

        """
        model = self._model
        if model is None or select_gigaam_device(self.device, self._torch) != "cuda":
            return
        self._cuda_used = True
        try:
            moved = model.to("cuda")
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                # ! Never keep a half-moved model resident.
                self._discard_model()
                raise
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
        self._active_device = None
        release_cuda_cache(self._torch)

    def unload(self, *, safe: bool = True) -> None:  # noqa: ARG002 - API parity
        """Release the resident model, cached VRAM when CUDA was used, and GC.

        Args:
            safe: Retained for WhisperXWrapper API parity; unused for GigaAM.

        """
        used_cuda = self._cuda_used
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
        self._load_model()
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
        files of a resident pipeline) run on CPU too.

        Args:
            model: Resident model.
            path: Window audio path.

        Returns:
            Raw ``.transcribe`` result.

        """
        try:
            return _call_model_transcribe(model, path)
        except Exception as exc:
            if self._active_device != "cuda" or not is_cuda_oom_error(exc):
                raise
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

        Args:
            audio_path: Prepared WAV (or other path accepted by GigaAM).
            language: Unused; retained for call-site compatibility.
            on_segment: Optional callback invoked per emitted segment.
            progress: Optional coarse progress callback ``(fraction, stage)``.

        Returns:
            Mapping with ``text`` and ``segments`` (list of start/end/text).

        """
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
