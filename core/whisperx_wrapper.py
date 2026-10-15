from __future__ import annotations

import contextlib
import gc as _gc
import importlib
import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from core.stt_errors import CudaOutOfMemoryError, DeviceUnavailableError
from core.whisper_batch import is_whisper_cuda_memory_error

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Optional heavy deps (loaded via importlib)
# * Ensure Hugging Face does not attempt symlinks on Windows (privilege issues)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    fw_mod = importlib.import_module("faster_whisper")
    fw_whisper_cls = getattr(fw_mod, "WhisperModel", None)
except ModuleNotFoundError:  # pragma: no cover
    fw_whisper_cls = None

try:
    torch_mod = importlib.import_module("torch")
except ModuleNotFoundError:  # pragma: no cover
    torch_mod = None  # type: ignore[assignment]

try:
    whisperx_mod = importlib.import_module("whisperx")
except ModuleNotFoundError:  # pragma: no cover
    whisperx_mod = None  # type: ignore[assignment]


# * Light wrapper around faster-whisper / whisperx alignment when available
# * VRAM thresholds (GiB) for auto model selection
_VRAM_THRESHOLD_LARGE_GB = 12.0
_VRAM_THRESHOLD_MEDIUM_GB = 8.0
# * Device tokens accepted by strict device mode (no ``auto``).
_STRICT_DEVICES = frozenset({"cpu", "cuda"})


def _strict_device_token(device: str) -> str:
    """Return the canonical explicit device for strict mode.

    Raises:
        ValueError: When *device* is not ``cpu`` or ``cuda``.

    """
    token = str(device).strip().lower()
    if token not in _STRICT_DEVICES:
        msg = (
            "Strict Whisper device mode requires an explicit device 'cpu' or "
            f"'cuda' (got device={device!r})"
        )
        raise ValueError(msg)
    return token


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


@dataclass
class AlignedWord:
    """A single aligned word with timestamps."""

    word: str
    start: float
    end: float


@dataclass
class AlignedSegment:
    """An aligned segment with per-word timing."""

    text: str
    start: float
    end: float
    words: list[AlignedWord]


class WhisperXWrapper:
    """Load model, transcribe, and optionally align with whisperx if installed.

    This wrapper prefers faster-whisper for speed and uses whisperx for alignment
    when available. All heavy deps are imported lazily.

    Concurrency contract: with ``strict_device=True``, ``num_workers=K`` and the
    model preloaded by :meth:`ensure_loaded`, up to K threads may call
    :meth:`transcribe` concurrently (faster-whisper runs them on CTranslate2
    replicas that share the weights). The default mode's CUDA→CPU recovery
    reloads the shared model and is not safe under concurrent calls.
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "auto",
        model_root: Path | None = None,
        *,
        strict_device: bool = False,
        cpu_threads: int | None = None,
        num_workers: int | None = None,
    ) -> None:
        """Store the configuration; the model loads lazily.

        Args:
            model_name: faster-whisper model name or path.
            device: ``auto``/``cuda``/``cpu`` (strict mode: ``cpu``/``cuda``).
            compute_type: CTranslate2 compute type, ``auto`` per device.
            model_root: Optional download root of the model.
            strict_device: Never fall back to CPU: ``cuda`` without a usable
                CUDA runtime raises
                :class:`~core.stt_errors.DeviceUnavailableError`, and a CUDA
                out-of-memory error at load or inference raises
                :class:`~core.stt_errors.CudaOutOfMemoryError`.
            cpu_threads: ``WhisperModel(cpu_threads=...)`` when set.
            num_workers: ``WhisperModel(num_workers=...)`` when set (the number
                of concurrent :meth:`transcribe` calls).

        Raises:
            ValueError: Strict mode without an explicit ``cpu``/``cuda``
                device, or a thread option that is not a positive integer.

        """
        self.model_name = model_name
        self.device = _strict_device_token(device) if strict_device else device
        self.compute_type = compute_type
        self.model_root = model_root
        self.strict_device = bool(strict_device)
        self.cpu_threads = _optional_positive_int("cpu_threads", cpu_threads)
        self.num_workers = _optional_positive_int("num_workers", num_workers)
        self._model: Any | None = None
        self._align_model: Any | None = None
        self._active_device: str | None = None
        self._active_compute_type: str | None = None
        # * Serializes lazy model loading per wrapper.
        self._load_lock = threading.Lock()

    @property
    def active_device(self) -> str | None:
        """Device of the loaded model, or ``None`` when no model is loaded."""
        return self._active_device

    def ensure_loaded(self) -> None:
        """Load the model once (thread-safe; a no-op when it is loaded).

        Raises:
            DeviceUnavailableError: Strict ``cuda`` without a usable CUDA runtime.
            CudaOutOfMemoryError: Strict CUDA load ran out of memory.

        """
        self._load_model()

    def loaded_model(self) -> Any:  # noqa: ANN401 - untyped faster-whisper model
        """Return the loaded faster-whisper ``WhisperModel``.

        Used by :func:`core.whisper_batch.transcribe_clips_batched`.

        Raises:
            RuntimeError: When the model is not loaded (call :meth:`ensure_loaded`).

        """
        model = getattr(self, "_model", None)
        if model is None:
            msg = "Whisper model is not loaded; call ensure_loaded() first"
            raise RuntimeError(msg)
        return model

    def _cuda_available(self) -> bool:
        return bool(
            torch_mod is not None
            and getattr(torch_mod, "cuda", None) is not None
            and torch_mod.cuda.is_available()
        )

    def _normalize_compute_type(self, device: str) -> str:
        requested = self.compute_type
        if requested == "auto":
            return "float16" if device == "cuda" else "int8"
        if device == "cpu" and requested in {"float16", "int8_float16"}:
            return "int8"
        return requested

    def _is_cuda_memory_error(self, exc: Exception) -> bool:
        return is_whisper_cuda_memory_error(exc)

    def _load_attempts(self, preferred_device: str | None = None) -> list[str]:
        if self.strict_device:
            return self._strict_load_attempts()
        if preferred_device is not None:
            return [preferred_device]
        if self.device == "cpu":
            return ["cpu"]
        if self.device in {"auto", "cuda"}:
            if self._cuda_available():
                return ["cuda", "cpu"]
            logging.getLogger(__name__).warning(
                "CUDA is unavailable; loading Whisper on CPU."
            )
            return ["cpu"]
        return [self.device]

    def _strict_load_attempts(self) -> list[str]:
        """Return the single strict load device, checking CUDA availability.

        Raises:
            DeviceUnavailableError: When ``cuda`` is requested and CUDA is unusable.

        """
        if self.device == "cuda" and not self._cuda_available():
            msg = (
                "CUDA is unavailable (strict device=cuda); Whisper is not loaded "
                "on CPU in strict mode"
            )
            raise DeviceUnavailableError(msg)
        return [self.device]

    def _model_init_kwargs(
        self, *, device_name: str, compute_type: str, download_root: str | None
    ) -> dict[str, object]:
        """Return ``WhisperModel`` keyword arguments; thread options only when set."""
        kwargs: dict[str, object] = {
            "device": device_name,
            "compute_type": compute_type,
            "download_root": download_root,
        }
        if self.cpu_threads is not None:
            kwargs["cpu_threads"] = self.cpu_threads
        if self.num_workers is not None:
            kwargs["num_workers"] = self.num_workers
        return kwargs

    def _load_model(self, *, preferred_device: str | None = None) -> None:
        # Defensive: some callers may reset internals; tolerate missing attrs
        if getattr(self, "_model", None) is not None:
            return
        lock = getattr(self, "_load_lock", None)
        with lock if lock is not None else contextlib.nullcontext():
            if getattr(self, "_model", None) is not None:
                return
            self._load_model_unlocked(preferred_device=preferred_device)

    def _load_model_unlocked(self, *, preferred_device: str | None) -> None:
        if fw_whisper_cls is None:
            msg = "faster-whisper is not installed"
            raise RuntimeError(msg)

        download_root = str(self.model_root) if self.model_root else None
        chosen_model = self.model_name
        logger = logging.getLogger(__name__)
        last_error: Exception | None = None
        attempts = self._load_attempts(preferred_device=preferred_device)

        for device_name in attempts:
            compute_type_final = self._normalize_compute_type(device_name)
            logger.info(
                "Using Whisper model: %s (device=%s, compute=%s)",
                chosen_model,
                device_name,
                compute_type_final,
            )
            try:
                self._model = fw_whisper_cls(
                    chosen_model,
                    **self._model_init_kwargs(
                        device_name=device_name,
                        compute_type=compute_type_final,
                        download_root=download_root,
                    ),
                )
            except Exception as exc:
                last_error = exc
                self._model = None
                self._active_device = None
                self._active_compute_type = None
                if (
                    device_name == "cuda"
                    and "cpu" in attempts
                    and self._is_cuda_memory_error(exc)
                ):
                    logger.warning(
                        "Whisper load hit CUDA memory pressure; retrying on CPU."
                    )
                    self.unload(safe=False)
                    continue
                if (
                    self.strict_device
                    and device_name == "cuda"
                    and self._is_cuda_memory_error(exc)
                ):
                    self.unload(safe=False)
                    msg = (
                        "Whisper load ran out of CUDA memory (strict device=cuda): "
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise CudaOutOfMemoryError(msg) from exc
                raise
            self._active_device = device_name
            self._active_compute_type = compute_type_final
            return

        if last_error is not None:
            raise last_error

    def _load_align_model(self) -> None:
        if self._align_model is None and whisperx_mod is not None:
            # whisperx module reference
            self._align_model = whisperx_mod

    # * Explicitly release heavy resources to avoid lingering VRAM usage between jobs
    def unload(self, *, safe: bool = True) -> None:
        """Release loaded models and free GPU memory (best effort).

        This method clears references to the underlying faster-whisper model
        and alignment backend and triggers Python/torch memory cleanup to reduce
        the chance of VRAM fragmentation or OOM across sequential jobs.
        """
        logger = logging.getLogger(__name__)
        logger.info(
            "WhisperXWrapper.unload: entering (safe=%s, device=%s)",
            safe,
            self.device,
        )
        # * Capture VRAM before unload for diagnostics
        _vram_before_mb: float | None = None
        if (
            self.device == "cuda"
            and torch_mod is not None
            and getattr(torch_mod, "cuda", None) is not None
        ):
            with contextlib.suppress(Exception):
                _vram_before_mb = torch_mod.cuda.memory_allocated() / (1024 * 1024)
        try:
            if getattr(self, "_model", None) is not None:
                logger.info("WhisperXWrapper.unload: before deleting self._model")
                with contextlib.suppress(Exception):
                    del self._model
            self._model = None
            self._align_model = None
            self._active_device = None
            self._active_compute_type = None
            logger.info("WhisperXWrapper.unload: after clearing model refs")
        finally:
            if not safe:
                with contextlib.suppress(Exception):
                    if (
                        self.device == "cuda"
                        and torch_mod is not None
                        and getattr(torch_mod, "cuda", None) is not None
                        and hasattr(torch_mod.cuda, "synchronize")
                    ):
                        logger.info(
                            "WhisperXWrapper.unload: before torch.cuda.synchronize()"
                        )
                        torch_mod.cuda.synchronize()
                        logger.info(
                            "WhisperXWrapper.unload: after torch.cuda.synchronize()"
                        )
            logger.info("WhisperXWrapper.unload: before gc.collect()")
            with contextlib.suppress(Exception):
                _gc.collect()
            logger.info("WhisperXWrapper.unload: after gc.collect()")
            # * Log VRAM delta to confirm model was released
            if (
                _vram_before_mb is not None
                and torch_mod is not None
                and getattr(torch_mod, "cuda", None) is not None
            ):
                with contextlib.suppress(Exception):
                    _vram_after_mb = torch_mod.cuda.memory_allocated() / (1024 * 1024)
                    _vram_freed_mb = _vram_before_mb - _vram_after_mb
                    logger.info(
                        "WhisperXWrapper.unload: vram_before_mb=%.1f vram_after_mb=%.1f freed_mb=%.1f",
                        _vram_before_mb,
                        _vram_after_mb,
                        _vram_freed_mb,
                    )
            if not safe:
                with contextlib.suppress(Exception):
                    if (
                        self.device == "cuda"
                        and torch_mod is not None
                        and getattr(torch_mod, "cuda", None) is not None
                    ):
                        logger.info(
                            "WhisperXWrapper.unload: before torch.cuda.empty_cache()"
                        )
                        torch_mod.cuda.empty_cache()
                        logger.info(
                            "WhisperXWrapper.unload: after torch.cuda.empty_cache()"
                        )
            logger.info("WhisperXWrapper.unload: finished")

    def _transcribe_once(
        self,
        *,
        audio_path: Path,
        language: str | None,
        on_segment: Callable[[dict[str, Any]], None] | None,
        progress: Callable[[float, str], None] | None,
        preferred_device: str | None,
        kwargs: dict[str, object],
    ) -> dict[str, Any]:
        self._load_model(preferred_device=preferred_device)
        model = self._model
        if model is None:
            msg = "Whisper model failed to load"
            raise RuntimeError(msg)
        segments_out: list[dict[str, Any]] = []
        text_parts: list[str] = []
        # * Marks an attribute that a segment object does not carry at all.
        absent = object()
        segments, _info = model.transcribe(str(audio_path), language=language, **kwargs)
        for seg in segments:
            start = float(seg.start)
            end = float(seg.end)
            txt = str(seg.text or "").strip()
            seg_dict: dict[str, Any] = {"start": start, "end": end, "text": txt}
            # * faster-whisper segments also carry scores and raw word timings;
            # * they are copied when present and never change ``text``.
            for score in ("no_speech_prob", "avg_logprob"):
                score_value: Any = getattr(seg, score, absent)
                if score_value is not absent and score_value is not None:
                    seg_dict[score] = float(score_value)
            words: Any = getattr(seg, "words", absent)
            if words is not absent:
                seg_dict["words"] = [
                    {
                        "word": str(word.word),
                        "start": float(word.start),
                        "end": float(word.end),
                        "probability": float(word.probability),
                    }
                    for word in words or []
                ]
            segments_out.append(seg_dict)
            if txt:
                text_parts.append(txt)
            # * Streaming callback for incremental handling (e.g., ETA/progress, partial files)
            if on_segment is not None:
                on_segment(seg_dict)
            if progress is not None and end > 0:
                # * Best-effort progress within transcription phase
                p = max(0.0, min(1.0, end))
                progress(p, "transcribe")
        return {"text": " ".join(text_parts).strip(), "segments": segments_out}

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        *,
        on_segment: Callable[[dict[str, Any]], None] | None = None,
        progress: Callable[[float, str], None] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        """Transcribe audio and return faster-whisper style result dict.

        Returns a dict with keys: text, segments (list of {start,end,text}).
        Segments of a faster-whisper model also carry ``no_speech_prob``,
        ``avg_logprob`` (floats) and ``words`` (raw
        ``{word,start,end,probability}`` dicts, ``[]`` without word timestamps).
        """
        # * Filter out non-faster-whisper kwargs (subtitle layout hints etc.)
        _kwargs_all = dict(kwargs)
        for _k in ("max_line_width", "max_line_count"):
            _kwargs_all.pop(_k, None)
        try:
            return self._transcribe_once(
                audio_path=audio_path,
                language=language,
                on_segment=on_segment,
                progress=progress,
                preferred_device=None,
                kwargs=_kwargs_all,
            )
        except Exception as exc:
            if self._active_device == "cuda" and self._is_cuda_memory_error(exc):
                if self.strict_device:
                    # * Strict mode: no unload, no CPU retry; the model stays loaded.
                    msg = (
                        "Whisper transcription ran out of CUDA memory "
                        f"(strict device=cuda): {type(exc).__name__}: {exc}"
                    )
                    raise CudaOutOfMemoryError(msg) from exc
                logging.getLogger(__name__).warning(
                    "Whisper transcription hit CUDA memory pressure; retrying on CPU."
                )
                self.unload(safe=False)
                return self._transcribe_once(
                    audio_path=audio_path,
                    language=language,
                    on_segment=on_segment,
                    progress=progress,
                    preferred_device="cpu",
                    kwargs=_kwargs_all,
                )
            raise

    def align(
        self,
        audio_path: Path,
        transcript_result: dict[str, Any],
        language: str | None = None,
    ) -> list[AlignedSegment]:
        """Align segments at word level using whisperx if available; otherwise return coarse segments."""
        self._load_align_model()
        segments = transcript_result.get("segments", [])
        # Fallback: no alignment available
        if self._align_model is None:
            return [
                AlignedSegment(
                    text=s.get("text", ""),
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                    words=[],
                )
                for s in segments
            ]

        device = (
            "cuda"
            if (
                torch_mod is not None
                and getattr(torch_mod, "cuda", None) is not None
                and torch_mod.cuda.is_available()
                and self.device == "cuda"
            )
            else "cpu"
        )

        wx = cast("Any", self._align_model)
        try:
            model_a, metadata = wx.load_align_model(
                language_code=language, device=device
            )
            aligned = wx.align(
                transcript_result,
                model_a,
                metadata,
                str(audio_path),
                device,
                return_char_alignments=False,
            )
        except (OSError, ValueError) as e:
            logging.getLogger(__name__).debug("Alignment fallback: %s", e)
            return [
                AlignedSegment(
                    text=s.get("text", ""),
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                    words=[],
                )
                for s in segments
            ]
        else:
            out: list[AlignedSegment] = []
            for seg in aligned.get("segments", []) or []:
                words: list[AlignedWord] = []
                for w in seg.get("words", []) or []:
                    try:
                        words.append(
                            AlignedWord(
                                word=str(w.get("word", "")),
                                start=float(w.get("start", 0.0)),
                                end=float(w.get("end", 0.0)),
                            )
                        )
                    except (TypeError, ValueError):
                        continue
                out.append(
                    AlignedSegment(
                        text=str(seg.get("text", "")),
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                        words=words,
                    )
                )
            return out
