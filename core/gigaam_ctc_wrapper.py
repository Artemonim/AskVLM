r"""Lazy CPU-only wrapper for GigaAM Multilingual CTC speech-to-text.

Uses the official Transformers API::

    AutoModel.from_pretrained(..., revision=\"ctc\", trust_remote_code=True)
    model.transcribe(path)

Results are normalized to the pipeline contract ``{\"text\", \"segments\"}``.
Upstream short-form ``.transcribe`` is limited to ~25 s; longer WAV files are
split into fixed external chunks so long-form media still works without
pyannote longform.
"""

from __future__ import annotations

import contextlib
import gc as _gc
import importlib
import logging
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from core.stt_providers import (
    GIGAAM_HF_REPO_ID,
    GIGAAM_HF_REVISION,
    resolve_gigaam_device,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# * Upstream short-form limit for model.transcribe (~25 s).
_SHORTFORM_LIMIT_S = 25.0
_CHUNK_S = 20.0

_MISSING_EXTRA_MSG = (
    "GigaAM Multilingual CTC requires the AskVLM ML extra "
    '`pip install -e ".[ml]"` '
    "(torch/torchaudio 2.10 via run.ps1/build.ps1 CUDA ensure, transformers 5, "
    "hydra-core, omegaconf, sentencepiece, pyannote.audio)."
)


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


class GigaAMCtcWrapper:
    """Load GigaAM CTC on CPU, transcribe, and release resources on close."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the wrapper without loading model weights.

        Args:
            device: Requested device; only ``cpu`` / ``auto`` are accepted.
            cache_dir: Optional Hugging Face cache directory override.

        Raises:
            ValueError: When *device* is not CPU-compatible.

        """
        # * Reject invalid devices before any model download or import.
        self.device = resolve_gigaam_device(device)
        self.cache_dir = cache_dir
        self._model: Any | None = None

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
        auto_model_cls, _torch = self._ensure_deps()
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
        # * Keep weights on CPU; do not call .to("cuda").
        if hasattr(model, "eval"):
            model.eval()
        self._model = model

    def unload(self, *, safe: bool = True) -> None:  # noqa: ARG002 - API parity
        """Release the resident model and run a best-effort GC pass.

        Args:
            safe: Retained for WhisperXWrapper API parity; unused for CPU GigaAM.

        """
        if self._model is not None:
            with contextlib.suppress(Exception):
                del self._model
            self._model = None
        with contextlib.suppress(Exception):
            _gc.collect()

    def close(self) -> None:
        """Unload resources held by this wrapper."""
        self.unload(safe=False)

    @staticmethod
    def _wav_duration_seconds(path: Path) -> float:
        """Return WAV duration in seconds via the stdlib wave module.

        Args:
            path: Path to a PCM WAV file.

        Returns:
            Duration in seconds (0.0 when the file has no frames).

        """
        with wave.open(str(path), "rb") as reader:
            frames = reader.getnframes()
            rate = reader.getframerate()
        if rate <= 0:
            return 0.0
        return float(frames) / float(rate)

    @staticmethod
    def _slice_wav(src: Path, dst: Path, start_s: float, dur_s: float) -> None:
        """Cut a mono/stereo PCM WAV slice into *dst*.

        Args:
            src: Source WAV path.
            dst: Destination WAV path.
            start_s: Slice start in seconds.
            dur_s: Slice duration in seconds.

        """
        with wave.open(str(src), "rb") as reader:
            rate = reader.getframerate()
            width = reader.getsampwidth()
            channels = reader.getnchannels()
            start_frame = int(start_s * rate)
            n_frames = max(0, int(dur_s * rate))
            reader.setpos(start_frame)
            frames = reader.readframes(n_frames)
        with wave.open(str(dst), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(width)
            writer.setframerate(rate)
            writer.writeframes(frames)

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
        """Run one short-form ``model.transcribe`` call.

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
        # * First .transcribe() can still ImportError on lazy remote-code deps.
        try:
            return self._coerce_text(model.transcribe(str(audio_path)))
        except ImportError as exc:
            _raise_missing_extra(exc)

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

        Args:
            audio_path: Prepared WAV (or other path accepted by GigaAM).
            language: Unused; retained for call-site compatibility.
            on_segment: Optional callback invoked per emitted segment.
            progress: Optional coarse progress callback ``(fraction, stage)``.

        Returns:
            Mapping with ``text`` and ``segments`` (list of start/end/text).

        """
        duration_s = self._wav_duration_seconds(audio_path)
        segments_out: list[dict[str, Any]] = []
        text_parts: list[str] = []

        def _emit(start: float, end: float, text: str) -> None:
            seg = {"start": start, "end": end, "text": text}
            segments_out.append(seg)
            if text:
                text_parts.append(text)
            if on_segment is not None:
                on_segment(seg)
            if progress is not None and duration_s > 0:
                progress(max(0.0, min(1.0, end / duration_s)), "transcribe")

        if duration_s <= _SHORTFORM_LIMIT_S + 0.05:
            text = self._transcribe_file(audio_path)
            _emit(0.0, duration_s, text)
        else:
            with tempfile.TemporaryDirectory(prefix="askvlm-gigaam-") as tmp:
                work = Path(tmp)
                start = 0.0
                idx = 0
                while start < duration_s - 1e-6:
                    dur = min(_CHUNK_S, duration_s - start)
                    chunk_path = work / f"chunk_{idx:03d}.wav"
                    self._slice_wav(audio_path, chunk_path, start, dur)
                    piece = self._transcribe_file(chunk_path)
                    _emit(start, start + dur, piece)
                    start += _CHUNK_S
                    idx += 1

        return {"text": " ".join(text_parts).strip(), "segments": segments_out}
