from __future__ import annotations

import importlib
import os
import threading
from dataclasses import dataclass
from typing import Any

from utils.logging import get_logger

# * Global limiter: only one diarization runs at a time to protect VRAM
_DIARIZATION_SEMAPHORE = threading.Semaphore(1)


# * Represent a single speaker segment
@dataclass
class Segment:
    """A single diarized segment with speaker and timing."""

    speaker: str
    start: float
    end: float


# * Pipeline for speaker diarization (pyannote.audio)
class DiarizationPipeline:
    """Diarization pipeline using pyannote.audio when available.

    Falls back to returning an empty list if pyannote is not installed or
    if the Hugging Face token is missing for gated models.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.0",
        *,
        hf_token: str | None = None,
        device: str = "auto",  # "auto" | "cuda" | "cpu"
    ) -> None:
        """Initialize diarization pipeline.

        Args:
            model_name: Hugging Face model id for diarization pipeline.
            hf_token: Optional HF token (reads HF_TOKEN from env if not provided).
            device: Preferred device (auto/cuda/cpu).

        """
        self.model_name = model_name
        # Accept common env names for HF/pyannote tokens
        self.hf_token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("PYANNOTE_TOKEN")
            or os.getenv("PYANNOTE_AUTH_TOKEN")
            or ""
        )
        self.device = device
        self._pipeline: Any | None = None
        # Initialize logger before any method might use it
        self._log = get_logger(__name__)
        self._try_load()

    def _apply_torchaudio_shim(self) -> None:
        """Apply compatibility shim for torchaudio >= 2.1."""
        try:
            import sys  # noqa: PLC0415
            import types  # noqa: PLC0415
            from dataclasses import dataclass  # noqa: PLC0415
            from typing import cast  # noqa: PLC0415

            import torchaudio  # noqa: PLC0415

            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]

            if not hasattr(torchaudio, "set_audio_backend"):
                torchaudio.set_audio_backend = lambda _: None

            if not hasattr(torchaudio, "get_audio_backend"):
                torchaudio.get_audio_backend = lambda: "soundfile"

            if "torchaudio.backend" not in sys.modules:
                sys.modules["torchaudio.backend"] = types.ModuleType(
                    "torchaudio.backend"
                )
                # Inject into torchaudio namespace as well
                torchaudio.backend = sys.modules["torchaudio.backend"]

            if "torchaudio.backend.common" not in sys.modules:
                common = types.ModuleType("torchaudio.backend.common")
                sys.modules["torchaudio.backend.common"] = common
                torchaudio.backend.common = common

            # Define AudioMetaData if missing
            if not hasattr(torchaudio, "AudioMetaData"):

                @dataclass
                class AudioMetaData:
                    sample_rate: int
                    num_frames: int
                    num_channels: int
                    bits_per_sample: int
                    encoding: str

                torchaudio.AudioMetaData = AudioMetaData
                # Also inject into backend.common where it used to be
                cast(
                    "Any", sys.modules["torchaudio.backend.common"]
                ).AudioMetaData = AudioMetaData

        except ImportError:
            pass

    def _try_load(self) -> None:
        """Attempt to load pyannote pipeline lazily with safe fallbacks."""
        self._apply_torchaudio_shim()

        try:
            pa = importlib.import_module("pyannote.audio")
        except ModuleNotFoundError:
            # * pyannote is optional; skip loading
            self._pipeline = None
            self._log.info("pyannote.audio is not installed; skipping diarization")
            return

        try:
            pipeline_cls = pa.Pipeline
        except AttributeError:
            # * Unknown API version; skip loading
            self._pipeline = None
            return

        # * Try to create pipeline; token may be optional for community pipeline
        kwargs: dict[str, Any] = {}
        if self.hf_token:
            kwargs["use_auth_token"] = self.hf_token
        # Warn once if likely gated
        elif "pyannote/" in self.model_name:
            self._log.warning(
                "HF token is not set; gated model '%s' may fail to load. "
                "Set HF_TOKEN in your environment.",
                self.model_name,
            )
        try:
            pipe = pipeline_cls.from_pretrained(self.model_name, **kwargs)
        except Exception as exc:  # noqa: BLE001
            # ! If loading fails (e.g., gated model without token), skip diarization
            self._log.warning(
                "Could not load diarization pipeline '%s': %s. Skipping.",
                self.model_name,
                exc,
            )
            self._pipeline = None
            return

        self._move_to_device(pipe)
        self._pipeline = pipe

    def _move_to_device(self, pipe: Any) -> None:  # noqa: ANN401
        """Move pipeline to requested device (best effort)."""
        try:
            use_cuda = False
            if self.device == "cuda":
                use_cuda = True
            elif self.device == "auto":
                try:
                    import torch  # noqa: PLC0415

                    if torch.cuda.is_available():
                        use_cuda = True
                except ImportError:
                    pass

            if use_cuda:
                pipe.to("cuda")
            elif self.device == "cpu":
                # * Enforce CUDA-only ML processing
                msg = "CUDA is required for ML processing, but CPU was requested."
                raise RuntimeError(msg)  # noqa: TRY301
        except Exception as exc:  # noqa: BLE001
            # * Device move is best-effort; log minimal info
            self._log.debug("Diarization device move failed: %s", exc)

    def diarize(self, audio_path: str) -> list[Segment]:
        """Perform diarization on audio and return list of segments.

        Returns empty list when pipeline is unavailable.
        """
        pipe = self._pipeline
        if pipe is None:
            return []
        # * Serialize heavy diarization to avoid VRAM spikes when multiple
        # * jobs run in parallel (Fast mode). This limits concurrency to 1.
        try:
            _diar_lock_acquire = _DIARIZATION_SEMAPHORE.acquire
            _diar_lock_release = _DIARIZATION_SEMAPHORE.release
            _diar_lock_acquire()
            try:
                annotation = pipe(audio_path)
            finally:
                _diar_lock_release()
        except Exception as exc:  # noqa: BLE001
            # ! Runtime failure (e.g., missing backends), return empty
            self._log.warning("Diarization failed at runtime: %s", exc)
            return []

        out: list[Segment] = []
        try:
            # pyannote Annotation API
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                try:
                    out.append(
                        Segment(
                            speaker=str(speaker),
                            start=float(getattr(turn, "start", 0.0) or 0.0),
                            end=float(getattr(turn, "end", 0.0) or 0.0),
                        )
                    )
                except Exception:  # noqa: BLE001,S112
                    continue
        except Exception:  # noqa: BLE001
            return []
        return out
