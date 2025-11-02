from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

from utils.logging import get_logger


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
        model_name: str = "pyannote/speaker-diarization-community-1",
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

    def _try_load(self) -> None:
        """Attempt to load pyannote pipeline lazily with safe fallbacks."""
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

        # * Move to device if possible
        try:
            if self.device == "cuda":
                pipe.to("cuda")
            elif self.device == "cpu":
                # * Enforce CUDA-only ML processing
                msg = "CUDA is required for ML processing, but CPU was requested."
                raise RuntimeError(msg)  # noqa: TRY301
        except Exception as exc:  # noqa: BLE001
            # * Device move is best-effort; log minimal info
            self._log.debug("Diarization device move failed: %s", exc)
        self._pipeline = pipe

    def diarize(self, audio_path: str) -> list[Segment]:
        """Perform diarization on audio and return list of segments.

        Returns empty list when pipeline is unavailable.
        """
        pipe = self._pipeline
        if pipe is None:
            return []
        try:
            annotation = pipe(audio_path)
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
