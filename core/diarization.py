from dataclasses import dataclass
from typing import List


# * Represent a single speaker segment
@dataclass
class Segment:
    speaker: str
    start: float
    end: float


# * Pipeline for speaker diarization (pyannote.audio)
class DiarizationPipeline:
    """Stub for diarization pipeline using pyannote.audio."""

    def __init__(self, model_name: str = "pyannote/speaker-diarization"):
        """Initialize diarization pipeline (model loading to be implemented)."""
        self.model_name = model_name
        # ! Model loading will be implemented in Phase 2

    def diarize(self, audio_path: str) -> List[Segment]:
        """Perform diarization on audio and return list of segments."""
        # ! Actual diarization implementation will be added in Phase 2
        # * For now, return empty list as placeholder
        _ = audio_path  # * Acknowledge unused parameter
        return []
