from pathlib import Path
from typing import Optional

from editing.text_model import TextSegment, Document

from .audio_io import prepare_audio
from .whisper_wrapper import WhisperWrapper
from .diarization import DiarizationPipeline
from .llm_formatter import LLMFormatter


# * Orchestrate local speech-to-text processing pipeline
class LocalPipeline:
    """Pipeline for local processing: FFmpeg -> Whisper -> Pyannote -> LLM formatting."""

    def __init__(
        self,
        model_root: Optional[Path] = None,
        whisper_model: str = "base",
        llm_model: str = "gguf-q4_0",
    ) -> None:
        """Initialize pipeline components."""
        self.model_root = model_root
        self.whisper = WhisperWrapper(model_name=whisper_model, model_root=model_root)
        self.diarizer = DiarizationPipeline()
        self.formatter = LLMFormatter(model_name=llm_model, model_path=None)

    def process(
        self,
        input_path: Path,
        work_dir: Path = Path("."),
    ) -> Document:
        """Process a media file through the pipeline and return a formatted document."""
        # * Step 1: Ensure audio is in WAV format
        audio_path = prepare_audio(input_path, work_dir)

        # * Step 2: Transcribe audio to raw text
        raw_text = self.whisper.transcribe(audio_path)

        # * Step 3: Perform speaker diarization
        diarization_segments = self.diarizer.diarize(str(audio_path))

        # * Step 4: Format text using LLM
        formatted_text = self.formatter.format_text(raw_text)

        # * Build document (simple single-segment fallback)
        doc = Document()
        if diarization_segments:
            # ! Split formatted_text by segments based on timestamps will be implemented in Phase 2
            for segment in diarization_segments:
                doc.add_segment(
                    TextSegment(
                        speaker_id=segment.speaker,
                        start_time=segment.start,
                        end_time=segment.end,
                        text=formatted_text,
                    )
                )
        else:
            doc.add_segment(
                TextSegment(
                    speaker_id="speaker_1",
                    start_time=0.0,
                    end_time=0.0,
                    text=formatted_text,
                )
            )
        return doc
