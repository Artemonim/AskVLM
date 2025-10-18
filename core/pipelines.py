from pathlib import Path

from editing.text_model import Document, TextSegment

from .audio_io import prepare_audio
from .diarization import DiarizationPipeline
from .llm_formatter import LLMFormatter
from .whisper_wrapper import WhisperWrapper
from .whisperx_wrapper import WhisperXWrapper


# * Orchestrate local speech-to-text processing pipeline
class LocalPipeline:
    """Pipeline for local processing: FFmpeg -> STT (Whisper/WhisperX) -> Diarization -> LLM formatting."""

    def __init__(
        self,
        model_root: Path | None = None,
        whisper_model: str = "base",
        llm_model: str = "gguf-q4_0",
        *,
        engine: str = "whisper",  # whisper | whisperx
        enable_diarization: bool = True,
        enable_dialog_blocks: bool = False,
        language: str | None = None,
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        """Initialize pipeline components."""
        self.model_root = model_root
        self.engine = engine
        self.language = language
        self.device = device
        self.compute_type = compute_type
        # STT engines
        w_device = (
            "cuda" if device in {"auto", "cuda"} else "cpu"
        )  # basic mapping for whisper
        self.whisper = WhisperWrapper(
            model_name=whisper_model, model_root=model_root, device=w_device
        )
        self.whisperx = WhisperXWrapper(
            model_name=whisper_model, model_root=model_root, device=w_device, compute_type=compute_type
        )
        # Diarization & LLM
        self.enable_diarization = enable_diarization
        self.enable_dialog_blocks = enable_dialog_blocks
        self.diarizer = DiarizationPipeline()
        self.formatter = LLMFormatter(model_name=llm_model, model_path=None)

    def process(
        self,
        input_path: Path,
        work_dir: Path = Path(),
    ) -> Document:
        """Process a media file through the pipeline and return a formatted document."""
        # * Step 1: Ensure audio is in WAV format
        audio_path = prepare_audio(input_path, work_dir)

        # * Step 2: Transcribe audio to raw text
        if self.engine == "whisperx":
            tx = self.whisperx.transcribe(audio_path, language=self.language)
            raw_text = tx.get("text", "")
            # * Alignment is optional; enable in later phase
        else:
            raw_text = self.whisper.transcribe(audio_path)

        # * Step 3: Perform speaker diarization (optional)
        diarization_segments = (
            self.diarizer.diarize(str(audio_path)) if self.enable_diarization else []
        )

        # * Step 4: Format text using LLM (optional blocks)
        formatted_text = (
            self.formatter.format_text(raw_text)
            if self.enable_dialog_blocks
            else raw_text
        )

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
