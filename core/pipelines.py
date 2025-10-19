from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from editing.text_model import Document, TextSegment

from .audio_io import prepare_audio
from .diarization import DiarizationPipeline
from .llm_formatter import LLMFormatter
from .settings import configure_ml_caches, get_project_cache_dir
from .whisper_wrapper import WhisperWrapper
from .whisperx_wrapper import WhisperXWrapper

if TYPE_CHECKING:
    from .diarization import Segment


# * Orchestrate local speech-to-text processing pipeline
class LocalPipeline:
    """Pipeline for local processing: FFmpeg -> STT (Whisper/WhisperX) -> Diarization -> LLM formatting."""

    def __init__(  # noqa: PLR0913
        self,
        model_root: Path | None = None,
        whisper_model: str = "base",
        llm_model: str = "gguf-q4_0",
        *,
        engine: str = "auto",  # whisper | whisperx | auto
        enable_diarization: bool = True,
        enable_dialog_blocks: bool = False,
        language: str | None = None,
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        """Initialize pipeline components."""
        # * Ensure ML caches are inside project directory
        cache_root = configure_ml_caches(get_project_cache_dir())
        # Prefer explicit model_root if provided, else use project cache/models
        self.model_root = model_root or (cache_root / "models")
        self.engine = engine
        self.language = language
        self.device = device
        self.compute_type = compute_type
        # STT engines
        w_device = (
            "cuda" if device in {"auto", "cuda"} else "cpu"
        )  # basic mapping for whisper
        self.whisper = WhisperWrapper(
            model_name=whisper_model, model_root=self.model_root, device=w_device
        )
        self.whisperx = WhisperXWrapper(
            model_name=whisper_model,
            model_root=self.model_root,
            device=w_device,
            compute_type=compute_type,
        )
        # Diarization & LLM
        self.enable_diarization = enable_diarization
        self.enable_dialog_blocks = enable_dialog_blocks
        # Lazy init to avoid importing heavy backends when not needed
        self.diarizer: DiarizationPipeline | None = None
        self.formatter = LLMFormatter(model_name=llm_model, model_path=None)

    def process(
        self,
        input_path: Path,
        work_dir: Path = Path(),
        progress: Callable[[str, float], None] | None = None,
    ) -> Document:
        """Process a media file through the pipeline and return a formatted document.

        Args:
            input_path: Path to media file (audio/video).
            work_dir: Working directory for intermediate artifacts.
            progress: Optional callback reporting (message, 0..1) progress.

        """

        def report(msg: str, frac: float) -> None:
            if progress is not None:
                # Clamp fraction into [0,1]
                pct = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
                progress(msg, pct)

        report("Preparing audio", 0.05)
        # * Step 1: Ensure audio is in WAV format
        audio_path = prepare_audio(input_path, work_dir)
        report("Audio prepared", 0.15)

        # * Step 2: Transcribe audio to raw text
        raw_text = ""
        engine_to_use = self.engine
        report("Transcribing (engine selection)", 0.2)
        if engine_to_use == "auto":
            # Prefer faster-whisper/whisperx when available, fallback to whisper
            try:
                tx = self.whisperx.transcribe(audio_path, language=self.language)
                raw_text = tx.get("text", "")
                engine_to_use = "whisperx"
            except Exception:  # noqa: BLE001
                raw_text = self.whisper.transcribe(audio_path)
                engine_to_use = "whisper"
        elif engine_to_use == "whisperx":
            tx = self.whisperx.transcribe(audio_path, language=self.language)
            raw_text = tx.get("text", "")
        else:
            raw_text = self.whisper.transcribe(audio_path)
        report(f"Transcription complete ({engine_to_use})", 0.6)

        # * Step 3: Perform speaker diarization (optional)
        diarization_segments: list[Segment] = []
        if self.enable_diarization:
            if self.diarizer is None:
                # Prefer CUDA per project policy
                self.diarizer = DiarizationPipeline(device="cuda")
            diarization_segments = self.diarizer.diarize(str(audio_path))
        report(
            "Diarization complete" if diarization_segments else "Diarization skipped",
            0.75,
        )

        # * Step 4: Format text using LLM (optional blocks)
        formatted_text = (
            self.formatter.format_text(raw_text)
            if self.enable_dialog_blocks
            else raw_text
        )
        report(
            "Formatting complete"
            if self.enable_dialog_blocks
            else "Formatting skipped",
            0.9,
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
        report("Document built", 0.98)
        return doc
