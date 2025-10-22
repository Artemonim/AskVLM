from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from editing.text_model import Document, TextSegment
from utils.env import load_env_file

from .audio_io import cleanup_intermediate_audio, prepare_audio
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
        # * Load .env for HF_TOKEN and other variables
        load_env_file()
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
        transcript_segments: list[dict[str, object]] = []
        engine_to_use = self.engine
        report("Transcribing (engine selection)", 0.2)
        if engine_to_use == "auto":
            # Prefer faster-whisper/whisperx when available, fallback to whisper
            try:
                tx = self.whisperx.transcribe(audio_path, language=self.language)
                raw_text = tx.get("text", "")
                # * Keep coarse segments for export
                transcript_segments = list(tx.get("segments", []) or [])
                engine_to_use = "whisperx"
            except Exception:  # noqa: BLE001
                tx2 = self.whisper.transcribe(audio_path)
                raw_text = str(tx2.get("text", ""))
                transcript_segments = list(tx2.get("segments", []) or [])
                engine_to_use = "whisper"
        elif engine_to_use == "whisperx":
            tx = self.whisperx.transcribe(audio_path, language=self.language)
            raw_text = tx.get("text", "")
            transcript_segments = list(tx.get("segments", []) or [])
        else:
            tx3 = self.whisper.transcribe(audio_path)
            raw_text = str(tx3.get("text", ""))
            transcript_segments = list(tx3.get("segments", []) or [])
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

        # * Build document with best available segmentation
        doc = _build_document(
            formatted_text=formatted_text,
            transcript_segments=transcript_segments,
            diarization_segments=diarization_segments,
            enable_dialog_blocks=self.enable_dialog_blocks,
            format_text_fn=self.formatter.format_text,
        )
        report("Document built", 0.98)
        # * Cleanup intermediates (_work WAV) after successful processing
        cleanup_intermediate_audio(input_path, work_dir)
        return doc


def _build_document(
    *,
    formatted_text: str,
    transcript_segments: list[dict[str, object]],
    diarization_segments: list["Segment"],
    enable_dialog_blocks: bool,
    format_text_fn: Callable[[str], str],
) -> Document:
    """Construct a Document from transcript and diarization results.

    Prefers transcript segments; assigns speakers by greatest overlap.
    Falls back to diarization-only, then to single-segment document.
    """
    doc = Document()

    def resolve_speaker(start: float, end: float) -> str:
        if not diarization_segments:
            return "speaker_1"
        best_speaker = "speaker_1"
        best_overlap = 0.0
        for seg in diarization_segments:
            ov_start = max(start, seg.start)
            ov_end = min(end, seg.end)
            overlap = ov_end - ov_start
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg.speaker
        return best_speaker

    if transcript_segments:
        for s in transcript_segments:
            try:
                start_val = cast("Any", s.get("start", 0.0))
                end_val = cast("Any", s.get("end", 0.0))
                text_val = s.get("text", "")
                s_start = (
                    float(start_val) if not isinstance(start_val, float) else start_val
                )
                s_end = float(end_val) if not isinstance(end_val, float) else end_val
                s_text_raw = str(text_val)
            except (TypeError, ValueError):
                continue
            s_text = format_text_fn(s_text_raw) if enable_dialog_blocks else s_text_raw
            doc.add_segment(
                TextSegment(
                    speaker_id=resolve_speaker(s_start, s_end),
                    start_time=s_start,
                    end_time=s_end,
                    text=s_text,
                )
            )
        return doc

    if diarization_segments:
        for segment in diarization_segments:
            doc.add_segment(
                TextSegment(
                    speaker_id=segment.speaker,
                    start_time=segment.start,
                    end_time=segment.end,
                    text=formatted_text,
                )
            )
        return doc

    doc.add_segment(
        TextSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=0.0,
            text=formatted_text,
        )
    )
    return doc
