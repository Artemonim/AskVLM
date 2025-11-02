import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, cast

from editing.text_model import Document, TextSegment
from utils.env import load_env_file

from .audio_io import cleanup_intermediate_audio, prepare_audio
from .diarization import DiarizationPipeline
from .ffmpeg import get_media_duration_seconds
from .llm_formatter import LLMFormatter
from .settings import configure_ml_caches, get_project_cache_dir
from .whisperx_wrapper import WhisperXWrapper

if TYPE_CHECKING:
    from .diarization import Segment


# * Orchestrate local speech-to-text processing pipeline
class LocalPipeline:
    """Pipeline for local processing: FFmpeg -> STT (Whisper/WhisperX) -> Diarization -> LLM formatting."""

    def __init__(  # noqa: PLR0913
        self,
        model_root: Path | None = None,
        whisper_model: str = "auto",
        llm_model: str = "gguf-q4_0",
        *,
        engine: str = "auto",  # whisper | whisperx | auto
        enable_diarization: bool = False,
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
        # STT engine (Faster-Whisper via WhisperX)
        w_device = "cuda" if device in {"auto", "cuda"} else "cpu"
        self.whisperx = WhisperXWrapper(
            model_name="large-v3" if whisper_model == "auto" else whisper_model,
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

    def process(  # noqa: C901, PLR0915
        self,
        input_path: Path,
        work_dir: Path = Path(),
        progress: Callable[[str, float], None] | None = None,
        *,
        subtitle_max_line_width: int | None = None,
        subtitle_max_lines: int | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Document:
        """Process a media file through the pipeline and return a formatted document.

        Args:
            input_path: Path to media file (audio/video).
            work_dir: Working directory for intermediate artifacts.
            progress: Optional callback reporting (message, 0..1) progress.
            subtitle_max_line_width: Desired max chars per subtitle line (hint for WhisperX).
            subtitle_max_lines: Desired max lines per subtitle cue (hint for WhisperX).
            should_cancel: Optional callback; when returns True, processing cancels.

        """

        def report(msg: str, frac: float) -> None:
            if progress is not None:
                # Clamp fraction into [0,1]
                pct = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
                progress(msg, pct)

        report("Preparing audio", 0.05)
        # * Step 1: Ensure audio is in WAV format (cancellable)
        audio_path = prepare_audio(
            input_path,
            work_dir,
            should_cancel=should_cancel,
        )
        report("Audio prepared", 0.15)

        # * Media duration (for ETA and streaming progress)
        duration_s = get_media_duration_seconds(input_path)

        # * Step 2: Transcribe audio to raw text
        raw_text = ""
        transcript_segments: list[dict[str, object]] = []
        report("Transcribing", 0.2)
        wx_kwargs: dict[str, object] = {}
        # * Default extreme recognition profile (fits on 8 GiB VRAM empirically)
        # * Higher quality / VRAM usage: beam search and word-level timestamps
        wx_kwargs["beam_size"] = 10
        wx_kwargs["vad_filter"] = True
        wx_kwargs["word_timestamps"] = True
        if subtitle_max_line_width is not None:
            wx_kwargs["max_line_width"] = int(subtitle_max_line_width)
        if subtitle_max_lines is not None:
            wx_kwargs["max_line_count"] = int(subtitle_max_lines)

        # * Optional streaming output of partial transcript
        partial_txt: Path | None = None
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            partial_txt = work_dir / f"{input_path.stem}.partial.txt"
            with contextlib.suppress(OSError):
                if partial_txt.exists():
                    partial_txt.unlink()
        except OSError:
            partial_txt = None

        # * Streaming callback from ASR segments
        t0_asr = monotonic()

        def _on_segment(seg: dict[str, object]) -> None:
            try:
                s_val = cast("Any", seg.get("start", 0.0))
                e_val = cast("Any", seg.get("end", 0.0))
                float(s_val) if not isinstance(s_val, float) else s_val
                e = float(e_val) if not isinstance(e_val, float) else e_val
                txt = str(seg.get("text", "")).strip()
            except (TypeError, ValueError):
                return
            # Responsive cancellation during ASR
            if should_cancel is not None and should_cancel():
                msg = "Canceled"
                raise RuntimeError(msg)
            # Update transcription progress progressively within [0.2, 0.6]
            if duration_s > 0:
                inner = max(0.0, min(1.0, e / duration_s))
            else:
                # Fallback on time elapsed ratio if duration is unknown
                elapsed = max(0.0, monotonic() - t0_asr)
                # Assume linear growth over an arbitrary 1.0 unit time
                inner = max(0.0, min(1.0, elapsed / max(1.0, elapsed + 1.0)))
            frac = 0.2 + inner * 0.4
            report("Transcribing", frac)
            # Append to partial transcript file
            if partial_txt is not None and txt:
                try:
                    with partial_txt.open("a", encoding="utf-8") as fh:
                        fh.write(txt + os.linesep)
                except OSError:
                    pass

        # Use WhisperX (Faster-Whisper) exclusively for ASR with segment callback
        # Help type checker via explicit expansion to avoid kwargs mis-binding
        try:
            tx = self.whisperx.transcribe(
                audio_path=audio_path,
                language=self.language,
                on_segment=_on_segment,
                progress=None,
                **wx_kwargs,
            )
        except Exception as exc:
            report(f"ASR failed: {exc}", 0.6)
            raise
        raw_text = tx.get("text", "")
        transcript_segments = list(tx.get("segments", []) or [])
        report("Transcription complete (whisperx)", 0.6)

        # * Step 3: Perform speaker diarization (optional)
        diarization_segments: list[Segment] = []
        if self.enable_diarization:
            report("Diarizing", 0.61)
            if self.diarizer is None:
                # Prefer CUDA per project policy
                self.diarizer = DiarizationPipeline(device="cuda")
            diarization_segments = self.diarizer.diarize(str(audio_path))
        report(
            "Diarization complete" if diarization_segments else "Diarization skipped",
            0.75,
        )

        # * Step 4: Format text using LLM (optional blocks)
        if self.enable_dialog_blocks:
            report("Formatting", 0.82)
            formatted_text = self.formatter.format_text(raw_text)
            report("Formatting complete", 0.9)
        else:
            formatted_text = raw_text
            report("Formatting skipped", 0.86)

        # * Build document with best available segmentation
        doc = _build_document(
            formatted_text=formatted_text,
            transcript_segments=transcript_segments,
            diarization_segments=diarization_segments,
            enable_dialog_blocks=self.enable_dialog_blocks,
            format_text_fn=self.formatter.format_text,
        )
        report("Document built", 0.98)
        # * Export subtitle artifacts are configured later using options passed from GUI
        # * Export-time subtitle layout rules are applied by exporters
        # * Cleanup intermediates (_work WAV) after successful processing
        cleanup_intermediate_audio(input_path, work_dir)
        return doc


def create_default_local_pipeline() -> LocalPipeline:
    """Return a `LocalPipeline` configured with application defaults.

    Defaults are aligned with the GUI/CLI behavior:
    - engine="auto" (prefer Faster-Whisper; WhisperX alignment when available)
    - device="auto" (prefer CUDA when available)
    - enable_diarization=False
    - enable_dialog_blocks=False
    - whisper_model="auto" (resolves to "large-v3")
    - compute_type="auto" (prefers float16 on CUDA)
    """
    return LocalPipeline()

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
