import contextlib
import enum
import os
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final, cast

from editing.text_model import Document, TextSegment
from utils.env import load_env_file

from .audio_io import cleanup_intermediate_audio, prepare_audio
from .diarization import DiarizationPipeline
from .ffmpeg import get_media_duration_seconds
from .llm_formatter import LLMFormatter
from .settings import configure_ml_caches, get_project_cache_dir
from .stt_providers import (
    STT_PROVIDER_GIGAAM_E2E_CTC,
    STT_PROVIDER_WHISPER,
    is_gigaam_provider,
    normalize_stt_provider,
    resolve_gigaam_requested_device,
)
from .whisperx_wrapper import WhisperXWrapper

if TYPE_CHECKING:
    from .diarization import Segment


# * Cancellation sentinel used across pipeline layers
class CancelledError(RuntimeError):
    """Raised to indicate user-requested cancellation during processing."""


class _Unset(enum.Enum):
    """Marker type of an argument that was not passed."""

    TOKEN = enum.auto()


# * Default of ``process(language=...)``: use the pipeline's own language.
_UNSET: Final = _Unset.TOKEN
# * Device tokens accepted by strict device mode (no ``auto`` coercion).
_STRICT_DEVICES = frozenset({"cpu", "cuda"})


def _strict_device(device: str) -> str:
    """Return the explicit device of strict mode.

    Raises:
        ValueError: When *device* is not ``cpu`` or ``cuda``.

    """
    token = str(device).strip().lower()
    if token not in _STRICT_DEVICES:
        msg = (
            "Strict device mode requires an explicit device 'cpu' or 'cuda' "
            f"(got device={device!r})"
        )
        raise ValueError(msg)
    return token


# * Orchestrate local speech-to-text processing pipeline
class LocalPipeline:
    """Pipeline for local processing: FFmpeg -> STT -> Diarization -> LLM formatting.

    Default STT provider is Whisper/faster-whisper. Optional GigaAM providers:
    ``gigaam-ctc`` runs on CPU only; ``gigaam-e2e-ctc`` accepts
    ``auto``/``cpu``/``cuda`` and resolves CUDA lazily at model load.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_root: Path | None = None,
        whisper_model: str = "auto",
        llm_model: str = "gguf-q4_0",
        *,
        engine: str = "auto",  # whisper | whisperx | auto
        stt_provider: str = STT_PROVIDER_WHISPER,
        enable_diarization: bool = False,
        enable_dialog_blocks: bool = False,
        language: str | None = None,
        device: str = "auto",
        compute_type: str = "auto",
        strict_device: bool = False,
        cpu_threads: int | None = None,
        num_workers: int | None = None,
        torch_threads: int | None = None,
        gigaam_threadsafe: bool = False,
        gigaam_warmup_batch: int = 1,
    ) -> None:
        """Initialize pipeline components.

        The keyword-only options after ``compute_type`` are opt-in settings of
        the STT worker pool; each is forwarded to the STT wrapper only when it
        differs from its default.

        Args:
            model_root: Model directory (default: project cache ``models``).
            whisper_model: Whisper model name; ``auto`` means ``large-v3``.
            llm_model: LLM formatter model name.
            engine: Legacy engine hint (``whisper``/``whisperx``/``auto``).
            stt_provider: ``whisper``, ``gigaam-ctc`` or ``gigaam-e2e-ctc``.
            enable_diarization: Run speaker diarization.
            enable_dialog_blocks: Format dialog blocks with the LLM.
            language: Default language of :meth:`process` (``None`` detects).
            device: ``auto``/``cpu``/``cuda`` (strict mode: ``cpu``/``cuda``).
            compute_type: Whisper compute type.
            strict_device: Strict device mode of the STT wrapper: an explicit
                ``cpu``/``cuda`` and no CUDA→CPU fallback (typed errors from
                :mod:`core.stt_errors` instead).
            cpu_threads: Whisper ``WhisperModel(cpu_threads=...)``.
            num_workers: Whisper ``WhisperModel(num_workers=...)`` (concurrent
                :meth:`process` calls on one model).
            torch_threads: GigaAM ``torch.set_num_threads`` at model load.
            gigaam_threadsafe: GigaAM lock-free thread-safe inference path.
            gigaam_warmup_batch: GigaAM CUDA warm-up batch size (thread-safe mode).

        Raises:
            ValueError: Invalid provider or device (strict mode: a device other
                than ``cpu``/``cuda``), or an invalid wrapper option.

        """
        if strict_device:
            device = _strict_device(device)
        # * Load .env for HF_TOKEN and other variables
        load_env_file()
        # * Ensure ML caches are inside project directory
        cache_root = configure_ml_caches(get_project_cache_dir())
        # Prefer explicit model_root if provided, else use project cache/models
        self.model_root = model_root or (cache_root / "models")
        self.engine = engine
        self.stt_provider = normalize_stt_provider(stt_provider)
        self.language = language
        self.compute_type = compute_type
        self.strict_device = bool(strict_device)
        self.whisperx: WhisperXWrapper | None = None
        self.gigaam: Any | None = None
        if is_gigaam_provider(self.stt_provider):
            # * Validate the device token before any GigaAM import or weight
            # * download; the CUDA-capable head keeps auto/cuda for load time.
            self.device = resolve_gigaam_requested_device(self.stt_provider, device)
            hf_cache = cache_root / "huggingface"
            gigaam_options = _gigaam_wrapper_options(
                strict_device=strict_device,
                torch_threads=torch_threads,
                threadsafe=gigaam_threadsafe,
                warmup_batch=gigaam_warmup_batch,
            )
            if self.stt_provider == STT_PROVIDER_GIGAAM_E2E_CTC:
                from . import gigaam_e2e_ctc_wrapper as gigaam_e2e_mod  # noqa: PLC0415

                self.gigaam = gigaam_e2e_mod.GigaAME2ECtcWrapper(
                    device=self.device,
                    cache_dir=hf_cache,
                    **gigaam_options,
                )
            else:
                from . import gigaam_ctc_wrapper as gigaam_mod  # noqa: PLC0415

                self.gigaam = gigaam_mod.GigaAMCtcWrapper(
                    device=self.device,
                    cache_dir=hf_cache,
                    **gigaam_options,
                )
        else:
            self.device = device
            # STT engine (Faster-Whisper via WhisperX)
            w_device = device if device in {"auto", "cuda", "cpu"} else "auto"
            self.whisperx = WhisperXWrapper(
                model_name="large-v3" if whisper_model == "auto" else whisper_model,
                model_root=self.model_root,
                device=w_device,
                compute_type=compute_type,
                **_whisper_wrapper_options(
                    strict_device=strict_device,
                    cpu_threads=cpu_threads,
                    num_workers=num_workers,
                ),
            )
        # Diarization & LLM
        self.enable_diarization = enable_diarization
        self.enable_dialog_blocks = enable_dialog_blocks
        # Lazy init to avoid importing heavy backends when not needed
        self.diarizer: DiarizationPipeline | None = None
        self._llm_model = llm_model
        self.formatter: LLMFormatter | None = None

    def require_whisperx(self) -> WhisperXWrapper:
        """Return the Whisper backend or raise when this pipeline uses GigaAM.

        Returns:
            The configured :class:`WhisperXWrapper`.

        Raises:
            RuntimeError: When the pipeline was created for a GigaAM provider.

        """
        if self.whisperx is None:
            msg = "Whisper backend is not configured for this STT provider"
            raise RuntimeError(msg)
        return self.whisperx

    def ensure_stt_loaded(self) -> None:
        """Load the active STT model now (thread-safe GigaAM also warms up).

        Call it single-threaded before concurrent :meth:`process` calls.

        Raises:
            RuntimeError: When the configured STT backend is missing.
            DeviceUnavailableError: Strict ``cuda`` without a usable CUDA
                runtime (see :mod:`core.stt_errors`).
            CudaOutOfMemoryError: Strict CUDA load ran out of memory.

        """
        if is_gigaam_provider(self.stt_provider):
            if self.gigaam is None:
                msg = "GigaAM CTC backend is not configured"
                raise RuntimeError(msg)
            self.gigaam.ensure_loaded()
            return
        self.require_whisperx().ensure_loaded()

    def _transcribe_audio(
        self,
        audio_path: Path,
        *,
        language: str | None,
        on_segment: Callable[[dict[str, object]], None],
        subtitle_max_line_width: int | None,
        subtitle_max_lines: int | None,
    ) -> tuple[dict[str, Any], str]:
        """Dispatch ASR to the selected provider without cross-wiring kwargs.

        Args:
            audio_path: Prepared WAV path.
            language: Language of this call (``None`` detects).
            on_segment: Segment streaming callback.
            subtitle_max_line_width: Whisper subtitle layout hint.
            subtitle_max_lines: Whisper subtitle layout hint.

        Returns:
            ``(transcript_dict, asr_label)`` for progress reporting.

        """
        if is_gigaam_provider(self.stt_provider):
            if self.gigaam is None:
                msg = "GigaAM CTC backend is not configured"
                raise RuntimeError(msg)
            # * Do not forward Whisper-only beam/VAD/compute kwargs to GigaAM.
            return (
                cast(
                    "dict[str, Any]",
                    self.gigaam.transcribe(
                        audio_path=audio_path,
                        language=language,
                        on_segment=on_segment,
                        progress=None,
                    ),
                ),
                self.stt_provider,
            )
        whisper = self.require_whisperx()
        wx_kwargs: dict[str, object] = {
            "beam_size": 10,
            "vad_filter": True,
            "word_timestamps": True,
        }
        if subtitle_max_line_width is not None:
            wx_kwargs["max_line_width"] = int(subtitle_max_line_width)
        if subtitle_max_lines is not None:
            wx_kwargs["max_line_count"] = int(subtitle_max_lines)
        return (
            whisper.transcribe(
                audio_path=audio_path,
                language=language,
                on_segment=on_segment,
                progress=None,
                **wx_kwargs,
            ),
            "whisperx",
        )

    def _get_formatter(self) -> LLMFormatter:
        if self.formatter is None:
            self.formatter = LLMFormatter(model_name=self._llm_model, model_path=None)
        return self.formatter

    @property
    def llm_model_name(self) -> str:
        """Return the configured LLM formatter model name."""
        return self._llm_model

    def close(self, *, aggressive: bool = False) -> None:
        """Release loaded ML backends held by the pipeline."""
        if self.whisperx is not None:
            self.whisperx.unload(safe=not aggressive)
        if self.gigaam is not None:
            self.gigaam.unload(safe=not aggressive)
        if self.diarizer is not None:
            with contextlib.suppress(Exception):
                self.diarizer.close()
            self.diarizer = None
        if self.formatter is not None:
            with contextlib.suppress(Exception):
                self.formatter.close()
            self.formatter = None

    def __enter__(self) -> "LocalPipeline":
        """Return the pipeline as a context-managed resource."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        """Release heavy ML resources when leaving the context manager."""
        self.close(aggressive=True)

    def process(  # noqa: C901, PLR0915
        self,
        input_path: Path,
        work_dir: Path = Path(),
        progress: Callable[[str, float], None] | None = None,
        *,
        subtitle_max_line_width: int | None = None,
        subtitle_max_lines: int | None = None,
        should_cancel: Callable[[], bool] | None = None,
        language: str | None | _Unset = _UNSET,
    ) -> Document:
        """Process a media file through the pipeline and return a formatted document.

        Concurrency contract: several threads may call ``process`` on one
        instance at the same time when diarization and dialog blocks are off
        (both lazily create shared backends), each call uses its own
        ``work_dir`` (intermediate WAV and partial transcript live there), the
        STT model was loaded beforehand by :meth:`ensure_stt_loaded` in strict
        device mode (the default mode's CUDA→CPU recovery reloads the shared
        model), Whisper runs with ``num_workers`` at least the number of
        concurrent calls, and GigaAM runs with ``gigaam_threadsafe=True``. The
        per-call ``language`` is passed by value and never stored on the
        instance.

        Args:
            input_path: Path to media file (audio/video).
            work_dir: Working directory for intermediate artifacts.
            progress: Optional callback reporting (message, 0..1) progress.
            subtitle_max_line_width: Max chars per subtitle line (Whisper hint).
            subtitle_max_lines: Max lines per subtitle cue (Whisper hint).
            should_cancel: Optional callback; when returns True, processing cancels.
            language: Language of this call only, overriding the pipeline's
                ``language`` (``None`` = auto-detect); omitted means the
                pipeline's ``language``.

        """
        call_language = self.language if isinstance(language, _Unset) else language

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
                raise CancelledError(msg)
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

        try:
            tx, asr_label = self._transcribe_audio(
                audio_path,
                language=call_language,
                on_segment=_on_segment,
                subtitle_max_line_width=subtitle_max_line_width,
                subtitle_max_lines=subtitle_max_lines,
            )
        except Exception as exc:
            report(f"ASR failed: {exc}", 0.6)
            raise
        raw_text = str(tx.get("text", "") or "")
        segments_raw = tx.get("segments", [])
        transcript_segments = (
            list(cast("list[dict[str, object]]", segments_raw))
            if isinstance(segments_raw, list)
            else []
        )
        report(f"Transcription complete ({asr_label})", 0.6)

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
            formatted_text = self._get_formatter().format_text(raw_text)
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
            format_text_fn=(
                self._get_formatter().format_text
                if self.enable_dialog_blocks
                else _identity_format_text
            ),
        )
        # * Mark document so text/JSON exporters can build dialog blocks when enabled
        doc.dialog_blocks_enabled = bool(self.enable_dialog_blocks)
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
    - stt_provider="whisper" (optional external path: gigaam-ctc / gigaam-e2e-ctc)
    - device="auto" (prefer CUDA when available; gigaam-ctc forces CPU,
      gigaam-e2e-ctc also checks free VRAM at model load)
    - enable_diarization=False
    - enable_dialog_blocks=False
    - whisper_model="auto" (resolves to "large-v3")
    - compute_type="auto" (prefers float16 on CUDA)
    """
    return LocalPipeline()


def _identity_format_text(text: str) -> str:
    """Return text unchanged."""
    return text


def _whisper_wrapper_options(
    *,
    strict_device: bool,
    cpu_threads: int | None,
    num_workers: int | None,
) -> dict[str, Any]:
    """Return the non-default Whisper wrapper options (empty for the defaults)."""
    options: dict[str, Any] = {}
    if strict_device:
        options["strict_device"] = True
    if cpu_threads is not None:
        options["cpu_threads"] = cpu_threads
    if num_workers is not None:
        options["num_workers"] = num_workers
    return options


def _gigaam_wrapper_options(
    *,
    strict_device: bool,
    torch_threads: int | None,
    threadsafe: bool,
    warmup_batch: int,
) -> dict[str, Any]:
    """Return the non-default GigaAM wrapper options (empty for the defaults)."""
    options: dict[str, Any] = {}
    if strict_device:
        options["strict_device"] = True
    if torch_threads is not None:
        options["torch_threads"] = torch_threads
    if threadsafe:
        options["threadsafe"] = True
    if warmup_batch != 1:
        options["warmup_batch"] = warmup_batch
    return options


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
