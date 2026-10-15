"""CPU speaker diarization with sherpa-onnx (the hardsub skill's backend).

The diarizer runs the same two networks with the same configuration as the
hardsub skill's ``diarize.py``: pyannote segmentation 3.0 and WeSpeaker
ResNet34-LM embeddings (ONNX, CPU provider, ``num_threads`` for both
networks), fast clustering with ``num_clusters = num_speakers or -1`` and the
distance threshold, and the ``min_duration_on``/``min_duration_off`` turn
filters. Audio is read from the prepared 16 kHz mono 16-bit WAV and decoded
exactly like the skill's ``read_audio`` (``int16 / 32768`` as float32), so
both produce the same turns for the same media.

Thread safety: every :meth:`SherpaDiarizer.diarize_wav` call builds its own
``OfflineSpeakerDiarization`` instance (about 0.1 s) and never shares it, so
several threads may use one :class:`SherpaDiarizer` at once; the shared state
is only the thread count and the verified model paths. ``set_config`` of
sherpa-onnx is never used.

! The sherpa-onnx 1.13.8 binding of ``process`` keeps the GIL for the whole
! native run. Concurrent calls from several threads of one process therefore
! run one after another, and other Python threads of the process run only
! while the progress callback executes (the callback is installed only when
! ``should_cancel`` is given, and is first called after the segmentation
! stage). Parallel diarization needs separate processes.

Cancellation: sherpa-onnx 1.13.8 ignores the callback's return value, so the
callback aborts the native run by raising :class:`DiarizationCancelledError`;
the exception unwinds the native frames and reaches the caller. The callback
runs once per embedding chunk, after the segmentation stage (roughly a fifth
of the run time), which cannot be interrupted.

``numpy`` and ``sherpa_onnx`` are imported on first use only, so importing
this module stays light. The module has no pool or CLI dependencies: the pool
engine and the one-shot CLI path both use it directly.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import threading
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from core.audio_io import cleanup_intermediate_audio, prepare_audio
from core.diarization_models import ensure_diarization_models
from core.stt_options import SAMPLE_RATE, DiarizeOptions
from core.stt_output import turns_json

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from core.diarization_models import DiarizationModelPaths
    from core.stt_options import MediaWindow

logger = logging.getLogger(__name__)

# * CPU threads of each network per diarization call (the skill's default).
DEFAULT_THREADS: Final = 4

# * The prepared WAV is mono 16-bit PCM; samples are scaled like the skill's
# * ``read_audio``: int16 / 32768 as float32.
_PCM16_SAMPLE_WIDTH: Final = 2
_MONO: Final = 1
_PCM16_SCALE: Final = 32768.0
# * Frames decoded per read (about 65 s of audio, 2 MiB of PCM).
_READ_CHUNK_FRAMES: Final = 1 << 20
# * Subfolder of the caller's work dir that receives the prepared WAV, so it
# * never collides with a transcription WAV of the same input.
_WORK_SUBDIR: Final = "diarize"

_MISSING_BACKEND_MSG: Final = (
    "speaker diarization needs sherpa-onnx and numpy from the AskVLM ML extra; "
    'install them with `pip install -e ".[ml]"`'
)


class DiarizationError(RuntimeError):
    """Diarization failed: unusable audio or a rejected backend configuration."""


class DiarizationBackendMissingError(DiarizationError):
    """``sherpa_onnx`` or ``numpy`` cannot be imported."""


class DiarizationCancelledError(RuntimeError):
    """The caller's ``should_cancel`` stopped the diarization."""


class _Turn(Protocol):
    """One speaker turn of a sherpa-onnx diarization result."""

    @property
    def start(self) -> float:
        """Turn start in seconds from the start of the samples."""
        ...

    @property
    def end(self) -> float:
        """Turn end in seconds from the start of the samples."""
        ...

    @property
    def speaker(self) -> int:
        """Raw cluster index (not necessarily contiguous or starting at zero)."""
        ...


class _DiarizationResult(Protocol):
    """Result object of ``OfflineSpeakerDiarization.process``."""

    def sort_by_start_time(self) -> Sequence[_Turn]:
        """Return the turns ordered by start time."""
        ...


class _DiarizationEngine(Protocol):
    """The used surface of ``sherpa_onnx.OfflineSpeakerDiarization``."""

    @property
    def sample_rate(self) -> int:
        """Sample rate the models expect."""
        ...

    def process(
        self, samples: object, callback: Callable[[int, int], int] | None = None
    ) -> _DiarizationResult:
        """Diarize float32 samples; an exception raised by the callback aborts.

        sherpa-onnx 1.13.8 ignores the callback's return value, so a
        cancellation must raise from inside the callback.
        """
        ...


def _import_backend() -> tuple[ModuleType, ModuleType]:
    """Import ``numpy`` and ``sherpa_onnx`` on first use.

    Returns:
        ``(numpy, sherpa_onnx)``.

    Raises:
        DiarizationBackendMissingError: When either package cannot be imported.

    """
    try:
        numpy = importlib.import_module("numpy")
        sherpa = importlib.import_module("sherpa_onnx")
    except ImportError as exc:
        msg = f"{_MISSING_BACKEND_MSG} (import failed: {exc})"
        raise DiarizationBackendMissingError(msg) from exc
    return numpy, sherpa


def _require_pcm16_mono(wav_path: Path, handle: wave.Wave_read) -> None:
    """Fail unless *handle* is 16 kHz mono 16-bit PCM.

    Raises:
        DiarizationError: When the format differs.

    """
    rate = handle.getframerate()
    channels = handle.getnchannels()
    width = handle.getsampwidth()
    if rate != SAMPLE_RATE or channels != _MONO or width != _PCM16_SAMPLE_WIDTH:
        msg = (
            f"{wav_path}: expected a {SAMPLE_RATE} Hz mono 16-bit PCM WAV, got "
            f"{rate} Hz, {channels} channel(s), {width * 8}-bit samples"
        )
        raise DiarizationError(msg)


def _decode_frames(
    numpy: ModuleType, handle: wave.Wave_read, max_frames: int
) -> tuple[object, int]:
    """Decode the PCM frames of *handle* into one float32 buffer.

    Frames are read in chunks of :data:`_READ_CHUNK_FRAMES` and converted into
    a preallocated float32 array that is then scaled in place, so the peak
    memory is 4 bytes per sample plus one chunk. The values are bit-identical
    to the skill's ``int16.astype(float32) / 32768.0`` (the int16 to float32
    conversion is exact and the division by 2**15 is exact in float32).

    Args:
        numpy: The numpy module.
        handle: Open 16-bit mono WAV reader.
        max_frames: Upper bound of the frame count (from the file size), so a
            bogus header cannot trigger a huge allocation.

    Returns:
        ``(samples, frame_count)``; *samples* is a float32 array in [-1, 1).

    """
    capacity = min(handle.getnframes(), max_frames)
    samples = numpy.empty(capacity, dtype=numpy.float32)
    filled = 0
    while filled < capacity:
        chunk = handle.readframes(min(_READ_CHUNK_FRAMES, capacity - filled))
        count = len(chunk) // _PCM16_SAMPLE_WIDTH
        if count == 0:
            break
        samples[filled : filled + count] = numpy.frombuffer(
            chunk, dtype="<i2", count=count
        )
        filled += count
    samples = samples[:filled]
    samples /= numpy.float32(_PCM16_SCALE)
    return samples, filled


def _read_samples(numpy: ModuleType, wav_path: Path) -> tuple[object, int]:
    """Return the float32 samples of a 16 kHz mono 16-bit WAV.

    Args:
        numpy: The numpy module.
        wav_path: The WAV file.

    Returns:
        ``(samples, frame_count)`` decoded like the skill's ``read_audio``.

    Raises:
        DiarizationError: When the file cannot be read, has another format, or
            holds no audio frames.

    """
    try:
        max_frames = wav_path.stat().st_size // _PCM16_SAMPLE_WIDTH
        with wave.open(str(wav_path), "rb") as handle:
            _require_pcm16_mono(wav_path, handle)
            samples, frames = _decode_frames(numpy, handle, max_frames)
    except (wave.Error, EOFError, OSError) as exc:
        msg = f"cannot read the WAV {wav_path}: {exc}"
        raise DiarizationError(msg) from exc
    if frames == 0:
        msg = f"no audio decoded from {wav_path}"
        raise DiarizationError(msg)
    return samples, frames


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    """Raise :class:`DiarizationCancelledError` when cancellation is requested."""
    if should_cancel is not None and should_cancel():
        msg = "speaker diarization was cancelled"
        raise DiarizationCancelledError(msg)


def _abort_callback(should_cancel: Callable[[], bool]) -> Callable[[int, int], int]:
    """Return the sherpa-onnx progress callback that aborts on cancellation.

    sherpa-onnx 1.13.8 ignores the returned value, so the callback aborts by
    raising: the Python exception unwinds the native embedding loop (no
    ``noexcept`` frames, RAII-owned buffers) and surfaces from ``process``.

    Args:
        should_cancel: Polled once per embedding chunk.

    Returns:
        ``callback(processed_chunks, num_chunks) -> 0``.

    """

    def callback(processed_chunks: int, num_chunks: int) -> int:
        """Raise :class:`DiarizationCancelledError` once cancellation is requested.

        Raises:
            DiarizationCancelledError: When ``should_cancel`` returns true.

        """
        logger.debug("diarization progress %s/%s", processed_chunks, num_chunks)
        _raise_if_cancelled(should_cancel)
        return 0

    return callback


class SherpaDiarizer:
    """CPU speaker diarizer over pinned sherpa-onnx models.

    One instance may serve several threads at once: each call builds and
    owns its sherpa-onnx engine. The native runs of such calls do not overlap
    because sherpa-onnx keeps the GIL (see the module notes).
    """

    def __init__(
        self,
        threads: int = DEFAULT_THREADS,
        model_paths: DiarizationModelPaths | None = None,
    ) -> None:
        """Configure the diarizer; nothing is loaded until :meth:`load`.

        Args:
            threads: CPU threads of each network per call, ``>= 1``.
            model_paths: Models already verified by
                :func:`~core.diarization_models.ensure_diarization_models`;
                ``None`` ensures them in the default cache on :meth:`load`.

        Raises:
            ValueError: When *threads* is not a positive integer.

        """
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            msg = f"threads must be a positive integer (got {threads!r})"
            raise ValueError(msg)
        self._threads = threads
        self._requested_paths = model_paths
        self._paths: DiarizationModelPaths | None = None
        self._load_lock = threading.Lock()

    @property
    def threads(self) -> int:
        """CPU threads of each network per call."""
        return self._threads

    @property
    def loaded(self) -> bool:
        """Whether :meth:`load` has completed."""
        return self._paths is not None

    def load(self) -> DiarizationModelPaths:
        """Ensure the models and validate the configuration once (idempotent).

        Returns:
            The verified model paths.

        Raises:
            DiarizationBackendMissingError: When sherpa-onnx or numpy is missing.
            DiarizationModelError: When the models cannot be provisioned.
            DiarizationError: When a model file is missing or sherpa-onnx
                rejects the configuration.

        """
        with self._load_lock:
            if self._paths is not None:
                return self._paths
            started = time.monotonic()
            _, sherpa = _import_backend()
            paths = self._requested_paths or ensure_diarization_models()
            for path in (paths.segmentation, paths.embedding):
                if not path.is_file():
                    msg = f"diarization model file not found: {path}"
                    raise DiarizationError(msg)
            # * Build one engine to fail early on a broken model or config.
            self._new_engine(sherpa, paths, DiarizeOptions())
            self._paths = paths
            logger.info(
                "sherpa-onnx diarization ready: threads=%s models=%s, %s (%.2f s)",
                self._threads,
                paths.segmentation,
                paths.embedding,
                time.monotonic() - started,
            )
            return paths

    def _new_engine(
        self, sherpa: ModuleType, paths: DiarizationModelPaths, options: DiarizeOptions
    ) -> _DiarizationEngine:
        """Build a fresh engine with the skill's configuration.

        Raises:
            DiarizationError: When sherpa-onnx rejects the configuration or the
                models expect another sample rate.

        """
        config = sherpa.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(paths.segmentation)
                ),
                num_threads=self._threads,
                provider="cpu",
            ),
            embedding=sherpa.SpeakerEmbeddingExtractorConfig(
                model=str(paths.embedding), num_threads=self._threads, provider="cpu"
            ),
            clustering=sherpa.FastClusteringConfig(
                num_clusters=options.num_speakers or -1, threshold=options.threshold
            ),
            min_duration_on=options.min_duration_on,
            min_duration_off=options.min_duration_off,
        )
        if not config.validate():
            msg = (
                "sherpa-onnx rejected the diarization configuration "
                f"(models {paths.segmentation}, {paths.embedding}; {options})"
            )
            raise DiarizationError(msg)
        engine: _DiarizationEngine = sherpa.OfflineSpeakerDiarization(config)
        if engine.sample_rate != SAMPLE_RATE:
            msg = (
                f"the diarization models expect {engine.sample_rate} Hz audio, "
                f"not {SAMPLE_RATE} Hz"
            )
            raise DiarizationError(msg)
        return engine

    def diarize_wav(
        self,
        wav_path: str | Path,
        options: DiarizeOptions,
        *,
        offset_s: float = 0.0,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[dict[str, object]]:
        """Diarize a prepared 16 kHz mono 16-bit WAV.

        Args:
            wav_path: The prepared WAV.
            options: Clustering and turn-filter settings.
            offset_s: Seconds from the media start to the WAV start; added to
                every turn so the times are media-relative.
            should_cancel: Polled before the run, once per embedding chunk
                and after the run; a true result aborts the native run (the
                segmentation stage before the first chunk is not
                interruptible). An exception it raises propagates.

        Returns:
            ``{start, end, speaker}`` turns in start order, with speakers
            renumbered from ``speaker_00`` in order of first appearance (see
            :func:`core.stt_output.turns_json`); empty when no speech is found.

        Raises:
            DiarizationCancelledError: When *should_cancel* returned true.
            DiarizationError: When the WAV is unusable or empty, or sherpa-onnx
                rejects the configuration.
            DiarizationBackendMissingError: When sherpa-onnx or numpy is missing.
            DiarizationModelError: When the models cannot be provisioned.

        """
        paths = self.load()
        _raise_if_cancelled(should_cancel)
        numpy, sherpa = _import_backend()
        source = Path(wav_path)
        samples, frames = _read_samples(numpy, source)
        _raise_if_cancelled(should_cancel)
        started = time.monotonic()
        engine = self._new_engine(sherpa, paths, options)
        if should_cancel is None:
            result = engine.process(samples)
        else:
            result = engine.process(samples, callback=_abort_callback(should_cancel))
            # * A request after the last callback (or for audio without any
            # * embedding chunk) still wins over the result.
            _raise_if_cancelled(should_cancel)
        turns = turns_json(
            (
                (float(turn.start), float(turn.end), int(turn.speaker))
                for turn in result.sort_by_start_time()
            ),
            offset_s,
        )
        logger.info(
            "diarized %s: %.1f s of audio, %d turns, %d speakers in %.2f s",
            source,
            frames / SAMPLE_RATE,
            len(turns),
            len({turn["speaker"] for turn in turns}),
            time.monotonic() - started,
        )
        return turns

    def diarize_media(
        self,
        input_path: str | Path,
        options: DiarizeOptions,
        *,
        window: MediaWindow,
        work_dir: Path,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[dict[str, object]]:
        """Decode a media window to WAV and diarize it.

        The WAV is prepared in a ``diarize`` subfolder of *work_dir* (so it
        never collides with a transcription WAV of the same input) and is
        removed afterwards.

        Args:
            input_path: Any media file ffmpeg can read.
            options: Clustering and turn-filter settings.
            window: Media window; turn times stay media-relative.
            work_dir: Folder for the intermediate WAV.
            should_cancel: Polled during decoding and diarization.

        Returns:
            ``{start, end, speaker}`` turns in start order.

        Raises:
            DiarizationCancelledError: When *should_cancel* returned true.
            DiarizationError: When the decoded audio is unusable or empty.
            ValueError: When the window holds no audio (from
                :func:`core.audio_io.prepare_audio`).
            RuntimeError: When ffmpeg fails on the cancellable path.
            ffmpeg.Error: When ffmpeg fails without *should_cancel*.

        """
        source = Path(input_path)
        prepare_dir = Path(work_dir) / _WORK_SUBDIR
        try:
            try:
                if window.is_default:
                    wav = prepare_audio(
                        source, prepare_dir, should_cancel=should_cancel
                    )
                else:
                    wav = prepare_audio(
                        source,
                        prepare_dir,
                        should_cancel=should_cancel,
                        start_s=window.start_s or None,
                        duration_s=window.duration_s,
                    )
            except RuntimeError as exc:
                if should_cancel is not None and should_cancel():
                    msg = "speaker diarization was cancelled while decoding the audio"
                    raise DiarizationCancelledError(msg) from exc
                raise
            return self.diarize_wav(
                wav, options, offset_s=window.start_s, should_cancel=should_cancel
            )
        finally:
            cleanup_intermediate_audio(source, prepare_dir)
            with contextlib.suppress(OSError):
                prepare_dir.rmdir()
