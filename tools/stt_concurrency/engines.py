"""Child-side STT engines: faster-whisper, GigaAM-v3 ``e2e_ctc``, GigaAM ``ctc``.

ML packages are imported lazily inside :meth:`load`, so importing this module
(or the parent orchestrator) never pulls in torch or CTranslate2.

Whisper batching (``B`` mode)
-----------------------------
Both Whisper ``B`` paths call ``BatchedInferencePipeline.transcribe`` with
explicit ``clip_timestamps`` (no VAD): on the clips every file becomes one chunk
of at most ``whisper_chunk_s`` (longer files become several chunks) and the
files of one dispatched batch are concatenated, so one forward pass decodes
chunks of different files (cross-file batching); with ``-long`` one file is cut
into fixed chunks. Segments are mapped back to their file through their
``seek`` (the chunk offset in frames). The batched decoder uses the same beam
size, language, and (disabled) temperature fallback as the sequential path and
gets ``without_timestamps=False`` like ``WhisperModel.transcribe`` unless a
``--whisper-option`` sets it; it always decodes every chunk independently
(``condition_on_previous_text`` is off in every mode of this benchmark).

Thread safety (``S`` mode)
--------------------------
faster-whisper: ``WhisperModel(num_workers=N)`` maps to CTranslate2
``inter_threads=N``; concurrent ``transcribe()`` calls from N Python threads
run on separate replicas that share the weights on one device. The returned
segment generator is lazy, so every worker consumes it inside its own thread.

GigaAM: the project wrapper is not thread-safe (its CUDA OOM fallback mutates
the resident model and device), so workers call the loaded remote-code model
directly: ``asr.forward(wav, lengths)`` followed by
``asr.decoding.decode(asr.head, encoded, encoded_len)`` - exactly what
``GigaAMASR.transcribe`` does for one file. After one single-threaded warm-up
(which creates the lazy rotary ``pos_enc.pe`` buffer of the encoder) the
forward pass only reads module state; activations are per call, and
``torch.inference_mode`` / ``torch.autocast`` are thread-local. The remote
``load_audio`` is avoided: it wraps ``torch.frombuffer`` in
``warnings.catch_warnings()``, which mutates the process-global warning filters
and is not thread-safe on Python 3.11. This module runs the same ffmpeg command
and decodes into a writable buffer (identical samples, no warning to silence).
On CUDA each ``S`` thread may get its own CUDA stream so kernels can overlap.

GigaAM batching (``B`` mode)
----------------------------
The remote code batches natively: zero-padded waveforms plus a lengths tensor
(as in its ``AudioDataset.collate``), length-aware feature lengths, attention
and convolution padding masks in the encoder, and per-item lengths in the CTC
greedy decoder. The one non-equivalence with single-item inference is the right
edge of shorter items: the striding sub-sampling convolutions are not masked,
so they see log-mel frames of the zero padding instead of the implicit zero
padding of a single item, which can change the last encoder frame. The summary
quantifies the effect through the transcript comparison.
"""

from __future__ import annotations

import bisect
import contextlib
import gc
import importlib
import math
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Protocol

# * These project modules are torch-free at import time (ML imports are lazy).
from core.gigaam_ctc_wrapper import GigaAMCtcWrapper
from core.gigaam_e2e_ctc_wrapper import GigaAME2ECtcWrapper
from core.settings import configure_ml_caches, get_project_cache_dir
from core.stt_providers import (
    GIGAAM_HF_REPO_ID,
    GIGAAM_HF_REVISION,
    GIGAAM_V3_HF_REPO_ID,
    GIGAAM_V3_HF_REVISION,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from .protocol import ChildSpec

# * Compute types used by the project for faster-whisper on each device.
WHISPER_CPU_COMPUTE_TYPE = "int8"
WHISPER_CUDA_COMPUTE_TYPE = "float16"
# * Mirrors core.gigaam_ctc_wrapper: 20 s windows, 40 ms minimum after padding.
GIGAAM_WINDOW_S = 20.0
GIGAAM_MIN_AUDIO_S = 0.04
GIGAAM_DEFAULT_SAMPLE_RATE = 16000

_PCM16_SCALE = 32768.0
_FLOAT_EPS_S = 1e-6
_FFMPEG_TIMEOUT_S = 300.0
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_GIGAAM_ASR_ATTRS = ("forward", "head", "decoding", "parameters")


class EngineError(RuntimeError):
    """Raised when an engine cannot load or decode under the benchmark contract."""


# * Stable tag of a model that ended up on another device than requested.
DEVICE_MISMATCH_TAG = "device-mismatch"


class DeviceMismatchError(EngineError):
    """The wrapper placed the model on another device than the configuration asks.

    This is an environment or placement problem (CUDA unavailable, the wrapper's
    CPU fallback), never a CUDA out-of-memory saturation point of the benchmark.
    """


class SttEngine(Protocol):
    """Engine contract used by the child worker."""

    def load(self) -> dict[str, object]:
        """Load the model and return descriptive info."""
        ...

    def warmup(self, path: Path, *, slot: int, primary: bool) -> dict[str, object]:
        """Run an untimed warm-up on *path* for worker *slot*."""
        ...

    def transcribe(self, paths: Sequence[Path], *, slot: int) -> list[str]:
        """Transcribe *paths* (one call = one dispatched work unit)."""
        ...

    def close(self) -> None:
        """Release the model."""
        ...


def _configure_project_caches() -> Path:
    """Point Hugging Face and torch caches at ``<project>/.cache`` like LocalPipeline.

    Returns:
        The project cache root.

    """
    return configure_ml_caches(get_project_cache_dir())


def _join_segments(segments: Iterable[Any]) -> str:
    """Consume a (lazy) faster-whisper segment iterable and join its texts.

    Args:
        segments: Segment iterable returned by ``transcribe``.

    Returns:
        Space-joined, stripped segment texts.

    """
    parts = [str(segment.text or "").strip() for segment in segments]
    return " ".join(part for part in parts if part)


def _decoded_text(entry: object) -> str:
    """Return the text of one GigaAM decoder or ``transcribe`` result.

    v3 returns plain strings; the Multilingual revision returns
    ``(text, token_ids, token_frames)`` tuples from the decoder and a
    ``TranscriptionResult`` (with ``.text``) from ``transcribe``.
    """
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (tuple, list)) and entry:
        return str(entry[0])
    text = getattr(entry, "text", None)
    return str(text) if text is not None else str(entry)


def chunk_timestamps(duration_s: float, chunk_s: float) -> list[dict[str, float]]:
    """Split ``[0, duration_s)`` into fixed chunks for BatchedInferencePipeline.

    Args:
        duration_s: Audio duration.
        chunk_s: Chunk length (at most 30 s for Whisper).

    Returns:
        ``clip_timestamps`` dictionaries in seconds.

    """
    clips: list[dict[str, float]] = []
    start = 0.0
    while duration_s - start > _FLOAT_EPS_S:
        end = min(start + chunk_s, duration_s)
        clips.append({"start": start, "end": end})
        start = end
    return clips


class WhisperEngine:
    """faster-whisper ``small``; ``B`` mode wraps it in BatchedInferencePipeline."""

    def __init__(self, spec: ChildSpec) -> None:
        """Store the spec; nothing is loaded until :meth:`load`.

        Args:
            spec: Child specification.

        """
        self._spec = spec
        self._model: Any = None
        self._pipeline: Any = None
        self._decode_audio: Any = None
        self._numpy: Any = None
        self._sample_rate = 16000

    def load(self) -> dict[str, object]:
        """Load ``WhisperModel`` offline from the project model root.

        Returns:
            Engine info for the result file.

        """
        cache_root = _configure_project_caches()
        model_root = cache_root / "models"
        faster_whisper = importlib.import_module("faster_whisper")
        audio_mod = importlib.import_module("faster_whisper.audio")
        ctranslate2 = importlib.import_module("ctranslate2")
        self._numpy = importlib.import_module("numpy")
        config = self._spec.config
        compute_type = (
            WHISPER_CUDA_COMPUTE_TYPE
            if config.device == "cuda"
            else WHISPER_CPU_COMPUTE_TYPE
        )
        # * num_workers -> CTranslate2 inter_threads (replicas sharing weights),
        # * cpu_threads -> intra_threads of every replica.
        self._model = faster_whisper.WhisperModel(
            self._spec.whisper_model,
            device=config.device,
            compute_type=compute_type,
            cpu_threads=config.threads,
            num_workers=config.slots,
            download_root=str(model_root),
            local_files_only=True,
        )
        self._sample_rate = int(self._model.feature_extractor.sampling_rate)
        self._decode_audio = audio_mod.decode_audio
        if config.mode == "B":
            self._pipeline = faster_whisper.BatchedInferencePipeline(self._model)
        return {
            "backend": "faster-whisper",
            "faster_whisper": str(getattr(faster_whisper, "__version__", "?")),
            "ctranslate2": str(getattr(ctranslate2, "__version__", "?")),
            "model": self._spec.whisper_model,
            "model_root": str(model_root),
            "compute_type": compute_type,
            "cpu_threads": config.threads,
            "num_workers": config.slots,
            "batched_pipeline": self._pipeline is not None,
            "batching": (
                ("chunks of one file" if config.long else "across files")
                if self._pipeline is not None
                else None
            ),
            "batch_size": config.batch_size,
            "chunk_s": self._spec.whisper_chunk_s if self._pipeline else None,
            "decode_options": (
                self._batched_options()
                if self._pipeline is not None
                else dict(self._spec.whisper_options)
            ),
        }

    def _require_model(self) -> Any:  # noqa: ANN401 - untyped faster-whisper model
        """Return the loaded model.

        Raises:
            EngineError: When :meth:`load` has not run.

        """
        if self._model is None:
            msg = "Whisper model is not loaded"
            raise EngineError(msg)
        return self._model

    def _batched_options(self) -> dict[str, object]:
        """Decode options of the batched path, aligned with the sequential one."""
        options = dict(self._spec.whisper_options)
        # * WhisperModel.transcribe predicts timestamps by default; the batched
        # * pipeline would not, so align it unless the user chose explicitly.
        options.setdefault("without_timestamps", False)
        return options

    def _transcribe_audios_batched(self, audios: Sequence[Any]) -> list[str]:
        """Decode several files as fixed chunks in batched forward passes.

        Every file becomes ``ceil(duration / chunk_s)`` chunks; the files are
        concatenated, ``BatchedInferencePipeline`` decodes ``batch_size`` chunks
        per forward pass (across file borders), and every segment returns to its
        file through ``segment.seek`` (computed exactly like the pipeline does).

        Args:
            audios: Mono float32 sample arrays at the model sample rate.

        Returns:
            One transcript per input array.

        """
        model = self._require_model()
        sample_rate = self._sample_rate
        frames_per_second = int(model.frames_per_second)
        min_samples = max(1, sample_rate // frames_per_second)
        clips: list[dict[str, float]] = []
        owner_by_seek: dict[int, int] = {}
        starts: list[float] = []
        owners: list[int] = []
        offset = 0
        for owner, audio in enumerate(audios):
            total = int(audio.shape[0])
            for chunk in chunk_timestamps(
                total / sample_rate, self._spec.whisper_chunk_s
            ):
                begin = round(chunk["start"] * sample_rate)
                end = min(total, round(chunk["end"] * sample_rate))
                if end - begin < min_samples:
                    continue
                start_s = (offset + begin) / sample_rate
                clips.append({"start": start_s, "end": (offset + end) / sample_rate})
                # * Same arithmetic as BatchedInferencePipeline.forward's `seek`.
                seek = int(
                    (int(start_s * sample_rate) / sample_rate) * frames_per_second
                )
                owner_by_seek.setdefault(seek, owner)
                starts.append(start_s)
                owners.append(owner)
            offset += total
        parts: list[list[str]] = [[] for _ in audios]
        if not clips:
            return ["" for _ in audios]
        segments, _info = self._pipeline.transcribe(
            self._numpy.concatenate(list(audios)),
            batch_size=self._spec.config.batch_size,
            clip_timestamps=clips,
            **self._batched_options(),
        )
        for segment in segments:
            found = owner_by_seek.get(int(segment.seek))
            if found is None:
                # * Fallback: the chunk whose start precedes the segment start.
                position = bisect.bisect_right(starts, float(segment.start) + 1e-3)
                found = owners[max(0, position - 1)]
            text = str(segment.text or "").strip()
            if text:
                parts[found].append(text)
        return [" ".join(part) for part in parts]

    def _decode(self, path: Path) -> Any:  # noqa: ANN401 - numpy array
        return self._decode_audio(str(path), sampling_rate=self._sample_rate)

    def warmup(self, path: Path, *, slot: int, primary: bool) -> dict[str, object]:
        """Warm up; the primary batched warm-up runs one full batch.

        Args:
            path: Warm-up clip.
            slot: Worker slot (unused: CTranslate2 picks a free replica).
            primary: First warm-up of this model instance.

        Returns:
            Warm-up info.

        """
        if self._pipeline is not None and primary:
            batch_size = self._spec.config.batch_size
            texts = self._transcribe_audios_batched([self._decode(path)] * batch_size)
            return {
                "kind": "full-batch",
                "batch_size": batch_size,
                "chars": len(texts[0]),
            }
        text = self.transcribe([path], slot=slot)[0]
        return {"kind": "single", "chars": len(text)}

    def transcribe(self, paths: Sequence[Path], *, slot: int) -> list[str]:
        """Transcribe *paths*; ``B`` mode decodes them in shared forward passes.

        The lazy segment generators are consumed inside the calling thread.

        Args:
            paths: Audio files.
            slot: Worker slot (unused: CTranslate2 picks a free replica).

        Returns:
            One transcript per path.

        """
        del slot
        if self._pipeline is not None:
            return self._transcribe_audios_batched([self._decode(p) for p in paths])
        model = self._require_model()
        texts: list[str] = []
        for path in paths:
            segments, _info = model.transcribe(str(path), **self._spec.whisper_options)
            texts.append(_join_segments(segments))
        return texts

    def close(self) -> None:
        """Drop the model references."""
        self._pipeline = None
        self._model = None
        gc.collect()


class GigaAMEngine:
    """GigaAM (v3 ``e2e_ctc`` or Multilingual ``ctc``) loaded by the project wrapper.

    Inference calls the loaded remote-code model directly (see the module
    docstring); both revisions share the forward/decoder layout.
    """

    def __init__(self, spec: ChildSpec) -> None:
        """Store the spec; nothing is loaded until :meth:`load`.

        Args:
            spec: Child specification.

        """
        self._spec = spec
        self._torch: Any = None
        self._wrapper: Any = None
        self._model: Any = None
        self._asr: Any = None
        self._device: Any = None
        self._dtype: Any = None
        self._ffmpeg: str | None = None
        self._sample_rate = GIGAAM_DEFAULT_SAMPLE_RATE
        self._streams: list[Any] = []

    def load(self) -> dict[str, object]:
        """Load the model via the project wrapper and validate the device.

        ``gigaam`` uses ``GigaAME2ECtcWrapper`` (v3 ``e2e_ctc``), ``gigaam-ctc``
        uses ``GigaAMCtcWrapper`` (Multilingual ``ctc``, CPU only).

        Returns:
            Engine info for the result file.

        Raises:
            EngineError: When ffmpeg is missing, the wrapper fell back to
                another device, or the remote-code layout is unexpected.

        """
        cache_root = _configure_project_caches()
        torch = importlib.import_module("torch")
        config = self._spec.config
        torch.set_num_threads(config.threads)
        self._ffmpeg = shutil.which("ffmpeg")
        if self._ffmpeg is None:
            msg = "ffmpeg is not on PATH; GigaAM audio decoding requires it"
            raise EngineError(msg)
        multilingual = config.engine == "gigaam-ctc"
        wrapper_cls = GigaAMCtcWrapper if multilingual else GigaAME2ECtcWrapper
        wrapper = wrapper_cls(
            device=config.device,
            cache_dir=cache_root / "huggingface",
        )
        # ! The wrapper has no public load-only entry point; _load_model applies
        # ! the transformers 5.x / hydra patches and the device placement.
        wrapper._load_model()  # noqa: SLF001
        self._wrapper = wrapper
        active = wrapper.active_device
        if active != config.device:
            # ! A placement problem, not a measured saturation point: the
            # ! message deliberately carries no CUDA OOM wording.
            msg = (
                f"{DEVICE_MISMATCH_TAG}: GigaAM model placed on {active!r} instead "
                f"of {config.device!r} (CUDA not usable for this process; see the "
                "wrapper warning in the child log)"
            )
            raise DeviceMismatchError(msg)
        model = wrapper._model  # noqa: SLF001 - the loaded remote-code model
        asr = getattr(model, "model", None)
        missing = [attr for attr in _GIGAAM_ASR_ATTRS if not hasattr(asr, attr)]
        if asr is None or missing:
            msg = f"Unexpected GigaAM remote-code layout; missing {missing}"
            raise EngineError(msg)
        remote = sys.modules.get(type(asr).__module__)
        self._sample_rate = int(
            getattr(remote, "SAMPLE_RATE", GIGAAM_DEFAULT_SAMPLE_RATE),
        )
        first_param = next(iter(asr.parameters()))
        self._torch = torch
        self._model = model
        self._asr = asr
        self._device = first_param.device
        self._dtype = first_param.dtype
        if config.device == "cuda":
            if self._spec.gigaam_cuda_streams and config.slots > 1:
                self._streams = [torch.cuda.Stream() for _ in range(config.slots)]
            torch.cuda.synchronize()
        return {
            "backend": "gigaam-multilingual-ctc"
            if multilingual
            else "gigaam-v3-e2e_ctc",
            "repo": GIGAAM_HF_REPO_ID if multilingual else GIGAAM_V3_HF_REPO_ID,
            "revision": GIGAAM_HF_REVISION if multilingual else GIGAAM_V3_HF_REVISION,
            "torch": str(torch.__version__),
            "active_device": active,
            "dtype": str(self._dtype),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "cuda_streams": len(self._streams),
            "batch_size": config.batch_size,
            "window_s": GIGAAM_WINDOW_S,
            "sample_rate": self._sample_rate,
            "audio_loader": "ffmpeg s16le mono -> torch.frombuffer(bytearray)",
        }

    def _require_ready(self) -> tuple[Any, Any, str]:
        """Return ``(torch, asr, ffmpeg)`` of the loaded engine.

        Raises:
            EngineError: When :meth:`load` has not run.

        """
        if self._torch is None or self._asr is None or self._ffmpeg is None:
            msg = "GigaAM engine is not loaded"
            raise EngineError(msg)
        return self._torch, self._asr, self._ffmpeg

    def _load_wave(self, path: Path) -> Any:  # noqa: ANN401 - torch tensor
        """Decode *path* exactly like the remote ``load_audio`` (ffmpeg s16le mono).

        Args:
            path: Audio file.

        Returns:
            1-D float32 tensor in ``[-1, 1)``.

        Raises:
            EngineError: When ffmpeg fails.

        """
        torch, _asr, ffmpeg = self._require_ready()
        cmd = [
            ffmpeg,
            "-nostdin",
            "-threads",
            "0",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self._sample_rate),
            "-",
        ]
        completed = subprocess.run(  # noqa: S603 - resolved ffmpeg, fixed argv, no shell
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=_FFMPEG_TIMEOUT_S,
            creationflags=_NO_WINDOW,
        )
        if completed.returncode != 0:
            tail = completed.stderr.decode("utf-8", errors="replace")[-500:]
            msg = (
                f"ffmpeg failed to decode {path} (code {completed.returncode}): {tail}"
            )
            raise EngineError(msg)
        pcm = bytearray(completed.stdout)
        del pcm[len(pcm) - len(pcm) % 2 :]
        if not pcm:
            return torch.zeros(0, dtype=torch.float32)
        return torch.frombuffer(pcm, dtype=torch.int16).float() / _PCM16_SCALE

    def _split_windows(self, wave: Any) -> list[Any]:  # noqa: ANN401 - torch tensor
        """Split a waveform into 20 s windows, padding each to the STFT minimum.

        Args:
            wave: 1-D waveform.

        Returns:
            Window tensors.

        """
        torch, _asr, _ffmpeg = self._require_ready()
        window = int(GIGAAM_WINDOW_S * self._sample_rate)
        min_samples = math.ceil(GIGAAM_MIN_AUDIO_S * self._sample_rate)
        total = int(wave.shape[-1])
        windows = [wave[start : start + window] for start in range(0, total, window)]
        if not windows:
            windows = [wave]
        padded: list[Any] = []
        for piece in windows:
            missing = min_samples - int(piece.shape[-1])
            padded.append(
                torch.nn.functional.pad(piece, (0, missing)) if missing > 0 else piece,
            )
        return padded

    def _infer(self, windows: Sequence[Any], *, slot: int) -> list[str]:
        """Run forward + CTC decoding over *windows* in chunks of the batch size.

        Args:
            windows: Window tensors (each at most 20 s).
            slot: Worker slot (selects the CUDA stream in ``S`` mode).

        Returns:
            One decoded text per window.

        """
        torch, asr, _ffmpeg = self._require_ready()
        stream = self._streams[slot] if self._streams else None
        stream_ctx: contextlib.AbstractContextManager[object] = (
            torch.cuda.stream(stream)
            if stream is not None
            else contextlib.nullcontext()
        )
        batch_size = self._spec.config.batch_size
        texts: list[str] = []
        with torch.inference_mode(), stream_ctx:
            for start in range(0, len(windows), batch_size):
                chunk = list(windows[start : start + batch_size])
                lengths = torch.tensor(
                    [int(w.shape[-1]) for w in chunk], dtype=torch.long
                )
                batch = torch.zeros(
                    (len(chunk), int(lengths.max())),
                    dtype=chunk[0].dtype,
                )
                for row, piece in enumerate(chunk):
                    batch[row, : int(piece.shape[-1])] = piece
                wav = batch.to(self._device).to(self._dtype)
                encoded, encoded_len = asr.forward(wav, lengths.to(self._device))
                decoded = asr.decoding.decode(asr.head, encoded, encoded_len)
                texts.extend(_decoded_text(entry) for entry in decoded)
            if self._device.type == "cuda":
                active_stream = (
                    stream if stream is not None else torch.cuda.current_stream()
                )
                active_stream.synchronize()
        return texts

    def transcribe(self, paths: Sequence[Path], *, slot: int) -> list[str]:
        """Decode, window, infer (batched across all windows), and join per path.

        Args:
            paths: Audio files.
            slot: Worker slot.

        Returns:
            One transcript per path (window texts joined like the wrapper).

        """
        per_item = [self._split_windows(self._load_wave(path)) for path in paths]
        flat = [window for windows in per_item for window in windows]
        texts = self._infer(flat, slot=slot)
        outputs: list[str] = []
        offset = 0
        for windows in per_item:
            parts = [text.strip() for text in texts[offset : offset + len(windows)]]
            offset += len(windows)
            outputs.append(" ".join(part for part in parts if part))
        return outputs

    def warmup(self, path: Path, *, slot: int, primary: bool) -> dict[str, object]:
        """Warm up; the primary warm-up also checks parity with ``model.transcribe``.

        The primary warm-up runs single-threaded before any concurrent call, so
        the lazy encoder state is created exactly once.

        Args:
            path: Warm-up clip.
            slot: Worker slot.
            primary: First warm-up of this model instance.

        Returns:
            Warm-up info (``api_parity`` compares the benchmark path with the
            remote-code ``transcribe(path)`` used by the project wrapper).

        """
        if not primary:
            self.transcribe([path], slot=slot)
            return {"kind": "single"}
        torch, _asr, _ffmpeg = self._require_ready()
        wave = self._load_wave(path)
        info: dict[str, object] = {"kind": "primary"}
        if int(wave.shape[-1]) <= int(GIGAAM_WINDOW_S * self._sample_rate):
            reference = _decoded_text(self._model.transcribe(str(path))).strip()
            ours = self.transcribe([path], slot=slot)[0]
            info.update(
                api_parity=reference == ours,
                api_reference_text=reference,
                benchmark_text=ours,
            )
        else:
            info["api_parity"] = None
            self.transcribe([path], slot=slot)
        batch_size = self._spec.config.batch_size
        if batch_size > 1:
            self._infer(self._split_windows(wave)[:1] * batch_size, slot=slot)
            info["full_batch_warmup"] = batch_size
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        return info

    def close(self) -> None:
        """Release the model and cached VRAM through the wrapper."""
        self._asr = None
        self._model = None
        self._streams = []
        wrapper, self._wrapper = self._wrapper, None
        if wrapper is not None:
            wrapper.close()
        gc.collect()


def make_engine(spec: ChildSpec) -> SttEngine:
    """Create the engine for *spec*.

    Args:
        spec: Child specification.

    Returns:
        Unloaded engine.

    """
    if spec.config.engine == "whisper":
        return WhisperEngine(spec)
    return GigaAMEngine(spec)
