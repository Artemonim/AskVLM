"""Controlled ML boundaries and synchronization for real pool-engine tests."""

from __future__ import annotations

import queue
import sys
import threading
import types
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import core
from core.stt_pool import engines
from core.stt_pool.engine_api import WorkerSpec, WorkItem
from core.stt_pool.keys import make_key

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import numpy as np
    import pytest
    from numpy.typing import NDArray

    from core.stt_pool.engine_api import EngineResult, PoolEngine
    from core.stt_pool.keys import Backend, Priority

RATE = 16000
# * Failure-only bounds allow slow media preparation on a contended Windows host.
RESULT_TIMEOUT_S = 30.0
GATE_TIMEOUT_S = 30.0
PREP_TIMEOUT_S = 30.0


@dataclass
class ProcessCall:
    """Observable input to the substituted one-shot ML pipeline."""

    path: Path
    work_dir: Path
    language: str | None
    thread_id: int


@dataclass
class BatchCall:
    """One external forward call and the clips or windows it consumes."""

    markers: list[int]
    lengths: list[int]
    thread_id: int
    language: str | None = None


class FakeDocument:
    """Document boundary carrying text from the fake recognizer."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.segments: list[object] = []

    def get_full_text(self) -> str:
        """Return the recognizer text before the engine normalizes it."""
        return self._text


class FakeGigaam:
    """Fake CUDA inference endpoint; WAV decoding and engine loops remain real."""

    def __init__(self, runtime: EngineRuntime, device: str) -> None:
        self._runtime = runtime
        self.active_device = device

    def transcribe_window_batch(
        self, windows: Sequence[NDArray[np.float32]]
    ) -> list[str]:
        """Record every real window and return its identity or a configured failure."""
        markers = [round(float(window[0]) * 32768) for window in windows]
        self._runtime.record_batch(markers, [len(window) for window in windows])
        return [
            self._runtime.gigaam_texts.get(marker, str(marker)) for marker in markers
        ]


class FakeWhisper:
    """Expose the loaded fake faster-whisper model without touching CUDA."""

    def __init__(self, device: str) -> None:
        self.active_device = device
        self._model = types.SimpleNamespace(
            model=types.SimpleNamespace(device=device),
            feature_extractor=types.SimpleNamespace(sampling_rate=RATE),
            frames_per_second=100,
        )

    def loaded_model(self) -> object:
        """Return model metadata used by the real cross-file batch planner."""
        return self._model


class FakePipeline:
    """Explicitly allowed LocalPipeline ML substitute with controllable inference."""

    def __init__(self, runtime: EngineRuntime, options: dict[str, object]) -> None:
        self._runtime = runtime
        self.options = options
        device = runtime.loaded_device or str(options["device"])
        provider = str(options["stt_provider"])
        self.gigaam = None if provider == "whisper" else FakeGigaam(runtime, device)
        self._whisper = FakeWhisper(device)

    def ensure_stt_loaded(self) -> None:
        """Count one shared weight load, optionally injecting an external failure."""
        self._runtime.load_count += 1
        if self._runtime.load_error is not None:
            raise self._runtime.load_error

    def require_whisperx(self) -> FakeWhisper:
        """Return the configured fake Whisper endpoint."""
        return self._whisper

    def process(
        self,
        path: Path,
        work_dir: Path,
        *,
        should_cancel: Callable[[], bool],
        language: str | None,
    ) -> FakeDocument:
        """Block only at the model boundary and preserve each job's language."""
        runtime = self._runtime
        runtime.process_calls.put(
            ProcessCall(path, work_dir, language, threading.get_ident())
        )
        runtime.inferences.append((path.name, threading.get_ident()))
        gate = runtime.process_gates.get(path.name)
        if gate is not None and not gate.wait(GATE_TIMEOUT_S):
            msg = "test did not release the fake pipeline inference"
            raise TimeoutError(msg)
        if should_cancel():
            msg = "fake inference cancelled"
            raise RuntimeError(msg)
        outcome = runtime.pipeline_texts.get(
            path.name, f" {language or 'auto'}:{path.name} \n"
        )
        if isinstance(outcome, Exception):
            raise outcome
        return FakeDocument(outcome)

    def close(self, *, aggressive: bool) -> None:  # noqa: ARG002 - production boundary keyword
        """Count external model unloading without changing engine bookkeeping."""
        self._runtime.unload_count += 1


class EngineRuntime:
    """Install controllable models and VAD while retaining real engine orchestration."""

    def __init__(self) -> None:
        self.pipelines: list[FakePipeline] = []
        self.load_count = 0
        self.unload_count = 0
        self.loaded_device: str | None = None
        self.load_error: Exception | None = None
        self.process_calls: queue.Queue[ProcessCall] = queue.Queue()
        self.batch_calls: queue.Queue[BatchCall] = queue.Queue()
        self.inferences: list[tuple[str, int]] = []
        self.process_gates: dict[str, threading.Event] = {}
        self.batch_gate: threading.Event | None = None
        self.decode_gates: dict[str, threading.Event] = {}
        self.pipeline_texts: dict[str, str | Exception] = {}
        self.gigaam_texts: dict[int, str] = {}
        self.batch_errors: dict[int, Exception] = {}
        self.audio: dict[str, NDArray[np.float32] | Exception] = {}
        self.vad_spans: dict[int, list[dict[str, int]] | Exception] = {}
        self.handles: list[EngineHandle] = []

    def new_pipeline(self, **options: object) -> FakePipeline:
        """Construct the stand-in at the production pipeline-constructor boundary."""
        pipeline = FakePipeline(self, options)
        self.pipelines.append(pipeline)
        return pipeline

    def record_batch(
        self, markers: list[int], lengths: list[int], language: str | None = None
    ) -> None:
        """Record one inference operation before its controlled return or failure."""
        self.batch_calls.put(
            BatchCall(markers, lengths, threading.get_ident(), language)
        )
        self.inferences.append((f"batch:{language or 'gigaam'}", threading.get_ident()))
        if self.batch_gate is not None and not self.batch_gate.wait(GATE_TIMEOUT_S):
            msg = "test did not release fake batch inference"
            raise TimeoutError(msg)
        for marker in markers:
            error = self.batch_errors.get(marker)
            if error is not None:
                raise error

    def decode_audio(self, path: str, *, sampling_rate: int) -> NDArray[np.float32]:
        """Provide bounded decoded samples or an external decoder failure."""
        if sampling_rate != RATE:
            msg = "fake decoder only accepts 16 kHz audio"
            raise ValueError(msg)
        name = Path(path).name
        gate = self.decode_gates.get(name)
        if gate is not None and not gate.wait(GATE_TIMEOUT_S):
            msg = "test did not release fake audio decoding"
            raise TimeoutError(msg)
        audio = self.audio[name]
        if isinstance(audio, Exception):
            raise audio
        return audio

    def get_speech_timestamps(
        self, audio: NDArray[np.float32], **_options: object
    ) -> list[dict[str, int]]:
        """Return VAD boundaries keyed by the constant identity of an audio clip."""
        marker = round(float(audio[0]) * 32768)
        spans = self.vad_spans.get(marker, [{"start": 0, "end": len(audio)}])
        if isinstance(spans, Exception):
            raise spans
        return spans

    def install(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Substitute only the allowed pipeline and external faster-whisper boundaries."""
        pipeline_module = types.ModuleType("core.pipelines")
        pipeline_module.LocalPipeline = self.new_pipeline
        monkeypatch.setitem(sys.modules, "core.pipelines", pipeline_module)
        monkeypatch.setattr(core, "pipelines", pipeline_module, raising=False)
        whisper_module = types.ModuleType("faster_whisper")
        audio_module = types.ModuleType("faster_whisper.audio")
        vad_module = types.ModuleType("faster_whisper.vad")
        audio_module.decode_audio = self.decode_audio
        vad_module.VadOptions = types.SimpleNamespace
        vad_module.get_speech_timestamps = self.get_speech_timestamps
        runtime = self

        class BatchPipeline:
            """External decoder substitute honoring faster-whisper seek metadata."""

            def __init__(self, model: object) -> None:
                pass

            def transcribe(
                self,
                audio: NDArray[np.float32],
                **options: Any,  # noqa: ANN401 - untyped external decoder boundary
            ) -> tuple[Iterator[types.SimpleNamespace], object]:
                """Generate independently identifiable text for each concatenated clip."""
                stamps = options["clip_timestamps"]
                starts = [int(stamp["start"] * RATE) for stamp in stamps]
                ends = [int(stamp["end"] * RATE) for stamp in stamps]
                markers = [round(float(audio[start]) * 32768) for start in starts]
                language = str(options["language"])

                def segments() -> Iterator[types.SimpleNamespace]:
                    runtime.record_batch(
                        markers,
                        [end - start for start, end in zip(starts, ends, strict=True)],
                        language,
                    )
                    for start, marker in zip(starts, markers, strict=True):
                        yield types.SimpleNamespace(
                            seek=int(start / RATE * 100), text=f" {language}:{marker} "
                        )

                return segments(), object()

        whisper_module.BatchedInferencePipeline = BatchPipeline
        for name, module in (
            ("faster_whisper", whisper_module),
            ("faster_whisper.audio", audio_module),
            ("faster_whisper.vad", vad_module),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        # * Production directories still use mkdtemp and are confined to the test root.
        monkeypatch.setattr(engines.tempfile, "tempdir", str(tmp_path))

    def open(self, spec: WorkerSpec) -> EngineHandle:
        """Load and start a real engine over the substituted ML endpoints."""
        handle = EngineHandle(engines.create_engine(spec))
        self.handles.append(handle)
        handle.engine.load()
        handle.engine.start(handle.results.put)
        return handle

    def release_and_close(self) -> None:
        """Release every artificial wait and join all engine threads on every exit."""
        for gate in (*self.process_gates.values(), *self.decode_gates.values()):
            gate.set()
        if self.batch_gate is not None:
            self.batch_gate.set()
        for handle in self.handles:
            handle.engine.close()


class EngineHandle:
    """Own a real engine and its observable result queue."""

    def __init__(self, engine: PoolEngine) -> None:
        self.engine = engine
        self.results: queue.Queue[EngineResult] = queue.Queue()

    def result(self) -> EngineResult:
        """Return one result with a liveness deadline, without sleeping or polling."""
        return self.results.get(timeout=RESULT_TIMEOUT_S)

    def wait_ready(self, item: WorkItem) -> None:
        """Arrange a cancel while the real prep thread has released the job."""
        engine = self.engine
        # * Internal state is only a barrier; assertions concern external results.
        with engine._cond:  # noqa: SLF001 - deterministic ownership barrier
            ready = engine._cond.wait_for(  # noqa: SLF001 - deterministic ownership barrier
                lambda: any(
                    job.key == (item.job_id, item.attempt_id)
                    for job in engine._ready  # noqa: SLF001 - deterministic ownership barrier
                ),
                timeout=PREP_TIMEOUT_S,
            )
        if not ready:
            msg = "fake inference blocker did not leave the next job prepared"
            raise TimeoutError(msg)


def spec(
    provider: str = "whisper", backend: Backend = "cpu", **options: object
) -> WorkerSpec:
    """Build a deterministic real spec with capacity two and caller-chosen options."""
    return WorkerSpec(
        "engine-test",
        "test-instance",
        make_key(provider, "Org/RequestedModel", "auto", backend),
        2,
        2,
        0 if backend == "cpu" else 3,
        "cpu" if backend == "cpu" else "cuda",
        options,
        "core.stt_pool.engines:create_engine",
    )


def item(
    path: Path,
    *,
    seq: int = 1,
    language: str | None = "ru",
    duration_s: float | None = 1.0,
    priority: Priority = "normal",
    call_id: str = "call-A",
) -> WorkItem:
    """Give one media input stable queue and attempt identities."""
    job_id = f"{seq:032x}"
    return WorkItem(
        job_id, f"{job_id}:1", path, language, duration_s, None, priority, call_id, seq
    )


def wav(path: Path, markers: Sequence[int], *, window_s: float = 20.0) -> Path:
    """Write tiny deterministic PCM inputs for the real WAV-window reader."""
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        for marker in markers:
            sample = marker.to_bytes(2, "little", signed=True)
            writer.writeframes(sample * round(window_s * RATE))
    return path
