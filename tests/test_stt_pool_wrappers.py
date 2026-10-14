"""Strict devices, opt-in inference and pipeline wiring with external ML substitutes."""

from __future__ import annotations

import contextlib
import importlib
import json
import shutil
import subprocess
import sys
import types
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pytest

from core.ffmpeg import MediaProbeError, probe_media_duration
from core.gigaam_ctc_wrapper import GigaAMCtcWrapper, iter_wav_windows
from core.gigaam_e2e_ctc_wrapper import GigaAME2ECtcWrapper
from core.stt_errors import CudaOutOfMemoryError, DeviceUnavailableError
from core.stt_pool.procs import default_worker_launch
from core.whisper_batch import speech_span, transcribe_clips_batched

if TYPE_CHECKING:
    from collections.abc import Iterator

_ROOT = Path(__file__).resolve().parents[1]
_RATE = 16000


class Tensor:
    """Tiny external tensor substitute; all memory stays in NumPy on CPU."""

    def __init__(self, values: object, device: str = "cpu") -> None:
        self.values = np.asarray(values)
        self.device = types.SimpleNamespace(type=device)
        self.dtype = "float32"

    def to(self, target: object) -> Self:
        """Represent a device move without touching CUDA."""
        device = getattr(target, "type", target)
        if device in {"cpu", "cuda"}:
            self.device.type = device
        return self


class Runtime:
    """Track only interactions at real-model and torch boundaries."""

    def __init__(self) -> None:
        self.available = True
        self.failure: str | None = None
        self.threads: list[int] = []
        self.loads: list[dict[str, object]] = []
        self.moves: list[str] = []
        self.inference_devices: list[str] = []
        self.languages: list[str | None] = []
        self.batches: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
        self.module = types.ModuleType("torch")
        self.module.cuda = types.SimpleNamespace(
            is_available=lambda: self.available,
            mem_get_info=lambda: (16 * 1024**3, 16 * 1024**3),
            empty_cache=lambda: None,
        )
        self.module.set_num_threads = self.threads.append
        self.module.device = lambda _target: contextlib.nullcontext()
        self.module.inference_mode = contextlib.nullcontext
        self.module.from_numpy = Tensor
        self.module.tensor = lambda values, **_kwargs: Tensor(values)
        self.module.long = "int64"

    def fail(self, stage: str) -> None:
        """Raise a realistic external CUDA OOM when the selected stage is reached."""
        if self.failure == stage:
            msg = f"CUDA out of memory: {stage}"
            raise RuntimeError(msg)


class Asr:
    """External GigaAM forward/decode substitute with per-call tensors."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.parameter = Tensor([0])
        self.head = object()
        self.decoding = types.SimpleNamespace(
            decode=lambda _head, encoded, _lengths: [
                str(float(row[0])) for row in encoded
            ]
        )

    def parameters(self) -> list[Tensor]:
        """Expose the resident device and dtype to the real wrapper."""
        return [self.parameter]

    def forward(
        self, wav: Tensor, lengths: Tensor
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Record prepared samples and optionally fail at the external inference boundary."""
        self.runtime.inference_devices.append(wav.device.type)
        self.runtime.fail("forward")
        self.runtime.batches.append((wav.values.copy(), lengths.values.copy()))
        return wav.values, lengths.values


class GigaModel:
    """External Hugging Face model substitute with observable device moves."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.model = Asr(runtime)

    def parameters(self) -> list[Tensor]:
        """Expose only fake tensors, never real model weights."""
        return self.model.parameters()

    def buffers(self) -> list[Tensor]:
        """Expose the empty buffer set of this stand-in."""
        return []

    def float(self) -> Self:
        """Keep the stand-in in float32."""
        return self

    def eval(self) -> Self:
        """Keep inference behavior independent of training state."""
        return self

    def to(self, device: str) -> Self:
        """Move the stand-in or reproduce a partially failed CUDA move."""
        self.runtime.moves.append(device)
        self.model.parameter.to(device)
        if device == "cuda":
            self.runtime.fail("move")
        return self

    def transcribe(self, path: str) -> str:
        """Fail or decode at the external single-window boundary."""
        device = self.model.parameter.device.type
        self.runtime.inference_devices.append(device)
        if device == "cuda":
            self.runtime.fail("window")
        return Path(path).stem


def _discard_fake_imports(cached: set[str]) -> None:
    """Discard modules first imported against a temporary ML substitute."""
    for name in (
        "core.pipelines",
        "core.whisperx_wrapper",
        "core.diarization",
        "core.llm_formatter",
    ):
        if name not in cached:
            module = sys.modules.pop(name, None)
            parent = sys.modules.get("core")
            attribute = name.rsplit(".", 1)[1]
            if (
                module is not None
                and parent is not None
                and getattr(parent, attribute, None) is module
            ):
                delattr(parent, attribute)


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[Runtime]:
    """Install only external ML modules before importing any heavy wrapper."""
    boundary = Runtime()
    cached = set(sys.modules)

    class WhisperModel:
        """External faster-whisper stand-in including lazy-generator failures."""

        def __init__(self, model: str, **kwargs: object) -> None:
            boundary.loads.append({"model": model, **kwargs})
            self.device = str(kwargs["device"])
            boundary.fail("load")

        def transcribe(
            self, path: str, *, language: str | None, **_kwargs: object
        ) -> tuple[Iterator[types.SimpleNamespace], dict[str, object]]:
            """Raise OOM on generator iteration or identify the per-call language."""
            boundary.languages.append(language)

            def segments() -> Iterator[types.SimpleNamespace]:
                boundary.inference_devices.append(self.device)
                if self.device == "cuda":
                    boundary.fail("inference")
                yield types.SimpleNamespace(
                    start=0.0, end=0.1, text=f"{language or 'auto'}:{Path(path).name}"
                )

            return segments(), {}

    class AutoModel:
        """External Hugging Face loader substitute."""

        @staticmethod
        def from_pretrained(repo: str, **kwargs: object) -> GigaModel:
            """Record weight loading without downloading or allocating a model."""
            boundary.loads.append({"repo": repo, **kwargs})
            return GigaModel(boundary)

    class PreTrainedModel:
        """Minimal external transformers compatibility API."""

        @staticmethod
        def _finalize_model_loading(*_args: object) -> None:
            """Complete stand-in loading without tied weights."""

    modules = {
        name: types.ModuleType(name)
        for name in (
            "faster_whisper",
            "whisperx",
            "transformers",
            "transformers.modeling_utils",
            "hydra",
            "hydra.utils",
        )
    }
    modules["faster_whisper"].WhisperModel = WhisperModel
    modules["transformers"].AutoModel = AutoModel
    modules["transformers.modeling_utils"].PreTrainedModel = PreTrainedModel
    modules["hydra.utils"].instantiate = lambda *_args, **_kwargs: None
    modules["hydra"].utils = modules["hydra.utils"]
    for name, module in {"torch": boundary.module, **modules}.items():
        monkeypatch.setitem(sys.modules, name, module)
    wrapper = importlib.import_module("core.whisperx_wrapper")
    monkeypatch.setattr(wrapper, "torch_mod", boundary.module)
    monkeypatch.setattr(wrapper, "fw_whisper_cls", WhisperModel)
    monkeypatch.setattr(wrapper, "whisperx_mod", None)
    try:
        yield boundary
    finally:
        _discard_fake_imports(cached)


def _wav(path: Path, *, frames: int = 800) -> Path:
    """Write a small canonical silence WAV for real streaming code."""
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(_RATE)
        writer.writeframes(b"\x00\x00" * frames)
    return path


@pytest.mark.parametrize("provider", ["whisper", "gigaam"])
def test_strict_cuda_unavailable_fails_before_loading_cpu_weights(
    runtime: Runtime, provider: str
) -> None:
    """Catch strict workers silently loading on CPU when the requested GPU is absent."""
    runtime.available = False
    wrapper = importlib.import_module("core.whisperx_wrapper")
    model = (
        wrapper.WhisperXWrapper(device="cuda", strict_device=True)
        if provider == "whisper"
        else GigaAME2ECtcWrapper(device="cuda", strict_device=True)
    )
    with pytest.raises(DeviceUnavailableError):
        model.ensure_loaded()
    assert not runtime.loads


@pytest.mark.parametrize("stage", ["load", "inference"])
def test_strict_whisper_cuda_oom_is_typed_and_never_retries_cpu(
    runtime: Runtime, tmp_path: Path, stage: str
) -> None:
    """Catch load or lazy-generator CUDA OOM causing unreserved CPU recovery."""
    runtime.failure = stage
    wrapper = importlib.import_module("core.whisperx_wrapper").WhisperXWrapper(
        device="cuda", strict_device=True
    )
    with pytest.raises(CudaOutOfMemoryError):
        wrapper.transcribe(tmp_path / "clip.wav")
    assert [entry["device"] for entry in runtime.loads] == ["cuda"]
    assert "cpu" not in runtime.inference_devices


@pytest.mark.parametrize("stage", ["move", "window"])
@pytest.mark.parametrize("strict", [False, True])
def test_gigaam_strict_device_preserves_default_recovery_only_when_opted_out(
    runtime: Runtime, tmp_path: Path, stage: str, *, strict: bool
) -> None:
    """Catch pool strict mode falling back on move/window OOM or legacy mode losing its recovery."""
    runtime.failure = stage
    wrapper = GigaAME2ECtcWrapper(device="cuda", strict_device=strict)
    audio = _wav(tmp_path / "clip.wav")
    if strict:
        with pytest.raises(CudaOutOfMemoryError):
            wrapper.transcribe(audio)
        assert "cpu" not in runtime.moves
        assert "cpu" not in runtime.inference_devices
        assert wrapper.active_device == (None if stage == "move" else "cuda")
    else:
        result = wrapper.transcribe(audio)
        assert result["segments"]
        assert wrapper.active_device == "cpu"
        assert runtime.inference_devices[-1] == "cpu"


@pytest.mark.parametrize("configured", [False, True])
def test_thread_settings_are_sent_only_when_explicit(
    runtime: Runtime, *, configured: bool
) -> None:
    """Catch default thread settings changing or pool capacity never reaching the ML runtime."""
    module = importlib.import_module("core.whisperx_wrapper")
    options = {"cpu_threads": 2, "num_workers": 3} if configured else {}
    whisper = module.WhisperXWrapper(device="cpu", **options)
    whisper.ensure_loaded()
    whisper.ensure_loaded()
    assert len(runtime.loads) == 1
    if configured:
        assert runtime.loads[0]["cpu_threads"] == 2
        assert runtime.loads[0]["num_workers"] == 3
    else:
        assert "cpu_threads" not in runtime.loads[0]
        assert "num_workers" not in runtime.loads[0]
    gigaam = GigaAMCtcWrapper(torch_threads=2 if configured else None)
    gigaam.ensure_loaded()
    gigaam.ensure_loaded()
    assert runtime.threads == ([2] if configured else [])


def test_gigaam_threadsafe_warmup_and_batches_use_real_window_lengths(
    runtime: Runtime,
) -> None:
    """Catch missing warm-up, shared per-call tensor state or lost zero-padded window lengths."""
    wrapper = GigaAMCtcWrapper(threadsafe=True)
    wrapper.ensure_loaded()
    wrapper.ensure_loaded()
    assert len(runtime.batches) == 1
    assert runtime.batches[0][1].tolist() == [20 * _RATE]
    windows = [
        np.full(640, -0.5, dtype=np.float32),
        np.full(1000, 0.25, dtype=np.float32),
    ]
    texts = wrapper.transcribe_window_batch(windows)
    assert len(texts) == 2
    samples, lengths = runtime.batches[-1]
    assert samples.shape == (2, 1000)
    assert lengths.tolist() == [640, 1000]
    np.testing.assert_array_equal(samples[0, :640], windows[0])
    assert not np.any(samples[0, 640:])
    np.testing.assert_array_equal(samples[1], windows[1])


@pytest.mark.parametrize("strict", [False, True])
def test_gigaam_cuda_lockfree_oom_never_retries_cpu(
    runtime: Runtime, *, strict: bool
) -> None:
    """Catch an opt-in batch bypassing scheduler admission with CPU OOM recovery."""
    wrapper = GigaAME2ECtcWrapper(device="cuda", strict_device=strict, threadsafe=True)
    wrapper.ensure_loaded()
    runtime.failure = "forward"
    with pytest.raises(CudaOutOfMemoryError):
        wrapper.transcribe_window_batch([np.zeros(800, dtype=np.float32)])
    assert wrapper.active_device == "cuda"
    assert "cpu" not in runtime.inference_devices


@pytest.mark.parametrize(
    ("frames", "expected_sizes"),
    [(0, [640]), (1, [640]), (320000, [320000]), (320001, [320000, 640])],
)
def test_gigaam_streaming_preserves_boundaries_and_padding(
    tmp_path: Path, frames: int, expected_sizes: list[int]
) -> None:
    """Catch exact-window tails being lost, empty audio crashing STFT or incorrect tail timestamps."""
    audio = _wav(tmp_path / "clip.wav", frames=frames)
    windows = list(iter_wav_windows(audio))
    assert [window.samples.size for window in windows] == expected_sizes
    assert windows[0].start_s == 0
    assert windows[-1].end_s == frames / _RATE
    assert all(window.samples.dtype == np.float32 for window in windows)


@pytest.mark.integration
@pytest.mark.xdist_group(name="pool_processes")
def test_pipeline_concurrent_language_overrides_do_not_leak_between_jobs(
    runtime: Runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch per-call language mutating shared state or concurrent jobs sharing intermediate files."""
    real_popen = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []
    launch = default_worker_launch(
        project_root=_ROOT, module="tests.stt_pool_fake_worker"
    )

    def media_process_boundary(
        argv: list[str],
        **kwargs: Any,  # noqa: ANN401 - subprocess boundary signature
    ) -> subprocess.Popen[bytes]:
        command = (*launch.argv, "--media-args", json.dumps(argv))
        child = real_popen(
            command,
            cwd=_ROOT,
            env=dict(launch.env),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            **kwargs,
        )
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", media_process_boundary)
    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "WHISPER_CACHE",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.setenv(key, str(tmp_path / "cache"))
    pipeline_module = importlib.import_module("core.pipelines")
    pipeline = pipeline_module.LocalPipeline(
        model_root=tmp_path / "models",
        language="en",
        device="cpu",
        strict_device=True,
        num_workers=2,
        enable_diarization=False,
        enable_dialog_blocks=False,
    )
    pipeline.ensure_stt_loaded()
    audio = _wav(tmp_path / "clip.wav")
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    pipeline.process, audio, tmp_path / label, language=language
                )
                for label, language in (("explicit", "ru"), ("detect", None))
            ]
            documents = [future.result(timeout=10) for future in futures]
        default = pipeline.process(audio, tmp_path / "default")
        assert set(runtime.languages[:2]) == {"ru", None}
        assert runtime.languages[2] == "en"
        assert pipeline.language == "en"
        assert "ru" in documents[0].get_full_text()
        assert "auto" in documents[1].get_full_text()
        assert "en" in default.get_full_text()
        assert (tmp_path / "explicit" / "clip.partial.txt").exists()
        assert (tmp_path / "detect" / "clip.partial.txt").exists()
        assert not tuple(tmp_path.glob("*/_work/*.wav"))
    finally:
        pipeline.close()
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()


@pytest.mark.integration
def test_probe_duration_uses_real_ffprobe_on_the_known_fixture() -> None:
    """Catch parsing or command construction making a valid duration unusable for ETA."""
    duration = probe_media_duration(
        _ROOT / "tests" / "fixtures" / "test_video_short.mp4"
    )
    assert 15.0 <= duration <= 17.5


@pytest.mark.integration
@pytest.mark.parametrize("exists", [False, True])
def test_probe_invalid_media_reports_a_typed_error(
    tmp_path: Path, *, exists: bool
) -> None:
    """Catch missing or corrupt media being reported as a zero-length successful probe."""
    path = tmp_path / "invalid.mp4"
    if exists:
        path.write_bytes(b"invalid media")
    with pytest.raises(MediaProbeError) as error:
        probe_media_duration(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_probe_invalid_timeout_fails_without_waiting(
    monkeypatch: pytest.MonkeyPatch, timeout: float
) -> None:
    """Catch exhausted client deadlines starting an unbounded probe."""

    def reject_probe_launch(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid timeout must be rejected before launching ffprobe")

    monkeypatch.setattr(shutil, "which", lambda executable: executable)
    monkeypatch.setattr(subprocess, "Popen", reject_probe_launch)
    with pytest.raises(MediaProbeError):
        probe_media_duration(
            _ROOT / "tests" / "fixtures" / "test_video_short.mp4", timeout_s=timeout
        )


def test_whisper_batch_maps_multiple_segments_and_skips_subframe_clips(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch segments attached to another file, dropped segment breaks or empty-clip index shifts."""

    class BatchPipeline:
        """External batched decoder honoring faster-whisper seek metadata."""

        def __init__(self, model: object) -> None:
            pass

        def transcribe(
            self,
            audio: np.ndarray[Any, Any],
            **kwargs: Any,  # noqa: ANN401 - external decoder signature
        ) -> tuple[list[types.SimpleNamespace], object]:
            """Return noncontiguous segments using the external decoder's frame seeks."""
            assert kwargs["language"] == "ru"
            stamps = kwargs["clip_timestamps"]
            assert len(stamps) == 2
            sample_starts = [int(stamp["start"] * _RATE) for stamp in stamps]
            assert audio[sample_starts[0]] == -0.5
            assert audio[sample_starts[1]] == 0.25
            # * The external API reports seek as frames at the decoded clip's start.
            seeks = [int(start / _RATE * 100) for start in sample_starts]
            return [
                types.SimpleNamespace(seek=seeks[0], text=" first "),
                types.SimpleNamespace(seek=seeks[1], text=" second "),
                types.SimpleNamespace(seek=seeks[0], text=" continuation "),
            ], object()

    monkeypatch.setattr(
        sys.modules["faster_whisper"],
        "BatchedInferencePipeline",
        BatchPipeline,
        raising=False,
    )
    model = types.SimpleNamespace(
        feature_extractor=types.SimpleNamespace(sampling_rate=_RATE),
        frames_per_second=100,
    )
    clips = [
        np.zeros(1, dtype=np.float32),
        np.full(320, -0.5, dtype=np.float32),
        np.full(480, 0.25, dtype=np.float32),
    ]
    texts = transcribe_clips_batched(model, clips, language="ru")
    assert texts == ["", "first\ncontinuation", "second"]


def test_whisper_batch_lazy_cuda_oom_is_typed_without_retry(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch lazy batched inference escaping the typed CUDA OOM contract."""

    class BatchPipeline:
        """External batched decoder whose iterator raises OOM."""

        def __init__(self, model: object) -> None:
            pass

        def transcribe(
            self, _audio: object, **_kwargs: object
        ) -> tuple[Iterator[object], object]:
            """Defer a realistic OOM until segment consumption."""

            def segments() -> Iterator[object]:
                yield from ()
                msg = "CUDA out of memory"
                raise RuntimeError(msg)

            return segments(), object()

    monkeypatch.setattr(
        sys.modules["faster_whisper"],
        "BatchedInferencePipeline",
        BatchPipeline,
        raising=False,
    )
    model = types.SimpleNamespace(
        feature_extractor=types.SimpleNamespace(sampling_rate=_RATE),
        frames_per_second=100,
        model=types.SimpleNamespace(device="cuda"),
    )
    with pytest.raises(CudaOutOfMemoryError):
        transcribe_clips_batched(
            model, [np.zeros(320, dtype=np.float32)], language="ru"
        )


def test_speech_span_uses_the_first_and_last_vad_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch trimmed short clips dropping later speech or treating silence as speech."""
    module = types.ModuleType("faster_whisper.vad")
    speeches = [{"start": 1600, "end": 3200}, {"start": 6400, "end": 8000}]
    module.VadOptions = types.SimpleNamespace
    module.get_speech_timestamps = lambda _audio, **_kwargs: speeches
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", module)
    samples = np.zeros(16000, dtype=np.float32)
    assert speech_span(samples) == (0.1, 0.5)
    speeches.clear()
    assert speech_span(samples) is None
