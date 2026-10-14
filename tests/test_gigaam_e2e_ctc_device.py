"""CUDA placement tests for the GigaAM-v3 e2e CTC wrapper (``--device auto|cpu|cuda``).

``torch`` and the model are replaced at the wrapper's ``_ensure_deps`` seam by a
tiny stand-in CUDA runtime, so the real windowing, load patches, device policy,
OOM fallback, and unload logic run without a GPU.

Expected values come from the requirement: ``cpu`` never uses CUDA; ``cuda``
uses CUDA when available (CPU otherwise); ``auto`` uses CUDA only with at least
2.5 GiB free VRAM; an out-of-memory error falls back to CPU for the whole model
(on move) or for the failing window and every later window (during transcribe).
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import types
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import pytest
from typer.testing import CliRunner

import cli
import core.pipelines as pl
from core.gigaam_ctc_wrapper import GigaAMCtcWrapper
from core.gigaam_device import is_cuda_oom_error
from core.gigaam_e2e_ctc_wrapper import GigaAME2ECtcWrapper

if TYPE_CHECKING:
    from collections.abc import Callable

_GIB = 1024**3
# * Requirement: ``--device auto`` picks CUDA only with >= 2.5 GiB free VRAM.
_AUTO_MIN_FREE_BYTES = int(2.5 * _GIB)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_RATE = 8000


class OutOfMemoryError(RuntimeError):
    """Stand-in for ``torch.OutOfMemoryError`` (recognised by its class name)."""


class _FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type


class _FakeTensor:
    def __init__(self, device_type: str = "cpu") -> None:
        self.device = _FakeDevice(device_type)


class _FakeCuda:
    """CUDA runtime counters; ``mem_get_info``/``empty_cache`` record their use."""

    def __init__(
        self,
        *,
        available: bool = True,
        free_bytes: int = 8 * _GIB,
        mem_info_error: BaseException | None = None,
    ) -> None:
        self.available = available
        self.free_bytes = free_bytes
        self.mem_info_error = mem_info_error
        self.mem_get_info_calls = 0
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self.available

    def mem_get_info(self) -> tuple[int, int]:
        self.mem_get_info_calls += 1
        if self.mem_info_error is not None:
            raise self.mem_info_error
        return self.free_bytes, 8 * _GIB

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda


class _FakeModel:
    """GigaAM stand-in whose tensors live on a named device.

    ``transcribe`` returns the audio file stem, so the output shows which
    window produced which text. ``transcribe_log`` records, per attempt, the
    device the model sat on and the window stem.
    """

    def __init__(
        self,
        *,
        patches_active: Callable[[], bool] | None = None,
        move_error: BaseException | None = None,
        oom_on_cuda_call: int | None = None,
        window_error: BaseException | None = None,
        fixed_text: str | None = None,
    ) -> None:
        self._tensors = [_FakeTensor() for _ in range(3)]
        self._patches_active = patches_active
        self._move_error = move_error
        self._oom_on_cuda_call = oom_on_cuda_call
        self._window_error = window_error
        self._fixed_text = fixed_text
        self._cuda_calls = 0
        self.dtype = "fp16"
        self.training = True
        self.half_called = False
        self.patches_active_during_cuda_move: bool | None = None
        self.transcribe_log: list[tuple[str, str]] = []

    @property
    def device_type(self) -> str:
        return self._tensors[0].device.type

    def parameters(self) -> list[_FakeTensor]:
        return self._tensors[:2]

    def buffers(self) -> list[_FakeTensor]:
        return self._tensors[2:]

    def float(self) -> Self:
        self.dtype = "fp32"
        return self

    def half(self) -> Self:
        self.half_called = True
        self.dtype = "fp16"
        return self

    def eval(self) -> Self:
        self.training = False
        return self

    def to(self, device: str) -> Self:
        if device == "cuda":
            if self._patches_active is not None:
                self.patches_active_during_cuda_move = self._patches_active()
            if self._move_error is not None:
                # * A failed move leaves the model split across devices.
                self._tensors[0].device.type = "cuda"
                raise self._move_error
        for tensor in self._tensors:
            tensor.device.type = device
        return self

    def transcribe(self, path: str) -> str:
        stem = Path(path).stem
        device = self.device_type
        self.transcribe_log.append((device, stem))
        if self._window_error is not None:
            raise self._window_error
        if device == "cuda":
            self._cuda_calls += 1
            if self._cuda_calls == self._oom_on_cuda_call:
                msg = "CUDA out of memory. Tried to allocate 20.00 MiB"
                raise OutOfMemoryError(msg)
        return self._fixed_text if self._fixed_text is not None else stem


class _CompatProbe:
    """Reports whether the v3 load-time global patches are currently installed."""

    def __init__(
        self, hydra_utils: types.ModuleType, pretrained_cls: type[Any]
    ) -> None:
        self._hydra_utils = hydra_utils
        self._pretrained_cls = pretrained_cls
        self._instantiate = hydra_utils.instantiate
        self._finalize = pretrained_cls._finalize_model_loading  # noqa: SLF001

    def patched(self) -> bool:
        current = self._pretrained_cls._finalize_model_loading  # noqa: SLF001
        return (
            self._hydra_utils.instantiate is not self._instantiate
            or current is not self._finalize
        )


@pytest.fixture
def compat(monkeypatch: pytest.MonkeyPatch) -> _CompatProbe:
    """Provide hydra / transformers modules (real when installed, else stubs)."""
    try:
        hydra_utils = importlib.import_module("hydra.utils")
        modeling = importlib.import_module("transformers.modeling_utils")
        importlib.import_module("torch")
    except ImportError:
        pass
    else:
        return _CompatProbe(hydra_utils, modeling.PreTrainedModel)

    hydra_pkg = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda *_a, **_k: None  # type: ignore[attr-defined]
    hydra_pkg.utils = hydra_utils  # type: ignore[attr-defined]
    transformers_pkg = types.ModuleType("transformers")
    modeling = types.ModuleType("transformers.modeling_utils")

    class _PreTrainedModel:
        @staticmethod
        def _finalize_model_loading(model: object, *_args: object) -> object:
            return model

    modeling.PreTrainedModel = _PreTrainedModel  # type: ignore[attr-defined]
    transformers_pkg.modeling_utils = modeling  # type: ignore[attr-defined]
    torch_mod = types.ModuleType("torch")

    class _Device:
        def __init__(self, _name: str) -> None:
            self.name = _name

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    torch_mod.device = _Device  # type: ignore[attr-defined]
    for name, module in (
        ("hydra", hydra_pkg),
        ("hydra.utils", hydra_utils),
        ("transformers", transformers_pkg),
        ("transformers.modeling_utils", modeling),
        ("torch", torch_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return _CompatProbe(hydra_utils, _PreTrainedModel)


def _write_silent_wav(path: Path, *, duration_s: float) -> None:
    """Write a mono 16-bit silent WAV (windows are ``_WINDOW_S`` = 20 s long)."""
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(_SAMPLE_RATE)
        writer.writeframes(b"\x00\x00" * int(duration_s * _SAMPLE_RATE))


def _install(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_cls: type[GigaAMCtcWrapper],
    model: _FakeModel,
    cuda: _FakeCuda,
) -> None:
    """Route the wrapper's model/torch imports to the stand-ins."""

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeModel:
            return model

    monkeypatch.setattr(
        wrapper_cls,
        "_ensure_deps",
        lambda _self: (_FakeAutoModel, _FakeTorch(cuda)),
    )


def _e2e(
    monkeypatch: pytest.MonkeyPatch,
    compat: _CompatProbe,
    *,
    device: str,
    model: _FakeModel | None = None,
    cuda: _FakeCuda | None = None,
) -> tuple[GigaAME2ECtcWrapper, _FakeModel, _FakeCuda]:
    """Build an e2e wrapper wired to a fake model and fake CUDA runtime."""
    fake_cuda = cuda if cuda is not None else _FakeCuda()
    fake_model = (
        model if model is not None else _FakeModel(patches_active=compat.patched)
    )
    _install(monkeypatch, GigaAME2ECtcWrapper, fake_model, fake_cuda)
    return GigaAME2ECtcWrapper(device=device), fake_model, fake_cuda


def _short_wav(tmp_path: Path) -> Path:
    path = tmp_path / "short.wav"
    _write_silent_wav(path, duration_s=3.0)
    return path


def _three_window_wav(tmp_path: Path) -> Path:
    """Return a 45 s WAV that the wrapper splits into windows chunk_000..002."""
    path = tmp_path / "long.wav"
    _write_silent_wav(path, duration_s=45.0)
    return path


# * --------------------------------------------------------------------------
# * Device resolution
# * --------------------------------------------------------------------------


def test_cpu_request_never_touches_cuda(
    monkeypatch: pytest.MonkeyPatch, compat: _CompatProbe, tmp_path: Path
) -> None:
    """``--device cpu`` must not query or allocate on CUDA, even when it is free."""
    wrapper, model, cuda = _e2e(monkeypatch, compat, device="cpu")

    result = wrapper.transcribe(_short_wav(tmp_path))
    wrapper.unload()

    assert result["text"] == "short"
    assert [device for device, _stem in model.transcribe_log] == ["cpu"]
    assert cuda.mem_get_info_calls == 0
    assert cuda.empty_cache_calls == 0


@pytest.mark.parametrize(
    ("free_bytes", "expected_device"),
    [
        (_AUTO_MIN_FREE_BYTES, "cuda"),
        (8 * _GIB, "cuda"),
        (_AUTO_MIN_FREE_BYTES - 1, "cpu"),
        (1 * _GIB, "cpu"),
        (0, "cpu"),
    ],
)
def test_auto_uses_cuda_only_with_enough_free_vram(
    monkeypatch: pytest.MonkeyPatch,
    compat: _CompatProbe,
    tmp_path: Path,
    free_bytes: int,
    expected_device: str,
) -> None:
    """``auto`` picks CUDA at >= 2.5 GiB free and CPU below it (boundary exact)."""
    wrapper, model, _cuda = _e2e(
        monkeypatch,
        compat,
        device="auto",
        cuda=_FakeCuda(free_bytes=free_bytes),
    )

    result = wrapper.transcribe(_short_wav(tmp_path))

    assert result["text"] == "short"
    assert [device for device, _stem in model.transcribe_log] == [expected_device]
    assert wrapper.active_device == expected_device


def test_auto_with_unknown_free_vram_loads_on_cpu(
    monkeypatch: pytest.MonkeyPatch, compat: _CompatProbe, tmp_path: Path
) -> None:
    """A failing VRAM probe must degrade to CPU, not abort the transcription."""
    wrapper, model, _cuda = _e2e(
        monkeypatch,
        compat,
        device="auto",
        cuda=_FakeCuda(mem_info_error=RuntimeError("driver hiccup")),
    )

    result = wrapper.transcribe(_short_wav(tmp_path))

    assert result["text"] == "short"
    assert [device for device, _stem in model.transcribe_log] == ["cpu"]


@pytest.mark.parametrize("requested", ["cuda", "auto"])
def test_unavailable_cuda_falls_back_to_cpu_with_warning(
    monkeypatch: pytest.MonkeyPatch,
    compat: _CompatProbe,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    requested: str,
) -> None:
    """No CUDA runtime: the request still succeeds on CPU and a warning is logged."""
    wrapper, model, cuda = _e2e(
        monkeypatch, compat, device=requested, cuda=_FakeCuda(available=False)
    )

    with caplog.at_level(logging.WARNING):
        result = wrapper.transcribe(_short_wav(tmp_path))

    assert result["text"] == "short"
    assert [device for device, _stem in model.transcribe_log] == ["cpu"]
    assert cuda.mem_get_info_calls == 0
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_explicit_cuda_ignores_the_auto_vram_threshold(
    monkeypatch: pytest.MonkeyPatch, compat: _CompatProbe, tmp_path: Path
) -> None:
    """``--device cuda`` is an explicit request: low free VRAM does not veto it."""
    wrapper, model, _cuda = _e2e(
        monkeypatch,
        compat,
        device="cuda",
        cuda=_FakeCuda(free_bytes=256 * 1024 * 1024),
    )

    wrapper.transcribe(_short_wav(tmp_path))

    assert [device for device, _stem in model.transcribe_log] == ["cuda"]


# * --------------------------------------------------------------------------
# * Load contract
# * --------------------------------------------------------------------------


def test_cuda_load_keeps_fp32_eval_and_restores_patches_before_the_move(
    monkeypatch: pytest.MonkeyPatch, compat: _CompatProbe, tmp_path: Path
) -> None:
    """The GPU move never extends the global load patches and never halves the model.

    A model left in the global patch window would block other loader threads for
    the whole (slow) CUDA transfer; an fp16 model breaks the n_fft=320 mel
    front-end.
    """
    wrapper, model, _cuda = _e2e(monkeypatch, compat, device="cuda")

    wrapper.transcribe(_short_wav(tmp_path))

    assert model.patches_active_during_cuda_move is False
    assert compat.patched() is False
    assert model.dtype == "fp32"
    assert model.half_called is False
    assert model.training is False
    assert model.device_type == "cuda"


# * --------------------------------------------------------------------------
# * OOM fallbacks
# * --------------------------------------------------------------------------


def test_oom_on_model_move_returns_the_whole_model_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
    compat: _CompatProbe,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A CUDA OOM while moving leaves nothing on the GPU and transcription works."""
    model = _FakeModel(
        patches_active=compat.patched,
        move_error=OutOfMemoryError("CUDA out of memory. Tried to allocate 900 MiB"),
    )
    wrapper, model, cuda = _e2e(monkeypatch, compat, device="cuda", model=model)

    with caplog.at_level(logging.WARNING):
        result = wrapper.transcribe(_short_wav(tmp_path))

    assert result["text"] == "short"
    assert {t.device.type for t in [*model.parameters(), *model.buffers()]} == {"cpu"}
    assert [device for device, _stem in model.transcribe_log] == ["cpu"]
    assert wrapper.active_device == "cpu"
    assert cuda.empty_cache_calls >= 1
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_non_oom_move_failure_propagates_and_drops_the_model(
    monkeypatch: pytest.MonkeyPatch, compat: _CompatProbe, tmp_path: Path
) -> None:
    """Only OOM is recoverable: other move failures surface and keep no half model."""
    model = _FakeModel(move_error=RuntimeError("CUDA error: device-side assert"))
    wrapper, _model, _cuda = _e2e(monkeypatch, compat, device="cuda", model=model)

    with pytest.raises(RuntimeError, match="device-side assert"):
        wrapper.transcribe(_short_wav(tmp_path))

    assert wrapper.active_device is None


def test_oom_during_a_window_retries_that_window_on_cpu_and_stays_on_cpu(
    monkeypatch: pytest.MonkeyPatch, compat: _CompatProbe, tmp_path: Path
) -> None:
    """The failing window is retried once on CPU; later windows never return to CUDA."""
    model = _FakeModel(patches_active=compat.patched, oom_on_cuda_call=2)
    wrapper, model, cuda = _e2e(monkeypatch, compat, device="cuda", model=model)
    segments: list[dict[str, Any]] = []

    result = wrapper.transcribe(_three_window_wav(tmp_path), on_segment=segments.append)

    assert model.transcribe_log == [
        ("cuda", "chunk_000"),
        ("cuda", "chunk_001"),  # * OOM here
        ("cpu", "chunk_001"),  # * same window, retried on CPU
        ("cpu", "chunk_002"),
    ]
    assert result["text"] == "chunk_000 chunk_001 chunk_002"
    assert [s["text"] for s in segments] == ["chunk_000", "chunk_001", "chunk_002"]
    assert {t.device.type for t in [*model.parameters(), *model.buffers()]} == {"cpu"}
    assert wrapper.active_device == "cpu"
    assert cuda.empty_cache_calls >= 1


def test_non_oom_window_error_is_not_retried_or_swallowed(
    monkeypatch: pytest.MonkeyPatch, compat: _CompatProbe, tmp_path: Path
) -> None:
    """A non-OOM failure must surface exactly once instead of silently re-running."""
    model = _FakeModel(window_error=ValueError("unreadable audio"))
    wrapper, model, _cuda = _e2e(monkeypatch, compat, device="cuda", model=model)

    with pytest.raises(ValueError, match="unreadable audio"):
        wrapper.transcribe(_short_wav(tmp_path))

    assert len(model.transcribe_log) == 1


# * --------------------------------------------------------------------------
# * Unload
# * --------------------------------------------------------------------------


def test_unload_releases_cuda_cache_after_cuda_use(
    monkeypatch: pytest.MonkeyPatch, compat: _CompatProbe, tmp_path: Path
) -> None:
    """Unloading a CUDA-resident model returns cached VRAM to the driver."""
    wrapper, _model, cuda = _e2e(monkeypatch, compat, device="cuda")
    wrapper.transcribe(_short_wav(tmp_path))
    assert cuda.empty_cache_calls == 0

    wrapper.unload()

    assert cuda.empty_cache_calls >= 1
    assert wrapper.active_device is None


@pytest.mark.parametrize(
    ("requested", "cuda_kwargs"),
    [
        ("cpu", {}),
        ("auto", {"free_bytes": 1 * _GIB}),
        ("cuda", {"available": False}),
    ],
)
def test_unload_leaves_cuda_alone_when_the_model_stayed_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    compat: _CompatProbe,
    tmp_path: Path,
    requested: str,
    cuda_kwargs: dict[str, Any],
) -> None:
    """A CPU-only run must not poke the CUDA allocator on unload."""
    wrapper, _model, cuda = _e2e(
        monkeypatch, compat, device=requested, cuda=_FakeCuda(**cuda_kwargs)
    )
    wrapper.transcribe(_short_wav(tmp_path))

    wrapper.unload()

    assert cuda.empty_cache_calls == 0


# * --------------------------------------------------------------------------
# * gigaam-ctc stays CPU-only
# * --------------------------------------------------------------------------


def test_multilingual_ctc_head_never_uses_cuda_even_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shared device machinery must not leak CUDA into ``gigaam-ctc``."""
    cuda = _FakeCuda(free_bytes=8 * _GIB)
    model = _FakeModel()
    _install(monkeypatch, GigaAMCtcWrapper, model, cuda)
    wrapper = GigaAMCtcWrapper(device="auto")

    result = wrapper.transcribe(_short_wav(tmp_path))
    wrapper.unload()

    assert result["text"] == "short"
    assert [device for device, _stem in model.transcribe_log] == ["cpu"]
    assert cuda.mem_get_info_calls == 0
    assert cuda.empty_cache_calls == 0


def test_multilingual_ctc_head_still_rejects_cuda() -> None:
    """``gigaam-ctc`` keeps refusing an explicit CUDA request."""
    with pytest.raises(ValueError, match="cuda"):
        GigaAMCtcWrapper(device="cuda")


# * --------------------------------------------------------------------------
# * OOM classification (decision table)
# * --------------------------------------------------------------------------


def _chained(outer: BaseException, cause: BaseException) -> BaseException:
    outer.__cause__ = cause
    return outer


@pytest.mark.parametrize(
    ("error", "is_oom"),
    [
        (OutOfMemoryError("anything"), True),
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
        (RuntimeError("CUDA error: out of memory"), True),
        (RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"), True),
        (RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"), True),
        (RuntimeError("cuFFT error: CUFFT_ALLOC_FAILED"), True),
        (
            _chained(RuntimeError("model.transcribe failed"), OutOfMemoryError("x")),
            True,
        ),
        # * CPU RAM exhaustion cannot be cured by moving the model to CPU.
        (RuntimeError("[enforce fail] DefaultCPUAllocator: not enough memory"), False),
        (MemoryError("cpu"), False),
        (RuntimeError("CUDA error: device-side assert triggered"), False),
        (ValueError("unreadable audio"), False),
    ],
)
def test_cuda_oom_classification(error: BaseException, is_oom: bool) -> None:  # noqa: FBT001
    """Only GPU allocation failures trigger the CPU fallback."""
    assert is_cuda_oom_error(error) is is_oom


def test_real_torch_out_of_memory_error_is_classified_as_oom() -> None:
    """The class torch actually raises is recognised (guards against renames)."""
    torch = pytest.importorskip("torch")
    assert is_cuda_oom_error(torch.OutOfMemoryError("CUDA out of memory"))


# * --------------------------------------------------------------------------
# * Real entry point: CLI -> LocalPipeline -> e2e wrapper (fake CUDA runtime)
# * --------------------------------------------------------------------------


def test_cli_no_daemon_scenario_runs_e2e_on_cuda_and_releases_it(
    monkeypatch: pytest.MonkeyPatch,
    compat: _CompatProbe,
    tmp_path: Path,
    short_audio_fixture: Path,
) -> None:
    """``external-transcribe --stt-provider gigaam-e2e-ctc --device cuda`` end to end.

    The text reaches stdout from a model that ran on CUDA, and the pipeline
    close hands the cached VRAM back.
    """
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(pl, "load_env_file", lambda: None)
    monkeypatch.setattr(pl, "configure_ml_caches", lambda root: root)
    monkeypatch.setattr(pl, "get_project_cache_dir", lambda: tmp_path)
    cuda = _FakeCuda()
    model = _FakeModel(patches_active=compat.patched, fixed_text="Привет, мир.")
    _install(monkeypatch, GigaAME2ECtcWrapper, model, cuda)

    result = CliRunner().invoke(
        cli.app,
        [
            "external-transcribe",
            str(short_audio_fixture),
            "--no-daemon",
            "--stt-provider",
            "gigaam-e2e-ctc",
            "--device",
            "cuda",
            "--work-dir",
            str(tmp_path / "work"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "Привет, мир.\n"
    assert model.transcribe_log
    assert {device for device, _stem in model.transcribe_log} == {"cuda"}
    assert cuda.empty_cache_calls >= 1


# * --------------------------------------------------------------------------
# * The thin client stays torch-free
# * --------------------------------------------------------------------------


def test_thin_client_path_does_not_import_torch() -> None:
    """Validating ``--device`` for the GPU-capable head must not pull in torch.

    The ``external-transcribe`` client is a thin process; importing torch there
    would cost seconds and hundreds of MiB per call and could create CUDA state.
    """
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import cli\n"
        "from core import stt_providers\nfrom core.stt_pool import client\n"
        "for provider, device in (('gigaam-e2e-ctc', 'cuda'),"
        " ('gigaam-e2e-ctc', 'auto'), ('gigaam-ctc', 'auto'), ('whisper', 'cuda')):\n"
        "    cli._resolve_external_device(stt_provider=provider, device=device)\n"
        "stt_providers.provider_may_use_cuda('gigaam-e2e-ctc')\n"
        "loaded = sorted(m for m in ('torch', 'transformers', 'hydra')"
        " if m in sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    completed = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        [sys.executable, "-I", "-c", code, str(_PROJECT_ROOT)],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
