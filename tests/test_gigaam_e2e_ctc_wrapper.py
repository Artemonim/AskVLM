"""Unit tests for the optional GigaAM-v3 e2e CTC wrapper."""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types
import wave
from pathlib import Path
from typing import Any, Self

import pytest

from core.gigaam_ctc_wrapper import _MIN_AUDIO_DURATION_S
from core.gigaam_e2e_ctc_wrapper import GigaAME2ECtcWrapper
from core.stt_providers import (
    GIGAAM_V3_HF_REPO_ID,
    GIGAAM_V3_HF_REVISION,
    STT_PROVIDER_GIGAAM_E2E_CTC,
    normalize_stt_provider,
)

_FINALIZE_ATTR = "_finalize_model_loading"


def _write_pcm_silence(
    path: Path,
    *,
    n_frames: int,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    """Write a 16-bit PCM WAV with the given frame count, rate, and channel count."""
    frame = b"\x00\x00" * channels
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frame * n_frames)


def _write_silent_wav(
    path: Path,
    *,
    duration_s: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    """Write a 16-bit silent WAV of the requested duration."""
    _write_pcm_silence(
        path,
        n_frames=int(duration_s * sample_rate),
        sample_rate=sample_rate,
        channels=channels,
    )


def _install_v3_compat_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any]:
    """Return hydra.utils and PreTrainedModel, stubbing only when missing.

    Prefer already-installed ``[ml]`` modules so tests do not replace live
    ``torch``/``transformers`` in ``sys.modules`` (that would break later
    Whisper/faster-whisper tests in the same worker).

    Returns:
        ``(hydra_utils_module, PreTrainedModel)`` visible to subsequent imports.

    """
    try:
        hydra_utils = importlib.import_module("hydra.utils")
        modeling = importlib.import_module("transformers.modeling_utils")
        importlib.import_module("torch")
    except ImportError:
        pass
    else:
        return hydra_utils, modeling.PreTrainedModel

    hydra_pkg = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")

    def _original_instantiate(*_args: object, **_kwargs: object) -> None:
        return None

    hydra_utils.instantiate = _original_instantiate
    hydra_pkg.utils = hydra_utils

    transformers_pkg = types.ModuleType("transformers")
    modeling = types.ModuleType("transformers.modeling_utils")

    class _PreTrainedModel:
        @staticmethod
        def _finalize_model_loading(
            model: object, _load_config: object, _loading_info: object
        ) -> object:
            return model

    modeling.PreTrainedModel = _PreTrainedModel
    transformers_pkg.modeling_utils = modeling

    torch_mod = types.ModuleType("torch")

    class _Device:
        def __init__(self, _name: str) -> None:
            self.name = _name

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    torch_mod.device = _Device

    monkeypatch.setitem(sys.modules, "hydra", hydra_pkg)
    monkeypatch.setitem(sys.modules, "hydra.utils", hydra_utils)
    monkeypatch.setitem(sys.modules, "transformers", transformers_pkg)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return hydra_utils, _PreTrainedModel


def _finalize_loading(cls: object) -> object:
    """Return the current ``_finalize_model_loading`` attribute of *cls*."""
    return getattr(cls, _FINALIZE_ATTR)


def _live_patch_flags(
    original_instantiate: object, original_finalize: object
) -> tuple[bool, bool]:
    """Return whether hydra instantiate and finalize loaders currently differ.

    Imports are deferred so they resolve the stub modules installed for the test.

    Args:
        original_instantiate: Pre-patch ``hydra.utils.instantiate``.
        original_finalize: Pre-patch ``_finalize_model_loading``.

    Returns:
        ``(instantiate_patched, finalize_patched)``.

    """
    import hydra.utils as hydra_utils  # noqa: PLC0415
    from transformers.modeling_utils import PreTrainedModel  # noqa: PLC0415

    instantiate_patched = hydra_utils.instantiate is not original_instantiate
    finalize_patched = _finalize_loading(PreTrainedModel) is not original_finalize
    return instantiate_patched, finalize_patched


class _FakeV3Model:
    """Tracks the precision, mode, and device a loaded model ends up in.

    The v3 checkpoint ships fp16 weights, so a freshly "loaded" model starts as
    ``fp16``; the wrapper must leave it fp32, in eval mode, on CPU.
    """

    def __init__(self) -> None:
        self.dtype = "fp16"
        self.training = True
        self.device = "cpu"
        self.half_called = False

    def float(self) -> _FakeV3Model:
        self.dtype = "fp32"
        return self

    def half(self) -> _FakeV3Model:
        self.half_called = True
        self.dtype = "fp16"
        return self

    def eval(self) -> _FakeV3Model:
        self.training = False
        return self

    def to(self, device: str) -> _FakeV3Model:
        self.device = device
        return self

    def transcribe(self, path: str) -> str:
        return f"transcribed:{path}"


def test_gigaam_e2e_rejects_unknown_device_before_load() -> None:
    """Tokens outside ``auto|cpu|cuda`` are rejected before any model import."""
    with pytest.raises(ValueError, match="tpu"):
        GigaAME2ECtcWrapper(device="tpu")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("auto", "auto"),
        ("cpu", "cpu"),
        ("cuda", "cuda"),
        (" CUDA ", "cuda"),
        ("", "auto"),
    ],
)
def test_gigaam_e2e_keeps_device_request_for_load_time(raw: str, expected: str) -> None:
    """The CUDA-capable head keeps the request; the model load resolves it."""
    wrapper = GigaAME2ECtcWrapper(device=raw)
    assert wrapper.device == expected
    assert wrapper.active_device is None


@pytest.mark.parametrize(
    "raw",
    ["gigaam-e2e-ctc", " GigaAM-E2E-CTC ", "GIGAAM-E2E-CTC"],
)
def test_normalize_accepts_gigaam_e2e_ctc(raw: str) -> None:
    """The v3 provider id is accepted case- and space-insensitively."""
    assert normalize_stt_provider(raw) == STT_PROVIDER_GIGAAM_E2E_CTC


def test_normalize_rejects_unknown_provider() -> None:
    """Garbage provider tokens stay rejected."""
    with pytest.raises(ValueError, match="Unsupported STT provider"):
        normalize_stt_provider("not-a-provider")


def test_gigaam_e2e_from_pretrained_identity_and_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load uses the v3 repo/revision with both compat patches, then restores."""
    hydra_utils, pretrained_cls = _install_v3_compat_stubs(monkeypatch)
    original_instantiate = hydra_utils.instantiate
    original_finalize = _finalize_loading(pretrained_cls)
    created: list[_FakeV3Model] = []
    seen: dict[str, Any] = {}

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs: object) -> _FakeV3Model:
            instantiate_patched, finalize_patched = _live_patch_flags(
                original_instantiate, original_finalize
            )
            seen["repo_id"] = repo_id
            seen["kwargs"] = kwargs
            seen["instantiate_patched"] = instantiate_patched
            seen["finalize_patched"] = finalize_patched
            model = _FakeV3Model()
            created.append(model)
            return model

    monkeypatch.setattr(
        GigaAME2ECtcWrapper,
        "_ensure_deps",
        lambda _self: (_FakeAutoModel, object()),
    )
    wrapper = GigaAME2ECtcWrapper(device="cpu")
    wrapper._load_model()  # noqa: SLF001

    assert seen["repo_id"] == GIGAAM_V3_HF_REPO_ID
    assert seen["kwargs"]["revision"] == GIGAAM_V3_HF_REVISION
    assert seen["kwargs"]["trust_remote_code"] is True
    assert seen["instantiate_patched"] is True
    assert seen["finalize_patched"] is True
    # * fp32 (Mel/STFT parity), eval mode, untouched by CUDA for a CPU request.
    assert created[0].dtype == "fp32"
    assert created[0].half_called is False
    assert created[0].training is False
    assert created[0].device == "cpu"
    assert hydra_utils.instantiate is original_instantiate
    assert _finalize_loading(pretrained_cls) is original_finalize


def test_gigaam_e2e_restores_patches_after_from_pretrained_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raised ``from_pretrained`` still restores both original callables."""
    hydra_utils, pretrained_cls = _install_v3_compat_stubs(monkeypatch)
    original_instantiate = hydra_utils.instantiate
    original_finalize = _finalize_loading(pretrained_cls)
    seen: dict[str, bool] = {}

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            instantiate_patched, finalize_patched = _live_patch_flags(
                original_instantiate, original_finalize
            )
            seen["instantiate_patched"] = instantiate_patched
            seen["finalize_patched"] = finalize_patched
            msg = "HF hub unreachable"
            raise OSError(msg)

    monkeypatch.setattr(
        GigaAME2ECtcWrapper,
        "_ensure_deps",
        lambda _self: (_FakeAutoModel, object()),
    )
    wrapper = GigaAME2ECtcWrapper(device="cpu")
    with pytest.raises(OSError, match="HF hub unreachable"):
        wrapper._load_model()  # noqa: SLF001
    assert seen["instantiate_patched"] is True
    assert seen["finalize_patched"] is True
    assert hydra_utils.instantiate is original_instantiate
    assert _finalize_loading(pretrained_cls) is original_finalize
    assert wrapper._model is None  # noqa: SLF001


def test_gigaam_e2e_transcribe_plain_str_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inherited short-file path normalizes a plain ``str`` transcribe return."""
    _install_v3_compat_stubs(monkeypatch)
    audio = tmp_path / "short.wav"
    _write_silent_wav(audio, duration_s=1.0)
    transcribed: list[str] = []

    class _FakeModel:
        def float(self) -> _FakeModel:
            return self

        def eval(self) -> _FakeModel:
            return self

        def transcribe(self, path: str) -> str:
            transcribed.append(path)
            return "  Hello, world.  "

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeModel:
            return _FakeModel()

    monkeypatch.setattr(
        GigaAME2ECtcWrapper,
        "_ensure_deps",
        lambda _self: (_FakeAutoModel, object()),
    )
    wrapper = GigaAME2ECtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert transcribed == [str(audio)]
    assert result["text"] == "Hello, world."
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "Hello, world."
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == pytest.approx(1.0)


def test_gigaam_e2e_tiny_wav_is_padded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inherited ``_transcribe_file`` pads tiny WAVs for the v3 e2e wrapper."""
    _install_v3_compat_stubs(monkeypatch)
    audio = tmp_path / "tiny.wav"
    _write_pcm_silence(audio, n_frames=1)
    transcribed: list[str] = []
    durations: list[float] = []

    class _FakeModel:
        def float(self) -> _FakeModel:
            return self

        def eval(self) -> _FakeModel:
            return self

        def transcribe(self, path: str) -> str:
            transcribed.append(path)
            with wave.open(path, "rb") as reader:
                rate = reader.getframerate()
                frames = reader.getnframes()
            durations.append(frames / float(rate) if rate > 0 else 0.0)
            return "tiny"

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeModel:
            return _FakeModel()

    monkeypatch.setattr(
        GigaAME2ECtcWrapper,
        "_ensure_deps",
        lambda _self: (_FakeAutoModel, object()),
    )
    wrapper = GigaAME2ECtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert result["text"] == "tiny"
    assert len(transcribed) == 1
    assert transcribed[0] != str(audio)
    assert Path(transcribed[0]).name.startswith("pad_")
    assert durations[0] >= _MIN_AUDIO_DURATION_S
    assert not Path(transcribed[0]).exists()
    assert list(tmp_path.glob("pad_*")) == []


def test_gigaam_e2e_concurrent_loads_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first loads never interleave the global compat patches.

    The external daemon runs one lazy pipeline per worker thread, so a cold
    burst can call ``_load_model`` from several threads at once.
    """
    hydra_utils, pretrained_cls = _install_v3_compat_stubs(monkeypatch)
    original_instantiate = hydra_utils.instantiate
    original_finalize = _finalize_loading(pretrained_cls)
    thread_count = 4
    barrier = threading.Barrier(thread_count)
    inside = 0
    max_inside = 0
    errors: list[str] = []
    failures: list[Exception] = []

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeV3Model:
            nonlocal inside, max_inside
            inside += 1
            max_inside = max(max_inside, inside)
            try:
                instantiate_patched, finalize_patched = _live_patch_flags(
                    original_instantiate, original_finalize
                )
                if not (instantiate_patched and finalize_patched):
                    errors.append("patches inactive during from_pretrained")
                # * Widen the race window so an unlocked implementation overlaps.
                time.sleep(0.05)
            finally:
                inside -= 1
            return _FakeV3Model()

    monkeypatch.setattr(
        GigaAME2ECtcWrapper,
        "_ensure_deps",
        lambda _self: (_FakeAutoModel, object()),
    )

    def _load() -> None:
        try:
            barrier.wait()
            GigaAME2ECtcWrapper(device="cpu")._load_model()  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            failures.append(exc)

    threads = [threading.Thread(target=_load) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures
    assert not errors
    assert max_inside == 1
    assert hydra_utils.instantiate is original_instantiate
    assert _finalize_loading(pretrained_cls) is original_finalize
