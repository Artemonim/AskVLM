import json
import os
from pathlib import Path

import builtins
import types

import pytest

from core import ffmpeg as ffm
from core import audio_io
from core import settings as core_settings
from core.gpu_guard import GPUResourceGuard
from utils import downloader


def test_find_project_root_points_to_repo_root():
    """Root discovery returns a directory containing pyproject.toml."""
    # Ensure it returns a directory containing pyproject.toml
    root = core_settings._find_project_root()
    assert (root / "pyproject.toml").exists()


def test_get_project_cache_dir_is_under_root():
    """Cache directory is .cache at project root."""
    cache = core_settings.get_project_cache_dir()
    assert cache.name == ".cache"
    assert cache.parent.exists()


def test_configure_ml_caches_sets_env_and_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """configure_ml_caches creates dirs and sets env variables."""
    used = core_settings.configure_ml_caches(cache_root=tmp_path)
    # Directories
    assert (used / "huggingface").exists()
    assert (used / "huggingface" / "hub").exists()
    assert (used / "torch").exists()
    assert (used / "whisper").exists()
    # Environment variables
    assert os.environ.get("HF_HOME") == str(used / "huggingface")
    assert os.environ.get("HF_HUB_DISABLE_SYMLINKS") == "1"
    assert os.environ.get("TORCH_HOME") == str(used / "torch")
    assert os.environ.get("XDG_CACHE_HOME") == str(used)


def test_load_settings_writes_and_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """load_settings writes defaults and applies env override for hf_token."""
    settings_path = tmp_path / "settings.json"
    # First load creates defaults and writes file
    s1 = core_settings.load_settings(settings_path)
    assert settings_path.exists()
    assert isinstance(s1.models_path, Path)
    # Second load reads and respects env override for hf_token when empty
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    data["hf_token"] = ""
    settings_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "TOKEN123")
    s2 = core_settings.load_settings(settings_path)
    assert s2.hf_token == "TOKEN123"


def test_downloader_ensure_and_check(tmp_path: Path):
    """ensure_models_dir creates dir; check_missing_models returns missing names."""
    models_dir = tmp_path / "models"
    downloader.ensure_models_dir(models_dir)
    assert models_dir.exists()
    cfg = {"m1.bin": "v1", "m2.bin": "v2"}
    # Create one of the files
    (models_dir / "m1.bin").write_bytes(b"x")
    missing = downloader.check_missing_models(models_dir, cfg)
    assert missing == ["m2.bin"]


def test_load_models_config(tmp_path: Path):
    """load_models_config loads JSON and returns empty dict for absent file."""
    cfg_path = tmp_path / "models.json"
    cfg_path.write_text(json.dumps({"a": "1"}), encoding="utf-8")
    cfg = downloader.load_models_config(cfg_path)
    assert cfg == {"a": "1"}
    assert downloader.load_models_config(tmp_path / "absent.json") == {}


def test_ffmpeg_get_media_duration_seconds_handles_error(monkeypatch: pytest.MonkeyPatch):
    """get_media_duration_seconds returns 0.0 when ffprobe raises."""
    def raise_err(_):  # noqa: ANN001
        raise RuntimeError("ffprobe error")

    monkeypatch.setattr(ffm.ffmpeg, "probe", raise_err)
    assert ffm.get_media_duration_seconds("nope.mp4") == 0.0


def test_prepare_audio_non_cancellable_creates_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """prepare_audio non-cancellable path writes expected WAV file."""
    # Arrange a fake extract_audio that writes a file to the expected destination
    created: list[Path] = []

    def fake_extract(inp, out, sample_rate=16000, channels=1):  # noqa: ANN001, ARG001
        Path(out).write_bytes(b"RIFF....WAVE")
        created.append(Path(out))

    monkeypatch.setattr(audio_io, "extract_audio", fake_extract)

    input_media = tmp_path / "in.mp3"
    input_media.write_bytes(b"data")
    out = audio_io.prepare_audio(input_media, tmp_path)
    assert out.exists()
    assert out in created


def test_cleanup_intermediate_audio_removes_files(tmp_path: Path):
    """cleanup_intermediate_audio removes wav and empty work dir."""
    media = tmp_path / "clip.mp3"
    media.write_bytes(b"x")
    work = tmp_path
    work_dir = work / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / f"{media.stem}.wav"
    wav.write_bytes(b"fake")
    audio_io.cleanup_intermediate_audio(media, work)
    assert not wav.exists()
    # Directory is removed if empty
    assert not work_dir.exists()


def test_gpu_guard_acquire_and_context(monkeypatch: pytest.MonkeyPatch):
    """GPUResourceGuard empties CUDA cache on switch and on context exit."""
    calls = {"empty": 0}

    class DummyCUDA:
        def empty_cache(self):  # noqa: D401
            calls["empty"] += 1

    # Patch torch.cuda
    dummy = types.SimpleNamespace(cuda=DummyCUDA())
    monkeypatch.setattr("core.gpu_guard.torch", dummy)

    guard = GPUResourceGuard()
    guard.acquire("m1")
    guard.acquire("m2")  # triggers unload
    assert calls["empty"] >= 1

    with guard.model("m3"):
        pass
    # Context exit triggers another empty_cache
    assert calls["empty"] >= 2


