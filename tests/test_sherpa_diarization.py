"""PCM decoding rules and real sherpa scenarios against a SAPI dialog oracle."""

from __future__ import annotations

import importlib
import re
import struct
import subprocess
import sys
import wave
from array import array
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import pytest

from core import sherpa_diarization as backend
from core.diarization_models import ensure_diarization_models
from core.sherpa_diarization import (
    DiarizationCancelledError,
    DiarizationError,
    SherpaDiarizer,
)
from core.stt_options import DiarizeOptions, MediaWindow
from core.stt_pool.client import ClientRequest, run_client
from core.stt_pool.engine_api import DEFAULT_ENGINE_FACTORY
from tests.bounded_scenario_support import run_bounded_scenario
from tests.stt_pool_supervisor_support import SupervisorHarness

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from core.diarization_models import DiarizationModelPaths

# * The hardsub reference uses voices from different engines and independently
# * known audible windows; voices from one engine are a weaker clustering oracle.
_UTTERANCES = (
    ("Microsoft Pavel", "Добрый день, меня зовут Павел."),
    ("Microsoft Zira Desktop", "Hello my name is Zira and I am very happy to meet you"),
    ("Microsoft Pavel", "Очень приятно познакомиться."),
)
_RATE = 16000
_QUIET_LEVEL = 300
_LEAD_FRAMES = 8000
_GAP_FRAMES = 12800
_TURN_TOLERANCE = 0.6


@dataclass(frozen=True)
class Recording:
    """Known speech windows of a synthesized male-female-male recording."""

    path: Path
    windows: tuple[tuple[float, float], ...]


def _speech(voice: str, text: str, path: Path) -> bytes:
    """Synthesize and trim SAPI padding at the external speech-system boundary."""
    escaped = [value.replace("'", "''") for value in (voice, text, str(path))]
    script = (
        "$ErrorActionPreference = 'Stop';"
        "Add-Type -AssemblyName System.Speech;"
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "try {{ $speaker.SelectVoice('{0}');"
        "$speaker.SetOutputToWaveFile('{2}', "
        "(New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo"
        "(16000, 'Sixteen', 'Mono')));"
        "$speaker.Speak('{1}') }} finally {{ $speaker.Dispose() }}"
    ).format(*escaped)
    try:
        subprocess.run(  # noqa: S603 - fixed fixture voices and escaped strings
            ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],  # noqa: S607 - pwsh from PATH hosts the SAPI voices
            check=True,
            capture_output=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(
            f"SAPI fixture failed: voice={voice!r}, exit={error.returncode}; "
            f"stderr={error.stderr.decode('utf-8', errors='replace')}",
            pytrace=False,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            f"SAPI fixture timed out after {error.timeout}s: voice={voice!r} (pwsh)",
            pytrace=False,
        )
    except FileNotFoundError:
        pytest.fail("SAPI fixture requires PowerShell 7 (pwsh) on PATH", pytrace=False)
    with wave.open(str(path), "rb") as reader:
        pcm = reader.readframes(reader.getnframes())
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    audible = [
        index for index, value in enumerate(samples) if abs(value) > _QUIET_LEVEL
    ]
    assert audible
    return pcm[2 * audible[0] : 2 * (audible[-1] + 1)]


@pytest.fixture(scope="module")
def two_voice_recording(tmp_path_factory: pytest.TempPathFactory) -> Recording:
    """Build the reference dialog while deriving boundaries from audible PCM frames."""
    folder = tmp_path_factory.mktemp("sherpa-sapi")
    pieces = [b"\x00\x00" * _LEAD_FRAMES]
    windows = []
    cursor = _LEAD_FRAMES
    for index, (voice, text) in enumerate(_UTTERANCES):
        pcm = _speech(voice, text, folder / f"utterance-{index}.wav")
        frames = len(pcm) // 2
        windows.append((cursor / _RATE, (cursor + frames) / _RATE))
        pieces.extend((pcm, b"\x00\x00" * _GAP_FRAMES))
        cursor += frames + _GAP_FRAMES
    path = folder / "dialog.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, _RATE, 0, "NONE", "not compressed"))
        writer.writeframes(b"".join(pieces))
    return Recording(path, tuple(windows))


@pytest.fixture(scope="module")
def diarization_models() -> DiarizationModelPaths:
    """Verify the cache through production provisioning before loading real networks."""
    return ensure_diarization_models()


@pytest.mark.integration
@pytest.mark.parametrize("chunk_frames", [1, 2, 1048576])
def test_pcm_decoding_matches_signed_little_endian_reference_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk_frames: int
) -> None:
    """Catch wrong signedness, byte order, scaling or dropped PCM chunks."""
    import numpy as np  # noqa: PLC0415 - audio math needs no real model

    path = tmp_path / "samples.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, _RATE, 0, "NONE", "not compressed"))
        writer.writeframes(struct.pack("<5h", -32768, -16384, 0, 16384, 32767))
    monkeypatch.setattr(backend, "_READ_CHUNK_FRAMES", chunk_frames)
    samples, frames = backend._read_samples(np, path)  # noqa: SLF001 - PCM rule
    assert frames == 5
    np.testing.assert_array_equal(
        samples, np.array([-1.0, -0.5, 0.0, 0.5, 0.999969482421875], dtype=np.float32)
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("rate", "channels", "width", "frames"),
    [(8000, 1, 2, 10), (16000, 2, 2, 10), (16000, 1, 1, 10), (16000, 1, 2, 0)],
)
def test_pcm_reader_rejects_unprepared_or_empty_audio(
    tmp_path: Path, rate: int, channels: int, width: int, frames: int
) -> None:
    """Catch invalid or empty WAVs silently producing empty diarization."""
    import numpy as np  # noqa: PLC0415 - binary decoding only, no network or weights

    path = tmp_path / "invalid.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x00" * (frames * channels * width))
    with pytest.raises(DiarizationError):
        backend._read_samples(np, path)  # noqa: SLF001 - dense WAV format validation


@pytest.mark.integration
@pytest.mark.heavy_ml
@pytest.mark.slow
@pytest.mark.xdist_group(name="ml_singleton")
@pytest.mark.skipif(sys.platform != "win32", reason="Windows SAPI dialog oracle")
def test_real_diarization_recovers_three_known_turns_and_two_voices(
    two_voice_recording: Recording,
    diarization_models: DiarizationModelPaths,
    tmp_path: Path,
) -> None:
    """Catch wrong sherpa configuration, collapsed voices or drifting turns."""
    diarizer = SherpaDiarizer(model_paths=diarization_models)
    turns = diarizer.diarize_media(
        two_voice_recording.path,
        DiarizeOptions(num_speakers=2),
        window=MediaWindow(),
        work_dir=tmp_path,
    )
    assert len(turns) == 3
    for turn, (start, end) in zip(turns, two_voice_recording.windows, strict=True):
        assert turn["start"] == pytest.approx(start, abs=_TURN_TOLERANCE)
        assert turn["end"] == pytest.approx(end, abs=_TURN_TOLERANCE)
    assert turns[0]["speaker"] == turns[2]["speaker"] != turns[1]["speaker"]
    assert all(
        re.fullmatch(r"speaker_[0-9]{2,}", str(turn["speaker"])) for turn in turns
    )
    assert not (tmp_path / "diarize").exists()


@pytest.mark.integration
@pytest.mark.heavy_ml
@pytest.mark.slow
@pytest.mark.xdist_group(name="ml_singleton")
@pytest.mark.skipif(sys.platform != "win32", reason="Windows SAPI dialog oracle")
def test_real_diarization_window_places_turns_at_media_time(
    two_voice_recording: Recording,
    diarization_models: DiarizationModelPaths,
    tmp_path: Path,
) -> None:
    """Catch omitted or duplicated window offsets in diarization JSON."""
    start, end = two_voice_recording.windows[1]
    window = MediaWindow(start - 0.3, end - start + 0.6)
    turns = SherpaDiarizer(model_paths=diarization_models).diarize_media(
        two_voice_recording.path,
        DiarizeOptions(num_speakers=1),
        window=window,
        work_dir=tmp_path,
    )
    assert turns
    assert turns[0]["start"] == pytest.approx(start, abs=_TURN_TOLERANCE)
    assert turns[-1]["end"] == pytest.approx(end, abs=_TURN_TOLERANCE)
    assert float(str(turns[0]["start"])) >= window.start_s
    assert not (tmp_path / "diarize").exists()


@pytest.mark.integration
@pytest.mark.heavy_ml
@pytest.mark.slow
@pytest.mark.xdist_group(name="ml_singleton")
@pytest.mark.skipif(sys.platform != "win32", reason="Windows SAPI dialog oracle")
def test_real_diarization_cancellation_is_typed_and_a_later_job_still_works(
    two_voice_recording: Recording,
    diarization_models: DiarizationModelPaths,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch ignored native cancellation or corrupt state/files after aborting."""
    sherpa = importlib.import_module("sherpa_onnx")
    real_factory = sherpa.OfflineSpeakerDiarization
    processing = False

    class ObservedNativeEngine:
        """Observe the real inference boundary without replacing native execution."""

        def __init__(self, config: object) -> None:
            self.engine = real_factory(config)

        @property
        def sample_rate(self) -> int:
            """Expose the real model's sample rate."""
            return int(self.engine.sample_rate)

        def process(
            self,
            samples: object,
            callback: Callable[[int, int], int] | None = None,
        ) -> object:
            """Make cancellation active only during actual native processing."""
            nonlocal processing
            processing = True
            try:
                if callback is None:
                    return self.engine.process(samples)
                return self.engine.process(samples, callback=callback)
            finally:
                processing = False

    monkeypatch.setattr(sherpa, "OfflineSpeakerDiarization", ObservedNativeEngine)
    diarizer = SherpaDiarizer(model_paths=diarization_models)
    with pytest.raises(DiarizationCancelledError):
        diarizer.diarize_media(
            two_voice_recording.path,
            DiarizeOptions(num_speakers=2),
            window=MediaWindow(),
            work_dir=tmp_path,
            should_cancel=lambda: processing,
        )
    assert not (tmp_path / "diarize").exists()
    turns = diarizer.diarize_media(
        two_voice_recording.path,
        DiarizeOptions(num_speakers=2),
        window=MediaWindow(),
        work_dir=tmp_path,
    )
    assert len(turns) == 3
    assert turns[0]["speaker"] == turns[2]["speaker"] != turns[1]["speaker"]
    assert not (tmp_path / "diarize").exists()


def _real_pool_dialog(recording: Recording, root: Path) -> None:
    """Exercise client, supervisor, worker, FFmpeg and sherpa in an owned process tree."""
    import threading  # noqa: PLC0415 - the bounded child owns the supervisor threads

    import psutil  # type: ignore[import-untyped]  # noqa: PLC0415 - capture only test-owned worker identities

    pool = SupervisorHarness(root)
    pool.options = replace(pool.options, engine_factory=DEFAULT_ENGINE_FACTORY)
    config_path = root / "pool-config.json"
    config_path.write_text("{}", encoding="utf-8")
    worker_temp = root / "worker-temp"
    worker_temp.mkdir()
    workers = []
    request = ClientRequest(
        inputs=(recording.path,),
        provider="whisper",
        whisper_model="small",
        compute_type="auto",
        device="cpu",
        language=None,
        priority="normal",
        max_parallel=1,
        client_timeout_s=60.0,
        mode="diarize",
        diarization=DiarizeOptions(num_speakers=2),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ASKVLM_EXTERNAL_QUEUE_DIR", str(pool.paths.root))
        patch.setenv("ASKVLM_STT_POOL_CONFIG", str(config_path))
        patch.setenv("TEMP", str(worker_temp))
        patch.setenv("TMP", str(worker_temp))

        def submit(call: ClientRequest) -> list[dict[str, object]]:
            outcome = run_client(
                call,
                queue_root=pool.paths.root,
                spawn=lambda _paths: pool.start(),
                clock=pool.clock,
                sleeper=lambda pause: threading.Event().wait(min(pause, 0.01)),
                poll_interval_s=0.01,
            ).outcomes[0]
            assert outcome.status == "ok", outcome.detail
            assert outcome.backend == "cpu"
            assert outcome.text == ""
            assert outcome.data is not None
            return outcome.data["turns"]

        try:
            turns = submit(request)
            assert len(turns) == 3
            for turn, (start, end) in zip(turns, recording.windows, strict=True):
                assert set(turn) == {"start", "end", "speaker"}
                assert turn["start"] == pytest.approx(start, abs=_TURN_TOLERANCE)
                assert turn["end"] == pytest.approx(end, abs=_TURN_TOLERANCE)
            assert [turn["speaker"] for turn in turns] == [
                "speaker_00",
                "speaker_01",
                "speaker_00",
            ]
            start, end = recording.windows[1]
            window = MediaWindow(start - 0.3, end - start + 0.6)
            cut = submit(
                replace(
                    request, window=window, diarization=DiarizeOptions(num_speakers=1)
                )
            )
            assert len(cut) == 1
            assert cut[0]["start"] == pytest.approx(start, abs=_TURN_TOLERANCE)
            assert cut[0]["end"] == pytest.approx(end, abs=_TURN_TOLERANCE)
            assert float(str(cut[0]["start"])) >= window.start_s
            heartbeat = pool.heartbeat(lambda value: bool(value["workers"]))
            workers = [psutil.Process(worker["pid"]) for worker in heartbeat["workers"]]
            assert len(workers) == 1
            assert all(
                worker["key"]["provider"] == "sherpa-diarize"
                for worker in heartbeat["workers"]
            )
            pool.finish()
            assert all(not worker.is_running() for worker in workers)
            assert not list(worker_temp.glob("askvlm-pool-*"))
        finally:
            pool.cleanup()


@pytest.mark.heavy_ml
@pytest.mark.slow
@pytest.mark.xdist_group(name="ml_singleton")
@pytest.mark.skipif(sys.platform != "win32", reason="Windows SAPI dialog oracle")
def test_real_pool_client_diarizes_known_dialog_and_media_relative_window(
    two_voice_recording: Recording,
    diarization_models: DiarizationModelPaths,
    tmp_path: Path,
) -> None:
    """Catch missing real worker dispatch, lost turn payloads or wrong pooled window times (E2E)."""
    run_bounded_scenario(partial(_real_pool_dialog, two_voice_recording), tmp_path)
