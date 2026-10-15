"""Real external CLI scenarios with SAPI oracles and a one-model worker budget."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import psutil  # type: ignore[import-untyped]
import pytest

import cli
from core.stt_pool.queue import QueuePaths, read_heartbeat
from tests.bounded_scenario_support import run_bounded_scenario
from tests.test_sherpa_diarization import Recording, _speech

pytestmark = [
    pytest.mark.heavy_ml,
    pytest.mark.slow,
    pytest.mark.xdist_group(name="ml_singleton"),
    pytest.mark.skipif(sys.platform != "win32", reason="Windows SAPI dialog oracle"),
]

_CLI_TIMEOUT_S = 90.0
_SCENARIO_TIMEOUT_S = 180.0
_RATE = 16000
_TURN_TOLERANCE_S = 0.6


def _recording(root: Path, *, russian: bool) -> Recording:
    """Derive utterance windows from trimmed SAPI frames, independently of both models."""
    root.mkdir()
    utterances = [
        ("Microsoft Pavel", "Добрый день, меня зовут Павел."),
        (
            "Microsoft Irina Desktop" if russian else "Microsoft Zira Desktop",
            "Здравствуйте, меня зовут Ирина."
            if russian
            else "Hello my name is Zira and I am very happy to meet you",
        ),
        ("Microsoft Pavel", "Очень приятно познакомиться."),
    ]
    cursor = 8000
    pieces = [b"\x00\x00" * cursor]
    windows = []
    for index, (voice, text) in enumerate(utterances):
        pcm = _speech(voice, text, root / f"utterance-{index}.wav")
        frames = len(pcm) // 2
        windows.append((cursor / _RATE, (cursor + frames) / _RATE))
        pieces.extend((pcm, b"\x00\x00" * 12800))
        cursor += frames + 12800
    path = root / "dialog.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, _RATE, 0, "NONE", "not compressed"))
        writer.writeframes(b"".join(pieces))
    return Recording(path, tuple(windows))


def _real_cli_scenario(root: Path) -> None:  # noqa: C901, PLR0915 - one bounded scenario owns serial CLI calls and process teardown
    """Own the supervisor process tree and exercise real CLI subprocesses serially."""
    russian = _recording(root / "русский", russian=True)
    reference = _recording(root / "reference", russian=False)
    paths = QueuePaths(root / "queue")
    config = root / "pool.json"
    config.write_text(
        json.dumps(
            {
                "cpu": {
                    "max_slots": 1,
                    "max_slots_per_worker": 1,
                    "threads_per_slot": 4,
                },
                "gpu": {"max_workers": 0},
                "threads": {"budget": 4},
                "diarization": {"threads_per_slot": 4},
                "memory": {"commit_budget_gib": 2.5},
                "lifecycle": {
                    "worker_idle_ttl_s": 2,
                    "supervisor_linger_s": 3,
                    "stop_grace_s": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    work = root / "work"
    work.mkdir()
    environment = {
        **os.environ,
        "ASKVLM_EXTERNAL_QUEUE_DIR": str(paths.root),
        "ASKVLM_STT_POOL_CONFIG": str(config),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TEMP": str(work),
        "TMP": str(work),
    }
    command = [sys.executable, "-X", "utf8", str(Path(cli.__file__).resolve())]
    stop = threading.Event()
    samples: list[list[dict[str, object]]] = []
    workers: dict[int, psutil.Process] = {}

    def observe() -> None:
        while not stop.wait(0.02):
            heartbeat = read_heartbeat(paths)
            if heartbeat is not None:
                active = heartbeat["workers"]
                samples.append(active)
                for worker in active:
                    pid = worker["pid"]
                    if pid not in workers:
                        with contextlib.suppress(psutil.NoSuchProcess):
                            workers[pid] = psutil.Process(pid)

    def invoke(arguments: list[str]) -> dict[str, object]:
        completed = subprocess.run(  # noqa: S603 - trusted CLI and pytest-owned media, no shell
            [*command, *arguments],
            env=environment,
            capture_output=True,
            timeout=_CLI_TIMEOUT_S,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert completed.returncode == 0, completed.stderr.decode(
            "utf-8", errors="replace"
        )
        assert completed.stdout.isascii()
        return json.loads(completed.stdout)

    with (
        (root / "supervisor.log").open("wb") as log,
        subprocess.Popen(  # noqa: S603 - own real daemon, fixed command and temporary queue
            [*command, "external-transcribe-daemon", "--queue-dir", str(paths.root)],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as supervisor,
    ):
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        try:
            deadline = time.monotonic() + 15
            while read_heartbeat(paths) is None:
                assert supervisor.poll() is None
                if time.monotonic() >= deadline:
                    msg = "test-owned supervisor did not publish a heartbeat"
                    raise TimeoutError(msg)
                stop.wait(0.02)
            output = root / "transcript.json"
            transcript = invoke(
                [
                    "external-transcribe",
                    str(russian.path),
                    "--device",
                    "cpu",
                    "--language",
                    "ru",
                    "--format",
                    "json",
                    "--diarization",
                    "--output-file",
                    str(output),
                ]
            )
            assert set(transcript) == {"media", "sample_rate", "segments", "turns"}
            assert transcript["media"] == str(russian.path)
            assert transcript["sample_rate"] == 16000
            segments = transcript["segments"]
            text = " ".join(segment["text"] for segment in segments).lower()
            assert all(word in text for word in ("добрый", "ирина", "приятно"))
            assert all(
                {
                    "start",
                    "end",
                    "text",
                    "words",
                    "no_speech_prob",
                    "avg_logprob",
                    "speaker",
                }
                <= set(segment)
                for segment in segments
            )
            assert all(
                segment["speaker"] is None or segment["speaker"].startswith("speaker_")
                for segment in segments
            )
            assert any(segment["words"] for segment in segments)
            assert json.loads(output.read_bytes()) == transcript
            assert "русский".encode() in output.read_bytes()
            turns = invoke(
                ["external-diarize", str(reference.path), "--num-speakers", "2"]
            )["turns"]
            assert len(turns) == 3
            assert [turn["speaker"] for turn in turns] == [
                "speaker_00",
                "speaker_01",
                "speaker_00",
            ]
            for turn, (start, end) in zip(turns, reference.windows, strict=True):
                assert turn["start"] == pytest.approx(start, abs=_TURN_TOLERANCE_S)
                assert turn["end"] == pytest.approx(end, abs=_TURN_TOLERANCE_S)
            start, end = reference.windows[1]
            cut = invoke(
                [
                    "external-diarize",
                    str(reference.path),
                    "--num-speakers",
                    "1",
                    "--start",
                    str(start - 0.3),
                    "--duration",
                    str(end - start + 0.6),
                ]
            )["turns"]
            assert len(cut) == 1
            assert cut[0]["start"] == pytest.approx(start, abs=_TURN_TOLERANCE_S)
            assert cut[0]["end"] == pytest.approx(end, abs=_TURN_TOLERANCE_S)
            supervisor.wait(timeout=30)
            assert supervisor.returncode == 0
        finally:
            stop.set()
            observer.join(timeout=2)
            assert not observer.is_alive()
            if supervisor.poll() is None:
                supervisor.terminate()
                supervisor.wait(timeout=10)
    assert samples
    assert all(len(active) <= 1 for active in samples)
    assert {worker["key"]["provider"] for active in samples for worker in active} == {
        "whisper",
        "sherpa-diarize",
    }
    assert all(not worker.is_running() for worker in workers.values())
    assert not paths.heartbeat_file.exists()
    assert not list(work.glob("askvlm-pool-*"))


def test_real_cli_json_diarization_and_diarize_recover_sapi_dialogs(
    tmp_path: Path,
) -> None:
    """Catch complete CLI wiring, real word/turn payload or pooled model eviction regressions (E2E)."""
    run_bounded_scenario(_real_cli_scenario, tmp_path, timeout_s=_SCENARIO_TIMEOUT_S)
