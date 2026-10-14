from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

import cli
from core.stt_pool import fsio
from core.stt_pool.queue import (
    JobRecord,
    QueuePaths,
    ResultRecord,
    read_heartbeat,
    read_job_payload,
    write_heartbeat,
    write_result,
)
from tests.stt_pool_client_support import ClientHarness
from tests.stt_pool_supervisor_support import configuration

if TYPE_CHECKING:
    from collections.abc import Iterator
    from threading import Event

    from core.stt_pool.config import PoolConfig

_RUNNER = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep every CLI test away from the user's queue, pool config and model loader."""
    config = tmp_path / "pool.json"
    config.write_text('{"gpu":{"max_workers":0}}', encoding="utf-8")
    monkeypatch.setenv("ASKVLM_EXTERNAL_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("ASKVLM_STT_POOL_CONFIG", str(config))

    def forbidden(_paths: QueuePaths) -> None:
        msg = "test must inject a fake supervisor launch"
        raise KeyboardInterrupt(msg)

    monkeypatch.setattr("core.stt_pool.client.spawn_supervisor", forbidden)
    handlers = {
        kind: signal.getsignal(kind) for kind in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        for kind, handler in handlers.items():
            signal.signal(kind, handler)


@pytest.fixture
def pool_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ClientHarness]:
    """Use real supervisor/worker entries while the model boundary stays fake."""
    harness = ClientHarness(tmp_path, monkeypatch)
    try:
        yield harness
    finally:
        harness.finish()


class _StubDocument:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_full_text(self) -> str:
        return self._text


@pytest.mark.integration
@pytest.mark.xdist_group(name="pool_supervisor")
@pytest.mark.parametrize(
    ("mode", "text", "code"),
    [
        ("ok", "Привет\nworld", 0),
        ("empty", "", 0),
        ("blank_ok", "", 0),
        ("error", "", 1),
    ],
)
def test_daemon_single_input_keeps_stdout_and_worker_error_contract(
    pool_client: ClientHarness, mode: str, text: str, code: int
) -> None:
    """Catch stdout noise, empty failures and lost worker detail."""
    media = pool_client.media(
        mode=mode,
        text=text,
        stray_stdout=True,
        error_text="Too long wav file: transcribe_longform\nworker detail: русский",
    )
    result = _RUNNER.invoke(
        cli.app, ["external-transcribe", str(media), "--device", "cpu"]
    )
    assert result.exit_code == code, result.output
    assert result.stdout_bytes == (
        (text.replace("\n", os.linesep) + os.linesep).encode("utf-8") if text else b""
    )
    if mode == "error":
        assert (
            result.stderr
            == pool_client.pool.events("emitted")[0]["error_detail"] + "\n"
        )
        assert "Too long wav file" in result.stderr
    else:
        assert result.stderr == ""
    assert len(pool_client.pool.events("submitted")) == 1


@pytest.mark.integration
@pytest.mark.xdist_group(name="pool_supervisor")
@pytest.mark.parametrize("provider", ["whisper", "gigaam-ctc", "gigaam-e2e-ctc"])
@pytest.mark.parametrize(
    ("mode", "code", "marker"),
    [
        ("hold", 10, "ASKVLM_CLIENT_TIMEOUT"),
        ("load_fail", 11, "ASKVLM_DAEMON_UNAVAILABLE"),
    ],
)
def test_degraded_daemon_returns_code_without_client_cpu_recovery(
    pool_client: ClientHarness,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    mode: str,
    code: int,
    marker: str,
) -> None:
    """Catch forbidden client CPU recovery after degraded service."""
    if mode == "load_fail":
        model = {
            "whisper": "small",
            "gigaam-ctc": "ai-sage/GigaAM-Multilingual@ctc",
            "gigaam-e2e-ctc": "ai-sage/GigaAM-v3@e2e_ctc",
        }[provider]
        pool_client.pool.plan(load_modes={model: "fail"})
    media = pool_client.media(mode="hold")
    local_loads = []

    def forbidden() -> object:
        local_loads.append(True)
        msg = "client must stay ML-free"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "_load_cli_runtime", forbidden)
    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(media),
            "--stt-provider",
            provider,
            "--device",
            "auto",
            "--client-timeout",
            "1",
        ],
    )
    assert result.exit_code == code, result.output
    assert marker in result.stderr
    assert result.stdout_bytes == b""
    assert not local_loads


@pytest.mark.integration
@pytest.mark.xdist_group(name="pool_supervisor")
def test_confirmed_cuda_absence_warns_only_on_stderr(
    pool_client: ClientHarness,
) -> None:
    """Catch stdout warnings or silent CUDA absence fallback."""
    pool_client.pool.options = replace(pool_client.pool.options, cuda_absent_grace_s=0)
    media = pool_client.media(text="cpu transcript")
    result = _RUNNER.invoke(
        cli.app, ["external-transcribe", str(media), "--device", "cuda"]
    )
    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == ("cpu transcript" + os.linesep).encode("utf-8")
    assert "ASKVLM_CUDA_UNAVAILABLE" in result.stderr
    assert {event["backend"] for event in pool_client.pool.events("submitted")} == {
        "cpu"
    }


@pytest.mark.integration
@pytest.mark.xdist_group(name="pool_supervisor")
def test_batch_streams_in_input_order_and_names_outputs_before_completion(
    pool_client: ClientHarness, tmp_path: Path
) -> None:
    """Catch reordered JSON, name collisions or lost call caps."""
    pool_client.pool.config = configuration(
        cpu={"max_slots": 2, "max_slots_per_worker": 2}
    )
    inputs = [
        pool_client.media(name=f"{index}/{name}.wav", mode="hold", text=text)
        for index, (name, text) in enumerate(
            [("A", "first"), ("A", "second"), ("a", "third"), ("A-2", "fourth")]
        )
    ]
    jobs_seen = []

    def release() -> None:
        jobs = pool_client.pool.until(pool_client.jobs, lambda jobs: len(jobs) == 4)
        jobs_seen.extend(jobs)
        pool_client.pool.until(
            lambda: pool_client.pool.events("submitted"),
            lambda events: len(events) == 2,
        )
        for job in jobs[1:]:
            pool_client.pool.release(job)
        pool_client.pool.until(
            lambda: pool_client.pool.events("emitted"), lambda events: len(events) == 3
        )
        pool_client.pool.release(jobs[0])

    pool_client.background(release)
    output_dir, output_file = tmp_path / "texts", tmp_path / "combined.txt"
    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            *(str(path) for path in inputs),
            "--device",
            "cpu",
            "--priority",
            "high",
            "--max-parallel",
            "2",
            "--output-dir",
            str(output_dir),
            "--output-file",
            str(output_file),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["input"] for row in rows] == [str(path) for path in inputs]
    assert [row["text"] for row in rows] == ["first", "second", "third", "fourth"]
    assert [row["status"] for row in rows] == ["ok"] * 4
    assert all(row["error"] is None for row in rows)
    assert [Path(row["output"]).name for row in rows] == [
        "A.txt",
        "A-2.txt",
        "a-3.txt",
        "A-2-2.txt",
    ]
    assert [Path(row["output"]).read_text(encoding="utf-8") for row in rows] == [
        row["text"] for row in rows
    ]
    assert output_file.read_text(encoding="utf-8") == "first\nsecond\nthird\nfourth"
    assert {job.priority for job in jobs_seen} == {"high"}
    assert {job.max_parallel for job in jobs_seen} == {2}
    assert len({job.call_id for job in jobs_seen}) == 1
    assert pool_client.pool.events("emitted")[0]["job_id"] != jobs_seen[0].job_id
    assert result.stderr == ""


@pytest.mark.integration
@pytest.mark.xdist_group(name="pool_supervisor")
@pytest.mark.parametrize("explicit", [None, 2])
def test_daemon_workers_alias_preserves_call_cap_and_warns(
    pool_client: ClientHarness, explicit: int | None
) -> None:
    """Catch broken bot aliases or overridden parallel caps."""
    media = pool_client.media(mode="hold", text="bot transcript")
    seen = []

    def release() -> None:
        jobs = pool_client.pool.until(pool_client.jobs, bool)
        seen.extend(jobs)
        pool_client.pool.until(lambda: pool_client.pool.events("submitted"), bool)
        pool_client.pool.release(jobs[0])

    pool_client.background(release)
    args = [
        "external-transcribe",
        str(media),
        "--device",
        "cpu",
        "--daemon-workers",
        "3",
    ]
    if explicit is not None:
        args.extend(["--max-parallel", str(explicit)])
    result = _RUNNER.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert seen[0].max_parallel == (explicit or 3)
    assert result.stdout_bytes == ("bot transcript" + os.linesep).encode("utf-8")
    assert "ASKVLM_DEPRECATED" in result.stderr
    assert "--daemon-workers" in result.stderr
    assert "--max-parallel" in result.stderr


@pytest.mark.integration
@pytest.mark.parametrize(
    ("statuses", "code", "blocked_output"),
    [
        (["error", "unavailable", "expired"], 10, False),
        (["error", "unavailable"], 11, False),
        (["ok", "error"], 1, False),
        (["ok", "empty"], 0, False),
        pytest.param(
            ["expired", "ok"],
            10,
            True,
            marks=pytest.mark.skipif(
                sys.platform != "win32", reason="real Windows file sharing"
            ),
        ),
        pytest.param(
            ["unavailable", "ok"],
            11,
            True,
            marks=pytest.mark.skipif(
                sys.platform != "win32", reason="real Windows file sharing"
            ),
        ),
    ],
)
def test_batch_exit_priority_and_partial_output_follow_terminal_queue_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[str],
    code: int,
    *,
    blocked_output: bool,
) -> None:
    """Catch hidden batch failures or output errors replacing exit 10/11."""
    inputs = [tmp_path / f"{index}.wav" for index in range(len(statuses))]
    for path in inputs:
        path.write_text("fake media", encoding="utf-8")
    root = QueuePaths(tmp_path / "queue")

    def publish(paths: QueuePaths) -> None:
        assert paths.root == root.root
        write_heartbeat(paths, {"state": "running"})
        for file in paths.incoming.glob("*.json"):
            payload = read_job_payload(paths, file.stem)
            assert payload is not None
            job = JobRecord.from_dict(payload, expected_job_id=file.stem)
            status = statuses[job.call_index]
            write_result(
                paths,
                ResultRecord(
                    job_id=job.job_id,
                    status=status,
                    text="success" if status == "ok" else "",
                    error_kind=None if status in {"ok", "empty"} else status,
                    error_detail=None
                    if status in {"ok", "empty"}
                    else f"detail {status}",
                    attempts=1,
                    finished_at=time.time(),
                ),
            )

    monkeypatch.setattr("core.stt_pool.client.spawn_supervisor", publish)
    monkeypatch.setattr("core.stt_pool.client._probe_with_ffprobe", lambda _p, _t: None)
    output = tmp_path / "joined.txt"
    args = [
        "external-transcribe",
        *(str(path) for path in inputs),
        "--output-file",
        str(output),
    ]
    if blocked_output:
        output.write_bytes(b"previous output")
    else:
        args.append("--no-stdout")
    with output.open("rb") if blocked_output else nullcontext():
        result = _RUNNER.invoke(cli.app, args)
    assert result.exit_code == code, result.output
    if blocked_output:
        assert isinstance(result.exception, SystemExit)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        assert [row["input"] for row in rows] == [str(path) for path in inputs]
        assert [row["status"] for row in rows] == [
            "timeout" if status == "expired" else status for status in statuses
        ]
        assert [row["text"] for row in rows] == ["", "success"]
        assert output.read_bytes() == b"previous output"
        diagnostic = [
            line for line in result.stderr.splitlines() if "--output-file" in line
        ]
        assert len(diagnostic) == 1
        assert str(output) in diagnostic[0]
        assert not tuple(tmp_path.glob("*.tmp"))
    else:
        assert result.stdout_bytes == b""
        assert output.read_text(encoding="utf-8") == "\n".join(
            "success" if status == "ok" else "" for status in statuses
        )
    assert not tuple(root.results.glob("*.json"))


@pytest.mark.parametrize(
    "flags",
    [
        ["--diarization"],
        ["--dialog-blocks"],
        ["--work-dir", "work"],
        ["--no-stdout"],
        ["--client-timeout", "nan"],
        ["--language", "not a language"],
        ["--priority", "urgent"],
        ["--max-parallel", "0"],
        ["--max-parallel", "65"],
        ["--stt-provider", "gigaam-ctc", "--device", "cuda"],
        ["--stt-provider", "gigaam-e2e-ctc", "--device", "cuda:0"],
    ],
)
def test_daemon_validation_fails_before_queue_submission(
    tmp_path: Path, flags: list[str]
) -> None:
    """Catch ignored options or malformed queued requests."""
    media = tmp_path / "clip.wav"
    media.write_text("fake media", encoding="utf-8")
    result = _RUNNER.invoke(cli.app, ["external-transcribe", str(media), *flags])
    assert result.exit_code == 2, result.output
    assert not tuple((tmp_path / "queue").rglob("incoming/*.json"))
    if flags[0] in {"--diarization", "--dialog-blocks", "--work-dir"}:
        assert flags[0] in result.output
        assert "--no-daemon" in result.output


def test_no_daemon_rejects_several_inputs(tmp_path: Path) -> None:
    """Catch the legacy one-shot flow silently dropping all but its first argument."""
    paths = [tmp_path / f"{index}.wav" for index in range(2)]
    for path in paths:
        path.touch()
    result = _RUNNER.invoke(
        cli.app, ["external-transcribe", *(str(path) for path in paths), "--no-daemon"]
    )
    assert result.exit_code == 2


@pytest.mark.integration
@pytest.mark.xdist_group(name="pool_supervisor")
def test_unknown_whisper_device_keeps_legacy_auto_coercion(
    pool_client: ClientHarness,
) -> None:
    """Catch rejecting the legacy Whisper device vocabulary before safe auto routing."""
    media = pool_client.media(mode="hold", text="auto")
    seen = []

    def release() -> None:
        jobs = pool_client.pool.until(pool_client.jobs, bool)
        seen.extend(jobs)
        pool_client.pool.until(lambda: pool_client.pool.events("submitted"), bool)
        pool_client.pool.release(jobs[0])

    pool_client.background(release)
    result = _RUNNER.invoke(
        cli.app, ["external-transcribe", str(media), "--device", "cuda:0"]
    )
    assert result.exit_code == 0, result.output
    assert seen[0].request.device == "auto"


@pytest.mark.integration
@pytest.mark.xdist_group(name="pool_supervisor")
def test_daemon_command_uses_explicit_config_overrides_and_ignores_legacy_flags(
    pool_client: ClientHarness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catch lost config overrides or worker keys from legacy flags."""
    import core.stt_pool.supervisor as module  # noqa: PLC0415 - replace the process boundary at this entry

    config_file = tmp_path / "explicit.json"
    config_file.write_text(
        '{"cpu":{"max_slots":1},"gpu":{"max_workers":0}}', encoding="utf-8"
    )
    job = pool_client.pool.job(text="daemon result")
    captured = []
    served = []
    real_entry = module.run_supervisor

    def serve(root: Path, config: PoolConfig, *, stop_event: Event) -> int:
        assert root == pool_client.pool.paths.root
        captured.append(config)

        def finish_on_result() -> None:
            try:
                served.append(pool_client.pool.result(job).text)
            finally:
                try:
                    handler = signal.getsignal(signal.SIGINT)
                    assert callable(handler)
                    handler(signal.SIGINT, None)
                    assert stop_event.is_set()
                finally:
                    stop_event.set()

        pool_client.background(finish_on_result)
        code = real_entry(
            root, config, options=pool_client.pool.options, stop_event=stop_event
        )
        pool_client.pool.code = code
        pool_client.pool.finish(request_stop=False)
        return code

    monkeypatch.setattr(module, "run_supervisor", serve)
    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe-daemon",
            "--queue-dir",
            str(pool_client.pool.paths.root),
            "--config",
            str(config_file),
            "--cpu-slots",
            "2",
            "--gpu-workers",
            "0",
            "--thread-budget",
            "4",
            "--commit-budget-gb",
            "10",
            "--vram-budget-gb",
            "3",
            "--idle-shutdown",
            "12",
            "--workers",
            "8",
            "--stt-provider",
            "gigaam-ctc",
            "--device",
            "cuda",
        ],
    )
    assert result.exit_code == 0, result.output
    assert served == ["daemon result"]
    assert captured[0].cpu.max_slots == 2
    assert captured[0].threads.budget == 4
    assert captured[0].memory.commit_budget_bytes == 10 * 1024**3
    assert captured[0].vram.budget_bytes == 3 * 1024**3
    assert captured[0].lifecycle.worker_idle_ttl_s == 12
    assert {
        event["key"]["provider"] for event in pool_client.pool.events("constructed")
    } == {"whisper"}
    assert "ASKVLM_DEPRECATED" in result.stderr
    assert "--stt-provider" in result.stderr
    assert "--device" in result.stderr


@pytest.mark.parametrize("args", [["--cpu-slots", "-1"], ["--config", "missing.json"]])
def test_daemon_bad_config_returns_two_without_starting(
    tmp_path: Path, args: list[str]
) -> None:
    """Catch config tracebacks or unconstrained pool startup."""
    if args[0] == "--config":
        args = ["--config", str(tmp_path / "missing.json")]
    result = _RUNNER.invoke(
        cli.app,
        ["external-transcribe-daemon", "--queue-dir", str(tmp_path / "queue"), *args],
    )
    assert result.exit_code == 2, result.output
    assert not (tmp_path / "queue").exists()


@pytest.mark.integration
def test_real_cli_timeout_has_pure_stdout_and_temp_queue_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch subprocess encoding, cancellation and exit-code drift."""
    paths = QueuePaths(tmp_path / "queue")
    paths.ensure()
    write_heartbeat(paths, {"state": "running"})
    beacon = read_heartbeat(paths)
    assert beacon is not None
    # * Future liveness covers process startup delays; the client never needs a spawn.
    beacon["ts"] = time.time() + 3600
    fsio.atomic_write_json(paths.heartbeat_file, beacon)
    media = tmp_path / "clip.wav"
    media.write_text("fake media", encoding="utf-8")
    monkeypatch.setenv("ASKVLM_EXTERNAL_QUEUE_DIR", str(paths.root))
    completed = subprocess.run(  # noqa: S603 - trusted CLI and temporary inputs, no shell
        [
            sys.executable,
            "-X",
            "utf8",
            str(Path(cli.__file__).resolve()),
            "external-transcribe",
            str(media),
            "--device",
            "cpu",
            "--client-timeout",
            "1",
        ],
        env=dict(os.environ),
        capture_output=True,
        check=False,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert completed.returncode == 10, completed.stderr
    assert completed.stdout == b""
    assert b"ASKVLM_CLIENT_TIMEOUT" in completed.stderr
    jobs = list(paths.incoming.glob("*.json"))
    assert len(jobs) == 1
    assert paths.cancel_file(jobs[0].stem).exists()
    assert not paths.supervisor_out_log.exists()


@pytest.mark.integration
@pytest.mark.xdist_group(name="pool_supervisor")
def test_cli_subprocess_streams_first_json_line_before_last_input_finishes(
    pool_client: ClientHarness,
) -> None:
    """Catch buffering an entire batch until its longest input completes."""
    first = pool_client.media(text="ready first")
    second = pool_client.media(mode="hold", text="ready second")
    pool_client.pool.start()
    pool_client.pool.heartbeat(lambda value: value["state"] == "running")
    argv = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "tests.stt_pool_cli_entry",
        "external-transcribe",
        str(first),
        str(second),
        "--device",
        "cpu",
    ]
    with subprocess.Popen(  # noqa: S603 - safe CLI shim forbids detached supervisor launches
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(os.environ),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as child:
        try:
            assert child.stdout is not None
            with ThreadPoolExecutor(max_workers=1) as reader:
                first_line = reader.submit(child.stdout.readline)
                try:
                    row = json.loads(first_line.result(timeout=30))
                    assert row["input"] == str(first)
                    assert row["text"] == "ready first"
                    assert child.poll() is None
                    held = pool_client.pool.until(
                        lambda: pool_client.pool.events("submitted"),
                        lambda events: len(events) == 2,
                    )[1]
                    assert not any(
                        event["job_id"] == held["job_id"]
                        for event in pool_client.pool.events("emitted")
                    )
                    (pool_client.pool.engine_root / "release" / held["job_id"]).touch()
                finally:
                    if not first_line.done() and child.poll() is None:
                        child.terminate()
            rest, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, stderr
            last_row = json.loads(rest)
            assert last_row["input"] == str(second)
            assert last_row["text"] == "ready second"
            assert stderr == b""
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)


@pytest.mark.integration
@pytest.mark.xdist_group(name="pool_supervisor")
@pytest.mark.skipif(sys.platform != "win32", reason="real Windows file sharing")
def test_failed_aggregate_output_keeps_existing_file_and_terminal_rows(
    pool_client: ClientHarness, tmp_path: Path
) -> None:
    """Catch failed atomic output destroying an earlier file or losing terminal rows."""
    inputs = [
        pool_client.media(text="first text"),
        pool_client.media(text="second text"),
    ]
    pool_client.pool.start()
    pool_client.pool.heartbeat(lambda value: value["state"] == "running")
    target = tmp_path / "transcript.txt"
    target.write_text("previous output", encoding="utf-8")
    with target.open("rb"):
        result = _RUNNER.invoke(
            cli.app,
            [
                "external-transcribe",
                *(str(path) for path in inputs),
                "--device",
                "cpu",
                "--output-file",
                str(target),
            ],
        )
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert [row["text"] for row in rows] == ["first text", "second text"]
    assert result.exit_code == 1, (result.exception, result.stderr)
    assert isinstance(result.exception, SystemExit)
    diagnostic = result.stderr.splitlines()
    assert len(diagnostic) == 1
    assert "--output-file" in diagnostic[0]
    assert str(target) in diagnostic[0]
    assert target.read_text(encoding="utf-8") == "previous output"
    assert not tuple(tmp_path.glob("*.tmp"))


def test_external_transcribe_uses_small_and_stdout_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The external CLI defaults to a small one-shot stdout contract."""
    monkeypatch.setattr(cli.sys, "platform", "linux")
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    work_dir = tmp_path / "work"
    calls: dict[str, object] = {}

    events: list[str] = []

    class _StubPipeline:
        def process(self, input_media: Path, process_work_dir: Path) -> _StubDocument:
            events.append("process")
            calls["process_input"] = input_media
            calls["process_work_dir"] = process_work_dir
            return _StubDocument("transcribed text")

        def close(self, *, aggressive: bool = False) -> None:
            events.append("close")
            calls["close_aggressive"] = aggressive

    def _fake_create_local_pipeline(**kwargs: object) -> _StubPipeline:
        calls["pipeline_kwargs"] = kwargs
        return _StubPipeline()

    monkeypatch.setattr(cli, "_create_local_pipeline", _fake_create_local_pipeline)

    real_echo = cli.typer.echo

    def _tracking_echo(message: str | None = None, **kwargs: object) -> None:
        events.append("stdout")
        real_echo(message, **kwargs)

    monkeypatch.setattr(cli.typer, "echo", _tracking_echo)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--no-daemon",
            "--work-dir",
            str(work_dir),
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == "transcribed text\n"
    assert events == ["process", "stdout", "close"]
    assert calls["pipeline_kwargs"] == {
        "whisper_model": "small",
        "engine": "whisperx",
        "diarization": False,
        "dialog_blocks": False,
        "language": None,
        "device": "auto",
        "compute_type": "auto",
        "stt_provider": "whisper",
    }
    assert calls["process_input"] == input_path
    assert calls["process_work_dir"] == work_dir
    assert calls["close_aggressive"] is False


def test_external_transcribe_writes_output_file_without_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The external CLI can write a file without emitting transcript stdout."""
    monkeypatch.setattr(cli.sys, "platform", "linux")
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    output_file = tmp_path / "nested" / "transcript.txt"

    events: list[str] = []
    calls: dict[str, object] = {}

    class _StubPipeline:
        def process(self, _input_media: Path, _process_work_dir: Path) -> _StubDocument:
            events.append("process")
            return _StubDocument("saved transcript")

        def close(self, *, aggressive: bool = False) -> None:
            events.append("close")
            calls["close_aggressive"] = aggressive

    def _fake_create_local_pipeline(**_kwargs: object) -> _StubPipeline:
        return _StubPipeline()

    real_write_plain = cli._write_plain_text  # noqa: SLF001

    def _tracing_write_plain(out_path: Path, text: str) -> None:
        events.append("write_file")
        real_write_plain(out_path, text)

    monkeypatch.setattr(cli, "_create_local_pipeline", _fake_create_local_pipeline)
    monkeypatch.setattr(cli, "_write_plain_text", _tracing_write_plain)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--no-daemon",
            "--work-dir",
            str(tmp_path / "work"),
            "--output-file",
            str(output_file),
            "--no-stdout",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert output_file.read_text(encoding="utf-8") == "saved transcript"
    assert events == ["process", "write_file", "close"]
    assert calls["close_aggressive"] is False


@pytest.mark.parametrize("stub_text", ["", "   \n\t  "])
def test_external_transcribe_accepts_empty_transcript_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_text: str,
) -> None:
    """Whitespace-only transcripts are successful and stay silent in stdout/stderr."""
    monkeypatch.setattr(cli.sys, "platform", "linux")
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    output_file = tmp_path / "out.txt"
    calls: dict[str, object] = {}
    events: list[str] = []

    class _StubPipeline:
        def process(self, _input_media: Path, _process_work_dir: Path) -> _StubDocument:
            return _StubDocument(stub_text)

        def close(self, *, aggressive: bool = False) -> None:
            calls["close_aggressive"] = aggressive

    real_echo = cli.typer.echo

    def _tracking_echo(message: str | None = None, **kwargs: object) -> None:
        events.append("stdout")
        real_echo(message, **kwargs)

    monkeypatch.setattr(
        cli,
        "_create_local_pipeline",
        lambda **_kwargs: _StubPipeline(),
    )
    monkeypatch.setattr(cli.typer, "echo", _tracking_echo)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--no-daemon",
            "--work-dir",
            str(tmp_path / "work"),
            "--output-file",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == ""
    assert events == []
    assert calls["close_aggressive"] is False


def test_external_transcribe_isolated_attempt_prefers_result_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Child JSON result is source of truth even with crash-like return code."""
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")

    def _fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result_path = Path(
            next(a for a in command if a.startswith("--_internal-result-file=")).split(
                "=", 1
            )[1]
        )
        result_path.write_text(
            '{"status":"ok","text":"from child"}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            command,
            3221226505,
            stdout="",
            stderr="native crash after write",
        )

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    run_isolated_attempt = cli._run_external_transcribe_isolated_attempt  # noqa: SLF001
    result = run_isolated_attempt(
        input_path=input_path,
        whisper_model="small",
        language=None,
        device="cuda",
        compute_type="auto",
        diarization=False,
        dialog_blocks=False,
        work_dir=tmp_path / "work",
    )

    assert result.success_text == "from child"
    assert result.return_code == 3221226505


def test_external_transcribe_windows_parent_uses_successful_child_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows parent accepts successful child result despite crash-like code."""
    monkeypatch.setattr(cli.sys, "platform", "win32")
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    output_file = tmp_path / "result.txt"
    called_devices: list[str] = []
    attempt_result_cls = cli._ExternalChildAttemptResult  # noqa: SLF001

    def _fake_attempt(**kwargs: object) -> object:
        called_devices.append(str(kwargs["device"]))
        return attempt_result_cls(
            success_text="isolated text",
            return_code=3221226505,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(cli, "_run_external_transcribe_isolated_attempt", _fake_attempt)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--no-daemon",
            "--output-file",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == "isolated text\n"
    assert result.stderr == ""
    assert output_file.read_text(encoding="utf-8") == "isolated text"
    assert called_devices == ["auto"]


def test_external_transcribe_windows_crash_like_failures_retry_once_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows parent does exactly one CPU retry for crash-like child failures."""
    monkeypatch.setattr(cli.sys, "platform", "win32")
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    called_devices: list[str] = []
    attempt_result_cls = cli._ExternalChildAttemptResult  # noqa: SLF001

    def _fake_attempt(**kwargs: object) -> object:
        called_devices.append(str(kwargs["device"]))
        if len(called_devices) == 1:
            return attempt_result_cls(
                success_text=None,
                return_code=3221226505,
                stdout="",
                stderr="native crash",
            )
        return attempt_result_cls(
            success_text="cpu recovered text",
            return_code=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(cli, "_run_external_transcribe_isolated_attempt", _fake_attempt)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--no-daemon",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == "cpu recovered text\n"
    assert result.stderr == ""
    assert called_devices == ["auto", "cpu"]


def test_external_transcribe_windows_non_crash_error_does_not_retry_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows parent does not retry CPU for regular child errors."""
    monkeypatch.setattr(cli.sys, "platform", "win32")
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    called_devices: list[str] = []
    attempt_result_cls = cli._ExternalChildAttemptResult  # noqa: SLF001

    def _fake_attempt(**kwargs: object) -> object:
        called_devices.append(str(kwargs["device"]))
        return attempt_result_cls(
            success_text=None,
            return_code=2,
            stdout="",
            stderr="regular child failure",
        )

    monkeypatch.setattr(cli, "_run_external_transcribe_isolated_attempt", _fake_attempt)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--no-daemon",
        ],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "regular child failure" in (result.stderr or "")
    assert called_devices == ["auto"]


def test_is_internal_child_ipc_error_detects_marker() -> None:
    """_is_internal_child_ipc_error returns True only for the IPC marker text."""
    _is_err = cli._is_internal_child_ipc_error  # noqa: SLF001
    assert _is_err("...Internal child mode requires --_internal-result-file...")
    assert not _is_err("some other error")
    assert not _is_err("")
    assert not _is_err(None)


def test_external_transcribe_internal_ipc_error_retries_on_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catch lost provider identity in Windows child commands."""
    monkeypatch.setattr(cli.sys, "platform", "win32")
    call_count = 0

    def _fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # ! First attempt: simulate IPC error (result file NOT written)
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="",
                stderr="Internal child mode requires --_internal-result-file.",
            )
        # * Second attempt (CPU retry): write result JSON and succeed
        result_path = Path(
            next(a for a in command if a.startswith("--_internal-result-file=")).split(
                "=", 1
            )[1]
        )
        result_path.write_text(
            '{"status":"ok","text":"cpu fallback text"}', encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    input_file = tmp_path / "audio.wav"
    input_file.write_bytes(b"fake")

    result = _RUNNER.invoke(
        cli.app,
        ["external-transcribe", str(input_file), "--no-daemon", "--device", "cuda"],
    )
    assert result.exit_code == 0
    assert "cpu fallback text" in result.output
    assert call_count == 2  # * first attempt + one CPU retry


def test_gigaam_no_daemon_on_windows_skips_child_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GigaAM CTC --no-daemon on Windows stays in-process (no GPU child path)."""
    monkeypatch.setattr(cli.sys, "platform", "win32")
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    isolated_calls = {"n": 0}
    once_calls: list[dict[str, object]] = []

    def _fake_isolated(**_kwargs: object) -> object:
        isolated_calls["n"] += 1
        msg = "Windows child isolation must not run for gigaam-ctc"
        raise AssertionError(msg)

    def _fake_once(**kwargs: object) -> str:
        once_calls.append(dict(kwargs))
        before_close = kwargs.get("before_close")
        text = "gigaam in-process"
        if callable(before_close):
            before_close(text)
        return text

    monkeypatch.setattr(
        cli, "_run_external_transcribe_isolated_attempt", _fake_isolated
    )
    monkeypatch.setattr(cli, "_run_external_transcribe_once", _fake_once)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--no-daemon",
            "--stt-provider",
            "gigaam-ctc",
            "--device",
            "auto",
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == "gigaam in-process\n"
    assert isolated_calls["n"] == 0
    assert len(once_calls) == 1
    assert once_calls[0]["stt_provider"] == "gigaam-ctc"
    assert once_calls[0]["device"] == "cpu"


def test_gigaam_e2e_no_daemon_on_windows_isolates_gpu_child_and_retries_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CUDA-capable GigaAM run on Windows gets Whisper's child isolation.

    A native crash of the GPU child must not take the caller down: exactly one
    CPU retry of the same provider recovers the transcript.
    """
    monkeypatch.setattr(cli.sys, "platform", "win32")
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    attempts: list[dict[str, object]] = []
    attempt_result_cls = cli._ExternalChildAttemptResult  # noqa: SLF001

    def _fake_attempt(**kwargs: object) -> object:
        attempts.append(dict(kwargs))
        if len(attempts) == 1:
            return attempt_result_cls(
                success_text=None,
                return_code=3221226505,
                stdout="",
                stderr="native crash",
            )
        return attempt_result_cls(
            success_text="e2e cpu recovered",
            return_code=0,
            stdout="",
            stderr="",
        )

    def _no_in_process(**_kwargs: object) -> str:
        msg = "a GPU-capable run must go through the isolated child"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "_run_external_transcribe_isolated_attempt", _fake_attempt)
    monkeypatch.setattr(cli, "_run_external_transcribe_once", _no_in_process)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--no-daemon",
            "--stt-provider",
            "gigaam-e2e-ctc",
            "--device",
            "auto",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "e2e cpu recovered\n"
    assert [a["device"] for a in attempts] == ["auto", "cpu"]
    assert {a["stt_provider"] for a in attempts} == {"gigaam-e2e-ctc"}


def test_gigaam_e2e_cpu_on_windows_skips_child_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit CPU request has no GPU state to isolate and stays in-process."""
    monkeypatch.setattr(cli.sys, "platform", "win32")
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    once_calls: list[dict[str, object]] = []

    def _fake_isolated(**_kwargs: object) -> object:
        msg = "child isolation must not run for --device cpu"
        raise AssertionError(msg)

    def _fake_once(**kwargs: object) -> str:
        once_calls.append(dict(kwargs))
        before_close = kwargs.get("before_close")
        text = "e2e in-process"
        if callable(before_close):
            before_close(text)
        return text

    monkeypatch.setattr(
        cli, "_run_external_transcribe_isolated_attempt", _fake_isolated
    )
    monkeypatch.setattr(cli, "_run_external_transcribe_once", _fake_once)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--no-daemon",
            "--stt-provider",
            "gigaam-e2e-ctc",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "e2e in-process\n"
    assert len(once_calls) == 1
    assert once_calls[0]["device"] == "cpu"
    assert once_calls[0]["stt_provider"] == "gigaam-e2e-ctc"


def test_build_external_child_command_includes_stt_provider(tmp_path: Path) -> None:
    """Windows child argv carries ``--stt-provider`` for Whisper isolation."""
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    command = cli._build_external_child_command(  # noqa: SLF001
        input_path=input_path,
        whisper_model="small",
        language=None,
        device="cuda",
        compute_type="auto",
        diarization=False,
        dialog_blocks=False,
        work_dir=None,
        child_result_file=tmp_path / "child_result.json",
        stt_provider="whisper",
    )
    assert "--stt-provider" in command
    idx = command.index("--stt-provider")
    assert command[idx + 1] == "whisper"

    gigaam_command = cli._build_external_child_command(  # noqa: SLF001
        input_path=input_path,
        whisper_model="small",
        language=None,
        device="cpu",
        compute_type="auto",
        diarization=False,
        dialog_blocks=False,
        work_dir=None,
        child_result_file=tmp_path / "child_result.json",
        stt_provider="gigaam-ctc",
    )
    assert "--stt-provider" in gigaam_command
    gidx = gigaam_command.index("--stt-provider")
    assert gigaam_command[gidx + 1] == "gigaam-ctc"

    e2e_command = cli._build_external_child_command(  # noqa: SLF001
        input_path=input_path,
        whisper_model="small",
        language=None,
        device="cpu",
        compute_type="auto",
        diarization=False,
        dialog_blocks=False,
        work_dir=None,
        child_result_file=tmp_path / "child_result.json",
        stt_provider="gigaam-e2e-ctc",
    )
    assert "--stt-provider" in e2e_command
    eidx = e2e_command.index("--stt-provider")
    assert e2e_command[eidx + 1] == "gigaam-e2e-ctc"


@pytest.mark.parametrize(
    ("provider", "device_flag", "expected_device"),
    [
        # * gigaam-ctc is CPU-only: ``auto`` resolves to ``cpu``.
        ("gigaam-ctc", "auto", "cpu"),
        # * gigaam-e2e-ctc may use CUDA: the request reaches the pipeline as given.
        ("gigaam-e2e-ctc", "auto", "auto"),
        ("gigaam-e2e-ctc", "cpu", "cpu"),
        ("gigaam-e2e-ctc", "cuda", "cuda"),
    ],
)
def test_external_transcribe_propagates_gigaam_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    device_flag: str,
    expected_device: str,
) -> None:
    """GigaAM ``--stt-provider`` reaches LocalPipeline with the allowed device."""
    monkeypatch.setattr(cli.sys, "platform", "linux")
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    calls: dict[str, object] = {}

    class _StubPipeline:
        def process(self, _input_media: Path, _process_work_dir: Path) -> _StubDocument:
            return _StubDocument("gigaam ok")

        def close(self, *, aggressive: bool = False) -> None:
            calls["closed"] = aggressive

    def _fake_create_local_pipeline(**kwargs: object) -> _StubPipeline:
        calls["pipeline_kwargs"] = kwargs
        return _StubPipeline()

    monkeypatch.setattr(cli, "_create_local_pipeline", _fake_create_local_pipeline)
    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--no-daemon",
            "--stt-provider",
            provider,
            "--device",
            device_flag,
            "--work-dir",
            str(tmp_path / "work"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == "gigaam ok\n"
    kwargs = calls["pipeline_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["stt_provider"] == provider
    assert kwargs["device"] == expected_device


@pytest.mark.parametrize(
    ("provider", "device_flag"),
    [
        # * gigaam-ctc stays CPU-only: CUDA is rejected.
        ("gigaam-ctc", "cuda"),
        # * Both heads reject tokens outside their device vocabulary.
        ("gigaam-ctc", "tpu"),
        ("gigaam-e2e-ctc", "tpu"),
    ],
)
def test_external_transcribe_rejects_invalid_gigaam_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    device_flag: str,
) -> None:
    """An invalid GigaAM ``--device`` fails before pipeline construction."""
    monkeypatch.setattr(cli.sys, "platform", "linux")
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    created = {"n": 0}

    def _boom(**_kwargs: object) -> object:
        created["n"] += 1
        msg = "should not create"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "_create_local_pipeline", _boom)
    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--no-daemon",
            "--stt-provider",
            provider,
            "--device",
            device_flag,
        ],
    )
    assert result.exit_code != 0
    assert created["n"] == 0
