from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli
from core.external_client import ClientOutcome
from core.external_queue import ENV_QUEUE_DIR, resolve_queue_dir, write_heartbeat

_RUNNER = CliRunner()


class _StubDocument:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_full_text(self) -> str:
        return self._text


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
    """exit_code=1 with 'Internal child mode' stderr triggers CPU retry, not fatal error."""
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


def test_external_transcribe_daemon_mode_echoes_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default daemon mode prints the transcript returned by the client."""
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")

    def _fake_client(_request: object) -> ClientOutcome:
        return ClientOutcome("ok", "daemon text", None)

    monkeypatch.setattr(cli, "run_client_transcribe", _fake_client)
    result = _RUNNER.invoke(cli.app, ["external-transcribe", str(input_path)])
    assert result.exit_code == 0
    assert result.stdout == "daemon text\n"


def test_external_transcribe_daemon_mode_timeout_cpu_device_uses_timeout_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A client timeout on an explicit CPU request maps to the timeout exit code.

    With ``--device cpu`` there is no GPU to recover from, so no CPU fallback runs
    and the dedicated timeout exit code / stderr marker is surfaced directly.
    """
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")

    def _fake_client(_request: object) -> ClientOutcome:
        return ClientOutcome("timeout", "", "slow")

    monkeypatch.setattr(cli, "run_client_transcribe", _fake_client)
    result = _RUNNER.invoke(
        cli.app, ["external-transcribe", str(input_path), "--device", "cpu"]
    )
    assert result.exit_code == cli.EXTERNAL_TRANSCRIBE_TIMEOUT_EXIT
    assert "ASKVLM_CLIENT_TIMEOUT" in (result.stderr or "")


def test_external_transcribe_daemon_mode_unavailable_cpu_device_uses_unavailable_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable daemon on an explicit CPU request maps to the exit code."""
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")

    def _fake_client(_request: object) -> ClientOutcome:
        return ClientOutcome("unavailable", "", "down")

    monkeypatch.setattr(cli, "run_client_transcribe", _fake_client)
    result = _RUNNER.invoke(
        cli.app, ["external-transcribe", str(input_path), "--device", "cpu"]
    )
    assert result.exit_code == cli.EXTERNAL_TRANSCRIBE_UNAVAILABLE_EXIT
    assert "ASKVLM_DAEMON_UNAVAILABLE" in (result.stderr or "")


@pytest.mark.parametrize("degraded_status", ["timeout", "unavailable"])
def test_external_transcribe_daemon_degraded_recovers_via_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, degraded_status: str
) -> None:
    """A degraded GPU-seeded daemon recovers the transcript via one CPU pass."""
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")

    def _fake_client(_request: object) -> ClientOutcome:
        return ClientOutcome(degraded_status, "", "degraded")

    fallback_devices: list[object] = []

    class _StubPipeline:
        def process(self, _input_media: Path, _process_work_dir: Path) -> _StubDocument:
            return _StubDocument("cpu recovered text")

        def close(self, *, aggressive: bool = False) -> None:  # noqa: ARG002
            return None

    def _fake_create_local_pipeline(**kwargs: object) -> _StubPipeline:
        fallback_devices.append(kwargs.get("device"))
        return _StubPipeline()

    monkeypatch.setattr(cli, "run_client_transcribe", _fake_client)
    monkeypatch.setattr(cli, "_create_local_pipeline", _fake_create_local_pipeline)

    # * Default device is "auto" (GPU-preferred), so the CPU fallback is eligible.
    result = _RUNNER.invoke(cli.app, ["external-transcribe", str(input_path)])

    assert result.exit_code == 0, result.output
    assert result.stdout == "cpu recovered text\n"
    assert fallback_devices == ["cpu"]


def test_external_transcribe_daemon_timeout_cpu_fallback_failure_surfaces_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the CPU fallback also fails, the timeout exit code is still surfaced."""
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")

    def _fake_client(_request: object) -> ClientOutcome:
        return ClientOutcome("timeout", "", "slow")

    def _boom(**_kwargs: object) -> object:
        msg = "faster-whisper unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(cli, "run_client_transcribe", _fake_client)
    monkeypatch.setattr(cli, "_create_local_pipeline", _boom)

    result = _RUNNER.invoke(cli.app, ["external-transcribe", str(input_path)])

    assert result.exit_code == cli.EXTERNAL_TRANSCRIBE_TIMEOUT_EXIT
    assert "ASKVLM_CPU_FALLBACK_FAILED" in (result.stderr or "")
    assert "ASKVLM_CLIENT_TIMEOUT" in (result.stderr or "")


@pytest.mark.parametrize("degraded_status", ["timeout", "unavailable"])
def test_gigaam_degraded_daemon_skips_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, degraded_status: str
) -> None:
    """GigaAM CTC never uses Whisper's degraded-daemon CPU fallback path."""
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    created = {"n": 0}

    def _fake_client(_request: object) -> ClientOutcome:
        return ClientOutcome(degraded_status, "", "degraded")

    def _fake_create_local_pipeline(**_kwargs: object) -> object:
        created["n"] += 1
        msg = "pipeline must not be built for GigaAM CPU fallback"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "run_client_transcribe", _fake_client)
    monkeypatch.setattr(cli, "_create_local_pipeline", _fake_create_local_pipeline)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--stt-provider",
            "gigaam-ctc",
            "--device",
            "auto",
        ],
    )
    assert created["n"] == 0
    if degraded_status == "timeout":
        assert result.exit_code == cli.EXTERNAL_TRANSCRIBE_TIMEOUT_EXIT
        assert "ASKVLM_CLIENT_TIMEOUT" in (result.stderr or "")
    else:
        assert result.exit_code == cli.EXTERNAL_TRANSCRIBE_UNAVAILABLE_EXIT
        assert "ASKVLM_DAEMON_UNAVAILABLE" in (result.stderr or "")
    assert "ASKVLM_CPU_FALLBACK_FAILED" not in (result.stderr or "")


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


@pytest.mark.parametrize("degraded_status", ["timeout", "unavailable"])
def test_gigaam_e2e_degraded_daemon_recovers_via_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, degraded_status: str
) -> None:
    """A degraded daemon on a GPU-preferring e2e request recovers on CPU.

    The recovery pass keeps the provider (never silently switches to Whisper)
    and forces the CPU device.
    """
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    built: list[dict[str, object]] = []

    def _fake_client(_request: object) -> ClientOutcome:
        return ClientOutcome(degraded_status, "", "degraded")

    class _StubPipeline:
        def process(self, _input_media: Path, _process_work_dir: Path) -> _StubDocument:
            return _StubDocument("e2e cpu recovered text")

        def close(self, *, aggressive: bool = False) -> None:  # noqa: ARG002
            return None

    def _fake_create_local_pipeline(**kwargs: object) -> _StubPipeline:
        built.append(dict(kwargs))
        return _StubPipeline()

    monkeypatch.setattr(cli, "run_client_transcribe", _fake_client)
    monkeypatch.setattr(cli, "_create_local_pipeline", _fake_create_local_pipeline)

    result = _RUNNER.invoke(
        cli.app,
        ["external-transcribe", str(input_path), "--stt-provider", "gigaam-e2e-ctc"],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "e2e cpu recovered text\n"
    assert len(built) == 1
    assert built[0]["device"] == "cpu"
    assert built[0]["stt_provider"] == "gigaam-e2e-ctc"


def test_gigaam_e2e_cpu_request_skips_degraded_daemon_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit CPU e2e request has no GPU to recover from: exit code only."""
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")

    def _fake_client(_request: object) -> ClientOutcome:
        return ClientOutcome("timeout", "", "slow")

    def _no_pipeline(**_kwargs: object) -> object:
        msg = "no CPU fallback pass for an explicit CPU request"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "run_client_transcribe", _fake_client)
    monkeypatch.setattr(cli, "_create_local_pipeline", _no_pipeline)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--stt-provider",
            "gigaam-e2e-ctc",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == cli.EXTERNAL_TRANSCRIBE_TIMEOUT_EXIT
    assert "ASKVLM_CPU_FALLBACK_FAILED" not in (result.stderr or "")


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


def test_external_transcribe_daemon_command_invokes_run_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The daemon subcommand forwards the worker count into the daemon config."""
    captured: dict[str, object] = {}

    def _fake_run_daemon(queue_dir: object, **kwargs: object) -> int:
        captured["queue_dir"] = queue_dir
        captured["config"] = kwargs.get("config")
        return 0

    monkeypatch.setattr(cli, "run_daemon", _fake_run_daemon)
    result = _RUNNER.invoke(
        cli.app,
        ["external-transcribe-daemon", "--workers", "2", "--device", "cpu"],
    )
    assert result.exit_code == 0
    config = captured["config"]
    assert isinstance(config, cli.DaemonConfig)
    assert config.max_workers == 2
    assert config.stt_provider == "whisper"


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


@pytest.mark.parametrize(
    ("provider", "device_flag", "expected_device"),
    [
        ("gigaam-ctc", "auto", "cpu"),
        ("gigaam-e2e-ctc", "auto", "auto"),
        ("gigaam-e2e-ctc", "cuda", "cuda"),
        ("gigaam-e2e-ctc", "cpu", "cpu"),
    ],
)
def test_external_transcribe_daemon_gigaam_config(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    device_flag: str,
    expected_device: str,
) -> None:
    """Daemon CLI stores the GigaAM provider and its allowed device on the config."""
    captured: dict[str, object] = {}

    def _fake_run_daemon(queue_dir: object, **kwargs: object) -> int:
        captured["config"] = kwargs.get("config")
        return 0

    monkeypatch.setattr(cli, "run_daemon", _fake_run_daemon)
    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe-daemon",
            "--stt-provider",
            provider,
            "--device",
            device_flag,
        ],
    )
    assert result.exit_code == 0
    config = captured["config"]
    assert isinstance(config, cli.DaemonConfig)
    assert config.stt_provider == provider
    assert config.device == expected_device


def test_external_transcribe_daemon_rejects_cuda_for_gigaam_ctc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resident gigaam-ctc daemon still refuses CUDA before serving."""
    served = {"n": 0}

    def _fake_run_daemon(_queue_dir: object, **_kwargs: object) -> int:
        served["n"] += 1
        return 0

    monkeypatch.setattr(cli, "run_daemon", _fake_run_daemon)
    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe-daemon",
            "--stt-provider",
            "gigaam-ctc",
            "--device",
            "cuda",
        ],
    )
    assert result.exit_code != 0
    assert served["n"] == 0


def test_device_mismatch_warning_reaches_stderr_of_the_real_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real CLI process warns on stderr about a daemon device mismatch.

    The live daemon is simulated by a fresh heartbeat in a throw-away queue; the
    request is still submitted to it (client timeout exit, not "unavailable").
    """
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    monkeypatch.setenv(ENV_QUEUE_DIR, str(queue_dir))
    # * Never let this test touch the real shared queue.
    assert resolve_queue_dir() == queue_dir.resolve()
    write_heartbeat(
        queue_dir,
        {"model": "small", "device": "cuda", "stt_provider": "whisper"},
    )
    media = tmp_path / "clip.wav"
    media.write_text("stub", encoding="utf-8")

    completed = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        [
            sys.executable,
            str(Path(cli.__file__).resolve()),
            "external-transcribe",
            str(media),
            "--device",
            "cpu",
            "--client-timeout",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == cli.EXTERNAL_TRANSCRIBE_TIMEOUT_EXIT, (
        completed.stderr
    )
    assert "ASKVLM_DAEMON_DEVICE_MISMATCH" in completed.stderr
    assert "ASKVLM_DAEMON_UNAVAILABLE" not in completed.stderr
