from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cli
from core.external_client import ClientOutcome

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
