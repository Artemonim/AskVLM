from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, Optional

import typer

from core.stt_pool.client import ClientRequest, run_client
from core.stt_pool.config import PoolConfigError, apply_overrides, load_pool_config
from core.stt_pool.keys import (
    normalize_device_request,
    normalize_language,
    normalize_priority,
)
from core.stt_pool.queue import MAX_PARALLEL_LIMIT
from core.stt_providers import (
    STT_PROVIDER_CHOICES,
    STT_PROVIDER_WHISPER,
    is_gigaam_provider,
    normalize_stt_provider,
    provider_may_use_cuda,
    resolve_gigaam_requested_device,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core.stt_pool.client import InputOutcome
    from core.stt_pool.keys import DeviceRequest

app = typer.Typer(
    help=(
        "AskVLM CLI for local transcription, subtitles, and machine-friendly "
        "external transcription."
    )
)


def _load_cli_runtime() -> dict[str, Any]:
    """Load heavy runtime modules lazily so `--help` stays lightweight."""
    return {
        "LocalPipeline": importlib.import_module("core.pipelines").LocalPipeline,
        "burn_subtitles": importlib.import_module("core.ffmpeg").burn_subtitles,
        "export_document": importlib.import_module("utils.exporters").export_document,
        "export_srt_with_rules": importlib.import_module(
            "utils.exporters"
        ).export_srt_with_rules,
        "SubtitleRules": importlib.import_module("utils.exporters").SubtitleRules,
    }


def _collect_files(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_dir():
        if recursive:
            return [path for path in input_path.rglob("*") if path.is_file()]
        return [path for path in input_path.iterdir() if path.is_file()]
    return [input_path]


def _create_local_pipeline(  # noqa: PLR0913
    *,
    whisper_model: str,
    engine: str,
    diarization: bool,
    dialog_blocks: bool,
    language: Optional[str],
    device: str,
    compute_type: str,
    stt_provider: str = STT_PROVIDER_WHISPER,
) -> Any:
    runtime = _load_cli_runtime()
    local_pipeline_cls = runtime["LocalPipeline"]
    return local_pipeline_cls(
        whisper_model=whisper_model,
        engine=engine,
        stt_provider=stt_provider,
        enable_diarization=diarization,
        enable_dialog_blocks=dialog_blocks,
        language=language,
        device=device,
        compute_type=compute_type,
    )


def _write_plain_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class _ExternalChildAttemptResult:
    success_text: Optional[str]
    return_code: int
    stdout: str
    stderr: str


def _normalize_transcript_text(text: Optional[str]) -> str:
    value = text or ""
    return value if value.strip() else ""


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
    ) as temp_file:
        json.dump(payload, temp_file)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_path = Path(temp_file.name)
    temp_path.replace(path)


def _read_child_success_result(path: Path) -> Optional[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("status") != "ok":
        return None
    text = data.get("text")
    if not isinstance(text, str):
        return None
    return _normalize_transcript_text(text)


def _resolve_external_device(*, stt_provider: str, device: str) -> str:
    """Resolve ``--device`` for the selected STT provider.

    Args:
        stt_provider: Canonical provider id.
        device: Raw ``--device`` token from the CLI.

    Validation is torch-free: ``gigaam-ctc`` resolves to ``cpu``;
    ``gigaam-e2e-ctc`` keeps ``auto``/``cpu``/``cuda`` and the model resolves
    it against the live CUDA state at load time.

    Returns:
        Device token safe for the selected provider.

    Raises:
        typer.BadParameter: When the device token is invalid for a GigaAM head.

    """
    if not is_gigaam_provider(stt_provider):
        return device
    try:
        return resolve_gigaam_requested_device(stt_provider, device)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _run_external_transcribe_once(  # noqa: PLR0913
    *,
    input_path: Path,
    whisper_model: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    diarization: bool,
    dialog_blocks: bool,
    work_dir: Optional[Path],
    stt_provider: str = STT_PROVIDER_WHISPER,
    before_close: Optional[Callable[[str], None]] = None,
) -> str:
    pipeline = _create_local_pipeline(
        whisper_model=whisper_model,
        engine="whisperx",
        diarization=diarization,
        dialog_blocks=dialog_blocks,
        language=language,
        device=device,
        compute_type=compute_type,
        stt_provider=stt_provider,
    )
    try:
        if work_dir is None:
            with tempfile.TemporaryDirectory(prefix="askvlm-cli-") as temp_dir:
                text = pipeline.process(input_path, Path(temp_dir)).get_full_text()
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            text = pipeline.process(input_path, work_dir).get_full_text()
        normalized_text = _normalize_transcript_text(text)
        if before_close is not None:
            before_close(normalized_text)
        return normalized_text
    finally:
        pipeline.close(aggressive=False)


def _build_external_child_command(  # noqa: PLR0913
    *,
    input_path: Path,
    whisper_model: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    diarization: bool,
    dialog_blocks: bool,
    work_dir: Optional[Path],
    child_result_file: Path,
    stt_provider: str = STT_PROVIDER_WHISPER,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "external-transcribe",
        str(input_path),
        "--stt-provider",
        stt_provider,
        "--whisper-model",
        whisper_model,
        "--device",
        device,
        "--compute-type",
        compute_type,
        "--_internal-child-mode",
        f"--_internal-result-file={child_result_file}",
    ]
    if language is not None:
        command.extend(["--language", language])
    if diarization:
        command.append("--diarization")
    if dialog_blocks:
        command.append("--dialog-blocks")
    if work_dir is not None:
        command.extend(["--work-dir", str(work_dir)])
    return command


def _run_external_transcribe_isolated_attempt(  # noqa: PLR0913
    *,
    input_path: Path,
    whisper_model: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    diarization: bool,
    dialog_blocks: bool,
    work_dir: Optional[Path],
    stt_provider: str = STT_PROVIDER_WHISPER,
) -> _ExternalChildAttemptResult:
    with tempfile.TemporaryDirectory(prefix="askvlm-ext-ipc-") as ipc_dir:
        child_result_file = Path(ipc_dir) / "child_result.json"
        command = _build_external_child_command(
            input_path=input_path,
            whisper_model=whisper_model,
            language=language,
            device=device,
            compute_type=compute_type,
            diarization=diarization,
            dialog_blocks=dialog_blocks,
            work_dir=work_dir,
            child_result_file=child_result_file,
            stt_provider=stt_provider,
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "_ASKVLM_CHILD_RESULT_FILE": str(child_result_file)},
        )
        return _ExternalChildAttemptResult(
            success_text=_read_child_success_result(child_result_file),
            return_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


def _is_windows_crash_like_return_code(return_code: int) -> bool:
    if return_code < 0:
        return True
    return (return_code & 0xFFFFFFFF) >= 0xC0000000


_INTERNAL_CHILD_IPC_ERROR = "Internal child mode requires"


def _is_internal_child_ipc_error(stderr: str) -> bool:
    """Return True when child stderr signals a missing IPC result-file path.

    Args:
        stderr: The captured stderr string from the child process, or None.

    Returns:
        True if the IPC setup error marker is present in stderr.
    """
    return _INTERNAL_CHILD_IPC_ERROR in (stderr or "")


def _raise_external_transcribe_error(attempt: _ExternalChildAttemptResult) -> None:
    details = (attempt.stderr or "").strip() or (attempt.stdout or "").strip()
    if details:
        typer.secho(details, err=True)
    raise typer.Exit(code=1)


def _emit_external_transcribe_outputs(
    *,
    text: str,
    output_file: Optional[Path],
    stdout: bool,
    output_dir_file: Path | None = None,
) -> None:
    if output_file is not None:
        _write_plain_text(output_file, text)
    if output_dir_file is not None:
        _write_plain_text(output_dir_file, text)
    if stdout and text:
        typer.echo(text)


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace *path* with *text* in one step (temporary file, then replace).

    Args:
        path: Destination file; missing parent directories are created.
        text: Content written as UTF-8 with platform line endings, like
            :func:`_write_plain_text`.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(text)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        temp_path.replace(path)
    except BaseException:
        if temp_path is not None:
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)
        raise


# * Distinct exit codes let a caller (for example the Telegram bot) react to a
# * busy/unavailable transcription service without parsing free-form stderr.
EXTERNAL_TRANSCRIBE_TIMEOUT_EXIT = 10
EXTERNAL_TRANSCRIBE_UNAVAILABLE_EXIT = 11
_CLIENT_TIMEOUT_MARKER = "ASKVLM_CLIENT_TIMEOUT"
_DAEMON_UNAVAILABLE_MARKER = "ASKVLM_DAEMON_UNAVAILABLE"
_ERROR_MARKER = "ERROR"
_DEPRECATED_MARKER = "ASKVLM_DEPRECATED"
_DAEMON_WORKERS_DEPRECATION = (
    f"{_DEPRECATED_MARKER}: --daemon-workers is deprecated; use --max-parallel"
)

# * Client statuses of a successful input; every other status is a failure.
_SUCCESS_STATUSES = frozenset({"ok", "empty"})

# * Exit code of a multi-input call: the first failure status present wins.
_FAILURE_EXIT_PRIORITY: tuple[tuple[str, int], ...] = (
    ("timeout", EXTERNAL_TRANSCRIBE_TIMEOUT_EXIT),
    ("unavailable", EXTERNAL_TRANSCRIBE_UNAVAILABLE_EXIT),
    ("error", 1),
)

# * stderr marker of each failure status in multi-input calls.
_FAILURE_MARKERS: dict[str, str] = {
    "timeout": _CLIENT_TIMEOUT_MARKER,
    "unavailable": _DAEMON_UNAVAILABLE_MARKER,
    "error": _ERROR_MARKER,
}


def _priority_callback(value: str) -> str:
    """Validate ``--priority`` case-insensitively.

    Args:
        value: Raw option value.

    Returns:
        The canonical priority class.

    Raises:
        typer.BadParameter: When the value names no priority class.

    """
    try:
        return normalize_priority(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _client_timeout_callback(value: float) -> float:
    """Validate ``--client-timeout``: a number of at least 1 or ``inf``.

    The range check of the option already rejects values below 1; ``nan``
    slips through every comparison and is refused here.

    Args:
        value: Parsed option value.

    Returns:
        The unchanged value; ``inf`` means the call has no deadline.

    Raises:
        typer.BadParameter: When the value is ``nan``.

    """
    if math.isnan(value):
        msg = "--client-timeout must be a number of seconds (or inf), not nan."
        raise typer.BadParameter(msg)
    return value


def _reject_supervisor_unsupported_flags(
    *, diarization: bool, dialog_blocks: bool, work_dir: Path | None
) -> None:
    """Refuse flags the transcription supervisor cannot honor.

    Args:
        diarization: ``--diarization`` was given.
        dialog_blocks: ``--dialog-blocks`` was given.
        work_dir: Value of ``--work-dir``.

    Raises:
        typer.BadParameter: Naming the first unsupported flag and ``--no-daemon``.

    """
    for flag, used in (
        ("--diarization", diarization),
        ("--dialog-blocks", dialog_blocks),
        ("--work-dir", work_dir is not None),
    ):
        if used:
            msg = (
                f"{flag} is not supported by the transcription supervisor; "
                "rerun with --no-daemon to use it."
            )
            raise typer.BadParameter(msg, param_hint=f"'{flag}'")


def _supervisor_device(provider: str, device: str) -> DeviceRequest:
    """Map ``--device`` onto the supervisor's device vocabulary.

    Args:
        provider: Canonical provider id.
        device: Device token already checked by :func:`_resolve_external_device`.

    Returns:
        ``auto``, ``cpu`` or ``cuda``. Whisper maps every other token to
        ``auto``.

    Raises:
        typer.BadParameter: When a GigaAM head rejects the token.

    """
    if not is_gigaam_provider(provider):
        # * Whisper keeps the legacy coercion of the in-process pipeline: a
        # * token outside auto|cpu|cuda (for example ``cuda:0``) means ``auto``.
        try:
            return normalize_device_request(provider, device)
        except ValueError:
            return "auto"
    try:
        return normalize_device_request(provider, device)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="'--device'") from exc


def _supervisor_language(language: str | None) -> str | None:
    """Validate ``--language`` for a supervisor job.

    Args:
        language: Raw ``--language`` value.

    Returns:
        The lowercased language code, or ``None`` when absent or blank.

    Raises:
        typer.BadParameter: When the value is not a language code.

    """
    try:
        return normalize_language(language)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="'--language'") from exc


def _build_supervisor_request(  # noqa: PLR0913 - one argument per CLI option
    *,
    input_paths: Sequence[Path],
    provider: str,
    whisper_model: str,
    compute_type: str,
    device: str,
    language: str | None,
    priority: str,
    max_parallel: int | None,
    client_timeout: float,
    diarization: bool,
    dialog_blocks: bool,
    work_dir: Path | None,
) -> ClientRequest:
    """Validate the supervisor-mode options and build the client request.

    Args:
        input_paths: Inputs in input order.
        provider: Canonical provider id.
        whisper_model: ``--whisper-model`` value.
        compute_type: ``--compute-type`` value.
        device: Device token already checked by :func:`_resolve_external_device`.
        language: ``--language`` value.
        priority: Canonical ``--priority`` value.
        max_parallel: Effective per-call cap (``--max-parallel``, else the
            deprecated ``--daemon-workers``), or ``None``.
        client_timeout: ``--client-timeout`` value.
        diarization: ``--diarization`` was given.
        dialog_blocks: ``--dialog-blocks`` was given.
        work_dir: ``--work-dir`` value.

    Returns:
        The request for :func:`~core.stt_pool.client.run_client`.

    Raises:
        typer.BadParameter: For a flag the supervisor cannot honor, a device
            token a GigaAM head rejects, or an invalid language token.

    """
    _reject_supervisor_unsupported_flags(
        diarization=diarization, dialog_blocks=dialog_blocks, work_dir=work_dir
    )
    return ClientRequest(
        inputs=tuple(input_paths),
        provider=provider,
        whisper_model=whisper_model,
        compute_type=compute_type,
        device=_supervisor_device(provider, device),
        language=_supervisor_language(language),
        priority=normalize_priority(priority),
        max_parallel=max_parallel,
        client_timeout_s=client_timeout,
    )


def _note_deprecated_daemon_workers(daemon_workers: int | None) -> None:
    """Print the ``--daemon-workers`` deprecation note when the flag was given.

    Args:
        daemon_workers: ``--daemon-workers`` value, or ``None``.

    """
    if daemon_workers is not None:
        typer.secho(_DAEMON_WORKERS_DEPRECATION, err=True)


def _resolve_output_dir_files(inputs: Sequence[Path], output_dir: Path) -> list[Path]:
    """Name the ``--output-dir`` file of every input before the run.

    The first input with a given stem gets ``<stem>.txt``; a later input whose
    name would collide (compared case-insensitively) gets ``<stem>-2.txt``,
    ``<stem>-3.txt``, and so on, skipping names already taken.

    Args:
        inputs: Inputs in input order.
        output_dir: Target directory.

    Returns:
        One distinct file path per input, in input order.

    """
    taken: set[str] = set()
    files: list[Path] = []
    for input_path in inputs:
        stem = input_path.stem
        name = f"{stem}.txt"
        suffix = 1
        while name.casefold() in taken:
            suffix += 1
            name = f"{stem}-{suffix}.txt"
        taken.add(name.casefold())
        files.append(output_dir / name)
    return files


def _single_line(text: str) -> str:
    """Join the non-blank lines of *text* into one stderr line.

    Args:
        text: Possibly multi-line diagnostic text.

    Returns:
        The stripped lines joined by `` | ``.

    """
    return " | ".join(line.strip() for line in text.splitlines() if line.strip())


def _report_single_supervisor_outcome(
    outcome: InputOutcome,
    *,
    output_file: Path | None,
    output_dir_file: Path | None,
    stdout: bool,
) -> None:
    """Apply the single-input output rules to a supervisor outcome.

    Args:
        outcome: The outcome of the only input.
        output_file: ``--output-file`` target, if any.
        output_dir_file: ``--output-dir`` file of the input, if any.
        stdout: Whether the transcript goes to stdout.

    Raises:
        typer.Exit: :data:`EXTERNAL_TRANSCRIBE_TIMEOUT_EXIT` on a timeout,
            :data:`EXTERNAL_TRANSCRIBE_UNAVAILABLE_EXIT` when the supervisor is
            unavailable, ``1`` on an error (stderr carries the worker's detail).

    """
    for warning in outcome.warnings:
        typer.secho(warning, err=True)
    if outcome.status in _SUCCESS_STATUSES:
        _emit_external_transcribe_outputs(
            text=outcome.text,
            output_file=output_file,
            stdout=stdout,
            output_dir_file=output_dir_file,
        )
        return
    if outcome.status == "timeout":
        typer.secho(
            f"{_CLIENT_TIMEOUT_MARKER}: {outcome.detail or ''}".strip(), err=True
        )
        raise typer.Exit(code=EXTERNAL_TRANSCRIBE_TIMEOUT_EXIT)
    if outcome.status == "unavailable":
        typer.secho(
            f"{_DAEMON_UNAVAILABLE_MARKER}: {outcome.detail or ''}".strip(), err=True
        )
        raise typer.Exit(code=EXTERNAL_TRANSCRIBE_UNAVAILABLE_EXIT)
    typer.secho(outcome.detail or "transcription failed", err=True)
    raise typer.Exit(code=1)


class _BatchReporter:
    """Report the outcomes of a multi-input call in input order.

    Outcomes arrive in completion order. Each ``--output-dir`` file is written
    as soon as its outcome arrives; the stdout JSON line and the stderr lines
    of an input are emitted once every earlier input is final.
    """

    def __init__(
        self,
        inputs: Sequence[Path],
        *,
        output_dir_files: Sequence[Path] | None,
        stdout: bool,
    ) -> None:
        self._count = len(inputs)
        self._output_dir_files = (
            None if output_dir_files is None else tuple(output_dir_files)
        )
        self._stdout = stdout
        self._received: dict[int, InputOutcome] = {}
        self._written: dict[int, Path] = {}
        self._next_index = 0
        self._warned: set[str] = set()

    def on_outcome(self, outcome: InputOutcome) -> None:
        """Accept one final outcome; a repeated index is ignored."""
        if outcome.index in self._received:
            return
        self._received[outcome.index] = self._write_output_dir_file(outcome)
        while self._next_index in self._received:
            self._emit(self._received[self._next_index])
            self._next_index += 1

    def outcomes(self) -> list[InputOutcome]:
        """Return the reported outcomes in input order."""
        return [self._received[index] for index in range(self._count)]

    def _write_output_dir_file(self, outcome: InputOutcome) -> InputOutcome:
        """Write ``<stem>.txt`` of a successful input; a failed write is an error."""
        files = self._output_dir_files
        if files is None or outcome.status not in _SUCCESS_STATUSES:
            return outcome
        target = files[outcome.index]
        try:
            _write_plain_text(target, outcome.text)
        except OSError as exc:
            return replace(
                outcome,
                status="error",
                text="",
                detail=f"cannot write {target}: {exc}",
            )
        self._written[outcome.index] = target
        return outcome

    def _emit(self, outcome: InputOutcome) -> None:
        """Write the stderr lines and the stdout JSON line of one input."""
        for warning in outcome.warnings:
            if warning not in self._warned:
                self._warned.add(warning)
                typer.secho(warning, err=True)
        failed = outcome.status not in _SUCCESS_STATUSES
        error = (outcome.detail or outcome.status) if failed else None
        if error is not None:
            typer.secho(
                f"{_FAILURE_MARKERS.get(outcome.status, _ERROR_MARKER)}: "
                f"input {outcome.index} ({outcome.input_path}): {_single_line(error)}",
                err=True,
            )
        if not self._stdout:
            return
        record: dict[str, object] = {
            "index": outcome.index,
            "input": str(outcome.input_path),
            "status": outcome.status,
            "text": "" if failed else outcome.text,
            "error": error,
        }
        if self._output_dir_files is not None:
            written = self._written.get(outcome.index)
            record["output"] = None if written is None else str(written)
        typer.echo(json.dumps(record))


def _batch_exit_code(outcomes: Sequence[InputOutcome]) -> int:
    """Return the exit code of a multi-input call.

    Args:
        outcomes: Final outcomes of every input.

    Returns:
        ``10`` when any input timed out, else ``11`` when any was unavailable,
        else ``1`` when any failed, else ``0``.

    """
    statuses = {outcome.status for outcome in outcomes}
    for status, code in _FAILURE_EXIT_PRIORITY:
        if status in statuses:
            return code
    return 0


def _run_external_transcribe_via_supervisor(
    request: ClientRequest,
    *,
    output_file: Path | None,
    output_dir: Path | None,
    stdout: bool,
) -> None:
    """Transcribe through the STT supervisor and apply the CLI output rules.

    One input keeps the plain single-file contract (transcript on stdout,
    exit codes 0/1/10/11). Several inputs produce one JSON line per input on
    stdout in input order, optional ``--output-dir`` files, one atomic
    ``--output-file`` with every transcript joined by newlines, and the exit
    code of :func:`_batch_exit_code`. A failed ``--output-file`` write prints one
    stderr line and turns an otherwise successful call into exit ``1``; the
    previous file content stays. There is no in-process recovery.

    Args:
        request: The client request.
        output_file: ``--output-file`` target, if any.
        output_dir: ``--output-dir`` target, if any.
        stdout: Whether transcripts (or JSON lines) go to stdout.

    Raises:
        typer.Exit: With the exit code of a failed call.

    """
    output_dir_files = (
        None
        if output_dir is None
        else _resolve_output_dir_files(request.inputs, output_dir)
    )
    if len(request.inputs) == 1:
        outcome = run_client(request).outcomes[0]
        _report_single_supervisor_outcome(
            outcome,
            output_file=output_file,
            output_dir_file=None if output_dir_files is None else output_dir_files[0],
            stdout=stdout,
        )
        return
    reporter = _BatchReporter(
        request.inputs, output_dir_files=output_dir_files, stdout=stdout
    )
    result = run_client(request, on_outcome=reporter.on_outcome)
    for outcome in result.outcomes:
        reporter.on_outcome(outcome)
    outcomes = reporter.outcomes()
    output_file_failed = False
    if output_file is not None:
        try:
            _write_text_atomic(
                output_file,
                "\n".join(
                    outcome.text if outcome.status in _SUCCESS_STATUSES else ""
                    for outcome in outcomes
                ),
            )
        except OSError as exc:
            # * A failed aggregate write counts as an error of the call but never
            # * outranks a timeout (10) or an unavailable supervisor (11).
            output_file_failed = True
            typer.secho(
                f"{_ERROR_MARKER}: cannot write --output-file {output_file}: "
                f"{_single_line(str(exc)) or type(exc).__name__}",
                err=True,
            )
    code = _batch_exit_code(outcomes) or (1 if output_file_failed else 0)
    if code:
        raise typer.Exit(code=code)


@app.command()
def transcribe(
    input_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Input media file or directory"
    ),
    output_dir: Path = typer.Option(
        Path("transcriptions"), "--output", "-o", help="Output directory"
    ),
    whisper_model: str = typer.Option(
        "large-v3",
        help="Whisper model name for batch transcription (default: large-v3)",
    ),
    engine: str = typer.Option(
        "whisper",
        help=(
            "Backend compatibility hint: whisper | whisperx | auto. The current "
            "local batch pipeline uses the Whisper/Faster-Whisper path."
        ),
    ),
    language: Optional[str] = typer.Option(None, help="Language code (optional)"),
    diarization: bool = typer.Option(True, help="Enable speaker diarization"),
    dialog_blocks: bool = typer.Option(False, help="Format text with LLM"),
    export: str = typer.Option("txt", help="Export format: txt|srt|vtt|json"),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Process directories recursively"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing output files"
    ),
    device: str = typer.Option(
        "auto", help="Device: auto|cuda|cpu (passed to engines where applicable)"
    ),
    compute_type: str = typer.Option(
        "float16",
        help=(
            "Compute type for faster-whisper: float16|int8|int8_float16|auto. "
            "Default float16 (extreme profile) for best quality on 8+ GiB VRAM."
        ),
    ),
) -> None:
    """Transcribe a file or directory and export results."""
    runtime = _load_cli_runtime()
    export_document = runtime["export_document"]
    output_dir.mkdir(parents=True, exist_ok=True)
    # * Normalize engine when auto is requested
    chosen_engine = "whisperx" if engine == "auto" else engine
    pipeline = _create_local_pipeline(
        whisper_model=whisper_model,
        engine=chosen_engine,
        diarization=diarization,
        dialog_blocks=dialog_blocks,
        language=language,
        device=device,
        compute_type=compute_type,
    )

    try:
        for media in _collect_files(input_path, recursive=recursive):
            out_file = output_dir / f"{media.stem}.{export.lower()}"
            if out_file.exists() and not overwrite:
                typer.echo(f"Exists, skip: {out_file}")
                continue
            typer.echo(f"Processing {media}...")
            doc = pipeline.process(media, output_dir)
            export_document(doc, export, out_file)
            typer.echo(f"Saved to {out_file}")
    finally:
        with contextlib.suppress(Exception):
            pipeline.close(aggressive=True)


@app.command()
def subtitle(
    input_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Input media file or directory"
    ),
    output_dir: Path = typer.Option(
        Path("transcriptions"), "--output", "-o", help="Output directory"
    ),
    whisper_model: str = typer.Option(
        "large-v3",
        help="Whisper model name for subtitle generation (default: large-v3)",
    ),
    language: Optional[str] = typer.Option(None, help="Language code (optional)"),
    device: str = typer.Option("auto", help="Device: auto|cuda|cpu"),
    compute_type: str = typer.Option(
        "float16",
        help="Compute type for faster-whisper (default float16; consider int8_float16 on 8–12 GiB if OOM)",
    ),
    diarization: bool = typer.Option(False, help="Enable speaker diarization"),
    burn_in: bool = typer.Option(True, help="Burn subtitles into the video"),
    save_srt: bool = typer.Option(True, help="Always save .srt sidecar"),
    format: str = typer.Option(
        "srt", help="Subtitle format: srt|vtt|ass (srt only for burn)"
    ),
    max_cps: float = typer.Option(18.0, help="Max characters per second"),
    max_line_chars: int = typer.Option(42, help="Max characters per line"),
    max_lines: int = typer.Option(2, help="Max lines per cue"),
    min_duration: float = typer.Option(1.2, help="Minimum cue duration (s)"),
    max_duration: float = typer.Option(6.0, help="Maximum cue duration (s)"),
) -> None:
    """Generate subtitles with readability rules and optionally burn them into the video."""
    runtime = _load_cli_runtime()
    export_srt_with_rules = runtime["export_srt_with_rules"]
    subtitle_rules_cls = runtime["SubtitleRules"]
    burn_subtitles = runtime["burn_subtitles"]
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _create_local_pipeline(
        whisper_model=whisper_model,
        engine="whisperx",
        diarization=diarization,
        dialog_blocks=False,
        language=language,
        device=device,
        compute_type=compute_type,
    )
    files = _collect_files(input_path, recursive=False)

    rules = subtitle_rules_cls(
        max_line_chars=max_line_chars,
        max_lines=max_lines,
        min_duration=min_duration,
        max_duration=max_duration,
        max_cps=max_cps,
    )

    try:
        for media in files:
            typer.echo(f"Processing {media}...")
            doc = pipeline.process(media, output_dir)
            srt_path = output_dir / f"{media.stem}.srt"
            # Export srt with rules
            srt_text = export_srt_with_rules(doc, rules)
            srt_path.write_text(srt_text, encoding="utf-8")
            if save_srt:
                typer.echo(f"Saved SRT: {srt_path}")
            if burn_in and media.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
                out_video = output_dir / f"{media.stem}_subbed.mp4"
                burn_subtitles(media, srt_path, out_video)
                typer.echo(f"Burned-in video: {out_video}")
    finally:
        with contextlib.suppress(Exception):
            pipeline.close(aggressive=True)


@app.command("external-transcribe")
def external_transcribe(  # noqa: PLR0913
    input_paths: list[Path] = typer.Argument(
        ...,
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        help=(
            "One or more audio or video files to transcribe for an external "
            "caller (--no-daemon accepts exactly one)."
        ),
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help=(
            "Optional plain-text file to write alongside stdout output. With "
            "several inputs it receives every transcript joined by newlines in "
            "input order (failed inputs contribute an empty line)."
        ),
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help=(
            "Optional directory that receives <stem>.txt for every successful "
            "input; repeated stems get -2, -3, ... suffixes."
        ),
    ),
    whisper_model: str = typer.Option(
        "small",
        help="Whisper model name for the external one-shot flow (default: small).",
    ),
    stt_provider: str = typer.Option(
        STT_PROVIDER_WHISPER,
        "--stt-provider",
        help=(
            "Speech-to-text provider: whisper (default), gigaam-ctc (CPU-only), "
            "or gigaam-e2e-ctc (CPU or CUDA); GigaAM is included in .[ml]."
        ),
    ),
    language: Optional[str] = typer.Option(
        None, help="Optional language code, for example: en, ru, de."
    ),
    device: str = typer.Option(
        "auto",
        help=(
            "Preferred device: auto|cuda|cpu. Through the supervisor, auto "
            "prefers the GPU and runs on CPU when that is expected to finish "
            "sooner, cpu never uses the GPU, and cuda runs on the GPU (on CPU "
            "only when no CUDA device exists, with a warning). gigaam-ctc "
            "accepts only cpu (auto resolves to cpu). With --no-daemon, Whisper "
            "and gigaam-e2e-ctc retry once on CPU after a GPU crash."
        ),
    ),
    compute_type: str = typer.Option(
        "auto",
        help=(
            "Compute type: auto|float16|int8|int8_float16. 'auto' uses float16 "
            "on CUDA and int8 on CPU. Ignored for GigaAM providers."
        ),
    ),
    diarization: bool = typer.Option(
        False,
        "--diarization/--no-diarization",
        help=(
            "Enable speaker diarization (requires --no-daemon). Disabled by "
            "default for external calls; this can require additional GPU memory."
        ),
    ),
    dialog_blocks: bool = typer.Option(
        False,
        "--dialog-blocks/--no-dialog-blocks",
        help=(
            "Enable LLM-based text formatting (requires --no-daemon). Disabled "
            "by default."
        ),
    ),
    stdout: bool = typer.Option(
        True,
        "--stdout/--no-stdout",
        help=(
            "Write only the final transcript text to stdout; with several inputs, "
            "one JSON line per input in input order. Enabled by default."
        ),
    ),
    work_dir: Optional[Path] = typer.Option(
        None,
        "--work-dir",
        help=(
            "Optional directory for intermediate files (requires --no-daemon). "
            "When omitted, AskVLM uses a temporary directory and deletes it "
            "after completion."
        ),
    ),
    no_daemon: bool = typer.Option(
        False,
        "--no-daemon/--daemon",
        help=(
            "Run the legacy in-process one-shot flow (one input) instead of "
            "routing the request through the shared transcription supervisor."
        ),
    ),
    client_timeout: float = typer.Option(
        300.0,
        "--client-timeout",
        min=1.0,
        callback=_client_timeout_callback,
        help=(
            "Supervisor mode: seconds for the whole call, queue wait included; "
            "unfinished inputs are then cancelled and reported as timeouts. "
            "inf waits without a deadline."
        ),
    ),
    priority: str = typer.Option(
        "normal",
        "--priority",
        callback=_priority_callback,
        help=(
            "Supervisor mode: priority class of this call: high, normal or low. "
            "Higher classes are admitted first; running jobs are never "
            "preempted. Ignored with --no-daemon."
        ),
    ),
    max_parallel: int | None = typer.Option(
        None,
        "--max-parallel",
        min=1,
        max=MAX_PARALLEL_LIMIT,
        help=(
            "Supervisor mode: run at most N inputs of this call at once. By "
            "default the supervisor decides. Ignored with --no-daemon."
        ),
    ),
    daemon_workers: int | None = typer.Option(
        None,
        "--daemon-workers",
        min=1,
        max=MAX_PARALLEL_LIMIT,
        help="Deprecated alias of --max-parallel (used when that is not given).",
    ),
    _internal_child_mode: bool = typer.Option(
        False,
        "--_internal-child-mode",
        hidden=True,
    ),
    _internal_result_file: Optional[Path] = typer.Option(
        None,
        "--_internal-result-file",
        hidden=True,
    ),
) -> None:
    """Transcribe media files and return plain text for external applications.

    By default the inputs go through the shared STT supervisor (a pool of
    worker processes, one per configuration). One input prints its transcript;
    several inputs print one JSON line per input. ``--no-daemon`` runs the
    legacy in-process flow for one input.
    """
    if not stdout and output_file is None and output_dir is None:
        msg = "Either keep --stdout enabled or provide --output-file or --output-dir."
        raise typer.BadParameter(msg)
    try:
        provider = normalize_stt_provider(stt_provider)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if provider not in STT_PROVIDER_CHOICES:
        raise typer.BadParameter(
            f"Unsupported --stt-provider {stt_provider!r}; "
            f"expected one of: {', '.join(STT_PROVIDER_CHOICES)}"
        )
    device = _resolve_external_device(stt_provider=provider, device=device)
    # * Env-var fallback: recover IPC path if CLI arg was not parsed
    if _internal_child_mode and _internal_result_file is None:
        _env_path = os.environ.get("_ASKVLM_CHILD_RESULT_FILE")
        if _env_path:
            _internal_result_file = Path(_env_path)

    input_path = input_paths[0]
    if _internal_child_mode:
        if _internal_result_file is None:
            raise typer.BadParameter(
                "Internal child mode requires --_internal-result-file."
            )
        if len(input_paths) != 1:
            msg = "Internal child mode accepts exactly one input."
            raise typer.BadParameter(msg)
        try:
            _run_external_transcribe_once(
                input_path=input_path,
                whisper_model=whisper_model,
                language=language,
                device=device,
                compute_type=compute_type,
                diarization=diarization,
                dialog_blocks=dialog_blocks,
                work_dir=work_dir,
                stt_provider=provider,
                before_close=lambda text: _write_json_atomic(
                    _internal_result_file,
                    {"status": "ok", "text": text},
                ),
            )
            return
        except Exception as exc:
            if not _internal_result_file.exists():
                with contextlib.suppress(Exception):
                    _write_json_atomic(
                        _internal_result_file,
                        {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
            raise

    if not no_daemon:
        request = _build_supervisor_request(
            input_paths=input_paths,
            provider=provider,
            whisper_model=whisper_model,
            compute_type=compute_type,
            device=device,
            language=language,
            priority=priority,
            max_parallel=max_parallel if max_parallel is not None else daemon_workers,
            client_timeout=client_timeout,
            diarization=diarization,
            dialog_blocks=dialog_blocks,
            work_dir=work_dir,
        )
        _note_deprecated_daemon_workers(daemon_workers)
        _run_external_transcribe_via_supervisor(
            request, output_file=output_file, output_dir=output_dir, stdout=stdout
        )
        return

    if len(input_paths) != 1:
        msg = (
            "--no-daemon accepts exactly one input; drop --no-daemon to "
            "transcribe several inputs through the supervisor."
        )
        raise typer.BadParameter(msg)
    _note_deprecated_daemon_workers(daemon_workers)
    # * --priority and --max-parallel only shape supervisor scheduling.
    output_dir_file = (
        None
        if output_dir is None
        else _resolve_output_dir_files(input_paths, output_dir)[0]
    )

    # * Windows GPU child isolation covers CUDA-capable providers (Whisper and
    # * gigaam-e2e-ctc); gigaam-ctc always runs in-process on CPU.
    windows_non_cpu = (
        provider_may_use_cuda(provider)
        and sys.platform.startswith("win")
        and device.lower() != "cpu"
    )
    if windows_non_cpu:
        first_attempt = _run_external_transcribe_isolated_attempt(
            input_path=input_path,
            whisper_model=whisper_model,
            language=language,
            device=device,
            compute_type=compute_type,
            diarization=diarization,
            dialog_blocks=dialog_blocks,
            work_dir=work_dir,
            stt_provider=provider,
        )
        if first_attempt.success_text is not None:
            _emit_external_transcribe_outputs(
                text=first_attempt.success_text,
                output_file=output_file,
                stdout=stdout,
                output_dir_file=output_dir_file,
            )
            return

        if not _is_windows_crash_like_return_code(
            first_attempt.return_code
        ) and not _is_internal_child_ipc_error(first_attempt.stderr):
            _raise_external_transcribe_error(first_attempt)

        retry_attempt = _run_external_transcribe_isolated_attempt(
            input_path=input_path,
            whisper_model=whisper_model,
            language=language,
            device="cpu",
            compute_type=compute_type,
            diarization=diarization,
            dialog_blocks=dialog_blocks,
            work_dir=work_dir,
            stt_provider=provider,
        )
        if retry_attempt.success_text is not None:
            _emit_external_transcribe_outputs(
                text=retry_attempt.success_text,
                output_file=output_file,
                stdout=stdout,
                output_dir_file=output_dir_file,
            )
            return
        _raise_external_transcribe_error(retry_attempt)

    _run_external_transcribe_once(
        input_path=input_path,
        whisper_model=whisper_model,
        language=language,
        device=device,
        compute_type=compute_type,
        diarization=diarization,
        dialog_blocks=dialog_blocks,
        work_dir=work_dir,
        stt_provider=provider,
        before_close=lambda text: _emit_external_transcribe_outputs(
            text=text,
            output_file=output_file,
            stdout=stdout,
            output_dir_file=output_dir_file,
        ),
    )


def _install_stop_handlers(stop_event: threading.Event) -> None:
    """Wire available stop signals to request a graceful daemon shutdown.

    Args:
        stop_event: Flag set when a stop signal is received.
    """

    def _handle(_signum: int, _frame: FrameType | None) -> None:
        stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handle)


@app.command("external-transcribe-daemon")
def external_transcribe_daemon(  # noqa: PLR0913
    queue_dir: Path | None = typer.Option(
        None,
        "--queue-dir",
        file_okay=False,
        help=(
            "Queue root to serve (default: ASKVLM_EXTERNAL_QUEUE_DIR, then the "
            "project cache)."
        ),
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        dir_okay=False,
        help=(
            "STT pool config JSON (default: ASKVLM_STT_POOL_CONFIG, then "
            "<project>/stt_pool.json, then built-in conservative limits)."
        ),
    ),
    idle_shutdown: float | None = typer.Option(
        None,
        "--idle-shutdown",
        help="Seconds an idle worker stays loaded (default: from the config).",
    ),
    cpu_slots: int | None = typer.Option(
        None, "--cpu-slots", help="Concurrent CPU jobs (overrides cpu.max_slots)."
    ),
    gpu_workers: int | None = typer.Option(
        None, "--gpu-workers", help="GPU worker processes (overrides gpu.max_workers)."
    ),
    thread_budget: int | None = typer.Option(
        None,
        "--thread-budget",
        help="CPU threads of the pool (overrides threads.budget).",
    ),
    commit_budget_gb: float | None = typer.Option(
        None,
        "--commit-budget-gb",
        help="Commit memory budget in GiB (overrides memory.commit_budget_gib).",
    ),
    vram_budget_gb: float | None = typer.Option(
        None,
        "--vram-budget-gb",
        help="VRAM budget in GiB (overrides vram.budget_gib).",
    ),
    workers: int | None = typer.Option(None, "--workers", hidden=True),
    stt_provider: str | None = typer.Option(None, "--stt-provider", hidden=True),
    whisper_model: str | None = typer.Option(None, "--whisper-model", hidden=True),
    device: str | None = typer.Option(None, "--device", hidden=True),
    compute_type: str | None = typer.Option(None, "--compute-type", hidden=True),
    language: str | None = typer.Option(None, "--language", hidden=True),
) -> None:
    """Run the STT supervisor that serves the shared transcription queue.

    One supervisor serves a queue root; a second invocation exits at once with
    code 0. The supervisor runs one worker process per configuration (provider,
    model, compute type, CPU or GPU) within the limits of the pool config and
    the overrides above, and exits after idling. ``external-transcribe`` starts
    it on demand. The legacy flags ``--workers``, ``--stt-provider``,
    ``--whisper-model``, ``--device``, ``--compute-type`` and ``--language``
    are accepted and ignored: every job carries its own configuration.
    """
    for flag, value in (
        ("--workers", workers),
        ("--stt-provider", stt_provider),
        ("--whisper-model", whisper_model),
        ("--device", device),
        ("--compute-type", compute_type),
        ("--language", language),
    ):
        if value is not None:
            typer.secho(
                f"{_DEPRECATED_MARKER}: {flag} is ignored by the pool supervisor",
                err=True,
            )
    try:
        config = apply_overrides(
            load_pool_config(config_file),
            cpu_slots=cpu_slots,
            gpu_workers=gpu_workers,
            thread_budget=thread_budget,
            commit_budget_gib=commit_budget_gb,
            vram_budget_gib=vram_budget_gb,
            worker_idle_ttl_s=idle_shutdown,
        )
    except PoolConfigError as exc:
        typer.secho(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    stop_event = threading.Event()
    _install_stop_handlers(stop_event)
    # * Imported on use: only the daemon process needs the supervisor.
    from core.stt_pool.supervisor import run_supervisor  # noqa: PLC0415

    raise typer.Exit(code=run_supervisor(queue_dir, config, stop_event=stop_event))


@app.command("external-extract-frames")
def external_extract_frames(  # noqa: PLR0913
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        help="Video file to extract frames from.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory where extracted frame images are written.",
    ),
    fps: float = typer.Option(
        0.5,
        "--fps",
        help="Target sampling rate in frames per second (default: 0.5).",
        min=0.001,
    ),
    fps_fallback: float = typer.Option(
        0.2,
        "--fps-fallback",
        help="Fallback FPS used when frame-budget would be exceeded (default: 0.2).",
        min=0.001,
    ),
    frame_budget: int = typer.Option(
        20,
        "--frame-budget",
        help=(
            "Maximum number of frames to extract. "
            "When the target FPS would produce more frames, fps-fallback is used instead. "
            "0 disables the cap."
        ),
        min=0,
    ),
    as_json: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Output a JSON object instead of one path per line.",
    ),
) -> None:
    """Extract video frames at adaptive FPS for external vision pipelines.

    Writes frame images to OUTPUT_DIR. Prints extracted frame paths to stdout
    (one per line), or a JSON manifest when --json is used.

    Exit code 0 on success (even if the video has zero frames). Exit code 1 on
    any processing failure.
    """
    import json as _json

    from core.ffmpeg import extract_frames_for_span, get_media_duration_seconds

    output_dir.mkdir(parents=True, exist_ok=True)

    duration_s = get_media_duration_seconds(input_path)
    if duration_s <= 0.0:
        typer.secho(
            f"Warning: could not determine duration of {input_path}; defaulting to 0 frames.",
            err=True,
        )
        if as_json:
            typer.echo(_json.dumps({"frames": [], "fps_used": fps, "duration_s": 0.0}))
        return

    # * Select FPS: use target unless frame budget would be exceeded
    effective_fps = fps
    if frame_budget > 0:
        estimated = math.ceil(duration_s * fps)
        if estimated > frame_budget:
            effective_fps = fps_fallback
        # * Hard cap: when even the fallback FPS overshoots the budget, derive the
        # * rate from the budget itself so sampling stays uniform over the full
        # * duration instead of over-producing frames for the caller to truncate.
        if math.ceil(duration_s * effective_fps) > frame_budget:
            effective_fps = frame_budget / duration_s

    frame_paths = extract_frames_for_span(
        video_file=input_path,
        start_s=0.0,
        end_s=duration_s,
        output_pattern=output_dir / "frame-%06d.png",
        fps=effective_fps,
    )
    # * ffmpeg fps rounding may still emit one extra frame; enforce the cap exactly.
    if frame_budget > 0 and len(frame_paths) > frame_budget:
        frame_paths = frame_paths[:frame_budget]

    if as_json:
        typer.echo(
            _json.dumps(
                {
                    "frames": [str(p) for p in frame_paths],
                    "fps_used": effective_fps,
                    "duration_s": duration_s,
                }
            )
        )
    else:
        for p in frame_paths:
            typer.echo(str(p))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
