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
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, NoReturn, Optional

import typer

from core.stt_options import (
    DIARIZE_DEFAULT_MIN_DURATION_OFF,
    DIARIZE_DEFAULT_MIN_DURATION_ON,
    DIARIZE_DEFAULT_THRESHOLD,
    MAX_BEAM_SIZE,
    MIN_BEAM_SIZE,
    DiarizeOptions,
    MediaWindow,
    WhisperDecodeOptions,
)
from core.stt_output import (
    MIN_VOICES_FOR_SPEAKER_PREFIXES,
    assign_speakers,
    diarization_document,
    document_json,
    segments_json,
    speaker_text,
    transcript_document,
    voice_count,
)
from core.stt_pool.client import MODE_DIARIZE, ClientRequest, run_client
from core.stt_pool.config import PoolConfigError, apply_overrides, load_pool_config
from core.stt_pool.keys import (
    DIARIZE_PROVIDER,
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
from editing.text_model import TextSegment, merge_consecutive_segments_by_speaker

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from core.stt_pool.client import ClientOutcome, InputOutcome, OutcomeFn
    from core.stt_pool.keys import DeviceRequest
    from editing.text_model import Document

# * Output formats of ``external-transcribe``: the plain transcript (the default)
# * or one JSON document per input.
OUTPUT_FORMAT_TEXT = "text"
OUTPUT_FORMAT_JSON = "json"
_OUTPUT_FORMATS = (OUTPUT_FORMAT_TEXT, OUTPUT_FORMAT_JSON)

# * Shortest media window: ffmpeg receives window times with millisecond precision.
_MIN_WINDOW_DURATION_S = 0.001

# * CPU threads of each network of a one-shot (``--no-daemon``) diarization.
_ONE_SHOT_DIARIZE_THREADS = 4

# * Model selection of diarize-mode requests; the diarization worker ignores it.
_EXTERNAL_DEFAULT_WHISPER_MODEL = "small"
_EXTERNAL_DEFAULT_COMPUTE_TYPE = "auto"

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
    # * Structured payload of a successful child (segments, turns, speakers).
    success_data: dict[str, object] | None = None


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


def _read_child_success_result(
    path: Path,
) -> tuple[str, dict[str, object] | None] | None:
    """Read the result file of a successful one-shot child.

    Args:
        path: The child's result file.

    Returns:
        ``(text, data)`` of an ``ok`` result, where ``data`` is the structured
        payload (a JSON object) or ``None`` when the result has none; ``None``
        when the file is missing, unreadable, not ``ok``, or malformed (a
        non-string ``text`` or a ``data`` that is not an object).

    """
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
    payload = data.get("data")
    if "data" in data and not isinstance(payload, dict):
        return None
    return _normalize_transcript_text(text), payload


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


@contextlib.contextmanager
def _local_work_dir(work_dir: Path | None) -> Iterator[Path]:
    """Yield the work directory of a one-shot run.

    Args:
        work_dir: ``--work-dir`` value; ``None`` uses a temporary directory
            (prefix ``askvlm-cli-``) that is deleted afterwards.

    Yields:
        The existing directory for intermediate files.

    """
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="askvlm-cli-") as temp_dir:
            yield Path(temp_dir)
        return
    work_dir.mkdir(parents=True, exist_ok=True)
    yield work_dir


def _diarize_one_shot(
    input_path: Path,
    options: DiarizeOptions,
    *,
    window: MediaWindow,
    work_dir: Path,
) -> list[dict[str, object]]:
    """Diarize one input in this process on the CPU (sherpa-onnx).

    Args:
        input_path: Media file.
        options: Diarization settings.
        window: Media window; turn times stay media-relative.
        work_dir: Folder for the intermediate WAV.

    Returns:
        ``{start, end, speaker}`` turns in start order (possibly empty).

    Raises:
        typer.Exit: Code ``1`` after one stderr line when the diarization
            fails for any reason (missing backend with its install hint, model
            download, undecodable media, an empty window).

    """
    try:
        # * Imported on use: only a one-shot diarization needs the backend.
        from core.sherpa_diarization import SherpaDiarizer  # noqa: PLC0415

        return SherpaDiarizer(threads=_ONE_SHOT_DIARIZE_THREADS).diarize_media(
            input_path, options, window=window, work_dir=work_dir
        )
    except Exception as exc:
        # * Every failure becomes the one-line exit-1 report of the one-shot
        # * contract (the pool reports worker failures the same way).
        typer.secho(
            f"speaker diarization failed: {str(exc).strip() or type(exc).__name__}",
            err=True,
        )
        raise typer.Exit(code=1) from exc


def _speaker_lines(
    segments: Sequence[TextSegment],
    speakers: Sequence[str | None],
    *,
    merge_runs: bool,
) -> str:
    """Label the transcript document's segments with their speakers.

    Args:
        segments: The text segments of the transcript document, one per
            structured segment.
        speakers: The speaker of each segment (``None`` when unassigned).
        merge_runs: Merge consecutive segments of the same speaker into one
            line first (``--dialog-blocks``).

    Returns:
        One ``"{speaker}: {text}"`` line (``"{text}"`` when unassigned) per
        segment or merged run with non-empty text.

    Raises:
        RuntimeError: When the document and the structured transcript differ
            in segment count, so speakers cannot be matched to the text.

    """
    if len(segments) != len(speakers):
        msg = (
            f"the transcript document has {len(segments)} segments but the "
            f"structured transcript has {len(speakers)}; cannot label speakers"
        )
        raise RuntimeError(msg)
    # * A speaker id of "" marks an unassigned segment (labels are never empty).
    labelled = [
        TextSegment(
            speaker_id=speaker or "",
            start_time=segment.start_time,
            end_time=segment.end_time,
            text=segment.text,
        )
        for segment, speaker in zip(segments, speakers, strict=True)
    ]
    if merge_runs:
        labelled = merge_consecutive_segments_by_speaker(labelled)
    return speaker_text(
        [segment.text.strip() for segment in labelled],
        [segment.speaker_id or None for segment in labelled],
    )


def _local_transcript(
    document: Document,
    *,
    output_format: str,
    window: MediaWindow,
    turns: list[dict[str, object]] | None,
    dialog_blocks: bool,
) -> tuple[str, dict[str, object] | None]:
    """Return the text and the structured payload of a one-shot transcription.

    Args:
        document: The document returned by ``LocalPipeline.process``.
        output_format: ``text`` or ``json``.
        window: Media window of the run (offset of the segment times).
        turns: Diarization turns, or ``None`` without ``--diarization``.
        dialog_blocks: ``--dialog-blocks`` was given.

    Returns:
        ``(text, data)``. The text is the normalized ``get_full_text()``; with
        diarization, the plain-text format and at least two voices among the
        segments it holds one speaker-labelled line per segment (per merged run
        with ``--dialog-blocks``). The JSON format never labels the text: its
        document carries the speakers per segment. ``data`` is ``None`` for a
        plain text run, else ``{"segments"}`` plus ``"turns"`` and
        ``"speakers"`` with diarization.

    Raises:
        RuntimeError: When the pipeline returned no structured segments, or
            (plain text with diarization) they do not match the document's
            segments.

    """
    text = _normalize_transcript_text(document.get_full_text())
    if output_format != OUTPUT_FORMAT_JSON and turns is None:
        return text, None
    raw_segments = document.transcript_segments
    if raw_segments is None:
        msg = "the transcription returned no structured segments"
        raise RuntimeError(msg)
    segments = segments_json(raw_segments, window.start_s)
    if turns is None:
        return text, {"segments": segments}
    speakers = assign_speakers(segments, turns)
    data: dict[str, object] = {
        "segments": segments,
        "turns": turns,
        "speakers": speakers,
    }
    if output_format == OUTPUT_FORMAT_JSON:
        return text, data
    # * One voice (or none) keeps the plain transcript, byte for byte.
    if voice_count(speakers) >= MIN_VOICES_FOR_SPEAKER_PREFIXES:
        labelled = _speaker_lines(document.segments, speakers, merge_runs=dialog_blocks)
        if labelled.strip():
            text = labelled
    return text, data


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
    before_close: Callable[[str, dict[str, object] | None], None] | None = None,
    output_format: str = OUTPUT_FORMAT_TEXT,
    media_window: MediaWindow | None = None,
    decode_options: WhisperDecodeOptions | None = None,
) -> str:
    """Transcribe one input in this process (the ``--no-daemon`` flow).

    With ``diarization`` the sherpa-onnx diarization runs first and finishes
    before the STT pipeline is created, so only one model is loaded at a time;
    the local pipeline itself never diarizes. ``before_close`` receives the
    text and the structured payload (see :func:`_local_transcript`) before the
    pipeline closes.

    Returns:
        The final transcript text.

    """
    window = media_window if media_window is not None else MediaWindow()
    process_kwargs: dict[str, object] = {}
    if not window.is_default:
        process_kwargs["media_window"] = window
    if decode_options is not None and not decode_options.is_default:
        process_kwargs["decode_options"] = decode_options
    with _local_work_dir(work_dir) as directory:
        turns = (
            _diarize_one_shot(
                input_path, DiarizeOptions(), window=window, work_dir=directory
            )
            if diarization
            else None
        )
        pipeline = _create_local_pipeline(
            whisper_model=whisper_model,
            engine="whisperx",
            diarization=False,
            dialog_blocks=dialog_blocks,
            language=language,
            device=device,
            compute_type=compute_type,
            stt_provider=stt_provider,
        )
        try:
            document = pipeline.process(input_path, directory, **process_kwargs)
            text, data = _local_transcript(
                document,
                output_format=output_format,
                window=window,
                turns=turns,
                dialog_blocks=dialog_blocks,
            )
            if before_close is not None:
                before_close(text, data)
            return text
        finally:
            pipeline.close(aggressive=False)


def _child_result_writer(
    result_file: Path,
) -> Callable[[str, dict[str, object] | None], None]:
    """Return the ``before_close`` callback of the one-shot child.

    Args:
        result_file: The child's result file read by the parent.

    Returns:
        A callback writing ``{"status": "ok", "text", ["data"]}`` atomically.

    """

    def write(text: str, data: dict[str, object] | None = None) -> None:
        payload: dict[str, Any] = {"status": "ok", "text": text}
        if data is not None:
            payload["data"] = data
        _write_json_atomic(result_file, payload)

    return write


def _decode_flags(decode_options: WhisperDecodeOptions | None) -> list[str]:
    """Return the CLI flags of explicitly set Whisper decoding options.

    Args:
        decode_options: Decoding overrides, or ``None``.

    Returns:
        Flags such as ``["--beam-size", "5", "--no-vad"]``; empty when nothing
        is set.

    """
    if decode_options is None:
        return []
    flags: list[str] = []
    if decode_options.beam_size is not None:
        flags.extend(["--beam-size", str(decode_options.beam_size)])
    for value, name in (
        (decode_options.vad_filter, "vad"),
        (decode_options.condition_on_previous_text, "condition-on-previous-text"),
        (decode_options.word_timestamps, "word-timestamps"),
    ):
        if value is not None:
            flags.append(f"--{name}" if value else f"--no-{name}")
    return flags


def _window_flags(media_window: MediaWindow | None) -> list[str]:
    """Return the ``--start``/``--duration`` flags of a non-default window.

    Args:
        media_window: Media window, or ``None``.

    Returns:
        The flags; empty for the whole media.

    """
    if media_window is None:
        return []
    flags: list[str] = []
    if media_window.start_s > 0:
        flags.extend(["--start", repr(media_window.start_s)])
    if media_window.duration_s is not None:
        flags.extend(["--duration", repr(media_window.duration_s)])
    return flags


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
    output_format: str = OUTPUT_FORMAT_TEXT,
    media_window: MediaWindow | None = None,
    decode_options: WhisperDecodeOptions | None = None,
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
    command.extend(["--format", output_format])
    command.extend(_decode_flags(decode_options))
    command.extend(_window_flags(media_window))
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
    output_format: str = OUTPUT_FORMAT_TEXT,
    media_window: MediaWindow | None = None,
    decode_options: WhisperDecodeOptions | None = None,
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
            output_format=output_format,
            media_window=media_window,
            decode_options=decode_options,
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "_ASKVLM_CHILD_RESULT_FILE": str(child_result_file)},
        )
        success = _read_child_success_result(child_result_file)
        return _ExternalChildAttemptResult(
            success_text=None if success is None else success[0],
            return_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            success_data=None if success is None else success[1],
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


def _raise_external_transcribe_error(attempt: _ExternalChildAttemptResult) -> NoReturn:
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


def _stdout_json(value: object, *, indent: int | None = None) -> str:
    """Serialize JSON for stdout so that it survives any console encoding.

    A pipe or console on Windows defaults to the ANSI code page, which cannot
    encode every character Whisper emits (Japanese text on a cp1251 console),
    so stdout JSON is plain ASCII with escaped non-ASCII characters. Files
    keep UTF-8 JSON without escapes (:func:`core.stt_output.document_json`).

    Args:
        value: A JSON-serializable value.
        indent: Indentation of the document; ``None`` renders one compact line.

    Returns:
        The JSON text without a trailing line feed.

    """
    return json.dumps(value, ensure_ascii=True, indent=indent)


def _emit_document(
    document: Mapping[str, object],
    *,
    output_file: Path | None,
    output_dir_file: Path | None,
    stdout: bool,
) -> None:
    """Write one JSON document to the single-input targets.

    stdout gets the indented ASCII-only document and a line feed;
    ``--output-file`` and the ``--output-dir`` file get the same document as
    UTF-8 text with unescaped characters (platform line endings, like stdout),
    also for a document without segments or turns.

    Args:
        document: Transcript or diarization document.
        output_file: ``--output-file`` target, if any.
        output_dir_file: ``--output-dir`` file of the input, if any.
        stdout: Whether the document goes to stdout.

    """
    file_text = document_json(document) + "\n"
    if output_file is not None:
        _write_plain_text(output_file, file_text)
    if output_dir_file is not None:
        _write_plain_text(output_dir_file, file_text)
    if stdout:
        typer.echo(_stdout_json(document, indent=2))


def _object_list(data: Mapping[str, object], key: str) -> list[dict[str, object]]:
    """Return ``data[key]`` when it is a list of JSON objects.

    Raises:
        ValueError: When the key is missing or holds anything else.

    """
    value = data.get(key)
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [item for item in value if isinstance(item, dict)]
    msg = f"malformed result: expected a {key!r} list of objects, got {value!r:.80}"
    raise ValueError(msg)


def _speaker_labels(value: object) -> list[str | None]:
    """Return *value* when it is a list of speaker labels or nulls.

    Raises:
        ValueError: When *value* is anything else.

    """
    if isinstance(value, list) and all(
        item is None or isinstance(item, str) for item in value
    ):
        return [item if isinstance(item, str) else None for item in value]
    msg = f"malformed result: expected a 'speakers' list, got {value!r:.80}"
    raise ValueError(msg)


def _transcript_document_from_data(
    media: str, data: Mapping[str, object] | None
) -> dict[str, object]:
    """Build the transcript document of one input from its structured payload.

    Args:
        media: The input path as given.
        data: ``{"segments"}``, plus ``"turns"`` and ``"speakers"`` with
            diarization.

    Returns:
        The document of :func:`core.stt_output.transcript_document`.

    Raises:
        ValueError: When the payload is missing or malformed.

    """
    if data is None:
        msg = "malformed result: the transcript segments are missing"
        raise ValueError(msg)
    segments = _object_list(data, "segments")
    if "turns" not in data:
        return transcript_document(media, segments)
    speakers = _speaker_labels(data["speakers"]) if "speakers" in data else None
    return transcript_document(
        media, segments, turns=_object_list(data, "turns"), speakers=speakers
    )


def _diarization_document_from_data(
    media: str, data: Mapping[str, object] | None
) -> dict[str, object]:
    """Build the diarization document of one input from its structured payload.

    Args:
        media: The input path as given.
        data: ``{"turns"}``.

    Returns:
        The document of :func:`core.stt_output.diarization_document`.

    Raises:
        ValueError: When the payload is missing or malformed.

    """
    if data is None:
        msg = "malformed result: the diarization turns are missing"
        raise ValueError(msg)
    return diarization_document(media, _object_list(data, "turns"))


@dataclass(frozen=True)
class _ResultFormat:
    """How a call renders the results of its successful inputs.

    Attributes:
        kind: ``text`` (plain transcripts, the historical contract),
            ``transcript`` (transcript JSON documents) or ``diarization``
            (diarization JSON documents).

    """

    kind: Literal["text", "transcript", "diarization"]

    @property
    def structured(self) -> bool:
        """Whether every successful input renders as a JSON document."""
        return self.kind != "text"

    @property
    def has_text(self) -> bool:
        """Whether batch rows carry the transcript ``text``."""
        return self.kind != "diarization"

    @property
    def output_suffix(self) -> str:
        """Extension of the ``--output-dir`` files."""
        return ".json" if self.structured else ".txt"

    @property
    def failure_detail(self) -> str:
        """Return the stderr text of a failed single input without a detail."""
        if self.kind == "diarization":
            return "speaker diarization failed"
        return "transcription failed"

    def document(self, outcome: InputOutcome) -> dict[str, object]:
        """Build the JSON document of a successful outcome.

        Raises:
            ValueError: When the outcome's payload is missing or malformed.

        """
        media = str(outcome.input_path)
        if self.kind == "diarization":
            return _diarization_document_from_data(media, outcome.data)
        return _transcript_document_from_data(media, outcome.data)


_TEXT_RESULTS = _ResultFormat("text")
_TRANSCRIPT_DOCUMENTS = _ResultFormat("transcript")
_DIARIZATION_DOCUMENTS = _ResultFormat("diarization")


def _transcribe_result_format(output_format: str) -> _ResultFormat:
    """Return the result rendering of an ``external-transcribe`` ``--format``."""
    if output_format == OUTPUT_FORMAT_JSON:
        return _TRANSCRIPT_DOCUMENTS
    return _TEXT_RESULTS


def _emit_local_transcript(
    text: str,
    data: dict[str, object] | None = None,
    *,
    input_path: Path,
    output_format: str,
    output_file: Path | None,
    output_dir_file: Path | None,
    stdout: bool,
) -> None:
    """Apply the single-input output rules to a one-shot transcription.

    Args:
        text: Final transcript text.
        data: Structured payload (see :func:`_local_transcript`); required for
            ``--format json``.
        input_path: The input as given (the document's ``media``).
        output_format: ``text`` or ``json``.
        output_file: ``--output-file`` target, if any.
        output_dir_file: ``--output-dir`` file of the input, if any.
        stdout: Whether the result goes to stdout.

    Raises:
        typer.Exit: Code ``1`` when ``--format json`` lacks a valid payload.

    """
    if output_format != OUTPUT_FORMAT_JSON:
        _emit_external_transcribe_outputs(
            text=text,
            output_file=output_file,
            stdout=stdout,
            output_dir_file=output_dir_file,
        )
        return
    try:
        document = _transcript_document_from_data(str(input_path), data)
    except ValueError as exc:
        typer.secho(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _emit_document(
        document,
        output_file=output_file,
        output_dir_file=output_dir_file,
        stdout=stdout,
    )


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


def _output_format_callback(value: str) -> str:
    """Validate ``--format`` case-insensitively.

    Args:
        value: Raw option value.

    Returns:
        ``text`` or ``json``.

    Raises:
        typer.BadParameter: When the value names no output format.

    """
    token = value.strip().lower()
    if token in _OUTPUT_FORMATS:
        return token
    msg = f"expected one of: {', '.join(_OUTPUT_FORMATS)} (got {value!r})"
    raise typer.BadParameter(msg)


def _whisper_decode_options(
    provider: str,
    *,
    beam_size: int | None,
    vad: bool | None,
    condition_on_previous_text: bool | None,
    word_timestamps: bool | None,
) -> WhisperDecodeOptions:
    """Validate the Whisper decoding flags of ``external-transcribe``.

    Args:
        provider: Canonical STT provider id.
        beam_size: ``--beam-size``, or ``None`` when not given.
        vad: ``--vad``/``--no-vad``, or ``None`` when not given.
        condition_on_previous_text: ``--[no-]condition-on-previous-text``.
        word_timestamps: ``--[no-]word-timestamps``.

    Returns:
        The decoding overrides; all ``None`` keeps the pipeline defaults.

    Raises:
        typer.BadParameter: When a value is out of range, or any of the flags
            is given with a GigaAM provider (they are Whisper-only).

    """
    try:
        options = WhisperDecodeOptions(
            beam_size=beam_size,
            vad_filter=vad,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=word_timestamps,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if is_gigaam_provider(provider) and not options.is_default:
        given = " ".join(_decode_flags(options))
        msg = (
            f"{given}: these decoding options apply to the whisper provider only; "
            f"--stt-provider {provider} does not support them (GigaAM JSON has "
            "only its own segments, without words or scores)"
        )
        raise typer.BadParameter(msg, param_hint="'--stt-provider'")
    return options


def _media_window(start: float, duration: float | None) -> MediaWindow:
    """Validate ``--start``/``--duration``.

    Args:
        start: Window start in seconds from the media start.
        duration: Window length in seconds, or ``None`` for "to the end".

    Returns:
        The media window (the default window for ``0`` and ``None``).

    Raises:
        typer.BadParameter: When the start is negative or not finite, or the
            duration is not finite or below :data:`_MIN_WINDOW_DURATION_S`.

    """
    if not math.isfinite(start) or start < 0:
        msg = f"must be a finite number of seconds >= 0 (got {start})"
        raise typer.BadParameter(msg, param_hint="'--start'")
    if duration is not None and (
        not math.isfinite(duration) or duration < _MIN_WINDOW_DURATION_S
    ):
        msg = (
            f"must be a finite number of seconds >= {_MIN_WINDOW_DURATION_S} "
            f"(ffmpeg cuts with millisecond precision; got {duration})"
        )
        raise typer.BadParameter(msg, param_hint="'--duration'")
    return MediaWindow(start_s=start, duration_s=duration)


def _diarize_options(
    *,
    num_speakers: int | None,
    threshold: float,
    min_on: float,
    min_off: float,
) -> DiarizeOptions:
    """Validate the clustering and turn-filter flags of ``external-diarize``.

    Args:
        num_speakers: ``--num-speakers``, or ``None`` to cluster by threshold.
        threshold: ``--threshold``.
        min_on: ``--min-on``.
        min_off: ``--min-off``.

    Returns:
        The diarization settings.

    Raises:
        typer.BadParameter: When a value is out of range or not finite.

    """
    if num_speakers is not None and num_speakers < 1:
        msg = f"must be at least 1 (got {num_speakers})"
        raise typer.BadParameter(msg, param_hint="'--num-speakers'")
    if not math.isfinite(threshold) or threshold <= 0:
        msg = f"must be a finite number > 0 (got {threshold})"
        raise typer.BadParameter(msg, param_hint="'--threshold'")
    for flag, value in (("--min-on", min_on), ("--min-off", min_off)):
        if not math.isfinite(value) or value < 0:
            msg = f"must be a finite number of seconds >= 0 (got {value})"
            raise typer.BadParameter(msg, param_hint=f"'{flag}'")
    return DiarizeOptions(
        num_speakers=num_speakers,
        threshold=threshold,
        min_duration_on=min_on,
        min_duration_off=min_off,
    )


def _diarize_device(device: str) -> DeviceRequest:
    """Validate ``--device`` of ``external-diarize`` (CPU only).

    Args:
        device: Raw ``--device`` token.

    Returns:
        ``cpu`` for ``auto`` and ``cpu``.

    Raises:
        typer.BadParameter: For ``cuda`` (speaker diarization runs on the CPU
            only) and for unknown tokens.

    """
    try:
        return normalize_device_request(DIARIZE_PROVIDER, device)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="'--device'") from exc


def _reject_supervisor_unsupported_flags(
    *, dialog_blocks: bool, work_dir: Path | None
) -> None:
    """Refuse flags the transcription supervisor cannot honor.

    Args:
        dialog_blocks: ``--dialog-blocks`` was given.
        work_dir: Value of ``--work-dir``.

    Raises:
        typer.BadParameter: Naming the first unsupported flag and ``--no-daemon``.

    """
    for flag, used in (
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
    want_segments: bool = False,
    decode: WhisperDecodeOptions | None = None,
    window: MediaWindow | None = None,
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
        diarization: ``--diarization`` was given (a CPU diarization job runs
            next to every transcription job, with the default settings).
        dialog_blocks: ``--dialog-blocks`` was given.
        work_dir: ``--work-dir`` value.
        want_segments: ``--format json`` was given.
        decode: Whisper decoding overrides (``None`` keeps the defaults).
        window: Media window (``None`` covers the whole media).

    Returns:
        The request for :func:`~core.stt_pool.client.run_client`.

    Raises:
        typer.BadParameter: For a flag the supervisor cannot honor, a device
            token a GigaAM head rejects, or an invalid language token.

    """
    _reject_supervisor_unsupported_flags(dialog_blocks=dialog_blocks, work_dir=work_dir)
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
        want_segments=want_segments,
        decode=decode if decode is not None else WhisperDecodeOptions(),
        window=window if window is not None else MediaWindow(),
        diarization=DiarizeOptions() if diarization else None,
    )


def _note_deprecated_daemon_workers(daemon_workers: int | None) -> None:
    """Print the ``--daemon-workers`` deprecation note when the flag was given.

    Args:
        daemon_workers: ``--daemon-workers`` value, or ``None``.

    """
    if daemon_workers is not None:
        typer.secho(_DAEMON_WORKERS_DEPRECATION, err=True)


def _resolve_output_dir_files(
    inputs: Sequence[Path], output_dir: Path, *, extension: str = ".txt"
) -> list[Path]:
    """Name the ``--output-dir`` file of every input before the run.

    The first input with a given stem gets ``<stem>.txt``; a later input whose
    name would collide (compared case-insensitively) gets ``<stem>-2.txt``,
    ``<stem>-3.txt``, and so on, skipping names already taken. JSON results
    use ``.json`` in the same way.

    Args:
        inputs: Inputs in input order.
        output_dir: Target directory.
        extension: File extension with its dot (``.txt`` or ``.json``).

    Returns:
        One distinct file path per input, in input order.

    """
    taken: set[str] = set()
    files: list[Path] = []
    for input_path in inputs:
        stem = input_path.stem
        name = f"{stem}{extension}"
        suffix = 1
        while name.casefold() in taken:
            suffix += 1
            name = f"{stem}-{suffix}{extension}"
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
    result_format: _ResultFormat = _TEXT_RESULTS,
) -> None:
    """Apply the single-input output rules to a supervisor outcome.

    Args:
        outcome: The outcome of the only input.
        output_file: ``--output-file`` target, if any.
        output_dir_file: ``--output-dir`` file of the input, if any.
        stdout: Whether the transcript (or the JSON document) goes to stdout.
        result_format: Plain text, or the JSON document of every success (also
            of an empty transcript or a diarization without turns).

    Raises:
        typer.Exit: :data:`EXTERNAL_TRANSCRIBE_TIMEOUT_EXIT` on a timeout,
            :data:`EXTERNAL_TRANSCRIBE_UNAVAILABLE_EXIT` when the supervisor is
            unavailable, ``1`` on an error (stderr carries the worker's detail)
            or a malformed structured result.

    """
    for warning in outcome.warnings:
        typer.secho(warning, err=True)
    if outcome.status in _SUCCESS_STATUSES:
        if not result_format.structured:
            _emit_external_transcribe_outputs(
                text=outcome.text,
                output_file=output_file,
                stdout=stdout,
                output_dir_file=output_dir_file,
            )
            return
        try:
            document = result_format.document(outcome)
        except ValueError as exc:
            typer.secho(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        _emit_document(
            document,
            output_file=output_file,
            output_dir_file=output_dir_file,
            stdout=stdout,
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
    typer.secho(outcome.detail or result_format.failure_detail, err=True)
    raise typer.Exit(code=1)


class _BatchReporter:
    """Report the outcomes of a multi-input call in input order.

    Outcomes arrive in completion order. Each ``--output-dir`` file is written
    as soon as its outcome arrives; the stdout JSON line and the stderr lines
    of an input are emitted once every earlier input is final. With a
    structured result format every row also carries ``result``: the input's
    compact JSON document, or ``null`` for a failed input (and diarization
    rows carry no ``text``).
    """

    def __init__(
        self,
        inputs: Sequence[Path],
        *,
        output_dir_files: Sequence[Path] | None,
        stdout: bool,
        result_format: _ResultFormat = _TEXT_RESULTS,
    ) -> None:
        self._count = len(inputs)
        self._output_dir_files = (
            None if output_dir_files is None else tuple(output_dir_files)
        )
        self._stdout = stdout
        self._format = result_format
        self._received: dict[int, InputOutcome] = {}
        self._documents: dict[int, dict[str, object]] = {}
        self._written: dict[int, Path] = {}
        self._next_index = 0
        self._warned: set[str] = set()

    def on_outcome(self, outcome: InputOutcome) -> None:
        """Accept one final outcome; a repeated index is ignored."""
        if outcome.index in self._received:
            return
        documented = self._with_document(outcome)
        self._received[outcome.index] = self._write_output_dir_file(documented)
        while self._next_index in self._received:
            self._emit(self._received[self._next_index])
            self._next_index += 1

    def outcomes(self) -> list[InputOutcome]:
        """Return the reported outcomes in input order."""
        return [self._received[index] for index in range(self._count)]

    def documents(self) -> list[dict[str, object] | None]:
        """Return the JSON document of every input, ``None`` for a failed input."""
        return [self._document_of(outcome) for outcome in self.outcomes()]

    def _document_of(self, outcome: InputOutcome) -> dict[str, object] | None:
        """Return the document of a successful input, else ``None``."""
        if outcome.status not in _SUCCESS_STATUSES:
            return None
        return self._documents.get(outcome.index)

    def _with_document(self, outcome: InputOutcome) -> InputOutcome:
        """Build the document of a successful input; a malformed payload is an error."""
        if not self._format.structured or outcome.status not in _SUCCESS_STATUSES:
            return outcome
        try:
            self._documents[outcome.index] = self._format.document(outcome)
        except ValueError as exc:
            return replace(outcome, status="error", text="", detail=str(exc), data=None)
        return outcome

    def _write_output_dir_file(self, outcome: InputOutcome) -> InputOutcome:
        """Write the ``--output-dir`` file of a successful input.

        A failed write turns the input into an error.
        """
        files = self._output_dir_files
        if files is None or outcome.status not in _SUCCESS_STATUSES:
            return outcome
        target = files[outcome.index]
        document = self._documents.get(outcome.index)
        content = outcome.text if document is None else document_json(document) + "\n"
        try:
            _write_plain_text(target, content)
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
        }
        if self._format.has_text:
            record["text"] = "" if failed else outcome.text
        record["error"] = error
        if self._output_dir_files is not None:
            written = self._written.get(outcome.index)
            record["output"] = None if written is None else str(written)
        if self._format.structured:
            record["result"] = self._document_of(outcome)
        typer.echo(_stdout_json(record))


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


def _run_client_checked(
    request: ClientRequest, *, on_outcome: OutcomeFn | None = None
) -> ClientOutcome:
    """Run the client; an invalid request is a usage error.

    Args:
        request: The client request.
        on_outcome: Called once per input as soon as its outcome is final.

    Returns:
        One outcome per input, in input order.

    Raises:
        typer.BadParameter: When the client rejects the request (exit ``2``)
            before any job is written.

    """
    try:
        return run_client(request, on_outcome=on_outcome)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _aggregate_output(reporter: _BatchReporter, result_format: _ResultFormat) -> str:
    """Return the batch ``--output-file`` content in input order.

    Args:
        reporter: The reporter that received every outcome.
        result_format: How results are rendered.

    Returns:
        Plain text: the transcripts joined by line feeds (a failed input
        contributes an empty line). Structured: JSON Lines of the compact
        documents, ``null`` for a failed input, each line ending with a line
        feed.

    """
    if not result_format.structured:
        return "\n".join(
            outcome.text if outcome.status in _SUCCESS_STATUSES else ""
            for outcome in reporter.outcomes()
        )
    return "".join(
        json.dumps(document, ensure_ascii=False) + "\n"
        for document in reporter.documents()
    )


def _run_via_supervisor(
    request: ClientRequest,
    *,
    output_file: Path | None,
    output_dir: Path | None,
    stdout: bool,
    result_format: _ResultFormat = _TEXT_RESULTS,
) -> None:
    """Run a call through the STT supervisor and apply the CLI output rules.

    One input keeps the single-file contract (the transcript, or the indented
    JSON document, on stdout; exit codes 0/1/10/11). Several inputs produce one
    JSON line per input on stdout in input order, optional ``--output-dir``
    files, one atomic ``--output-file`` (transcripts joined by newlines, or
    JSON Lines of the documents), and the exit code of
    :func:`_batch_exit_code`. A failed ``--output-file`` write prints one
    stderr line and turns an otherwise successful call into exit ``1``; the
    previous file content stays. There is no in-process recovery.

    Args:
        request: The client request.
        output_file: ``--output-file`` target, if any.
        output_dir: ``--output-dir`` target, if any.
        stdout: Whether results (or JSON lines) go to stdout.
        result_format: Plain transcripts or JSON documents.

    Raises:
        typer.Exit: With the exit code of a failed call.
        typer.BadParameter: When the client rejects the request.

    """
    output_dir_files = (
        None
        if output_dir is None
        else _resolve_output_dir_files(
            request.inputs, output_dir, extension=result_format.output_suffix
        )
    )
    if len(request.inputs) == 1:
        outcome = _run_client_checked(request).outcomes[0]
        _report_single_supervisor_outcome(
            outcome,
            output_file=output_file,
            output_dir_file=None if output_dir_files is None else output_dir_files[0],
            stdout=stdout,
            result_format=result_format,
        )
        return
    reporter = _BatchReporter(
        request.inputs,
        output_dir_files=output_dir_files,
        stdout=stdout,
        result_format=result_format,
    )
    result = _run_client_checked(request, on_outcome=reporter.on_outcome)
    for outcome in result.outcomes:
        reporter.on_outcome(outcome)
    outcomes = reporter.outcomes()
    output_file_failed = False
    if output_file is not None:
        try:
            _write_text_atomic(output_file, _aggregate_output(reporter, result_format))
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


@dataclass(frozen=True)
class _LocalTranscribeJob:
    """Options of one ``external-transcribe --no-daemon`` run.

    Attributes:
        input_path: The only input, as given.
        provider: Canonical STT provider id.
        whisper_model: ``--whisper-model``.
        language: ``--language``.
        device: Device token checked by :func:`_resolve_external_device`.
        compute_type: ``--compute-type``.
        diarization: ``--diarization`` (sherpa-onnx, CPU).
        dialog_blocks: ``--dialog-blocks``.
        work_dir: ``--work-dir``.
        output_format: ``text`` or ``json``.
        window: Media window.
        decode: Whisper decoding overrides.

    """

    input_path: Path
    provider: str
    whisper_model: str
    language: str | None
    device: str
    compute_type: str
    diarization: bool
    dialog_blocks: bool
    work_dir: Path | None
    output_format: str
    window: MediaWindow
    decode: WhisperDecodeOptions


def _run_external_transcribe_no_daemon(
    job: _LocalTranscribeJob,
    *,
    output_file: Path | None,
    output_dir_file: Path | None,
    stdout: bool,
) -> None:
    """Run the one-shot flow of ``--no-daemon`` and apply the output rules.

    On Windows a CUDA-capable provider (Whisper, gigaam-e2e-ctc) without
    ``--device cpu`` runs in an isolated child process that writes its text
    and structured payload before its pipeline closes; a crash-like child
    failure is retried once on the CPU. Otherwise the run stays in-process.
    Both ways print the same output.

    Args:
        job: The run's options.
        output_file: ``--output-file`` target, if any.
        output_dir_file: ``--output-dir`` file of the input, if any.
        stdout: Whether the result goes to stdout.

    Raises:
        typer.Exit: Code ``1`` when the run fails.

    """

    def emit(text: str, data: dict[str, object] | None = None) -> None:
        _emit_local_transcript(
            text,
            data,
            input_path=job.input_path,
            output_format=job.output_format,
            output_file=output_file,
            output_dir_file=output_dir_file,
            stdout=stdout,
        )

    def attempt(device: str) -> _ExternalChildAttemptResult:
        return _run_external_transcribe_isolated_attempt(
            input_path=job.input_path,
            whisper_model=job.whisper_model,
            language=job.language,
            device=device,
            compute_type=job.compute_type,
            diarization=job.diarization,
            dialog_blocks=job.dialog_blocks,
            work_dir=job.work_dir,
            stt_provider=job.provider,
            output_format=job.output_format,
            media_window=job.window,
            decode_options=job.decode,
        )

    # * Windows GPU child isolation covers CUDA-capable providers (Whisper and
    # * gigaam-e2e-ctc); gigaam-ctc always runs in-process on CPU.
    windows_non_cpu = (
        provider_may_use_cuda(job.provider)
        and sys.platform.startswith("win")
        and job.device.lower() != "cpu"
    )
    if windows_non_cpu:
        first_attempt = attempt(job.device)
        if first_attempt.success_text is not None:
            emit(first_attempt.success_text, first_attempt.success_data)
            return

        if not _is_windows_crash_like_return_code(
            first_attempt.return_code
        ) and not _is_internal_child_ipc_error(first_attempt.stderr):
            _raise_external_transcribe_error(first_attempt)

        retry_attempt = attempt("cpu")
        if retry_attempt.success_text is not None:
            emit(retry_attempt.success_text, retry_attempt.success_data)
            return
        _raise_external_transcribe_error(retry_attempt)

    _run_external_transcribe_once(
        input_path=job.input_path,
        whisper_model=job.whisper_model,
        language=job.language,
        device=job.device,
        compute_type=job.compute_type,
        diarization=job.diarization,
        dialog_blocks=job.dialog_blocks,
        work_dir=job.work_dir,
        stt_provider=job.provider,
        before_close=emit,
        output_format=job.output_format,
        media_window=job.window,
        decode_options=job.decode,
    )


def _run_external_diarize_no_daemon(
    input_path: Path,
    options: DiarizeOptions,
    *,
    window: MediaWindow,
    output_file: Path | None,
    output_dir: Path | None,
    stdout: bool,
) -> None:
    """Diarize one input in this process and print its JSON document.

    Args:
        input_path: The only input, as given.
        options: Diarization settings.
        window: Media window.
        output_file: ``--output-file`` target, if any.
        output_dir: ``--output-dir`` target, if any.
        stdout: Whether the document goes to stdout.

    Raises:
        typer.Exit: Code ``1`` when the diarization fails.

    """
    output_dir_file = (
        None
        if output_dir is None
        else _resolve_output_dir_files(
            [input_path], output_dir, extension=_DIARIZATION_DOCUMENTS.output_suffix
        )[0]
    )
    with _local_work_dir(None) as work_dir:
        turns = _diarize_one_shot(input_path, options, window=window, work_dir=work_dir)
    _emit_document(
        diarization_document(str(input_path), turns),
        output_file=output_file,
        output_dir_file=output_dir_file,
        stdout=stdout,
    )


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
            "input order (failed inputs contribute an empty line). With --format "
            "json it receives the same document as stdout; with several inputs "
            "JSON Lines of the documents (null for a failed input)."
        ),
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help=(
            "Optional directory that receives <stem>.txt (<stem>.json with "
            "--format json) for every successful input; repeated stems get -2, "
            "-3, ... suffixes."
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
            "CPU speaker diarization (sherpa-onnx). Voices are labelled "
            "speaker_00, speaker_01, ... in order of first appearance. With two "
            "or more voices every text line starts with 'speaker_xx: '; with one "
            "voice the text stays unprefixed. With --format json the document "
            "gains turns and a speaker per segment (null without overlap). "
            "Disabled by default."
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
            "Write only the final transcript text (or the JSON document with "
            "--format json) to stdout; with several inputs, one JSON line per "
            "input in input order. Enabled by default."
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
    *,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            callback=_output_format_callback,
            help=(
                "Output format: text (default; the plain transcript) or json: one "
                "document {media, sample_rate, segments} per input with "
                "media-relative segment times, no_speech_prob, avg_logprob and "
                "words (Whisper; GigaAM segments carry start, end and text only). "
                "One input prints the indented document, also for an empty "
                "transcript; several inputs add it to every JSON line as result "
                "(null for a failed input)."
            ),
        ),
    ] = OUTPUT_FORMAT_TEXT,
    beam_size: Annotated[
        int | None,
        typer.Option(
            "--beam-size",
            min=MIN_BEAM_SIZE,
            max=MAX_BEAM_SIZE,
            help=(
                "Whisper only: beam search width (pipeline default: 10). An "
                "error with a GigaAM provider."
            ),
        ),
    ] = None,
    vad: Annotated[
        bool | None,
        typer.Option(
            "--vad/--no-vad",
            help=(
                "Whisper only: voice-activity filter (pipeline default: on). An "
                "error with a GigaAM provider."
            ),
        ),
    ] = None,
    condition_on_previous_text: Annotated[
        bool | None,
        typer.Option(
            "--condition-on-previous-text/--no-condition-on-previous-text",
            help=(
                "Whisper only: use the previous text as the decoding prompt "
                "(default: on). An error with a GigaAM provider."
            ),
        ),
    ] = None,
    word_timestamps: Annotated[
        bool | None,
        typer.Option(
            "--word-timestamps/--no-word-timestamps",
            help=(
                "Whisper only: word-level timestamps, the words of JSON segments "
                "(pipeline default: on). An error with a GigaAM provider."
            ),
        ),
    ] = None,
    start: Annotated[
        float,
        typer.Option(
            "--start",
            help=(
                "Transcribe from this many seconds after the media start "
                "(default: 0). JSON times stay relative to the media start."
            ),
        ),
    ] = 0.0,
    duration: Annotated[
        float | None,
        typer.Option(
            "--duration",
            help=(
                "Transcribe at most this many seconds from --start (at least "
                "0.001; default: to the end)."
            ),
        ),
    ] = None,
) -> None:
    """Transcribe media files for external applications (plain text or JSON).

    By default the inputs go through the shared STT supervisor (a pool of
    worker processes, one per configuration). One input prints its transcript
    (its JSON document with --format json); several inputs print one JSON line
    per input. ``--no-daemon`` runs the legacy in-process flow for one input.
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
    decode = _whisper_decode_options(
        provider,
        beam_size=beam_size,
        vad=vad,
        condition_on_previous_text=condition_on_previous_text,
        word_timestamps=word_timestamps,
    )
    window = _media_window(start, duration)
    if dialog_blocks and output_format == OUTPUT_FORMAT_JSON:
        msg = (
            "formats only the plain-text transcript and has no effect on "
            "--format json; drop one of them."
        )
        raise typer.BadParameter(msg, param_hint="'--dialog-blocks'")
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
                before_close=_child_result_writer(_internal_result_file),
                output_format=output_format,
                media_window=window,
                decode_options=decode,
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
            want_segments=output_format == OUTPUT_FORMAT_JSON,
            decode=decode,
            window=window,
        )
        _note_deprecated_daemon_workers(daemon_workers)
        _run_via_supervisor(
            request,
            output_file=output_file,
            output_dir=output_dir,
            stdout=stdout,
            result_format=_transcribe_result_format(output_format),
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
        else _resolve_output_dir_files(
            input_paths,
            output_dir,
            extension=_transcribe_result_format(output_format).output_suffix,
        )[0]
    )
    _run_external_transcribe_no_daemon(
        _LocalTranscribeJob(
            input_path=input_path,
            provider=provider,
            whisper_model=whisper_model,
            language=language,
            device=device,
            compute_type=compute_type,
            diarization=diarization,
            dialog_blocks=dialog_blocks,
            work_dir=work_dir,
            output_format=output_format,
            window=window,
            decode=decode,
        ),
        output_file=output_file,
        output_dir_file=output_dir_file,
        stdout=stdout,
    )


@app.command("external-diarize")
def external_diarize(  # noqa: PLR0913 - one argument per CLI option
    input_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            help=(
                "One or more audio or video files to diarize for an external "
                "caller (--no-daemon accepts exactly one)."
            ),
        ),
    ],
    *,
    num_speakers: Annotated[
        int | None,
        typer.Option(
            "--num-speakers",
            min=1,
            help=(
                "Exact number of speakers; by default the voices are clustered "
                "by --threshold."
            ),
        ),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help=(
                "Clustering distance threshold (> 0) used without --num-speakers; "
                "a larger value merges more voices."
            ),
        ),
    ] = DIARIZE_DEFAULT_THRESHOLD,
    min_on: Annotated[
        float,
        typer.Option("--min-on", help="Shortest kept speaker turn in seconds (>= 0)."),
    ] = DIARIZE_DEFAULT_MIN_DURATION_ON,
    min_off: Annotated[
        float,
        typer.Option(
            "--min-off",
            help=(
                "Shortest pause in seconds (>= 0) that splits two turns of one speaker."
            ),
        ),
    ] = DIARIZE_DEFAULT_MIN_DURATION_OFF,
    start: Annotated[
        float,
        typer.Option(
            "--start",
            help=(
                "Diarize from this many seconds after the media start (default: "
                "0). Turn times stay relative to the media start."
            ),
        ),
    ] = 0.0,
    duration: Annotated[
        float | None,
        typer.Option(
            "--duration",
            help=(
                "Diarize at most this many seconds from --start (at least 0.001; "
                "default: to the end)."
            ),
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help=(
                "auto|cpu: speaker diarization runs on the CPU only (auto means "
                "cpu); cuda is an error."
            ),
        ),
    ] = "auto",
    priority: Annotated[
        str,
        typer.Option(
            "--priority",
            callback=_priority_callback,
            help=(
                "Supervisor mode: priority class of this call: high, normal or "
                "low. Ignored with --no-daemon."
            ),
        ),
    ] = "normal",
    max_parallel: Annotated[
        int | None,
        typer.Option(
            "--max-parallel",
            min=1,
            max=MAX_PARALLEL_LIMIT,
            help=(
                "Supervisor mode: run at most N inputs of this call at once. By "
                "default the supervisor decides. Ignored with --no-daemon."
            ),
        ),
    ] = None,
    client_timeout: Annotated[
        float,
        typer.Option(
            "--client-timeout",
            min=1.0,
            callback=_client_timeout_callback,
            help=(
                "Supervisor mode: seconds for the whole call, queue wait "
                "included; unfinished inputs are then cancelled and reported as "
                "timeouts. inf waits without a deadline."
            ),
        ),
    ] = 300.0,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output-file",
            "-o",
            help=(
                "Optional file that receives the same document as stdout; with "
                "several inputs JSON Lines of the documents in input order (null "
                "for a failed input)."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            help=(
                "Optional directory that receives <stem>.json for every "
                "successful input; repeated stems get -2, -3, ... suffixes."
            ),
        ),
    ] = None,
    stdout: Annotated[
        bool,
        typer.Option(
            "--stdout/--no-stdout",
            help=(
                "Write only the JSON document to stdout; with several inputs, one "
                "JSON line per input in input order. Enabled by default."
            ),
        ),
    ] = True,
    no_daemon: Annotated[
        bool,
        typer.Option(
            "--no-daemon/--daemon",
            help=(
                "Diarize one input in this process instead of routing the request "
                "through the shared supervisor."
            ),
        ),
    ] = False,
) -> None:
    """Find who speaks when (CPU speaker diarization) for external applications.

    One input prints its document {media, turns} with turns {start, end,
    speaker} in seconds from the media start and speakers speaker_00,
    speaker_01, ... in order of first appearance (no turns is a success).
    Several inputs print one JSON line per input {index, input, status, error,
    result}. Exit codes: 0 success, 1 failure, 2 usage error, 10 client
    timeout, 11 supervisor unavailable.
    """
    if not stdout and output_file is None and output_dir is None:
        msg = "Either keep --stdout enabled or provide --output-file or --output-dir."
        raise typer.BadParameter(msg)
    options = _diarize_options(
        num_speakers=num_speakers, threshold=threshold, min_on=min_on, min_off=min_off
    )
    window = _media_window(start, duration)
    device_request = _diarize_device(device)
    if no_daemon:
        if len(input_paths) != 1:
            msg = (
                "--no-daemon accepts exactly one input; drop --no-daemon to "
                "diarize several inputs through the supervisor."
            )
            raise typer.BadParameter(msg)
        # * --priority, --max-parallel and --client-timeout only shape
        # * supervisor scheduling.
        _run_external_diarize_no_daemon(
            input_paths[0],
            options,
            window=window,
            output_file=output_file,
            output_dir=output_dir,
            stdout=stdout,
        )
        return
    request = ClientRequest(
        inputs=tuple(input_paths),
        provider=STT_PROVIDER_WHISPER,
        whisper_model=_EXTERNAL_DEFAULT_WHISPER_MODEL,
        compute_type=_EXTERNAL_DEFAULT_COMPUTE_TYPE,
        device=device_request,
        language=None,
        priority=normalize_priority(priority),
        max_parallel=max_parallel,
        client_timeout_s=client_timeout,
        mode=MODE_DIARIZE,
        window=window,
        diarization=options,
    )
    _run_via_supervisor(
        request,
        output_file=output_file,
        output_dir=output_dir,
        stdout=stdout,
        result_format=_DIARIZATION_DOCUMENTS,
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
