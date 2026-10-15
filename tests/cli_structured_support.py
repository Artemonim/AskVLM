"""Model and daemon boundaries for structured external CLI scenarios."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from typing import TYPE_CHECKING

import cli
from core.stt_pool.queue import (
    JobRecord,
    read_job_payload,
    write_heartbeat,
    write_result,
)
from editing.text_model import Document, TextSegment

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    import pytest

    from core.stt_options import DiarizeOptions, MediaWindow
    from core.stt_pool.queue import QueuePaths, ResultRecord


def queue_peer(
    templates: Mapping[tuple[int, str], ResultRecord],
) -> Callable[[QueuePaths], None]:
    """Publish terminal records at the external daemon's actual filesystem boundary."""

    def publish(paths: QueuePaths) -> None:
        write_heartbeat(paths, {"state": "running"})
        for path in paths.incoming.glob("*.json"):
            payload = read_job_payload(paths, path.stem)
            assert payload is not None
            job = JobRecord.from_dict(payload, expected_job_id=path.stem)
            template = templates.get((job.call_index, job.kind))
            if template is not None:
                write_result(paths, replace(template, job_id=job.job_id))

    return publish


class LocalModels:
    """Substitute only local ML endpoints; CLI composition and document logic stay real."""

    def __init__(self) -> None:
        self.raw = [
            {"start": 0.1, "end": 0.9, "text": " first "},
            {"start": 1.1, "end": 1.9, "text": " second "},
            {"start": 3.1, "end": 3.9, "text": " last "},
        ]
        self.document = Document(
            segments=[
                TextSegment("speaker_1", 0.1, 0.9, "first"),
                TextSegment("speaker_1", 1.1, 1.9, "second"),
                TextSegment("speaker_1", 3.1, 3.9, "last"),
            ],
            transcript_segments=self.raw,
        )
        self.turns: list[dict[str, object]] = [
            {"start": 0.0, "end": 2.0, "speaker": "speaker_00"},
            {"start": 3.0, "end": 4.0, "speaker": "speaker_01"},
        ]
        self.require_diarize = True
        self.diarized = False
        self.pipeline_options: list[dict[str, object]] = []
        self.process_options: list[dict[str, object]] = []
        self.diarize_options: list[tuple[DiarizeOptions, MediaWindow, int]] = []
        self.directories: list[Path] = []
        self.closed = False
        self.error: Exception | None = None
        self.close_error: Exception | None = None
        self.child_result: Path | None = None
        self.result_at_close: dict[str, object] | None = None

    def pipeline(self, **options: object) -> LocalModels:
        """Reject overlapping model lifetimes at the substituted allocation boundary."""
        if self.require_diarize and not self.diarized:
            msg = "STT allocation precedes completion of diarization"
            raise AssertionError(msg)
        self.pipeline_options.append(options)
        self.document.dialog_blocks_enabled = bool(options.get("enable_dialog_blocks"))
        return self

    def process(self, _path: Path, directory: Path, **options: object) -> Document:
        """Return real document objects carrying independently chosen model output."""
        self.directories.append(directory)
        self.process_options.append(options)
        return self.document

    def close(self, *, aggressive: bool) -> None:  # noqa: ARG002 - production model boundary
        """Expose whether the child published its result before model teardown."""
        self.closed = True
        if self.child_result is not None and self.child_result.exists():
            self.result_at_close = json.loads(
                self.child_result.read_text(encoding="utf-8")
            )
        if self.close_error is not None:
            raise self.close_error

    def diarizer(self, *, threads: int) -> object:
        """Build the fake native diarizer with the requested thread reservation."""
        runtime = self

        class Diarizer:
            """Fake sherpa inference with explicit completion and media options."""

            def diarize_media(
                self,
                _path: Path,
                options: DiarizeOptions,
                *,
                window: MediaWindow,
                work_dir: Path,
            ) -> list[dict[str, object]]:
                runtime.diarize_options.append((options, window, threads))
                runtime.directories.append(work_dir)
                if runtime.error is not None:
                    raise runtime.error
                runtime.diarized = True
                return runtime.turns

        return Diarizer()

    def install(self, patch: pytest.MonkeyPatch, root: Path) -> None:
        """Keep every temporary model artifact inside this scenario's pytest root."""
        patch.setattr(
            cli, "_load_cli_runtime", lambda: {"LocalPipeline": self.pipeline}
        )
        patch.setattr("core.sherpa_diarization.SherpaDiarizer", self.diarizer)
        patch.setattr(tempfile, "tempdir", str(root))
