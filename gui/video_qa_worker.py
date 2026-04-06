"""QObject worker: local Video QA (ASR, ffmpeg frames, LM Studio, aggregate)."""

from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, Slot

from core.pipelines import CancelledError
from core.video_qa_local_run import (
    VideoQALocalRunParams,
    VideoQAPreflightBlockedError,
    default_video_qa_local_run_options,
    run_local_video_qa,
)

if TYPE_CHECKING:
    from pathlib import Path

    from core.video_qa_context import VideoQAContextBundle
    from core.video_qa_local_run import (
        VideoQALMHttpTarget,
        VideoQALocalRunOptions,
    )
    from core.whisperx_wrapper import WhisperXWrapper

logger = logging.getLogger(__name__)


class VideoQALocalRunWorker(QObject):
    """Runs :func:`run_local_video_qa` on a worker thread."""

    progress = Signal(float, str)
    pipeline_log_line = Signal(str)
    finished = Signal(object)
    error = Signal(str)
    canceled = Signal()

    def __init__(  # noqa: PLR0913
        self,
        *,
        context: VideoQAContextBundle,
        output_dir: Path,
        context_window_tokens: int,
        frame_sample_fps: float,
        whisper: WhisperXWrapper,
        chunk_lm: VideoQALMHttpTarget,
        final_lm: VideoQALMHttpTarget,
        video_chunking_enabled: bool = True,
        run_options: VideoQALocalRunOptions | None = None,
    ) -> None:
        super().__init__()
        resolved_run = run_options or default_video_qa_local_run_options()
        self._params = VideoQALocalRunParams(
            context=context,
            output_dir=output_dir,
            context_window_tokens=context_window_tokens,
            chunk_lm=chunk_lm,
            final_lm=final_lm,
            frame_sample_fps=frame_sample_fps,
            video_chunking_enabled=video_chunking_enabled,
            run_options=resolved_run,
        )
        self._whisper = whisper
        self._cancel = False

    def request_cancel(self) -> None:
        """Request cooperative cancellation (checked during ASR and between chunks)."""
        self._cancel = True

    @Slot()
    def run(self) -> None:
        """Execute the Video QA pipeline and emit the outcome or an error signal."""

        def _progress(msg: str, frac: float) -> None:
            self.progress.emit(float(frac), msg)

        def _pipeline_log(line: str) -> None:
            self.pipeline_log_line.emit(line)

        logger.info("Video QA worker thread run() started")
        _pipeline_log("→ Stage: worker_thread (run_local_video_qa)")
        try:
            outcome = run_local_video_qa(
                params=self._params,
                whisper=self._whisper,
                should_cancel=lambda: self._cancel,
                progress=_progress,
                pipeline_log=_pipeline_log,
            )
            self.finished.emit(outcome)
        except VideoQAPreflightBlockedError as exc:
            self.error.emit(str(exc))
        except CancelledError:
            self.canceled.emit()
        except RuntimeError as exc:
            if str(exc) == "Canceled":
                self.canceled.emit()
            else:
                tb = traceback.format_exc()
                logger.exception("Video QA RuntimeError (non-cancel)")
                _pipeline_log(tb)
                self.error.emit(str(exc))
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("Video QA worker failed before normal completion")
            _pipeline_log(tb)
            self.error.emit(f"{exc!s}\n(see progress log above for traceback)")
