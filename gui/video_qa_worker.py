"""Background QObject worker for local Video QA (ASR + ffmpeg + LM Studio + aggregate)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from core.pipelines import CancelledError
from core.video_qa_local_run import (
    VideoQALocalRunParams,
    VideoQAPreflightBlockedError,
    run_local_video_qa,
)

if TYPE_CHECKING:
    from pathlib import Path

    from core.video_qa_context import VideoQAContextBundle
    from core.whisperx_wrapper import WhisperXWrapper


class VideoQALocalRunWorker(QObject):
    """Runs :func:`run_local_video_qa` on a worker thread."""

    progress = Signal(float, str)
    finished = Signal(object)
    error = Signal(str)
    canceled = Signal()

    def __init__(
        self,
        *,
        context: VideoQAContextBundle,
        output_dir: Path,
        context_window_tokens: int,
        frame_sample_fps: float,
        whisper: WhisperXWrapper,
        lm_base_url: str,
        lm_model_id: str,
    ) -> None:
        super().__init__()
        self._params = VideoQALocalRunParams(
            context=context,
            output_dir=output_dir,
            context_window_tokens=context_window_tokens,
            lm_base_url=lm_base_url,
            lm_model_id=lm_model_id,
            frame_sample_fps=frame_sample_fps,
        )
        self._whisper = whisper
        self._cancel = False

    def request_cancel(self) -> None:
        """Request cooperative cancellation (checked during ASR and between chunks)."""
        self._cancel = True

    def run(self) -> None:
        """Execute the Video QA pipeline and emit the outcome or an error signal."""

        def _progress(msg: str, frac: float) -> None:
            self.progress.emit(float(frac), msg)

        try:
            outcome = run_local_video_qa(
                params=self._params,
                whisper=self._whisper,
                should_cancel=lambda: self._cancel,
                progress=_progress,
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
                self.error.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
