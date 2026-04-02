import contextlib
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QTableWidgetItem
from pytestqt.qtbot import QtBot

from core.ffmpeg import get_media_duration_seconds
from core.lm_studio_rest import (
    lm_studio_llm_loaded_instance_count,
    lm_studio_unload_all_llm_instances,
    openai_chat_base_to_local_rest_root,
)
from core.video_qa_answer_bundle import answer_bundle_path_for_manifest
from core.video_qa_local_run import DEFAULT_LM_STUDIO_OPENAI_BASE_URL
from core.video_qa_orchestration import build_video_qa_chunk_plan
from core.whisperx_wrapper import fw_whisper_cls as _live_fw_whisper_cls
from gui.main_window import MainWindow
from gui.video_qa_worker import VideoQALocalRunWorker
from utils.exporters import extract_askvlm_metadata_from_srt

_FIXTURE_SHORT_MP4 = Path(__file__).parent / "fixtures" / "test_video_short.mp4"
_LM_STUDIO_MODELS_URL = "http://127.0.0.1:1234/v1/models"
_LIVE_E2E_VIDEO_QA_MODEL_ENV = "ASKVLM_TEST_LIVE_VIDEO_QA_MODEL_ID"


def _select_inputs(tmp_path: Path, fixtures_dir: Path, num_videos: int) -> list[Path]:
    short = fixtures_dir / "test_video_short.mp4"
    if not short.is_file():
        pytest.skip("Short fixture not found")
    inputs: list[Path] = []
    for index in range(num_videos):
        copy_path = tmp_path / "e2e_inputs" / f"input_{index}.mp4"
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(short, copy_path)
        inputs.append(copy_path)
    return inputs


def _create_subtitle_live_window(
    qtbot: QtBot,
    out_dir: Path,
    inputs: list[Path],
) -> MainWindow:
    """Build a main window configured for one successful subtitle/transcript GUI run."""
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    window.out_dir_edit.setText(str(out_dir))
    window.chk_diar.setChecked(False)
    window.chk_dialog.setChecked(False)
    window.chk_save_srt.setChecked(True)
    window.format_combo.setCurrentText("txt")
    qtbot.mouseClick(window.btn_quality, Qt.LeftButton)

    for media_path in inputs:
        window.last_input_dir = media_path.parent
        row = window.input_list.rowCount()
        window.input_list.insertRow(row)
        window.input_list.setItem(row, 1, QTableWidgetItem(str(media_path)))

    return window


def _force_uniform_video_qa_chunk_plan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    segment_seconds: float,
) -> None:
    """Force deterministic uniform chunking for both GUI preflight and worker reruns."""
    original = build_video_qa_chunk_plan

    def _forced_chunk_plan(
        duration_seconds: float,
        *,
        scene_spans: object = None,
        uniform_segment_seconds: float = 30.0,
    ) -> object:
        _ = (scene_spans, uniform_segment_seconds)
        return original(
            duration_seconds,
            scene_spans=None,
            uniform_segment_seconds=segment_seconds,
        )

    monkeypatch.setattr(
        "core.video_qa_orchestration.build_video_qa_chunk_plan",
        _forced_chunk_plan,
    )


def _require_live_subtitle_prereqs() -> None:
    """Skip when live subtitle GUI prerequisites are missing."""
    if not _FIXTURE_SHORT_MP4.is_file():
        pytest.skip(f"Fixture not found: {_FIXTURE_SHORT_MP4}")
    if _live_fw_whisper_cls is None:
        pytest.skip("Live subtitle GUI E2E requires faster-whisper to be installed.")
    try:
        duration_s = float(get_media_duration_seconds(_FIXTURE_SHORT_MP4))
    except OSError:
        pytest.skip(
            "Live subtitle GUI E2E requires ffmpeg/ffprobe to read the short fixture."
        )
    if duration_s <= 0.0:
        pytest.skip(
            "Live subtitle GUI E2E requires ffmpeg/ffprobe to report a non-zero "
            "duration for the short fixture."
        )


def _create_video_qa_live_window(
    qtbot: QtBot,
    out_dir: Path,
    media_path: Path,
) -> MainWindow:
    """Build MainWindow on the real Video QA tab with source and question."""
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    window.out_dir_edit.setText(str(out_dir))
    window.shell_tabs.setCurrentIndex(1)
    # * Large token budget so the short fixture passes preflight offline.
    window.video_qa_panel.set_context_window_tokens(200_000)
    window.video_qa_panel.set_source_path(media_path)
    window.video_qa_panel.set_question_text(
        "E2E live probe: describe one visible action in this clip.",
    )
    window.video_qa_panel.refresh_preflight()
    return window


def _video_qa_thread_running_or_terminal_status(window: MainWindow) -> bool:
    """Return True when Video QA is in-flight or has a terminal status message."""
    thread = window._video_qa_thread  # noqa: SLF001
    if thread is not None and thread.isRunning():
        return True
    msg = window.status.currentMessage()
    return (
        "Video QA error" in msg
        or "Video QA completed" in msg
        or "Video QA canceled" in msg
    )


def _lm_studio_http_reachable() -> bool:
    """Return True if LM Studio responds on the OpenAI-compatible models endpoint."""
    try:
        with urllib.request.urlopen(_LM_STUDIO_MODELS_URL, timeout=2.0) as response:  # noqa: S310
            response.read()
    except OSError:
        return False
    return True


def _require_live_video_qa_prereqs() -> None:
    """Skip when live Video QA prerequisites (fixture, ASR, LM Studio, ffmpeg) are missing."""
    if not _FIXTURE_SHORT_MP4.is_file():
        pytest.skip(f"Fixture not found: {_FIXTURE_SHORT_MP4}")

    if _live_fw_whisper_cls is None:
        pytest.skip("Live Video QA E2E requires faster-whisper to be installed.")
    if not _lm_studio_http_reachable():
        pytest.skip("Live Video QA E2E requires LM Studio on 127.0.0.1:1234.")
    rest_root = openai_chat_base_to_local_rest_root(DEFAULT_LM_STUDIO_OPENAI_BASE_URL)
    if rest_root is not None:
        # * Drop LM Studio weights from VRAM before CUDA Whisper loads (same GPU policy).
        lm_studio_unload_all_llm_instances(rest_root)
    try:
        duration_s = float(get_media_duration_seconds(_FIXTURE_SHORT_MP4))
    except OSError:
        pytest.skip(
            "Live Video QA E2E requires ffmpeg/ffprobe to read the short fixture."
        )
    if duration_s <= 0.0:
        pytest.skip(
            "Live Video QA E2E requires ffmpeg/ffprobe to report a non-zero "
            "duration for the short fixture."
        )
    if duration_s <= 10.0:
        pytest.skip(
            "Live Video QA E2E expects the short fixture to stay above 10 seconds "
            "so forced 10s planning yields more than one chunk."
        )


def _video_qa_progress_log_text(window: MainWindow) -> str:
    """Return the current Video QA progress log text."""
    return window.video_qa_panel.progress_log_edit.toPlainText()


def _video_qa_status(window: MainWindow) -> str:
    """Return the current Video QA status bar text."""
    return window.status.currentMessage()


def _video_qa_local_model_ids(window: MainWindow) -> tuple[str, ...]:
    """Return distinct non-placeholder LM Studio model ids from the GUI combo."""
    models: list[str] = []
    for idx in range(window.video_qa_panel.chunk_model_combo.count()):
        text = window.video_qa_panel.chunk_model_combo.itemText(idx).strip()
        if not text or text in {"LM Studio not running/reachable", "No models found"}:
            continue
        models.append(text)
    return tuple(dict.fromkeys(models))


def _select_live_video_qa_model(
    window: MainWindow,
    qtbot: QtBot,
) -> str:
    """Populate both Video QA model rows with one local model for a lighter live E2E."""
    qtbot.mouseClick(window.video_qa_panel.btn_refresh_lm_models, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: window.video_qa_panel.chunk_model_combo.count() > 0,
        timeout=30_000,
    )
    models = _video_qa_local_model_ids(window)
    if not models:
        pytest.skip("Live Video QA E2E requires local LM Studio models in the catalog.")
    requested_model = os.environ.get(_LIVE_E2E_VIDEO_QA_MODEL_ENV, "").strip()
    if requested_model:
        if requested_model not in models:
            pytest.skip(
                "Live Video QA E2E requested model "
                f"{requested_model!r} via {_LIVE_E2E_VIDEO_QA_MODEL_ENV}, but LM Studio "
                "did not expose it."
            )
        model_id = requested_model
    else:

        def _model_sort_key(model_id: str) -> tuple[int, int, float, int, str]:
            lower = model_id.lower()
            multimodal_hints = (
                "vl",
                "vision",
                "llava",
                "pixtral",
                "internvl",
                "minicpm-v",
                "gemma-3",
            )
            small_hints = ("nano", "mini", "small")
            size_matches = re.findall(r"(\d+(?:\.\d+)?)\s*b", lower)
            size = (
                min(float(match) for match in size_matches) if size_matches else 999.0
            )
            return (
                0 if any(hint in lower for hint in multimodal_hints) else 1,
                0 if any(hint in lower for hint in small_hints) else 1,
                size,
                len(model_id),
                lower,
            )

        model_id = sorted(models, key=_model_sort_key)[0]
    chunk_idx = window.video_qa_panel.chunk_model_combo.findText(
        model_id, Qt.MatchFlag.MatchExactly
    )
    final_idx = window.video_qa_panel.final_model_combo.findText(
        model_id, Qt.MatchFlag.MatchExactly
    )
    if chunk_idx < 0 or final_idx < 0:
        pytest.skip("Live Video QA E2E could not select the requested local model.")
    window.video_qa_panel.chunk_model_type_combo.setCurrentIndex(0)
    window.video_qa_panel.final_model_type_combo.setCurrentIndex(0)
    window.video_qa_panel.chunk_model_combo.setCurrentIndex(chunk_idx)
    window.video_qa_panel.final_model_combo.setCurrentIndex(final_idx)
    pair = window.video_qa_panel.lm_runtime_settings_pair()
    assert pair.chunk.model_id == model_id
    assert pair.final_answer.model_id == model_id
    return model_id


def _video_qa_backend_capability_issue(log_text: str) -> str | None:
    """Return a skip reason when the live backend is unavailable."""
    for needle, reason in (
        (
            "faster-whisper is not installed",
            "Live Video QA E2E requires faster-whisper to be installed.",
        ),
        (
            "Whisper model failed to load",
            "Live Video QA E2E could not load the Whisper model.",
        ),
        (
            "Connection refused",
            "Live Video QA E2E requires LM Studio on 127.0.0.1:1234.",
        ),
        (
            "No route to host",
            "Live Video QA E2E requires LM Studio on 127.0.0.1:1234.",
        ),
        (
            "Name or service not known",
            "Live Video QA E2E requires LM Studio on 127.0.0.1:1234.",
        ),
        (
            "HTTP 404",
            "Live Video QA E2E requires LM Studio on 127.0.0.1:1234.",
        ),
        (
            "HTTP 503",
            "Live Video QA E2E requires LM Studio on 127.0.0.1:1234.",
        ),
        (
            "Model does not support images",
            "Live Video QA E2E requires a multimodal LM Studio chunk model.",
        ),
        (
            "Model does not exist",
            "Live Video QA E2E requires a loaded LM Studio model for the selected row.",
        ),
        (
            "HTTP 500",
            "Live Video QA E2E hit an LM Studio internal server error.",
        ),
        (
            "Internal Server Error",
            "Live Video QA E2E hit an LM Studio internal server error.",
        ),
    ):
        if needle in log_text:
            return reason
    return None


def _skip_if_live_video_qa_backend_unavailable(
    window: MainWindow,
    log_text: str,
) -> None:
    """Skip after cleanup when the live Video QA backend is unavailable."""
    reason = _video_qa_backend_capability_issue(log_text)
    if reason is None:
        return
    with contextlib.suppress(Exception):
        window.close()
        window.await_worker_shutdown(timeout_ms=30000)
    pytest.skip(reason)


def _assert_lm_studio_at_most_one_llm_loaded() -> None:
    """Fail when LM Studio reports more than one loaded LLM instance (localhost control API)."""
    rest_root = openai_chat_base_to_local_rest_root(DEFAULT_LM_STUDIO_OPENAI_BASE_URL)
    if rest_root is None:
        return
    count = lm_studio_llm_loaded_instance_count(rest_root)
    if count is None:
        return
    assert count <= 1, (
        "Expected at most one loaded LLM instance in LM Studio after the scenario; "
        f"reported {count}. (If prompts alone hot-swap models, unload/load handoffs should "
        "still keep a single active LLM.)"
    )


def _assert_live_video_qa_live_run_completion_after_pre_vlm(
    window: MainWindow,
    qtbot: QtBot,
    out_dir: Path,
    *,
    expected_chunk_count: int,
) -> None:
    """After pre-VLM, wait for completion and validate the successful multi-chunk run."""
    assert window.progress.maximum() == 200
    assert window.progress.value() >= 100
    qtbot.waitUntil(
        lambda: _video_qa_progress_log_text(window).count("→ llm_pass:")
        >= expected_chunk_count
        or _video_qa_status(window)
        in {
            "Video QA completed",
            "Video QA error",
            "Video QA canceled",
        },
        timeout=600000,
    )
    log_text = _video_qa_progress_log_text(window)
    if _video_qa_status(window) == "Video QA error":
        _skip_if_live_video_qa_backend_unavailable(window, log_text)
        pytest.fail(
            "Video QA errored before the expected multi-chunk VLM passes completed.\n"
            f"Log:\n{log_text}"
        )
    qtbot.waitUntil(
        lambda: _video_qa_status(window)
        in {
            "Video QA completed",
            "Video QA error",
            "Video QA canceled",
        },
        timeout=600000,
    )
    log_text = _video_qa_progress_log_text(window)
    terminal_status = _video_qa_status(window)
    if terminal_status == "Video QA error":
        _skip_if_live_video_qa_backend_unavailable(window, log_text)
        pytest.fail(
            f"Video QA errored unexpectedly before completion.\nLog:\n{log_text}"
        )
    assert terminal_status == "Video QA completed"
    log_text = _video_qa_progress_log_text(window)
    assert "→ Stage: run_local_video_qa (preflight + executor)" in log_text
    assert "→ Stage: executor (transcript → chunks → synthesis)" in log_text
    assert "→ Stage: VLM inference (post pre-VLM)" in log_text
    assert log_text.count("→ llm_pass:") >= expected_chunk_count
    assert "Traceback" not in log_text

    manifest_files = sorted(out_dir.glob("*.manifest.json"))
    assert len(manifest_files) == 1
    manifest_path = manifest_files[0]
    answer_path = answer_bundle_path_for_manifest(manifest_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_payload.get("status") != "completed":
        failure_text = "\n".join(
            [
                _video_qa_progress_log_text(window),
                json.dumps(manifest_payload, ensure_ascii=False),
            ]
        )
        _skip_if_live_video_qa_backend_unavailable(window, failure_text)
        pytest.fail(
            "Video QA completed the GUI path but the backend run failed.\n"
            f"Manifest:\n{json.dumps(manifest_payload, indent=2, ensure_ascii=False)}"
        )
    assert answer_path.is_file()
    answer_payload = json.loads(answer_path.read_text(encoding="utf-8"))
    assert len(manifest_payload["chunks"]) == expected_chunk_count
    assert manifest_payload["question"] == (
        "E2E live probe: describe one visible action in this clip."
    )
    assert all(chunk["status"] == "completed" for chunk in manifest_payload["chunks"])
    assert answer_payload["manifest_run_id"] == manifest_payload["run_id"]
    assert answer_payload["question"] == manifest_payload["question"]
    assert str(answer_payload["answer"]).strip()
    assert window.video_qa_panel.answer_text().strip()
    assert "✓ Pipeline completed (manifest + answer bundle saved)" in log_text


def _make_dummy_thread(*, should_stop: bool) -> SimpleNamespace:
    thread = SimpleNamespace(quitted=False)

    def _is_running() -> bool:
        return True

    def _quit() -> None:
        thread.quitted = True

    def _wait(_timeout_ms: int) -> bool:
        return should_stop

    thread.isRunning = _is_running
    thread.quit = _quit
    thread.wait = _wait
    return thread


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.heavy_ml
@pytest.mark.xdist_group(name="ml_singleton")
def test_e2e_subtitles_live_gui_run_completes(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the real subtitle GUI to a successful transcript + subtitle export."""
    _require_live_subtitle_prereqs()
    temp_settings_path = tmp_path / "test_settings.ini"

    class MockSettings(QSettings):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__(str(temp_settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr("gui.main_window.QSettings", MockSettings)

    fixtures_dir = Path(__file__).parent / "fixtures"
    inputs = _select_inputs(tmp_path, fixtures_dir, 1)
    media_path = inputs[0]

    out_dir = tmp_path / "e2e_subtitles_live"
    out_dir.mkdir()

    window = _create_subtitle_live_window(qtbot, out_dir, inputs)
    txt_path = out_dir / f"{media_path.stem}.txt"
    srt_path = out_dir / f"{media_path.stem}.srt"
    try:
        qtbot.mouseClick(window.btn_start, Qt.LeftButton)
        qtbot.waitUntil(
            lambda: window._thread is not None and window._thread.isRunning(),  # noqa: SLF001
            timeout=60000,
        )
        qtbot.waitUntil(
            lambda: (txt_path.is_file() and srt_path.is_file())
            or window.status.currentMessage() in {"Error", "Canceled"},
            timeout=600000,
        )
        qtbot.waitUntil(
            lambda: window._thread is None,  # noqa: SLF001
            timeout=600000,
        )

        assert window.status.currentMessage() == "Done"
        assert txt_path.read_text(encoding="utf-8").strip()
        srt_text = srt_path.read_text(encoding="utf-8")
        assert srt_text.strip()
        meta = extract_askvlm_metadata_from_srt(srt_text)
        assert isinstance(meta, dict)
        assert meta.get("completed") is True
        assert str(meta.get("quality", "")).strip() == "fast"

        window.close()
        assert window.await_worker_shutdown(timeout_ms=30000)
    finally:
        with contextlib.suppress(Exception):
            window.close()
            window.await_worker_shutdown(timeout_ms=30000)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.heavy_ml
@pytest.mark.xdist_group(name="ml_singleton")
def test_e2e_video_qa_real_gui_run_completes(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the real Video QA GUI through pre-LLM and completion.

    Uses the short committed fixture with forced 10s planning so the live path executes
    more than one chunk on a deterministic fixture.
    """
    _require_live_video_qa_prereqs()
    _force_uniform_video_qa_chunk_plan(monkeypatch, segment_seconds=10.0)
    temp_settings_path = tmp_path / "test_settings.ini"

    class MockSettings(QSettings):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__(str(temp_settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr("gui.main_window.QSettings", MockSettings)

    fixtures_dir = Path(__file__).parent / "fixtures"
    inputs = _select_inputs(tmp_path, fixtures_dir, 1)
    media_path = inputs[0]

    out_dir = tmp_path / "e2e_vqa_out_live"
    out_dir.mkdir()

    window = _create_video_qa_live_window(qtbot, out_dir, media_path)
    try:
        preflight_text = window.video_qa_panel.preflight_edit.toPlainText()
        assert "Chunks: 2" in preflight_text
        model_id = _select_live_video_qa_model(window, qtbot)
        run_btn = window.video_qa_panel.btn_run_qa
        assert run_btn.isEnabled()

        qtbot.mouseClick(run_btn, Qt.LeftButton)
        qtbot.waitUntil(
            lambda: _video_qa_thread_running_or_terminal_status(window),
            timeout=600000,
        )
        assert isinstance(window._video_qa_worker, VideoQALocalRunWorker)  # noqa: SLF001

        qtbot.waitUntil(
            lambda: (
                "→ Stage: VLM inference (post pre-VLM)"
                in _video_qa_progress_log_text(window)
                or _video_qa_status(window)
                in {
                    "Video QA completed",
                    "Video QA error",
                    "Video QA canceled",
                }
            ),
            timeout=600000,
        )
        log_text = _video_qa_progress_log_text(window)
        if "→ Stage: VLM inference (post pre-VLM)" not in log_text:
            terminal_status = _video_qa_status(window)
            if terminal_status == "Video QA error":
                _skip_if_live_video_qa_backend_unavailable(window, log_text)
            pytest.fail(
                "Video QA did not reach the pre-LLM boundary before stopping. "
                f"Status: {terminal_status}\nLog:\n{log_text}"
            )

        _assert_live_video_qa_live_run_completion_after_pre_vlm(
            window,
            qtbot,
            out_dir,
            expected_chunk_count=2,
        )
        assert (
            window.video_qa_panel.lm_runtime_settings_pair().chunk.model_id == model_id
        )
        _assert_lm_studio_at_most_one_llm_loaded()

        window.close()
        assert window.await_worker_shutdown(timeout_ms=30000)
    finally:
        with contextlib.suppress(Exception):
            window.close()
            window.await_worker_shutdown(timeout_ms=30000)


@pytest.mark.parametrize(
    ("shutdown_result", "expected_accept"),
    [(False, False), (True, True)],
)
def test_close_event_respects_shutdown_result(
    qapp: QApplication,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shutdown_result: bool,
    expected_accept: bool,
) -> None:
    """CloseEvent keeps the window open until shutdown completes."""
    monkeypatch.setattr(MainWindow, "_load_settings", lambda _self: None)
    monkeypatch.setattr(MainWindow, "_save_settings", lambda _self: None)

    window = MainWindow()
    qtbot.addWidget(window)

    class DummyWorker:
        def __init__(self) -> None:
            self.closing = False

        def set_closing(self) -> None:
            self.closing = True

    dummy_worker = DummyWorker()
    dummy_thread = _make_dummy_thread(should_stop=shutdown_result)
    monkeypatch.setattr(window, "_worker", dummy_worker)
    monkeypatch.setattr(window, "_thread", dummy_thread)
    monkeypatch.setattr(window, "_burn_thread", None)
    cancel_calls: list[bool] = []
    monkeypatch.setattr(window, "request_cancel", lambda: cancel_calls.append(True))

    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted() is expected_accept
    assert dummy_worker.closing is True
    assert cancel_calls == [True]


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
