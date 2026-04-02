import contextlib
import json
import shutil
import sys
import time
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
from core.whisperx_wrapper import fw_whisper_cls as _live_fw_whisper_cls
from gui.main_window import MainWindow
from gui.video_qa_worker import VideoQALocalRunWorker


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


def _create_subtitle_crash_window(
    qtbot: QtBot,
    out_dir: Path,
    inputs: list[Path],
    quality: str,
) -> MainWindow:
    """Build a main window configured for the Text + Subtitles transcribe path."""
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    window.out_dir_edit.setText(str(out_dir))
    window.chk_diar.setChecked(False)
    window.chk_dialog.setChecked(False)
    if quality == "fast":
        qtbot.mouseClick(window.btn_quality, Qt.LeftButton)

    for media_path in inputs:
        window.last_input_dir = media_path.parent
        row = window.input_list.rowCount()
        window.input_list.insertRow(row)
        window.input_list.setItem(row, 1, QTableWidgetItem(str(media_path)))

    return window


_LM_STUDIO_MODELS_URL = "http://127.0.0.1:1234/v1/models"

# * Live Video QA GUI E2E uses fixed LM Studio model ids to exercise the dual-row swap path.
# * Chunk vision pass: ``qwen/qwen3.5-35b-a3b``. Final synthesis: ``nvidia/nemotron-3-nano-4b``
# * (picked as a recent small model so the unload/load handoff stays real without loading a
# * second full 35B). This pairing exists only to test REST load/unload wiring; it is not a
# * product recommendation—real runs often keep one strong multimodal model (e.g. the same
# * Qwen 35B) for both chunks and the final answer, or a larger model for synthesis.
_LIVE_E2E_CHUNK_MODEL_ID = "qwen/qwen3.5-35b-a3b"
_LIVE_E2E_FINAL_MODEL_ID = "nvidia/nemotron-3-nano-4b"


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
    qtbot.mouseClick(window.btn_quality, Qt.LeftButton)
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
    fixture_path = Path(__file__).parent / "fixtures" / "test_video_short.mp4"
    if not fixture_path.is_file():
        pytest.skip(f"Fixture not found: {fixture_path}")

    if _live_fw_whisper_cls is None:
        pytest.skip("Live Video QA E2E requires faster-whisper to be installed.")
    if not _lm_studio_http_reachable():
        pytest.skip("Live Video QA E2E requires LM Studio on 127.0.0.1:1234.")
    rest_root = openai_chat_base_to_local_rest_root(DEFAULT_LM_STUDIO_OPENAI_BASE_URL)
    if rest_root is not None:
        # * Drop LM Studio weights from VRAM before CUDA Whisper loads (same GPU policy).
        lm_studio_unload_all_llm_instances(rest_root)
    try:
        duration_s = float(get_media_duration_seconds(fixture_path))
    except OSError:
        pytest.skip(
            "Live Video QA E2E requires ffmpeg/ffprobe to read the short fixture."
        )
    if duration_s <= 0.0:
        pytest.skip(
            "Live Video QA E2E requires ffmpeg/ffprobe to report a non-zero "
            "duration for the short fixture."
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


def _select_distinct_local_models(
    window: MainWindow,
    qtbot: QtBot,
) -> tuple[str, str]:
    """Populate both Video QA model rows using fixed ids when LM Studio exposes them."""
    qtbot.mouseClick(window.video_qa_panel.btn_refresh_lm_models, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: window.video_qa_panel.chunk_model_combo.count() > 0,
        timeout=30_000,
    )
    models = _video_qa_local_model_ids(window)
    if not models:
        pytest.skip("Live Video QA E2E requires local LM Studio models in the catalog.")
    chunk_model = _LIVE_E2E_CHUNK_MODEL_ID
    final_model = _LIVE_E2E_FINAL_MODEL_ID
    if chunk_model not in models:
        pytest.skip(
            f"Live Video QA E2E requires chunk model {chunk_model!r} in LM Studio."
        )
    if final_model not in models:
        pytest.skip(
            f"Live Video QA E2E requires final model {final_model!r} in LM Studio."
        )
    chunk_idx = window.video_qa_panel.chunk_model_combo.findText(
        chunk_model, Qt.MatchFlag.MatchExactly
    )
    final_idx = window.video_qa_panel.final_model_combo.findText(
        final_model, Qt.MatchFlag.MatchExactly
    )
    if chunk_idx < 0 or final_idx < 0:
        pytest.skip("Live Video QA E2E could not select the requested local models.")
    window.video_qa_panel.chunk_model_type_combo.setCurrentIndex(0)
    window.video_qa_panel.final_model_type_combo.setCurrentIndex(0)
    window.video_qa_panel.chunk_model_combo.setCurrentIndex(chunk_idx)
    window.video_qa_panel.final_model_combo.setCurrentIndex(final_idx)
    pair = window.video_qa_panel.lm_runtime_settings_pair()
    assert pair.chunk.model_id == chunk_model
    assert pair.final_answer.model_id == final_model
    assert chunk_model != final_model
    return chunk_model, final_model


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
    chunk_model: str,
    final_model: str,
) -> None:
    """After pre-VLM, wait for LM Studio swap + completion and validate manifest/answer."""
    assert window.progress.maximum() == 200
    assert window.progress.value() >= 100
    assert chunk_model != final_model
    qtbot.waitUntil(
        lambda: "LM Studio REST: unload chunk model"
        in _video_qa_progress_log_text(window)
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
            "Video QA errored before the first LM Studio unload/load handoff.\n"
            f"Log:\n{log_text}"
        )
    assert "LM Studio REST: unload chunk model" in log_text
    qtbot.waitUntil(
        lambda: "LM Studio REST: loaded final model"
        in _video_qa_progress_log_text(window)
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
            "Video QA errored before LM Studio reported the final model load.\n"
            f"Log:\n{log_text}"
        )
    assert "LM Studio REST: loaded final model" in log_text

    qtbot.waitUntil(
        lambda: "→ llm_pass:" in _video_qa_progress_log_text(window)
        or _video_qa_status(window)
        in {
            "Video QA completed",
            "Video QA error",
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
    assert "→ llm_pass:" in log_text
    qtbot.waitUntil(
        lambda: _video_qa_status(window) == "Video QA completed",
        timeout=600000,
    )
    log_text = _video_qa_progress_log_text(window)
    assert "→ Stage: run_local_video_qa (preflight + executor)" in log_text
    assert "→ Stage: executor (transcript → chunks → synthesis)" in log_text
    assert "→ Stage: VLM inference (post pre-VLM)" in log_text
    assert "→ llm_pass:" in log_text

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


# * E2E crash detection covers subtitle shutdown; Video QA has separate live GUI E2E.
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.heavy_ml
@pytest.mark.xdist_group(name="ml_singleton")
@pytest.mark.parametrize("quality", ["fast", "good"])
@pytest.mark.parametrize("num_videos", [1, 2])
@pytest.mark.parametrize("strategy", ["implicit_cancel", "explicit_cancel"])
def test_e2e_subtitles_crash_scenarios(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quality: str,
    num_videos: int,
    strategy: str,
) -> None:
    """Parametric E2E for crash/hang detection on exit via the subtitle transcribe path.

    Covers Fast and Good modes, single and multi-file inputs, and implicit and
    explicit cancellation (main input table + Start Transcribe).
    """
    temp_settings_path = tmp_path / "test_settings.ini"

    class MockSettings(QSettings):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__(str(temp_settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr("gui.main_window.QSettings", MockSettings)

    fixtures_dir = Path(__file__).parent / "fixtures"
    inputs = _select_inputs(tmp_path, fixtures_dir, num_videos)

    out_dir = tmp_path / f"e2e_out_{quality}_{num_videos}_{strategy}"
    out_dir.mkdir()

    window = _create_subtitle_crash_window(qtbot, out_dir, inputs, quality)

    qtbot.mouseClick(window.btn_start, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: "Transcribing" in window.status.currentMessage(),
        timeout=60000,
    )
    time.sleep(2.0)

    if strategy == "explicit_cancel":
        qtbot.mouseClick(window.btn_cancel, Qt.LeftButton)
        time.sleep(0.5)

    window.close()
    assert window.await_worker_shutdown(timeout_ms=30000)


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

    Uses a two-model swap only to validate unload/load plumbing; production may use one
    strong multimodal model for both stages (see ``_LIVE_E2E_*`` comments).
    """
    _require_live_video_qa_prereqs()
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
        chunk_model, final_model = _select_distinct_local_models(window, qtbot)
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
            chunk_model=chunk_model,
            final_model=final_model,
        )
        _assert_lm_studio_at_most_one_llm_loaded()

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
def test_e2e_video_qa_shutdown_after_pre_llm_boundary(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the real Video QA GUI to the pre-LLM boundary and shut down cleanly.

    Same fixed two-model selection as ``test_e2e_video_qa_real_gui_run_completes``; see
    ``_LIVE_E2E_*`` module comments for why this is test-only wiring.
    """
    _require_live_video_qa_prereqs()
    temp_settings_path = tmp_path / "test_settings.ini"

    class MockSettings(QSettings):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__(str(temp_settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr("gui.main_window.QSettings", MockSettings)

    fixtures_dir = Path(__file__).parent / "fixtures"
    inputs = _select_inputs(tmp_path, fixtures_dir, 1)
    media_path = inputs[0]

    out_dir = tmp_path / "e2e_vqa_shutdown_out_live"
    out_dir.mkdir()

    window = _create_video_qa_live_window(qtbot, out_dir, media_path)
    try:
        chunk_model, final_model = _select_distinct_local_models(window, qtbot)
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
                "Video QA did not reach the pre-LLM boundary before shutdown. "
                f"Status: {terminal_status}\nLog:\n{log_text}"
            )

        assert chunk_model != final_model
        assert window.progress.maximum() == 200
        assert window.progress.value() >= 100
        assert "→ Stage: run_local_video_qa (preflight + executor)" in log_text
        assert "→ Stage: executor (transcript → chunks → synthesis)" in log_text
        assert "→ Stage: VLM inference (post pre-VLM)" in log_text
        assert "LM Studio REST: loaded final model" not in log_text
        if _video_qa_status(window) == "Video QA error":
            _skip_if_live_video_qa_backend_unavailable(window, log_text)
            pytest.fail(
                f"Video QA errored unexpectedly during shutdown E2E.\nLog:\n{log_text}"
            )

        window.close()
        assert window.await_worker_shutdown(timeout_ms=30000)
        assert _video_qa_status(window) != "Video QA completed"
        assert "✓ Pipeline completed (manifest + answer bundle saved)" not in (
            _video_qa_progress_log_text(window)
        )
        assert not list(out_dir.glob("*.manifest.json"))
        assert not list(out_dir.glob("*.answer.json"))
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
