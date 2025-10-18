# TODO - Artemonim's Speech Kit

## Phase 1: Core Structure and Local Processing MVP

-   [x] **Project Setup**
    -   [x] Initialize `TODO.md`
    -   [x] Create directory structure (`core`, `gui`, `editing`, `utils`, `tests`)
    -   [x] Create empty Python files for each module
    -   [x] Setup `.gitignore`
    -   [x] Create `main.py` entry point
    -   [x] Create basic `README.md`
-   [x] **Core Modules - Local Processing**
    -   [x] `core/ffmpeg.py`: Basic FFmpeg wrapper for audio extraction (WAV 16kHz mono).
    -   [x] `core/audio_io.py`: Implement audio extraction and resampling using `ffmpeg.py`.
    -   [x] `core/whisper_wrapper.py`: Integrate the existing Whisper script.
        -   [x] Define interface for transcription.
    -   [ ] `core/diarization.py`: Basic `pyannote.audio` pipeline setup.
        -   [ ] VAD + Diarization.
        -   [ ] Define interface for speaker identification.
        -   [ ] Plan for ONNX/TorchScript optimization.
    -   [ ] `core/llm_formatter.py`: Basic `llama-cpp-python` wrapper for Mistral-3B (GGUF).
        -   [ ] Text formatting (punctuation, paragraphs).
        -   [ ] Define interface for formatting.
    -   [x] `core/gpu_guard.py`:
        -   [x] Implement `acquire(model_name)` logic.
        -   [ ] Basic VRAM check (nvml).
        -   [x] Model unloading (`del`, `torch.cuda.empty_cache()`, `llama.reset()`).
        -   [x] Context manager (`__enter__`, `__exit__`) for heavy models.
    -   [x] `editing/text_model.py`:
        -   [x] Define data structures for text, speakers, timestamps (e.g., using dataclasses).
        -   [x] Representation for paragraphs with speaker IDs and timecodes.
    -   [x] `core/pipelines.py`:
        -   [x] `LocalPipeline`: Orchestrate local FFmpeg -> Whisper -> Pyannote -> LLM.
        -   [x] Logic to combine transcription and diarization based on timestamps.
-   [x] **Basic GUI - Qt (PySide6)**
    -   [x] `gui/main_window.py`:
        -   [x] Basic window structure.
        -   [x] Menu bar (File -> Open, File -> Exit).
        -   [x] Button/mechanism to trigger local processing pipeline.
        -   [x] Status bar for progress and messages.
    -   [x] `gui/wysiwyg_editor.py`:
        -   [x] Basic `QTextEdit` for displaying transcribed and formatted text.
    -   [x] `utils/logging.py`: Setup basic logging.
-   [x] **Utilities**
    -   [x] `utils/downloader.py`:
        -   [x] Basic functions to download models (e.g., Whisper, Mistral GGUF, Pyannote) from HuggingFace/GitHub.
        -   [x] Placeholder for `models.json` definition.
-   [x] **Settings**
    -   [x] Define Pydantic model for `settings.json`.
    -   [x] Initial `settings.json` structure (e.g., model paths, default mode).
-   [ ] **Packaging (Initial Setup)**
    -   [ ] Basic `PyInstaller` spec file considerations.

## Phase 1.5: Migration & Cleanup

-   [x] **Licensing & Dependencies**
    -   [x] Use WhisperX strictly as a dependency (BSD-2-Clause, no code copying).
    -   [x] Add attribution/acknowledgments for WhisperX and referenced models. (README updated)
    -   [x] Integrate Faster-Whisper backend option for FP16/INT8 on 8GB VRAM. (WhisperX wrapper)
    -   [x] Integrate `pyannote.audio` (community pipeline); manage HF token via ENV.
-   [x] **Speach Kit Backend Integration**
    -   [x] Add `core/whisperx_wrapper.py` with load/transcribe/align API.
    -   [x] Implement `core/diarization.py` using `pyannote/speaker-diarization-community-1`.
    -   [x] Implement `core/llm_formatter.py` (llama-cpp, 3B/7B GGUF) for dialog blocks.
    -   [x] Update `core/pipelines.py` to support engine switch (whisper/whisperx).
    -   [x] Add options: diarization on/off, dialog-blocks on/off, batch params.
    -   [x] Exporters: TXT, SRT, VTT, JSON (timestamps, speakers, text).
-   [x] **CLI for Batch Processing**
    -   [x] Add Typer-based CLI in Speach Kit for folders/files.
    -   [x] Support recursive mode, overwrite policy, language, device, formats.
    -   [ ] Progress/ETA using existing utils patterns (no LittleTools dependency).
-   [x] **Decommission Whisper Repos**
    -   [x] Migrate only ideas (no code) from `G:\GitHub\Whisper`.
    -   [x] Migrate only ideas (no code) from `G:\GitHub\WhisperX`.
    -   [x] Remove `Whisper/` after feature parity validation.
    -   [x] Remove `WhisperX/` after feature parity validation.
-   [x] **LittleTools Cleanup**
    -   [x] Remove `littletools_speech` package from LittleTools.
    -   [x] Remove ML deps added only for Whisper from LittleTools.
    -   [x] Ensure `littletools_cli/menu.py` handles missing speech plugin.
    -   [x] Update LittleTools docs to reflect removal of speech plugin.
-   [x] **Secrets & Compliance**
    -   [x] Read HF token from ENV (`HF_TOKEN`), add `.env` to `.gitignore`.
    -   [x] Document gated model acceptance steps (pyannote models).
-   [x] **VRAM & Performance**
    -   [x] Compute type auto-select (FP16 ➜ fallback FP32) with INT8 option.
    -   [ ] Batch size control and OOM handling guidance.
-   [ ] **Testing**
    -   [x] Unit tests for wrappers (whisperx, diarization, formatter) — minimal.
    -   [ ] Integration test: WAV ➜ segments+speakers ➜ dialog blocks export.

## Phase 1.7: Simple GUI MVP (Quick Transcribe)

-   [ ] Minimal end-to-end GUI flow
    -   [ ] `gui/main_window.py`: Quick Transcribe panel
        -   [ ] Pick file OR folder; pick output dir (default: `transcriptions/`)
        -   [ ] Toggles: Diarization on/off, Dialog blocks on/off
        -   [ ] Export format: txt|srt|vtt|json (default: txt)
        -   [ ] Start/Cancel buttons; status bar messages
    -   [ ] Non-blocking execution
        -   [ ] Run `LocalPipeline` in background (QtConcurrent/QThread)
        -   [ ] Disable controls during processing; enable on finish/cancel
    -   [ ] Result viewing
        -   [ ] Open output in `gui/wysiwyg_editor.py` for quick read-only viewing
        -   [ ] Button: Open output folder
    -   [ ] Error handling
        -   [ ] Message dialogs for common failures (unsupported media, OOM hint, no GPU)
    -   [ ] Persistence
        -   [ ] Remember last chosen input/output paths (session-scoped)

-   [ ] Pipeline integration
    -   [ ] Use `core/pipelines.LocalPipeline` with Engine=Auto (faster-whisper; whisperx alignment if available)
    -   [ ] Auto audio prep via `core/audio_io.prepare_audio`

-   [ ] Progress & logs
    -   [ ] Status bar updates; basic step progress (ETA in later phase)
    -   [ ] Optional console log pane (collapsed by default)

-   [ ] Doc
    -   [ ] Update `doc/Main UX.md` with Quick Transcribe flow details

## Phase 2: Advanced Editing and Cloud Integration

-   [ ] **Advanced Editing Features**
    -   [ ] `editing/operations.py`:
        -   [ ] Undo/Redo stack (`QUndoStack`).
        -   [ ] Change speaker for a segment.
        -   [ ] Merge/split segments.
        -   [ ] Delete timestamps.
    -   [ ] `gui/wysiwyg_editor.py` (Enhancements):
        -   [ ] Custom block data for `QTextDocument` (`speaker_id`, `start_ts`, `end_ts`).
        -   [ ] Context menu for block operations (Split, Merge, Delete Timestamps).
        -   [ ] Search/replace.
        -   [ ] Spell checker (`pyspellchecker`).
    -   [ ] `gui/speaker_sidebar.py`:
        -   [ ] `QListView` to display speakers (name, color).
        -   [ ] Mechanism to edit speaker names.
        -   [ ] Drag-resize handles for time-slice adjustments (ruler concept).
        -   [ ] Shortcuts Ctrl+1..9 for speaker change.
-   [ ] **Cloud Integration**
    -   [ ] `core/cloud_speechkit.py`:
        -   [ ] Yandex SpeechKit client (gRPC/REST) for transcription and diarization.
        -   [ ] IAM token authentication from settings.
        -   [ ] Retry policy and request limiting.
    -   [ ] `core/llm_formatter.py`: Add cloud LLM option for formatting.
    -   [ ] `core/pipelines.py`:
        -   [ ] `CloudPipeline`: Orchestrate FFmpeg -> Yandex SpeechKit -> Cloud LLM.
        -   [x] `LocalPipeline`: Orchestrate local FFmpeg -> Whisper -> Pyannote -> LLM.
        -   [ ] Logic to combine transcription and diarization based on timestamps.
    -   [ ] `gui/main_window.py`: Add UI elements to switch between Local/Cloud modes.
    -   [ ] `core/gpu_guard.py`: Logic for when GPU is not available (recommend Cloud).
-   [ ] **Settings Enhancements**
    -   [ ] `gui/preferences_dialog.py`:
        -   [ ] Tab "Models": path to models folder, auto-download checkbox.
        -   [ ] Tab "Processing": Default mode (Local/Cloud).
        -   [ ] Tab "Export": Default export format.
        -   [ ] Tab "Performance": GPU memory for LLM (low-mem/high-perf).
        -   [ ] Tab "General": UI Language (Qt Linguist groundwork).
        -   [ ] Yandex Cloud settings (OAuth token, folder_id).
    -   [ ] Define Pydantic model for `settings.json`.
    -   [ ] Update `settings.json` Pydantic model.
-   [ ] **Export Functionality**
    -   [ ] `gui/export_dialog.py`: Dialog to choose format and options.
    -   [ ] Implement exporters:
        -   [ ] TXT
        -   [ ] DOCX (`python-docx`, with speaker styles)
        -   [ ] ODT (`odfpy`)
        -   [ ] SRT/VTT
        -   [ ] Markdown ("Speaker: text")
        -   [ ] Option: Export audio with overlaid subtitles (FFmpeg + SRT).

## Phase 3: Refinement, Testing, and Deployment

-   [ ] **Model Management**
    -   [ ] `utils/downloader.py`:
        -   [ ] `check()` function for missing weights.
        -   [ ] Background download with `tqdm` progress in status bar.
        -   [ ] `models.json` for versioning.
        -   [ ] GUI for deleting/updating models (in Preferences).
-   [x] **Testing**
    -   [x] `tests/`: Setup `pytest`.
        -   [ ] Unit tests (mock Torch, mock SpeechKit).
        -   [ ] Integration tests (WAV -> JSON comparison).
        -   [ ] GUI autotests (`pyside-tester`, signal/slot checks).
        -   [ ] Enable code coverage requirements (80% minimum coverage).
    -   [ ] `tools/benchmark.py` script.
-   [x] **CI/CD (GitHub Actions)**
    -   [x] Linting (`ruff`).
    -   [x] Run tests.
    -   [ ] Build onefile-exe.
    -   [ ] Release: exe + models downloader script.
    -   [ ] Auto-versioning via git tags.
-   [x] **Documentation**
    -   [x] `README.md`: Installation, requirements, FAQ.
    -   [ ] Wiki: "How to connect Yandex key", "Text formatting, hotkeys".
    -   [x] Docstrings (Google Style for Python).
    -   [ ] Sphinx HTML documentation.
-   [ ] **Packaging & Distribution**
    -   [ ] Finalize `PyInstaller` setup (`--onefile`, `--add-data`).
    -   [ ] Logic for first-run model download to `~/.mytranscriber/models` or `%APPDATA%`.
    -   [ ] System requirements check (CUDA, VRAM via `nvml`) on startup.
-   [ ] **User Experience**
    -   [ ] Internationalization (Qt Linguist).
    -   [ ] Error handling and reporting.
    -   [ ] Polishing UI/UX.

## Future Enhancements (Post-MVP)

-   [ ] Live-preview subtitles over video (`QtMultimedia`).
-   [ ] Grammalecte/RuGPT integration for literary correction.
-   [ ] Collaborative editing (CRDT).
-   [ ] Plugin system for exporters (YouTube chapters, Podcast JSON RSS).

---

**Notes:**

-   Prioritize GPU resource management (`gpu_guard.py`) early.
-   Ensure modularity for easier testing and maintenance.
-   Keep UI responsive using QtConcurrent and/or asyncio for long tasks.
-   Adhere to specified style guides.
-   Security: Remind user about `.gitignore` for `settings.json` if it contains sensitive keys, and use placeholders for keys in code/docs.
-   Use `tqdm` for progress bars in CLI or status bar for GUI.
-   Configuration variables at the beginning of scripts.
-   Better Comments style.

## Current Status

**Phase 1: ~95% Complete** ✅

Core structure and basic local processing pipeline implemented:

-   ✅ All core modules have implementations (diarization needs completion)
-   ✅ GUI main window with file selection and processing capability
-   ✅ Settings system with Pydantic models
-   ✅ CLI entry point for batch processing
-   ✅ Proper project structure with type hints and documentation

**Phase 1.5: ~70% Complete** 🟡

Migration and backend integration partially complete:

-   ✅ WhisperX integration and faster-whisper backend
-   ✅ Typer-based CLI for batch processing
-   ✅ Export functionality (TXT, SRT, VTT, JSON)
-   ✅ Compute type auto-selection for GPU optimization
-   ❌ Diarization implementation (still stub)
-   ❌ HF_TOKEN environment support
-   ❌ Unit tests

**Code Quality Status:** ✅

-   ✅ Ruff Format Check
-   ✅ Ruff Lint
-   ✅ MyPy Type Check
-   ✅ Bandit Security Check
-   ⚠️ Pylint Analysis (import errors for optional ML libraries)
-   ⚠️ Pytest Tests (framework ready, no actual tests)

**Next Steps:**

1. ~~Install dependencies: `make setup-dev`~~ ✅ Completed
2. ~~Run quality checks: `python test.py`~~ ✅ Completed
3. Complete Phase 1.5 (diarization, HF_TOKEN, basic tests)
4. Begin Phase 2 implementation (advanced editing features)
