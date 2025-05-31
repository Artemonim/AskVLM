# TODO - Artemonim's Speech Kit

## Phase 1: Core Structure and Local Processing MVP

-   [ ] **Project Setup**
    -   [x] Initialize `TODO.md`
    -   [ ] Create directory structure (`core`, `gui`, `editing`, `utils`, `tests`)
    -   [ ] Create empty Python files for each module
    -   [ ] Setup `.gitignore`
    -   [ ] Create `main.py` entry point
    -   [ ] Create basic `README.md`
-   [ ] **Core Modules - Local Processing**
    -   [ ] `core/ffmpeg.py`: Basic FFmpeg wrapper for audio extraction (WAV 16kHz mono).
    -   [ ] `core/audio_io.py`: Implement audio extraction and resampling using `ffmpeg.py`.
    -   [ ] `core/whisper_wrapper.py`: Integrate the existing Whisper script.
        -   [ ] Define interface for transcription.
    -   [ ] `core/diarization.py`: Basic `pyannote.audio` pipeline setup.
        -   [ ] VAD + Diarization.
        -   [ ] Define interface for speaker identification.
        -   [ ] Plan for ONNX/TorchScript optimization.
    -   [ ] `core/llm_formatter.py`: Basic `llama-cpp-python` wrapper for Mistral-3B (GGUF).
        -   [ ] Text formatting (punctuation, paragraphs).
        -   [ ] Define interface for formatting.
    -   [ ] `core/gpu_guard.py`:
        -   [ ] Implement `acquire(model_name)` logic.
        -   [ ] Basic VRAM check (nvml).
        -   [ ] Model unloading (`del`, `torch.cuda.empty_cache()`, `llama.reset()`).
        -   [ ] Context manager (`__enter__`, `__exit__`) for heavy models.
    -   [ ] `editing/text_model.py`:
        -   [ ] Define data structures for text, speakers, timestamps (e.g., using dataclasses).
        -   [ ] Representation for paragraphs with speaker IDs and timecodes.
    -   [ ] `core/pipelines.py`:
        -   [ ] `LocalPipeline`: Orchestrate local FFmpeg -> Whisper -> Pyannote -> LLM.
        -   [ ] Logic to combine transcription and diarization based on timestamps.
-   [ ] **Basic GUI - Qt (PySide6)**
    -   [ ] `gui/main_window.py`:
        -   [ ] Basic window structure.
        -   [ ] Menu bar (File -> Open, File -> Exit).
        -   [ ] Button/mechanism to trigger local processing pipeline.
        -   [ ] Status bar for progress and messages.
    -   [ ] `gui/wysiwyg_editor.py`:
        -   [ ] Basic `QTextEdit` for displaying transcribed and formatted text.
    -   [ ] `utils/logging.py`: Setup basic logging.
-   [ ] **Utilities**
    -   [ ] `utils/downloader.py`:
        -   [ ] Basic functions to download models (e.g., Whisper, Mistral GGUF, Pyannote) from HuggingFace/GitHub.
        -   [ ] Placeholder for `models.json` definition.
-   [ ] **Settings**
    -   [ ] Define Pydantic model for `settings.json`.
    -   [ ] Initial `settings.json` structure (e.g., model paths, default mode).
-   [ ] **Packaging (Initial Setup)**
    -   [ ] Basic `PyInstaller` spec file considerations.

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
-   [ ] **Testing**
    -   [ ] `tests/`: Setup `pytest`.
        -   [ ] Unit tests (mock Torch, mock SpeechKit).
        -   [ ] Integration tests (WAV -> JSON comparison).
        -   [ ] GUI autotests (`pyside-tester`, signal/slot checks).
    -   [ ] `tools/benchmark.py` script.
-   [ ] **CI/CD (GitHub Actions)**
    -   [ ] Linting (`ruff`).
    -   [ ] Run tests.
    -   [ ] Build onefile-exe.
    -   [ ] Release: exe + models downloader script.
    -   [ ] Auto-versioning via git tags.
-   [ ] **Documentation**
    -   [ ] `README.md`: Installation, requirements, FAQ.
    -   [ ] Wiki: "How to connect Yandex key", "Text formatting, hotkeys".
    -   [ ] Docstrings (Google Style for Python).
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
