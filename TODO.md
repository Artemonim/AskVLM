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

-   [x] Runner flags for launch control (SkipLaunch, FastLaunch)
-   [x] Minimal end-to-end GUI flow
    -   [x] `gui/main_window.py`: Quick Transcribe panel
        -   [x] Pick file OR folder; pick output dir (default: `transcriptions/`)
        -   [x] Toggles: Diarization on/off, Dialog blocks on/off
        -   [x] Export format: txt|srt|vtt|json (default: txt)
        -   [x] Start/Cancel buttons; status bar messages
    -   [x] Non-blocking execution
        -   [x] Run `LocalPipeline` in background (QtConcurrent/QThread)
        -   [x] Disable controls during processing; enable on finish/cancel
    -   [x] Result viewing
        -   [x] Open output in `gui/wysiwyg_editor.py` for quick read-only viewing
        -   [x] Button: Open output folder
    -   [x] Error handling
        -   [x] Message dialogs for common failures (unsupported media, OOM hint, no GPU)
    -   [x] Persistence
        -   [x] Remember last chosen input/output paths (session-scoped)

-   [x] Pipeline integration
    -   [x] Use `core/pipelines.LocalPipeline` with Engine=Auto (faster-whisper; whisperx alignment if available)
    -   [x] Auto audio prep via `core/audio_io.prepare_audio`

-   [ ] Progress & logs
    -   [x] Status bar updates; basic step progress (ETA in later phase)
    -   [ ] Optional console log pane (collapsed by default)

-   [ ] Doc
    -   [ ] Update `doc/Main UX.md` with Quick Transcribe flow details

## Phase 1.8: AutoSubtitles & Resolve Workflow

-   [x] Decision: Default artifact is sidecar SRT; burn-in is done in DaVinci Resolve.
-   [x] CLI: `subtitle` command (batch: files/folders) producing SRT with CPS/duration rules.
-   [ ] Optional: watch-folder mode to auto-generate SRT on new exports from Resolve.
-   [x] GUI: quick action "Generate SRT for Resolve" (minimal controls).
-   [x] Documentation: Add `doc/AutoSubtitles.md` (pipeline, RAM↔VRAM, Resolve flow).
-   [ ] **VRAM & Performance**
    -   [ ] RAM residency for Align/Diari via ModelRegistry (CPU-resident, migrate to CUDA).
    -   [ ] Prefetch on idle (Align/Diari CPU load, OS cache warm-up for ASR weights).
    -   [ ] Single heavy-model on GPU guarantee across pipeline (enforce via `gpu_guard`).

## Phase 1.81: Preview
-   [x] Поле ввода длины строки субтитра (по умолчанию 42)
-   [x] Поле ввода количества строк субтитра (по умолчанию 2)
-   [x] **Subtitle preview**
    -   [x] Новая правая часть главного окна приложения
    -   [x] Превью субтитров: Отображает кадр, соответствующий стартовому таймингу выбранного субтитра, и сам субтитр на этом кадре.
    -   [x] Extract frame at subtitle start using FFmpeg (`extract_frame_to_file`)
    -   [x] Layout text according to UI limits (`max_line_width`, `max_line_count`) and display in preview
    -   [x] Preview font selection applied to preview pane
    -   [x] Editor selection → preview wiring (use `WysiwygEditor.get_row_data` and `SubtitlePreview.layout_text`)

## Phase 1.82: Export, Burn & Persistence
-   [x] Export SRT respects layout rules and UI hints (`SubtitleRules`, `export_srt_with_rules`)
-   [x] Inline cancellable burn worker integrated into GUI (`_InlineBurnWorker`) with progress parsing
-   [x] Pass subtitle layout options from UI to pipeline/exporters (`subtitle_max_line_width`, `subtitle_max_lines`)
-   [x] UI persistence: save/load window geometry, splitter state, last input dir, table column widths
-   [x] Windows HF symlink workaround: set `HF_HUB_DISABLE_SYMLINKS=1` (settings and wrapper)
-   [x] Pipeline/CLI defaults updated: default Whisper model `large-v3`, default compute_type `float16`
-   [ ] Review: separate BurnWorker vs inline worker refactor (suggestion)

## Phase 1.83: Progress, Preview Parity, Streaming

-   [x] Richer progress bar with elapsed/ETA and granular steps
-   [x] High-level console logging initialization
-   [x] Font size +/- controls affecting preview and burn-in
-   [x] Preview: robust media-to-tab mapping and frame extraction (bugfix)
-   [x] Preview matches burn styling (font size, outline, autoscale)
-   [x] Stream transcription text to partial file during ASR

## Phase 1.84: Input Status, Autosave, Metadata

-   [x] Input list: status icons (empty/❌/⏩/📄/🔥) per file
-   [x] Move "Line length" controls into Burn row
-   [x] Autosave SRT on editor changes with ASK metadata
-   [x] SRT metadata: tool name, quality (fast/good), completed flag
-   [x] Scan output folder for ASK metadata to pre-mark input statuses

## Phase 1.85: Disfluency Cleanup (Auto)

-   [ ] Core detection & cutlist
    -   [ ] `editing/disfluency.py`: detection of fillers and breaths
        -   [ ] RU filler-words via dictionary + WhisperX alignment (word timestamps)
        -   [ ] Filler-sounds ("э-э", "мм") via transcript tokens + simple audio heuristics
        -   [ ] Breaths: heuristic (VAD + spectral slope/energy, 200–1200 ms) with optional PANNs(AudioSet)
    -   [ ] Cutlist JSON schema: `start_ms`, `end_ms`, `type`, `confidence`, `action`, `margin_before_ms`, `margin_after_ms`
    -   [ ] Audio apply: attenuate or cut with adaptive crossfade via FFmpeg wrapper
    -   [ ] Integrate step into `core/pipelines.py` (optional, after Alignment)
    -   [ ] Exporters: EDL for Resolve + updated SRT/VTT post-edits

-   [ ] UI/Settings/CLI
    -   [ ] `gui/main_window.py`: toggles (remove breaths, sounds, words), mode (attenuate/cut), parameters
    -   [ ] `settings.json`/model: thresholds, durations, margins, per-type limits
    -   [ ] CLI flags in Typer: `--cleanup-disfluencies` with granular options

-   [ ] Quality & tests
    -   [ ] Unit tests for detectors (mock audio, mock alignment)
    -   [ ] Integration test: WAV → cutlist → processed WAV + SRT consistency
    -   [ ] Metrics: per-type precision/recall on a small reviewed set

-   [ ] Data & automation (minimal)
    -   [ ] Auto-label from aligned transcripts: RU filler-words → positive spans; candidate breaths via heuristics
    -   [ ] Manual review subset (30–60 мин): только low-confidence/граничные случаи
    -   [ ] `tools/benchmark.py`: офлайн замер метрик и аудио A/B-отчёт

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
    -   [ ] Update `env.example` with SK_* variables (device, compute_type, prefetch, RAM residency, output, logs).
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
    -   [x] Add `doc/AutoSubtitles.md` (AutoSubtitles pipeline and Resolve workflow).
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
