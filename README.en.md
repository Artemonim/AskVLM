[🇷🇺 Русский](README.ru.md) | [🇬🇧 English](README.md)

# AskVLM

AI-powered desktop toolkit for speech transcription, subtitle generation, and multimodal video analysis (Video QA). Built around local ML pipelines with optional cloud LLM support.

## Overview

AskVLM combines three workflows in a single PySide6-based desktop application:

- **Text mode** — transcribe audio/video to plain text with optional speaker diarization and LLM-based formatting.
- **Subtitle mode** — generate subtitles (SRT/VTT) with readability rules, preview them, and burn into video via FFmpeg.
- **Video QA mode** — ask a natural-language question about a video; AskVLM chunks the video, extracts representative frames, runs ASR, and queries a VLM (LM Studio or OpenRouter) to produce a grounded answer.

## Quick Start

### Prerequisites

- **Python**: 3.11.x (required; 3.12+ is not supported)
- **OS**: Windows (primary), Linux (community)
- **RAM**: 8 GB minimum, 16 GB+ recommended
- **GPU**: NVIDIA with CUDA (optional, CPU fallback available)
- **VRAM**: 4 GB minimum, 8 GB+ recommended for larger models
- **Storage**: 2 GB+ for models and temp files

See [PYTHON_SETUP.md](PYTHON_SETUP.md) for installing or selecting the right Python version.

### Installation

```bash
git clone <repository-url>
cd AskVLM

# Core dependencies
pip install -e .

# ML dependencies (transcription, diarization, LLM formatting)
pip install -e .[ml]

# Development tools (linting, testing)
pip install -e .[dev]
```

### Launch

```bash
# Run quality checks and launch GUI (recommended)
pwsh -File run.ps1

# Launch GUI directly (skip checks)
pwsh -File run.ps1 -FastLaunch
# or
python -m gui.main_window
```

On first run the application creates default settings and downloads required models automatically. If ML dependencies are missing, install them with `pip install -e .[ml]`.

## GUI

The application opens with a mode selector (Text + Subtitles / Video QA) and remembers the last choice between sessions.

### Text + Subtitles

- Choose a single file or an entire folder for batch processing.
- Toggle diarization (speaker identification) and LLM-based dialog formatting.
- Export to TXT, SRT, VTT, or JSON.
- Preview subtitles and burn them into video.
- Real-time progress with cancel support; output goes to `transcriptions/`.

### Video QA

- Provide a local video file (or YouTube URL, experimental).
- Type a natural-language question and optionally attach context files (txt, md, code, images).
- Review the preflight summary (source, chunk count, estimated token budget).
- Run the analysis: AskVLM chunks the video, extracts representative frames, runs ASR via WhisperX, and sends each chunk to a VLM.
- View the grounded Markdown answer and evidence log.
- Supports LM Studio (local) and OpenRouter (cloud) as LLM backends.

![Video QA GUI](doc/media/VideoQA%20GUI.png)

## CLI

AskVLM provides three CLI commands via [Typer](https://typer.tiangolo.com/):

### `transcribe` — batch transcription

```bash
python cli.py transcribe PATH -o output_dir --whisper-model large-v3 --export txt
```

Key options: `--whisper-model`, `--diarization/--no-diarization`, `--dialog-blocks`, `--export` (txt/srt/vtt/json), `--device` (auto/cuda/cpu), `--language`, `--engine` (whisper/whisperx/auto), `--recursive`, `--compute-type`.

### `subtitle` — subtitle generation with burn-in

```bash
python cli.py subtitle PATH -o output_dir --burn-in --whisper-model large-v3
```

Generates SRT with configurable readability rules (max CPS, max line length, cue duration limits) and optionally burns subtitles into the video via FFmpeg.

Key options: `--max-cps`, `--max-line-chars`, `--max-lines`, `--min-duration`, `--max-duration`, `--burn-in/--no-burn-in`, `--save-srt/--no-save-srt`, `--diarization`.

### `external-transcribe` — single-file transcription for integrations

```bash
python cli.py external-transcribe PATH_TO_MEDIA
```

Prints plain transcript text to stdout. Designed as a machine-friendly endpoint for external applications.

- Default Whisper model: `small`.
- JIT model loading: Whisper loads at transcription start and unloads before exit.
- CUDA safety: on Windows, if the GPU child process crashes (OOM), AskVLM retries on CPU automatically in an isolated subprocess.
- Optional file output: `--output-file transcript.txt`.
- Diarization is off by default (saves VRAM).

See [doc/EXTERNAL_CLI_TRANSCRIBER.md](doc/EXTERNAL_CLI_TRANSCRIBER.md) for detailed integration instructions.

### Environment Variables

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | Hugging Face token for PyAnnote diarization models (optional) |
| `LLM_GGUF_PATH` | Path to a local GGUF file for LLM-based text formatting (optional) |
| `OPENROUTER_API_KEY` | OpenRouter API key for cloud VLM in Video QA mode (optional) |

## LLM Backends

AskVLM uses LLMs in two independent contexts. Both are optional — the application works without any LLM configured, but some features degrade to identity (no formatting) or become unavailable (Video QA).

### Text Formatting (dialog blocks)

When `--dialog-blocks` is enabled (CLI) or the toggle is on (GUI), raw ASR output is sent to a local LLM for punctuation restoration, casing, and paragraph splitting. The formatter uses **llama-cpp-python** with a GGUF model file.

Setup:

1. Install the ML extras: `pip install -e .[ml]` (includes `llama-cpp-python`).
2. Set the `LLM_GGUF_PATH` environment variable to point at your `.gguf` file, or pass `--llm-model` in the legacy CLI.
3. If a CUDA GPU is available, the formatter offloads layers automatically; otherwise it runs on CPU.

If no GGUF path is set or `llama-cpp-python` is missing, the formatter falls back silently — transcription proceeds without formatting.

### Video QA (multimodal analysis)

In Video QA mode each video chunk (representative frames + transcript excerpt) is sent to a Vision-Language Model for grounded analysis. Two backends are supported:

**LM Studio (local)**

- Start [LM Studio](https://lmstudio.ai/) and load a VLM (e.g. a Qwen-VL or LLaVA variant).
- AskVLM connects via the OpenAI-compatible endpoint at `http://127.0.0.1:1234/v1`.
- The application manages model lifecycle through the LM Studio Developer REST API: it can list, load, and unload model instances to share a single GPU between Whisper and the VLM.

**OpenRouter (cloud)**

- Set `OPENROUTER_API_KEY` in `.env` or as an environment variable.
- Select a multimodal model in the GUI (e.g. `qwen/qwen3.6-plus:free`).
- Supports OpenRouter's `reasoning` parameter with configurable effort levels (`none`, `low`, `medium`, `high`).
- See [doc/OPENROUTER_INTEGRATION.md](doc/OPENROUTER_INTEGRATION.md) for API details and verified models.

Both backends use the same prompt contract (`core/llm_prompts.py`): per-chunk structured JSON analysis followed by a final synthesis step that produces a grounded answer with evidence and uncertainty markers.

### GPU memory doctrine

Only one heavy neural network occupies VRAM at a time. When the pipeline transitions between stages (e.g. Whisper → VLM), the previous model is unloaded to RAM or released entirely before the next one loads. This lets AskVLM run the full Video QA pipeline on a single 8 GB GPU.

## Architecture

```
core/           Core processing modules
  ffmpeg.py             FFmpeg audio/video conversion
  whisper_wrapper.py    OpenAI Whisper backend
  whisperx_wrapper.py   WhisperX backend (word-level timestamps)
  diarization.py        PyAnnote speaker diarization
  llm_formatter.py      LLM-based text formatting
  pipelines.py          LocalPipeline orchestration
  gpu_guard.py          VRAM checks and OOM prevention
  settings.py           Application settings
  lm_studio_rest.py     LM Studio REST client
  video_qa_*.py         Video QA pipeline (chunking, frames, orchestration, manifest, policy)
gui/            PySide6 user interface
  main_window.py        Main window with mode routing
  video_qa.py           Video QA screen
  wysiwyg_editor.py     WYSIWYG transcript editor
  subtitle_preview.py   Subtitle preview widget
  speaker_sidebar.py    Speaker management sidebar
  preferences_dialog.py Settings dialog
  export_dialog.py      Export dialog
editing/        Text model and editing operations
utils/          Exporters, logging, downloader, helpers
tools/          Benchmarking utilities (STT benchmarks, OOM threshold finder)
tests/          Pytest test suite (unit, integration, E2E)
doc/            Design documents, integration guides
```

![Multimodal GUI Architecture](doc/media/Multimodal%20GUI%20Design%2001%20-%20%D0%90%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D0%BD%D0%B0%D1%8F%20%D1%81%D1%85%D0%B5%D0%BC%D0%B0.png)

## Code Quality

All checks run via `run.ps1` → `build.py`. No separate pre-commit setup needed.

```bash
# Full pipeline: auto-fix → lint → type-check → test → security audit
pwsh -File run.ps1 -SkipLaunch

# Quick mode (skip slow tests)
pwsh -File run.ps1 -SkipLaunch -Fast

# Single tool
pwsh -File run.ps1 -Tool ruff

# Launch only
pwsh -File run.ps1 -FastLaunch
```

**Toolchain**: Ruff (format + lint), MyPy (strict), Pyright, Bandit (security), Pytest (with coverage), pip-audit.

```bash
# Makefile shortcuts
make setup-dev     # Full dev environment setup
make check-all     # Run all checks
make format        # Auto-format
make clean         # Clean generated files
```

## Troubleshooting

### "No module named 'faster_whisper'"

Install ML dependencies: `pip install -e .[ml]`

### "CUDA out of memory"

The application falls back to CPU automatically. You can also try a smaller Whisper model (`base` or `small` instead of `large-v3`) or close other GPU-intensive applications.

### CUDA Not Detected (PyTorch CPU-only)

PyTorch was installed without CUDA wheels. Reinstall with an explicit CUDA suffix:

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --extra-index-url https://pypi.org/simple
```

Or use the build script: `.\build.ps1 -EnsureCUDA`

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

See [doc/CUDA_SETUP.md](doc/CUDA_SETUP.md) for detailed CUDA troubleshooting.

### GUI doesn't start

```bash
python -m gui.main_window
```

Check console output for error messages.

### Performance Tips

- Use an NVIDIA GPU with 8 GB+ VRAM for best performance.
- Start with the `base` or `small` Whisper model; upgrade if quality is insufficient.
- Batch-process multiple files for efficiency.
- For Video QA, a local LM Studio instance avoids cloud API latency.

## Documentation

| Document | Description |
| --- | --- |
| [PYTHON_SETUP.md](PYTHON_SETUP.md) | Python 3.11 installation guide |
| [doc/CUDA_SETUP.md](doc/CUDA_SETUP.md) | CUDA and PyTorch GPU setup |
| [doc/EXTERNAL_CLI_TRANSCRIBER.md](doc/EXTERNAL_CLI_TRANSCRIBER.md) | Integration guide for `external-transcribe` |
| [doc/AutoSubtitles.md](doc/AutoSubtitles.md) | Subtitle pipeline design |
| [doc/Multimodal GUI Design.md](doc/Multimodal%20GUI%20Design.md) | Multimodal GUI architecture |
| [doc/OPENROUTER_INTEGRATION.md](doc/OPENROUTER_INTEGRATION.md) | OpenRouter integration reference |
| [doc/Disfluency-Cleanup-Design.md](doc/Disfluency-Cleanup-Design.md) | Disfluency cleanup design |
| [TODO.md](TODO.md) | Roadmap and current progress |
