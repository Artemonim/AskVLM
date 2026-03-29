# AskVLM

AI-powered speech transcription and editing toolkit with comprehensive code quality assurance.

**Current Status**: Phase 1.7 (Simple GUI MVP) - Quick Transcribe interface with batch processing capabilities.

## 🚀 Quick Start

### Prerequisites

**⚠️ Important:** This project requires **Python 3.11 or 3.12**. Python 3.13+ is not yet supported by ML libraries.

#### System Requirements

- **Python**: 3.11 or 3.12
- **RAM**: Minimum 8GB, recommended 16GB+
- **GPU**: NVIDIA GPU with CUDA support (optional, CPU fallback available)
- **VRAM**: Minimum 4GB, recommended 8GB+ for larger models
- **Storage**: 2GB+ for models and temporary files

Check your Python version:

```bash
python --version
```

If you need to switch Python versions, see [PYTHON_SETUP.md](PYTHON_SETUP.md) for detailed instructions.

#### Dependencies

The project uses optional ML dependencies that are installed separately:

```bash
# Core dependencies (always installed)
pip install -e .

# ML dependencies (for transcription/diarization)
pip install -e .[ml]
```

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd AskVLM

# Install development dependencies and setup environment
make setup-dev

# Run local CI (auto-fix + lint + type-check + tests)
pwsh -NoProfile -ExecutionPolicy Bypass -File run.ps1
```

### First Run

The easiest way to start is using the PowerShell wrapper, which runs quality checks and launches the GUI:

```bash
# Run quality checks and launch GUI (recommended)
pwsh -File run.ps1

# Or launch GUI directly (skips checks)
python main.py
# Or: pwsh -File run.ps1 -FastLaunch
```

**Note**: On first run, the application will create default settings and download required models automatically. If you haven't installed ML dependencies yet, you'll need to run:

```bash
pip install -e .[ml]
```

### Code Quality Tools

This project uses strict code quality standards with multiple linters and static analyzers:

#### 🔍 **Automated Checks**

-   **Ruff**: Fast Python linter and formatter (replaces flake8, isort, black)
-   **MyPy**: Static type checking with strict mode
-   **Bandit**: Security vulnerability analysis
-   **Pylint**: Comprehensive code quality analysis
-   **Pytest**: Testing framework with coverage reporting

#### ⚡ **Quick Commands**

```bash
# Run all quality checks
python test.py

# Fast checks only (skip pylint)
python test.py --quick

# Auto-format code
python test.py --format-only

# Type checking only
python test.py --type-check

# Security analysis only
python test.py --security

# Run tests only
python test.py --tests

# Clean up cache and temporary files
python test.py --clean

# Install dev dependencies
python test.py --install-deps
```

#### 🛠️ **Using Makefile**

```bash
make help          # Show all available commands
make setup-dev     # Complete development setup
make check-all     # Run all quality checks
make format        # Auto-format code
make fix           # Auto-fix issues where possible
make clean         # Clean up generated files
make clean-all     # Deep clean including temp files
make clean-verbose # Clean with detailed output
```

#### 🔧 **Automated Quality Checks**

All quality checks are integrated into `build.py` and PowerShell wrappers - no need for separate pre-commit setup. By default, successful checks automatically launch the GUI application.

```bash
# Run all checks and launch GUI if successful
pwsh -File run.ps1

# Run checks only (no GUI launch)
pwsh -File run.ps1 -SkipLaunch

# Launch GUI directly (skip checks)
pwsh -File run.ps1 -FastLaunch

# Single tool example
pwsh -File run.ps1 -- --tool ruff

### GUI Usage (Recommended)

```bash
# Launch the Quick Transcribe interface
python main.py

# Or use the PowerShell wrapper
pwsh -File run.ps1 -FastLaunch
```

**Quick Transcribe Features:**
- Choose single file or entire folder for batch processing
- Toggle diarization (speaker identification) on/off
- Toggle dialog blocks formatting on/off
- Export formats: TXT, SRT, VTT, JSON
- Real-time progress with cancel support
- Automatic output to `transcriptions/` folder

### CLI Usage

```bash
python -m pip install .[ml]
python cli.py -i PATH_TO_MEDIA -o output_dir --whisper-model base --enable-diarization --enable-dialog-blocks --export-format txt
```

**Available Options:**
- `--whisper-model`: Model size (tiny, base, small, medium, large)
- `--enable-diarization`: Enable speaker identification
- `--enable-dialog-blocks`: Enable LLM-based text formatting
- `--export-format`: Output format (txt, srt, vtt, json)
- `--device`: Processing device (auto, cuda, cpu)
- `--language`: Language code for transcription

**Environment Variables:**
- `HF_TOKEN`: Hugging Face token for pyannote models (optional, community pipeline preferred)
- `LLM_GGUF_PATH`: Path to local llama.cpp GGUF file for LLM formatting (optional)

**Notes:**
- Engine automatically selects fastest available: WhisperX → faster-whisper → OpenAI Whisper
- Automatic fallback on OOM: GPU compute type → CPU → smaller model
- GPU memory management with VRAM checks

## 📋 Code Quality Standards

**Current Status**: ✅ All checks passing

-   **Type Hints**: All code must include comprehensive type annotations (MyPy strict mode)
-   **Documentation**: Google-style docstrings for all public APIs
-   **Security**: No security vulnerabilities (enforced by Bandit)
-   **Formatting**: Consistent code style (enforced by Ruff)
-   **Linting**: Strict Ruff rules with auto-fix capabilities
-   **Testing**: Pytest framework ready (tests to be added in Phase 1.5 completion)
-   **Comments**: Better Comments style with semantic markers

**Quality Check Results:**
- ✅ Ruff Format Check
- ✅ Ruff Lint
- ✅ MyPy Type Check
- ✅ Bandit Security Check
- ⚠️ Pytest Tests (framework ready, tests pending)
- ⚠️ Pylint Analysis (optional, ML import warnings)

## 🎯 Project Status

**Phase 1.7 (Simple GUI MVP) - COMPLETED** ✅

Core functionality implemented with Quick Transcribe interface:

- **Local Processing Pipeline**: FFmpeg → Whisper/WhisperX → PyAnnote → LLM formatting
- **GUI Interface**: PySide6-based Quick Transcribe with progress tracking
- **Batch Processing**: Support for single files and entire folders
- **Export Formats**: TXT, SRT, VTT, JSON with timestamps and speakers
- **Smart Engine Selection**: Automatic WhisperX/faster-whisper/OpenAI Whisper fallback
- **GPU Optimization**: VRAM checks and compute type auto-selection
- **CLI Tools**: Batch processing with flexible options

**Next Phase**: Phase 2 - Advanced Editing (WYSIWYG editor, undo/redo, speaker management)

See [TODO.md](TODO.md) for the complete roadmap and current progress.

## 🏗️ Architecture

The project follows a modular architecture:

-   `core/`: Core processing modules (FFmpeg, Whisper, PyAnnote, LLM)
-   `gui/`: PySide6-based user interface
-   `editing/`: Text editing and manipulation tools
-   `utils/`: Utility functions and helpers
-   `tests/`: Test suite and fixtures

## ✅ Current Features

### Core Processing
- **Multi-Engine STT**: WhisperX, faster-whisper, OpenAI Whisper with automatic fallback
- **Speaker Diarization**: PyAnnote-based speaker identification (optional)
- **Text Formatting**: LLM-based punctuation and paragraph formatting (optional)
- **Audio Preprocessing**: FFmpeg-based conversion to optimal WAV format

### User Interface
- **Quick Transcribe GUI**: Drag-and-drop file/folder selection
- **Real-time Progress**: Progress bar with step-by-step status updates
- **Batch Processing**: Process multiple files with single click
- **Settings Control**: Toggle diarization, formatting, export formats

### Export & Output
- **Multiple Formats**: TXT, SRT, VTT, JSON
- **Speaker Metadata**: Timestamps and speaker identification in exports
- **Organized Output**: Automatic folder creation and file naming

### Performance & Reliability
- **GPU Memory Management**: VRAM checks and OOM prevention
- **Smart Fallbacks**: Automatic device/model selection based on hardware
- **Background Processing**: Non-blocking GUI with cancel support
- **Error Handling**: Graceful degradation and user-friendly error messages

## 🚀 Planned Features

### Advanced Editing (Phase 2)
- **WYSIWYG Editor**: Rich text editing with speaker management
- **Undo/Redo**: Full editing history with QUndoStack
- **Speaker Management**: Rename speakers, merge/split segments
- **Timeline View**: Visual timeline with speaker colors

### Cloud Integration (Phase 2)
- **Yandex SpeechKit**: Cloud-based transcription and diarization
- **Hybrid Processing**: Local + cloud fallback options

### Additional Formats (Phase 2)
- **Office Documents**: DOCX, ODT with speaker styling
- **Web Subtitles**: Advanced VTT with positioning
- **Markdown Export**: Speaker-attributed markdown format


## 🐛 Troubleshooting

### Common Issues

**"No module named 'faster_whisper'"**
```bash
pip install -e .[ml]
```
ML dependencies are required for transcription features.

**"CUDA out of memory"**
- The application automatically falls back to CPU
- Try smaller Whisper models (base/small instead of medium/large)
- Close other GPU-intensive applications

**"Permission denied" when creating output folder**
- Check write permissions in the output directory
- Try running as administrator (Windows) or with sudo (Linux)

**GUI doesn't start**
```bash
python main.py
```
Check console output for error messages.

### Environment Variables

```bash
# Hugging Face token for PyAnnote models (optional)
export HF_TOKEN="your_huggingface_token"

# Local LLM model path (optional)
export LLM_GGUF_PATH="/path/to/model.gguf"
```

### Performance Tips

- **GPU Processing**: Use NVIDIA GPU with 8GB+ VRAM for best performance
- **CPU Fallback**: Works on any system but slower
- **Model Size**: Start with "base" model, upgrade if quality is insufficient
- **Batch Processing**: Process multiple files at once for efficiency

## 📚 FAQ

**Q: What's the difference between Whisper, faster-whisper, and WhisperX?**
A: WhisperX provides word-level timestamps, faster-whisper is optimized for speed, OpenAI Whisper is the original. The app automatically selects the best available option.

**Q: Can I use this without a GPU?**
A: Yes, CPU processing is supported but significantly slower. GPU with CUDA is recommended for practical usage.

**Q: How accurate is speaker diarization?**
A: Accuracy depends on audio quality and number of speakers. PyAnnote provides industry-standard diarization but may not be perfect for overlapping speech.

**Q: Can I edit the transcription after processing?**
A: Basic viewing is available now. Full WYSIWYG editing with undo/redo is planned for Phase 2.

**Q: What audio/video formats are supported?**
A: Any format supported by FFmpeg: MP3, MP4, WAV, AVI, MKV, MOV, etc. Audio is automatically converted to optimal format for processing.

**Q: How much storage space do models require?**
A: Whisper models range from ~100MB (tiny) to ~3GB (large). PyAnnote models are ~20MB. Total: 1-4GB depending on selected models.

## Troubleshooting

### CUDA Not Detected (PyTorch CPU-only version)

**Symptom:** Error message: "CUDA is required for ML processing, but no compatible GPU is available."

**Cause:** PyTorch was installed with CPU-only support instead of CUDA-enabled wheels. This happens when pip cannot find CUDA wheels or encounters network issues, falling back to CPU-only version from PyPI.

**Solution:**

**⚠️ Important:** Always specify explicit CUDA version suffix (e.g., `+cu128`) to prevent CPU fallback!

1. **For CUDA 12.8** (recommended for RTX 30/40 series):
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   pip cache purge
   pip install --no-cache-dir `
     torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 `
     --index-url https://download.pytorch.org/whl/cu128 `
     --extra-index-url https://pypi.org/simple
   ```

2. **For CUDA 12.4**:
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   pip cache purge
   pip install --no-cache-dir `
     torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 `
     --index-url https://download.pytorch.org/whl/cu124 `
     --extra-index-url https://pypi.org/simple
   ```

3. **For CUDA 12.1**:
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   pip cache purge
   pip install --no-cache-dir `
     torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 `
     --index-url https://download.pytorch.org/whl/cu121 `
     --extra-index-url https://pypi.org/simple
   ```

4. **Using the build script** (automatic with fallback):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   .\build.ps1 -EnsureCUDA
   ```

**Verification:**
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch: 2.9.0+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 3070
```

**For detailed troubleshooting** (network issues, manual downloads, etc.), see [doc/CUDA_SETUP.md](doc/CUDA_SETUP.md).

### Installation Requirements

- **NVIDIA GPU:** GeForce GTX 960 or better (Maxwell architecture or newer)
  - Your System: **NVIDIA GeForce RTX 3070** ✅ (fully supported)
- **NVIDIA Driver:** Latest version (visit https://www.nvidia.com/Download/driverDetails.aspx)
- **CUDA Toolkit:** Optional (PyTorch includes CUDA runtime with wheels)
- **cuDNN:** Included with PyTorch wheels

To check your NVIDIA driver version:
```powershell
nvidia-smi
```

If `nvidia-smi` is not found, install the latest NVIDIA driver from: https://www.nvidia.com/Download/driverDetails.aspx
