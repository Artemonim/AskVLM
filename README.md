# Artemonim's Speech Kit

AI-powered speech transcription and editing toolkit with comprehensive code quality assurance.

## 🚀 Quick Start

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks (recommended)
make setup-dev

# Run comprehensive code quality checks
python test.py
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
```

#### 🔧 **Pre-commit Hooks**

Automatically run quality checks before each commit:

```bash
# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 📋 Code Quality Standards

-   **Type Hints**: All code must include comprehensive type annotations
-   **Documentation**: Google-style docstrings for all public APIs
-   **Security**: No security vulnerabilities (enforced by Bandit)
-   **Formatting**: Consistent code style (enforced by Ruff)
-   **Testing**: Minimum 80% code coverage
-   **Comments**: Better Comments style with semantic markers

## 🎯 Project Status

This project is in early development. See [TODO.md](TODO.md) for the complete roadmap and current progress.

## 🏗️ Architecture

The project follows a modular architecture:

-   `core/`: Core processing modules (FFmpeg, Whisper, PyAnnote, LLM)
-   `gui/`: PySide6-based user interface
-   `editing/`: Text editing and manipulation tools
-   `utils/`: Utility functions and helpers
-   `tests/`: Test suite and fixtures

## 📦 Features (Planned)

-   **Local Processing**: Whisper + PyAnnote + Local LLM
-   **Cloud Integration**: Yandex SpeechKit support
-   **Advanced Editing**: WYSIWYG editor with speaker management
-   **Multiple Export Formats**: TXT, DOCX, ODT, SRT, VTT, Markdown
-   **GPU Optimization**: Smart GPU memory management
