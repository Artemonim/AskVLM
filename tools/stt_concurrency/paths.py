"""Filesystem anchors shared by the parent orchestrator and the child workers."""

from __future__ import annotations

from pathlib import Path

# * tools/stt_concurrency/paths.py -> tools/ -> project root.
TOOLS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_DIR.parent
ENTRY_SCRIPT = TOOLS_DIR / "benchmark_stt_concurrency.py"

# * Hidden argv mode that turns the entry script into a benchmark worker.
CHILD_MODE_ARG = "_child"

RESULTS_DIRNAME = "results"
LOGS_DIRNAME = "logs"
SUMMARY_MD_NAME = "summary.md"
SUMMARY_JSON_NAME = "summary.json"
