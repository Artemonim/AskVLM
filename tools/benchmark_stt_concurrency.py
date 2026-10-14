r"""Benchmark STT concurrency modes for a supervisor-managed STT worker pool.

Measures, on this PC, how resident STT engines scale:

* ``P`` - N separate processes, each holding its own model;
* ``S`` - one process holding one model, served by N concurrent Python threads;
* ``B`` - one process batching N items per forward pass.

Engines: faster-whisper ``small`` (offline from ``<project>/.cache/models``),
GigaAM-v3 ``e2e_ctc`` (``core.gigaam_e2e_ctc_wrapper``) and GigaAM-Multilingual
``ctc`` (``core.gigaam_ctc_wrapper``, CPU only).

Usage (PowerShell, repository root, venv active)::

    python tools/benchmark_stt_concurrency.py list
    python tools/benchmark_stt_concurrency.py run --clips-dir <dir> \
        --out-dir <dir outside the repo> [--configs whisper-cpu-*,gigaam-cuda-B8] \
        [--long-files a.wav --long-files b.wav] [--manifest refs.jsonl] [--repeat 2]
    python tools/benchmark_stt_concurrency.py sweep --clips-dir <dir> \
        --out-dir <dir> --manifest refs.jsonl [--groups gigaam-cuda-*] [--dry-run]
    python tools/benchmark_stt_concurrency.py combo --clips-dir <dir> \
        --out-dir <dir> --manifest refs.jsonl (--pairs A+B | --auto-combos)
    python tools/benchmark_stt_concurrency.py summarize --out-dir <dir>

Each measurement runs in fresh child processes (this script in the hidden
``_child`` mode). ``run`` writes ``<out-dir>/results`` and ``summary.md``;
``sweep`` writes ``<out-dir>/sweep``; ``combo`` writes ``<out-dir>/combos``.
Re-running resumes: results with the same input fingerprint are reused
(``--retry-failed`` re-runs non-ok ones; ``run --clear`` starts over); results
with another fingerprint are moved to ``stale/`` and measured again.
"""

from __future__ import annotations

import sys
from pathlib import Path

# * Make direct script execution work without a preconfigured PYTHONPATH:
# * the project root provides `core`, the tools directory provides the package.
_TOOLS_DIR = Path(__file__).resolve().parent
for _path in (_TOOLS_DIR.parent, _TOOLS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def main() -> None:
    """Dispatch to the hidden child mode or to the Typer CLI."""
    # * Package imports follow the sys.path bootstrap above.
    from stt_concurrency.paths import CHILD_MODE_ARG  # noqa: PLC0415

    if len(sys.argv) > 1 and sys.argv[1] == CHILD_MODE_ARG:
        # * Children skip Typer/psutil imports and never return (os._exit).
        from stt_concurrency.child import run_child  # noqa: PLC0415

        run_child()
    from stt_concurrency.cli import app  # noqa: PLC0415

    app()


if __name__ == "__main__":
    main()
