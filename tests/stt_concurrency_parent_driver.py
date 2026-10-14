"""Run the benchmark CLI with fake workers; used to kill the parent on purpose.

Usage: ``python stt_concurrency_parent_driver.py <cli args...>``. The tests kill
this process abruptly and check that its worker processes do not survive it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from stt_concurrency import cli, runner  # noqa: E402

runner.ENTRY_SCRIPT = (
    Path(__file__).resolve().with_name("stt_concurrency_fake_worker.py")
)

if __name__ == "__main__":
    cli.app(args=sys.argv[1:])
