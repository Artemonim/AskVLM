"""Benchmark worker with a fake engine: the real child protocol loop, no ML models.

The orchestrator of ``tools/benchmark_stt_concurrency.py`` spawns
``<python> -X utf8 <ENTRY_SCRIPT> _child``. Tests point ``runner.ENTRY_SCRIPT`` at
this file, so the parent, the job object, the pipes, and the child event loop
(``stt_concurrency.child``) are all production code; only the speech engine (an
external, heavy system) is replaced.

Behavior is controlled through environment variables (inherited from the test).
``FAKE_HANG``, ``FAKE_FAIL_NAME`` and ``FAKE_ITEM_S`` also exist per engine as
``<NAME>_<ENGINE>`` (for example ``FAKE_ITEM_S_GIGAAM``), which takes precedence:

* ``FAKE_DIR`` - directory for markers: ``worker-<child>.pid`` (interpreter pid),
  ``grandchild-<child>.pid``, ``order.txt`` (one config name per load), and
  ``summary-<child>.json`` written on close (load count, peak thread overlap).
  Each ``<kind>-<child>.pid`` has a ``<kind>-<child>.created`` twin holding the
  creation time of that process, recorded while it is certainly the process: a
  pid file outlives its process, only the creation time tells it from a stranger
  that took over the pid.
* ``FAKE_LOAD_S`` - comma list of load seconds, indexed by child index.
* ``FAKE_LOAD_FAIL`` - raise while loading.
* ``FAKE_ITEM_S`` - seconds one ``transcribe`` call takes.
* ``FAKE_FAIL_NAME`` - raise in ``transcribe`` when a path name contains this.
* ``FAKE_OOM_FROM`` - raise a CUDA out-of-memory error when the width is >= N.
* ``FAKE_DEGRADE_FROM`` - corrupt every transcript when the width is >= N.
* ``FAKE_DIE`` - end the whole process abruptly inside ``transcribe``.
* ``FAKE_HANG`` - block ``transcribe`` forever.
* ``FAKE_GRANDCHILD`` - start a sleeping grandchild process while loading.
* ``FAKE_COMMIT_MB`` - retain up to 512 MiB in 16 MiB blocks while loading;
  ``<NAME>_<ENGINE>`` overrides apply. Larger requests are rejected for safety.
* ``FAKE_ITEM_COMMIT_MB`` - grow retained buffers to this size during decoding;
  the same 512 MiB bound applies to the combined load and decoding buffers.
* ``FAKE_CLOSE_S`` - hold the engine for up to 5 s during cleanup, after a
  0.5 s settling phase; ``closing-<child>`` marks the hold phase.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from stt_concurrency import child  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stt_concurrency.protocol import ChildSpec

_HANG_S = 3600.0
_MAX_COMMIT_MB = 512
_COMMIT_BLOCK_MB = 16
_MAX_CLOSE_S = 5.0
_CLOSE_SETTLE_S = 0.5


class FakeEngine:
    """Deterministic engine: the transcript of a file is ``text <stem>``."""

    def __init__(self, spec: ChildSpec) -> None:
        self._spec = spec
        self._dir = Path(os.environ["FAKE_DIR"])
        self._loads = 0
        self._active = 0
        self._peak = 0
        self._lock = threading.Lock()
        self._memory: list[bytearray] = []

    def _env(self, name: str) -> str | None:
        """Return ``<name>_<ENGINE>`` (engine upper-cased) or else ``<name>``."""
        engine = self._spec.config.engine.upper().replace("-", "_")
        return os.environ.get(f"{name}_{engine}") or os.environ.get(name)

    def _retain_commit(self, commit_mb: int) -> None:
        """Grow retained buffers to a bounded target without exceeding 512 MiB."""
        if not 0 <= commit_mb <= _MAX_COMMIT_MB:
            msg = f"Commit target must be between 0 and {_MAX_COMMIT_MB} MiB"
            raise ValueError(msg)
        with self._lock:
            retained_mb = sum(len(block) for block in self._memory) // 1024**2
            while retained_mb < commit_mb:
                block_mb = min(_COMMIT_BLOCK_MB, commit_mb - retained_mb)
                self._memory.append(bytearray(block_mb * 1024**2))
                retained_mb += block_mb

    def _record_process(self, kind: str, pid: int) -> None:
        """Publish *pid* as ``<kind>-<child>.pid`` with its creation time beside it.

        The creation time goes first, so a reader that finds the pid file finds the
        identity of the process that wrote it.
        """
        index = self._spec.child_index
        created = psutil.Process(pid).create_time()
        (self._dir / f"{kind}-{index}.created").write_text(repr(created))
        (self._dir / f"{kind}-{index}.pid").write_text(str(pid))

    def load(self) -> dict[str, object]:
        """Record the load, optionally start a grandchild, sleep, or fail."""
        index = self._spec.child_index
        self._record_process("worker", os.getpid())
        with (self._dir / "order.txt").open("a", encoding="utf-8") as handle:
            handle.write(self._spec.config.name + "\n")
        if os.environ.get("FAKE_GRANDCHILD"):
            grandchild = subprocess.Popen(  # noqa: S603
                [sys.executable, "-c", f"import time; time.sleep({_HANG_S})"],
            )
            self._record_process("grandchild", grandchild.pid)
        delays = [float(x) for x in os.environ.get("FAKE_LOAD_S", "0").split(",")]
        time.sleep(delays[index % len(delays)])
        if os.environ.get("FAKE_LOAD_FAIL"):
            msg = "fake load failure"
            raise RuntimeError(msg)
        self._retain_commit(int(self._env("FAKE_COMMIT_MB") or "0"))
        self._loads += 1
        return {"backend": "fake", "python_prefix": sys.prefix}

    def warmup(self, path: Path, *, slot: int, primary: bool) -> dict[str, object]:
        """Accept the warm-up without doing anything."""
        del path, slot, primary
        return {"kind": "fake"}

    def transcribe(self, paths: Sequence[Path], *, slot: int) -> list[str]:
        """Return ``text <stem>`` per path; hang or fail on request."""
        del slot
        with self._lock:
            self._active += 1
            self._peak = max(self._peak, self._active)
        try:
            if os.environ.get("FAKE_DIE"):
                os._exit(9)
            self._retain_commit(int(self._env("FAKE_ITEM_COMMIT_MB") or "0"))
            if self._env("FAKE_HANG"):
                time.sleep(_HANG_S)
            width = self._spec.config.width
            oom_from = os.environ.get("FAKE_OOM_FROM")
            if oom_from and width >= int(oom_from):
                msg = "CUDA out of memory. Tried to allocate 2.00 MiB"
                raise RuntimeError(msg)
            marker = self._env("FAKE_FAIL_NAME")
            if marker and any(marker in path.name for path in paths):
                msg = f"fake failure on {marker}"
                raise RuntimeError(msg)
            time.sleep(float(self._env("FAKE_ITEM_S") or "0"))
            texts = [f"text {path.stem}" for path in paths]
            degrade_from = os.environ.get("FAKE_DEGRADE_FROM")
            if degrade_from and width >= int(degrade_from):
                texts = [text.replace("text", "zzzz", 1) for text in texts]
            return texts
        finally:
            with self._lock:
                self._active -= 1

    def close(self) -> None:
        """Write the load count and the peak thread overlap."""
        close_s = float(os.environ.get("FAKE_CLOSE_S", "0"))
        if not 0 <= close_s <= _MAX_CLOSE_S:
            msg = f"FAKE_CLOSE_S must be between 0 and {_MAX_CLOSE_S}"
            raise ValueError(msg)
        if close_s:
            time.sleep(_CLOSE_SETTLE_S)
            (self._dir / f"closing-{self._spec.child_index}").touch()
            time.sleep(close_s)
        self._memory.clear()
        summary = {"loads": self._loads, "peak_overlap": self._peak}
        path = self._dir / f"summary-{self._spec.child_index}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")


child.make_engine = FakeEngine  # type: ignore[assignment]

if __name__ == "__main__":
    child.run_child()
