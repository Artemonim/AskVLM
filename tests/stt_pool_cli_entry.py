"""Run the actual CLI while forbidding model-supervisor spawns in a test subprocess."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cli
from core.stt_pool import client

if TYPE_CHECKING:
    from core.stt_pool.queue import QueuePaths


def forbid_spawn(_paths: QueuePaths) -> None:
    """Abort safely if the parent-owned fake supervisor becomes unreachable."""
    msg = "test CLI cannot launch a model supervisor"
    raise KeyboardInterrupt(msg)


if __name__ == "__main__":
    client.spawn_supervisor = forbid_spawn
    client._probe_with_ffprobe = lambda _path, _timeout: None  # noqa: SLF001 - replace only the external ffprobe boundary
    cli.main()
