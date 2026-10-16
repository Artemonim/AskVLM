"""Failure deadlines for scenarios that exercise real processes and thread locks."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import signal
import sys
import traceback
from typing import TYPE_CHECKING

import pytest

from core.stt_pool.winjob import WorkerJob

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.connection import Connection
    from multiprocessing.synchronize import Event
    from pathlib import Path

_SCENARIO_TIMEOUT_S = 30.0
_REAP_TIMEOUT_S = 5.0


def _scenario_peer(
    scenario: Callable[[Path], None],
    root: Path,
    release: Event,
    connection: Connection,
) -> None:
    """Wait for process ownership before running a scenario and its assertions."""
    try:
        if sys.platform != "win32":
            os.setsid()
        if not release.wait(_REAP_TIMEOUT_S):
            msg = "scenario ownership gate was not released"
            raise TimeoutError(msg)  # noqa: TRY301 - the except clause reports it to the parent
        scenario(root)
    except BaseException:  # noqa: BLE001 - child assertion failures must reach pytest
        connection.send(traceback.format_exc()[-4000:])
    else:
        connection.send(None)
    finally:
        connection.close()


def run_bounded_scenario(
    scenario: Callable[[Path], None],
    root: Path,
    *,
    timeout_s: float = _SCENARIO_TIMEOUT_S,
) -> None:
    """Bound the entire scenario, including executor shutdown and native pipes.

    The parent owns the child and its descendants before any media process
    starts. Closing the Windows Job Object reaps only the processes started by
    this scenario. On POSIX the process group is killed only while the peer has
    not exited yet (its group id cannot belong to a stranger before the join);
    descendants left behind by a peer that exited on its own are not reaped.

    Args:
        scenario: Picklable scenario whose assertions execute in the child.
        root: Pytest temporary directory used by the scenario.
        timeout_s: Failure deadline of the complete scenario, including native calls.

    """
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    release = context.Event()
    peer = context.Process(target=_scenario_peer, args=(scenario, root, release, child))
    job = WorkerJob(memory_limit_bytes=None)
    started = False
    try:
        if sys.platform == "win32" and not job.active:
            pytest.fail(f"scenario Job Object is unavailable: {job.error}")
        peer.start()
        started = True
        child.close()
        pid = peer.pid
        assert pid is not None
        if sys.platform == "win32" and not job.assign(pid):
            pytest.fail(f"scenario process ownership failed: {job.assign_error}")
        release.set()
        if not parent.poll(timeout_s):
            name = getattr(scenario, "__name__", type(scenario).__name__)
            pytest.fail(f"scenario exceeded {timeout_s}s: {name}")
        failure = parent.recv()
        if failure is not None:
            pytest.fail(failure, pytrace=False)
        peer.join(_REAP_TIMEOUT_S)
        assert peer.exitcode == 0
    finally:
        job.close()
        if started:
            # * Signal only a peer that has not been reaped: its pid, and the group id
            # * it leads after ``setsid``, cannot belong to a stranger before the join.
            if peer.is_alive():
                if sys.platform != "win32" and peer.pid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(peer.pid, signal.SIGKILL)
                peer.kill()
            peer.join(_REAP_TIMEOUT_S)
            assert not peer.is_alive()
            peer.close()
        parent.close()
        child.close()
