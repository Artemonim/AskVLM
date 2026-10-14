"""Surviving v1 lifecycle contracts exercised through the real v2 supervisor."""

from __future__ import annotations

import os
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from core.stt_pool.queue import SupervisorLock, read_heartbeat
from core.stt_pool.supervisor import run_supervisor
from core.temp_gc import TempGcSummary, purge_stale_askvlm_temp_dirs
from tests.stt_pool_supervisor_support import SupervisorHarness

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group(name="pool_supervisor")]


def test_run_supervisor_runs_temp_gc_once_after_lock(tmp_path: Path) -> None:
    """Catch GC before ownership, duplicate GC or GC after lock denial."""
    pool = SupervisorHarness(tmp_path)
    calls = []

    def sweep() -> TempGcSummary:
        with SupervisorLock(pool.paths) as other:
            assert not other.try_acquire()
        calls.append(True)
        assert run_supervisor(pool.paths.root, pool.config, options=pool.options) == 0
        return TempGcSummary()

    pool.options = replace(pool.options, temp_gc=sweep)
    job = pool.job(text="after GC")
    try:
        pool.start()
        assert pool.result(job).text == "after GC"
        assert calls == [True]
        pool.finish()
    finally:
        pool.cleanup()


def test_run_supervisor_survives_real_temp_gc_filesystem_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch GC failure preventing work; scan only a test temp root."""
    pool = SupervisorHarness(tmp_path)
    temp_root = tmp_path / "isolated-temp"
    temp_root.mkdir()
    original_iterdir = Path.iterdir
    failures = []

    def inaccessible(path: Path) -> Iterator[Path]:
        if path == temp_root:
            failures.append(True)
            msg = "test GC root access denied"
            raise PermissionError(msg)
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", inaccessible)
    pool.options = replace(
        pool.options, temp_gc=partial(purge_stale_askvlm_temp_dirs, temp_root)
    )
    job = pool.job(text="GC failure is tolerated")
    try:
        pool.start()
        assert pool.result(job).text == "GC failure is tolerated"
        assert failures == [True]
        pool.finish()
    finally:
        pool.cleanup()


def test_run_supervisor_releases_os_lock_if_startup_is_interrupted(
    tmp_path: Path,
) -> None:
    """Catch interruption leaking ownership, liveness or attempts."""
    pool = SupervisorHarness(tmp_path)
    job = pool.job()

    def interrupt() -> TempGcSummary:
        raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            run_supervisor(
                pool.paths.root,
                pool.config,
                options=replace(pool.options, temp_gc=interrupt),
            )
        assert read_heartbeat(pool.paths) is None
        with SupervisorLock(pool.paths) as lock:
            assert lock.try_acquire()
        assert pool.paths.incoming_file(job.job_id).exists()
        assert not pool.paths.attempts_file(job.job_id).exists()
        assert not pool.events("constructed")
    finally:
        pool.cleanup()


def test_run_supervisor_keeps_serving_after_heartbeat_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch transient heartbeat filesystem errors aborting the real serve loop."""
    pool = SupervisorHarness(tmp_path)
    original_replace = os.replace
    failures = []

    def failed_beacon(source: object, destination: object) -> None:
        if Path(destination) == pool.paths.heartbeat_file:
            failures.append(True)
            msg = "test beacon access denied"
            raise OSError(msg)
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", failed_beacon)
    job = pool.job(text="heartbeat failure is tolerated")
    try:
        pool.start()
        assert pool.result(job).text == "heartbeat failure is tolerated"
        assert failures
        pool.finish()
    finally:
        pool.cleanup()
