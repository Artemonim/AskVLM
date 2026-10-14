"""V2 client scenarios with real worker processes and deterministic policy time."""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from core.stt_pool.client import run_client, spawn_supervisor
from core.stt_pool.queue import SupervisorLock, read_heartbeat, write_heartbeat
from core.stt_pool.supervisor import run_supervisor
from tests.stt_pool_client_support import ClientHarness, PolicyClock
from tests.stt_pool_supervisor_support import configuration

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from core.stt_pool.client import InputOutcome
    from core.stt_pool.queue import QueuePaths
    from core.temp_gc import TempGcSummary

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group(name="pool_supervisor")]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ClientHarness]:
    """Own only temporary queues and fake workers for every client scenario."""
    harness = ClientHarness(tmp_path, monkeypatch)
    try:
        yield harness
    finally:
        harness.finish()


@pytest.mark.parametrize(
    ("mode", "text", "status"),
    [("ok", "Привет\nworld", "ok"), ("empty", "", "empty"), ("error", "", "error")],
)
def test_run_client_transcribe_result_from_real_worker(
    client: ClientHarness, mode: str, text: str, status: str
) -> None:
    """Catch lost transcripts, empty failures or missing worker detail."""
    media = client.media(mode=mode, text=text)
    reported: list[InputOutcome] = []
    result = client.run(client.request(media), on_outcome=reported.append)
    outcome = result.outcomes[0]
    assert outcome.status == status
    assert outcome.text == text
    if mode == "error":
        assert outcome.detail == client.pool.events("emitted")[0]["error_detail"]
        assert "Too long wav file" in (outcome.detail or "")
        assert "transcribe_longform" in (outcome.detail or "")
    assert reported == [outcome]
    assert not tuple(client.pool.paths.results.glob("*.json"))
    assert len(client.pool.events("submitted")) == 1
    client.pool.until(
        lambda: tuple(client.pool.paths.incoming.glob("*.json")),
        lambda paths: not paths,
    )


def test_repeated_inputs_have_one_call_and_distinct_ordered_jobs(
    client: ClientHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch path deduplication, per-job call IDs or loss of scheduling fields."""
    client.pool.config = configuration(cpu={"max_slots": 2, "max_slots_per_worker": 2})
    media = client.media(mode="hold", text="same path")
    snapshot = []

    def release() -> None:
        jobs = client.pool.until(client.jobs, lambda jobs: len(jobs) == 2)
        snapshot.extend(jobs)
        client.pool.until(lambda: client.pool.events("submitted"), bool)
        client.pool.heartbeat(
            lambda value: value["queue"] == {"running": 1, "pending": 1}
        )
        for job in jobs:
            client.pool.release(job)

    client.background(release)
    client.pool.start()
    client.pool.heartbeat(lambda value: value["state"] == "running")
    other_queue = client.pool.paths.root.parent / "environment-queue"
    monkeypatch.setenv("ASKVLM_EXTERNAL_QUEUE_DIR", str(other_queue))

    def forbidden_spawn(_paths: QueuePaths) -> None:
        pytest.fail("a fresh explicit queue must not spawn or use the environment root")

    request = client.request(
        media,
        media,
        max_parallel=1,
        priority="high",
        language="EN",
        whisper_model="Org/CaseModel",
    )
    outcome = client.run(request, spawn=forbidden_spawn)
    assert not other_queue.exists()
    assert [item.text for item in outcome.outcomes] == ["same path", "same path"]
    assert len({job.job_id for job in snapshot}) == 2
    assert len({job.call_id for job in snapshot}) == 1
    assert [job.call_index for job in snapshot] == [0, 1]
    assert snapshot[0].submitted_at < snapshot[1].submitted_at
    assert {job.max_parallel for job in snapshot} == {1}
    assert {job.priority for job in snapshot} == {"high"}
    assert {job.language for job in snapshot} == {"en"}
    assert {event["key"]["model"] for event in client.pool.events("constructed")} == {
        "Org/CaseModel"
    }


@pytest.mark.parametrize(
    ("budget", "expected_probes"), [(4, [1.0]), (40, [10.0]), (100, [10.0, 10.0, 5.0])]
)
def test_deadline_includes_probing_and_cancels_every_unfinished_job(
    client: ClientHarness,
    monkeypatch: pytest.MonkeyPatch,
    budget: float,
    expected_probes: list[float],
) -> None:
    """Catch resetting the deadline after probes or probes spending the whole budget."""
    clock = PolicyClock()
    monkeypatch.setattr("core.stt_pool.queue.time", clock)
    media = [client.media() for _ in range(4)]
    timeouts: list[float] = []

    def probe(_path: Path, timeout: float) -> float:
        timeouts.append(timeout)
        clock.sleep(timeout)
        return 12.0

    def spawn(paths: QueuePaths) -> None:
        write_heartbeat(paths, {"state": "running"})

    result = run_client(
        client.request(*media, client_timeout_s=budget),
        queue_root=client.pool.paths.root,
        spawn=spawn,
        clock=clock,
        sleeper=clock.sleep,
        probe_duration=probe,
        poll_interval_s=5,
        stale_after_s=100,
    )
    assert timeouts == expected_probes
    assert clock() == 1000 + budget
    jobs = client.jobs()
    assert len(jobs) == 4
    assert {job.deadline_at for job in jobs} == {1000 + budget}
    assert [job.duration_s for job in jobs] == [12.0] * len(expected_probes) + [
        None
    ] * (4 - len(expected_probes))
    assert {item.status for item in result.outcomes} == {"timeout"}
    assert {path.name for path in client.pool.paths.cancel.iterdir()} == {
        job.job_id for job in jobs
    }


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_abandoned_call_cancels_its_jobs(
    client: ClientHarness, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    """Catch Ctrl+C or a failed poll leaving the supervisor doing abandoned work."""
    clock = PolicyClock()
    monkeypatch.setattr("core.stt_pool.queue.time", clock)

    def interrupt(_delay: float) -> None:
        msg = "abandoned"
        raise failure(msg)

    def spawn(paths: QueuePaths) -> None:
        write_heartbeat(paths, {"state": "running"})

    with pytest.raises(failure, match="abandoned"):
        run_client(
            client.request(client.media(), client.media()),
            queue_root=client.pool.paths.root,
            spawn=spawn,
            clock=clock,
            sleeper=interrupt,
            probe_duration=lambda _p, _t: None,
        )
    jobs = client.jobs()
    assert len(jobs) == 2
    assert {path.name for path in client.pool.paths.cancel.iterdir()} == {
        job.job_id for job in jobs
    }


@pytest.mark.parametrize("spawn_fails", [False, True])
def test_unavailable_names_supervisor_log_and_respawns_at_ten_seconds(
    client: ClientHarness, *, spawn_fails: bool
) -> None:
    """Catch endless startup waits, spawn storms or lost log paths."""
    clock = PolicyClock()
    spawns: list[float] = []

    def spawn(_paths: QueuePaths) -> None:
        spawns.append(clock())
        if spawn_fails:
            msg = "fake process creation denied"
            raise OSError(msg)

    result = run_client(
        client.request(client.media(), client_timeout_s=math.inf),
        queue_root=client.pool.paths.root,
        spawn=spawn,
        clock=clock,
        sleeper=clock.sleep,
        probe_duration=lambda _p, _t: None,
        poll_interval_s=5,
    )
    outcome = result.outcomes[0]
    assert outcome.status == "unavailable"
    assert str(client.pool.paths.supervisor_out_log) in (outcome.detail or "")
    if spawn_fails:
        assert "fake process creation denied" in (outcome.detail or "")
    assert spawns == [1000, 1010, 1020]
    assert clock() == 1030
    assert len(tuple(client.pool.paths.cancel.iterdir())) == 1


def test_infinite_timeout_has_no_job_deadline_and_still_receives_results(
    client: ClientHarness,
) -> None:
    """Catch serializing infinity into jobs or assigning a finite deadline to inf."""
    media = client.media(mode="hold", text="after a long wait")
    seen = []

    def release() -> None:
        jobs = client.pool.until(client.jobs, bool)
        seen.extend(jobs)
        client.pool.until(lambda: client.pool.events("submitted"), bool)
        client.pool.clock.advance(10000)
        client.pool.release(jobs[0])

    client.background(release)
    result = client.run(
        client.request(media, client_timeout_s=math.inf), stale_after_s=20000
    )
    assert result.outcomes[0].text == "after a long wait"
    assert seen[0].deadline_at is None
    assert not tuple(client.pool.paths.cancel.iterdir())


@pytest.mark.parametrize("timeout", [math.nan, -math.inf, 0, -1])
def test_invalid_timeout_never_submits_or_spawns(
    client: ClientHarness, timeout: float
) -> None:
    """Catch NaN or non-positive budgets entering an unbounded queue wait."""
    with pytest.raises(ValueError, match="client_timeout_s"):
        client.run(client.request(client.media(), client_timeout_s=timeout))
    assert not client.jobs()
    assert client.spawned == 0


def test_failed_outcome_callback_cancels_remaining_running_attempt(
    client: ClientHarness,
) -> None:
    """Catch an output callback leaving another live job abandoned."""
    client.pool.config = configuration(cpu={"max_slots": 2, "max_slots_per_worker": 2})
    inputs = [client.media(mode="hold", text=f"text {index}") for index in range(2)]
    seen = []

    def release_first() -> None:
        jobs = client.pool.until(client.jobs, lambda jobs: len(jobs) == 2)
        seen.extend(jobs)
        client.pool.until(
            lambda: client.pool.events("submitted"), lambda events: len(events) == 2
        )
        client.pool.release(jobs[0])

    def broken_output(_outcome: InputOutcome) -> None:
        msg = "output sink failed"
        raise OSError(msg)

    client.background(release_first)
    with pytest.raises(OSError, match="output sink failed"):
        client.run(client.request(*inputs), on_outcome=broken_output)
    assert client.pool.result(seen[1]).status == "cancelled"
    assert len(client.pool.events("cancel", seen[1])) == 1
    assert len(client.pool.events("submitted", seen[1])) == 1


def test_spawn_supervisor_passes_queue_and_log_to_the_process_boundary(
    client: ClientHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch launching on the default queue or losing the startup error log."""
    captured = []

    def launch(argv: list[str], **kwargs: object) -> int:
        captured.append((argv, kwargs))
        return 123

    monkeypatch.setattr("core.stt_pool.client.spawn_detached", launch)
    spawn_supervisor(client.pool.paths)
    argv, kwargs = captured[0]
    assert argv[argv.index("--queue-dir") + 1] == str(client.pool.paths.root)
    assert "external-transcribe-daemon" in argv
    assert kwargs["log_path"] == client.pool.paths.supervisor_out_log
    assert (
        kwargs["env"]["ASKVLM_STT_POOL_CONFIG"] == os.environ["ASKVLM_STT_POOL_CONFIG"]
    )


def test_eight_client_startup_burst_has_one_owner_and_answers_all_calls(  # noqa: PLR0915 - one bounded scenario owns eight calls and all concurrent supervisors
    client: ClientHarness,
) -> None:
    """Catch lock bypass at an eight-client burst, lost calls or orphaned workers."""
    barrier = threading.Barrier(8)
    stop = threading.Event()
    supervisors: list[threading.Thread] = []
    codes: list[int] = []
    spawn_lock = threading.Lock()
    entered = threading.Event()
    release = threading.Event()
    gc_count = 0

    def hold_owner() -> TempGcSummary:
        nonlocal gc_count
        gc_count += 1
        entered.set()
        if not release.wait(30):
            msg = "startup burst was not released"
            raise TimeoutError(msg)
        return client.pool.options.temp_gc()

    options = replace(client.pool.options, temp_gc=hold_owner)

    def spawn(paths: QueuePaths) -> None:
        barrier.wait(timeout=30)

        def serve() -> None:
            codes.append(
                run_supervisor(
                    paths.root, client.pool.config, options=options, stop_event=stop
                )
            )

        thread = threading.Thread(target=serve, daemon=True)
        with spawn_lock:
            supervisors.append(thread)
        thread.start()

    requests = [
        client.request(client.media(text=f"answer {index}")) for index in range(8)
    ]
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    run_client,
                    request,
                    queue_root=client.pool.paths.root,
                    spawn=spawn,
                    probe_duration=lambda _p, _t: None,
                    poll_interval_s=0.01,
                )
                for request in requests
            ]
            assert entered.wait(30)
            client.pool.until(lambda: len(supervisors), lambda count: count == 8)
            client.pool.until(lambda: len(codes), lambda count: count == 7)
            assert gc_count == 1
            release.set()
            outcomes = [future.result(timeout=30).outcomes[0] for future in futures]
        assert [outcome.text for outcome in outcomes] == [
            f"answer {index}" for index in range(8)
        ]
        assert {outcome.status for outcome in outcomes} == {"ok"}
        events = client.pool.events("submitted")
        assert len({event["call_id"] for event in events}) == 8
        assert len(events) == 8
        assert len(client.pool.events("constructed")) == 1
    finally:
        stop.set()
        release.set()
        for thread in supervisors:
            thread.join(timeout=30)
            assert not thread.is_alive()
        try:
            client.pool.assert_processes_stopped()
        finally:
            client.pool.cleanup()
    assert codes == [0] * 8
    assert read_heartbeat(client.pool.paths) is None
    with SupervisorLock(client.pool.paths) as lock:
        assert lock.try_acquire()


def test_client_respawns_after_idle_exit_with_lock_still_held(
    client: ClientHarness,
) -> None:
    """Catch a one-off respawn during teardown leaving the next call unanswered."""
    first = client.run(client.request(client.media(text="first")))
    assert first.outcomes[0].text == "first"
    client.pool.hardware.block_gpu_after = client.pool.hardware.gpu_calls
    client.pool.hardware.gpu_gate.clear()
    client.pool.until(
        lambda: client.pool.hardware.gpu_calls,
        lambda count: count > client.pool.hardware.block_gpu_after,
    )
    client.pool.clock.advance(31)
    client.pool.heartbeat(lambda value: not value["workers"])
    client.pool.clock.advance(6)
    client.pool.until(
        client.pool.paths.heartbeat_file.exists, lambda exists: not exists
    )
    codes = []
    clock = PolicyClock(client.pool.clock.peek())

    def spawn(paths: QueuePaths) -> None:
        if not codes:
            codes.append(
                run_supervisor(
                    paths.root, client.pool.config, options=client.pool.options
                )
            )
            assert codes == [0]
            assert client.pool.thread is not None
            assert client.pool.thread.is_alive()
            client.pool.hardware.gpu_gate.set()
            client.pool.finish(request_stop=False)
        else:
            client.pool.start()
            codes.append(0)

    def pause(delay: float) -> None:
        clock.sleep(delay)
        threading.Event().wait(0.01)

    result = run_client(
        client.request(client.media(text="second")),
        queue_root=client.pool.paths.root,
        spawn=spawn,
        clock=clock,
        sleeper=pause,
        probe_duration=lambda _p, _t: None,
        poll_interval_s=1,
        stale_after_s=1000,
    )
    assert result.outcomes[0].text == "second"
    assert codes == [0, 0]
    assert len(client.pool.events("submitted")) == 2
