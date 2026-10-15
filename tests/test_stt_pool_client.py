"""V2 client scenarios with real worker processes and deterministic policy time."""

from __future__ import annotations

import errno
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from core.stt_options import DiarizeOptions, MediaWindow, WhisperDecodeOptions
from core.stt_pool.client import run_client, spawn_supervisor
from core.stt_pool.queue import (
    ResultRecord,
    SupervisorLock,
    is_cancel_requested,
    read_heartbeat,
    write_heartbeat,
    write_result,
)
from core.stt_pool.supervisor import run_supervisor
from tests.stt_pool_client_support import ClientHarness, PolicyClock
from tests.stt_pool_supervisor_support import configuration

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from core.stt_pool.client import InputOutcome
    from core.stt_pool.queue import QueuePaths, ResultStatus
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
    assert outcome.data is None
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


def _result_peer(
    client: ClientHarness, templates: Mapping[str, ResultRecord]
) -> Callable[[QueuePaths], None]:
    """Substitute an external daemon at its real on-disk result boundary."""

    def publish(paths: QueuePaths) -> None:
        write_heartbeat(paths, {"state": "running"})
        for job in client.jobs():
            template = templates.get(job.kind)
            if template is not None:
                write_result(paths, replace(template, job_id=job.job_id))

    return publish


@pytest.mark.parametrize(
    ("turns", "expected_speakers", "labelled"),
    [
        (
            [
                {"start": 10.0, "end": 13.0, "speaker": "speaker_00"},
                {"start": 100.0, "end": 101.0, "speaker": "speaker_01"},
            ],
            ["speaker_00", "speaker_00", None, None],
            False,
        ),
        (
            [
                {"start": 10.0, "end": 11.0, "speaker": "speaker_00"},
                {"start": 12.0, "end": 13.0, "speaker": "speaker_01"},
                {"start": 14.0, "end": 15.0, "speaker": "speaker_01"},
            ],
            ["speaker_00", "speaker_01", None, None],
            True,
        ),
        ([], [None, None, None, None], False),
    ],
)
def test_client_merges_real_worker_parts_by_segment_without_changing_one_voice_text(
    client: ClientHarness,
    turns: list[dict[str, object]],
    expected_speakers: list[str | None],
    *,
    labelled: bool,
) -> None:
    """Catch labels based on unassigned voices, dropped nulls or lost text/turns across IPC."""
    plain_text = "  original\nformatting stays\t intact  "
    segments = [
        {"start": 10.0, "end": 11.0, "text": "first"},
        {"start": 12.0, "end": 13.0, "text": "second"},
        {"start": 14.0, "end": 14.0, "text": "instant"},
        {"start": 16.0, "end": 17.0, "text": "unassigned"},
    ]
    media = client.media(
        stt={"text": plain_text, "data": {"segments": segments}},
        diarize={"data": {"turns": turns}},
    )
    reported: list[InputOutcome] = []
    result = client.run(
        client.request(media, diarization=DiarizeOptions()), on_outcome=reported.append
    ).outcomes[0]
    assert result.status == "ok"
    assert result.backend == "cpu"
    assert result.data == {
        "segments": segments,
        "turns": turns,
        "speakers": expected_speakers,
    }
    if labelled:
        assert result.text.splitlines() == [
            "speaker_00: first",
            "speaker_01: second",
            "instant",
            "unassigned",
        ]
    else:
        assert result.text == plain_text.strip()
    assert reported == [result]
    events = client.pool.events("submitted")
    assert {event["kind"] for event in events} == {"stt", "diarize"}
    assert len({event["call_id"] for event in events}) == 1
    assert all(event["want_segments"] for event in events if event["kind"] == "stt")


@pytest.mark.parametrize("mode", ["transcribe", "diarize"])
def test_empty_transcript_and_zero_turn_diarization_keep_structured_success(
    client: ClientHarness, mode: str
) -> None:
    """Catch empty text discarding data or a zero-turn diarize job becoming empty/error."""
    media = client.media(
        stt={"mode": "empty", "data": {"segments": []}},
        diarize={"data": {"turns": []}},
    )
    result = client.run(
        client.request(media, mode=mode, language=None, diarization=DiarizeOptions())
    ).outcomes[0]
    assert result.text == ""
    assert result.status == ("empty" if mode == "transcribe" else "ok")
    assert result.data == (
        {"segments": [], "turns": [], "speakers": []}
        if mode == "transcribe"
        else {"turns": []}
    )


def test_structured_transcribe_without_diarization_returns_only_segments(
    client: ClientHarness,
) -> None:
    """Catch want_segments failing to reach the worker or requiring a diarization sibling."""
    segments = [{"start": 0.125, "end": 0.875, "text": "spoken"}]
    media = client.media(data={"segments": segments}, text="spoken")
    result = client.run(client.request(media, want_segments=True)).outcomes[0]
    assert result.status == "ok"
    assert result.text == "spoken"
    assert result.data == {"segments": segments}
    assert len(client.pool.events("submitted")) == 1
    assert client.pool.events("submitted")[0]["want_segments"] is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_paired_jobs_share_call_order_options_deadline_and_parallel_cap(
    client: ClientHarness, device: str
) -> None:
    """Catch each part receiving its own call cap, reset deadline or wrong media/options."""
    client.pool.config = configuration(
        cpu={"max_slots": 2, "max_slots_per_worker": 2},
        gpu={"max_workers": 1 if device == "cuda" else 0},
    )
    client.pool.hardware.gpu_available = device == "cuda"
    media = client.media(
        stt={"mode": "hold", "text": "retained", "data": {"segments": []}},
        diarize={"mode": "hold", "data": {"turns": []}},
    )
    snapshot = []
    request = client.request(
        media,
        media,
        device=device,
        max_parallel=1,
        priority="high",
        window=MediaWindow(3.25, 2.5),
        decode=WhisperDecodeOptions(vad_filter=False),
        diarization=DiarizeOptions(num_speakers=2, threshold=0.7),
    )

    def release() -> None:
        jobs = client.pool.until(client.jobs, lambda jobs: len(jobs) == 4)
        jobs.sort(key=lambda job: job.submitted_at)
        snapshot.extend(jobs)
        client.pool.heartbeat(
            lambda value: value["queue"] == {"running": 1, "pending": 3}
        )
        for job in jobs:
            client.pool.release(job)

    client.background(release)
    result = client.run(request, probe_duration=lambda _path, _timeout: 8.0)
    assert [outcome.text for outcome in result.outcomes] == ["retained", "retained"]
    assert [job.call_index for job in snapshot] == [0, 0, 1, 1]
    assert [job.kind for job in snapshot] == ["stt", "diarize", "stt", "diarize"]
    assert len({job.job_id for job in snapshot}) == 4
    assert len({job.call_id for job in snapshot}) == 1
    assert len({job.deadline_at for job in snapshot}) == 1
    assert all(a.submitted_at < b.submitted_at for a, b in pairwise(snapshot))
    assert all(
        job.duration_s == 2.5 and job.window == request.window for job in snapshot
    )
    assert all(job.max_parallel == 1 and job.priority == "high" for job in snapshot)
    assert {job.request.device for job in snapshot if job.kind == "stt"} == {device}
    assert {job.request.device for job in snapshot if job.kind == "diarize"} == {"cpu"}
    events = client.pool.events("submitted")
    stt_events = [event for event in events if event["kind"] == "stt"]
    diarize_events = [event for event in events if event["kind"] == "diarize"]
    assert all(event["decode"] == {"vad_filter": False} for event in stt_events)
    assert all(event["diarize"] is None for event in stt_events)
    assert all(event["diarize"]["num_speakers"] == 2 for event in diarize_events)
    assert all(
        event["language"] is None and not event["decode"] for event in diarize_events
    )


@pytest.mark.parametrize("failed_kind", ["stt", "diarize"])
@pytest.mark.parametrize(
    ("status", "expected"),
    [("error", "error"), ("expired", "timeout"), ("unavailable", "unavailable")],
)
def test_one_failed_part_finishes_input_once_and_cancels_sibling(
    client: ClientHarness, failed_kind: str, status: ResultStatus, expected: str
) -> None:
    """Catch waiting for a doomed sibling, repeated callbacks or unattributed diarize failures."""
    peer = _result_peer(
        client, {failed_kind: ResultRecord("", status, error_detail="part failure")}
    )
    reported: list[InputOutcome] = []
    clock = PolicyClock()
    result = client.run(
        client.request(client.media(), diarization=DiarizeOptions()),
        spawn=peer,
        clock=clock,
        sleeper=clock.sleep,
        on_outcome=reported.append,
    ).outcomes[0]
    assert result.status == expected
    assert result.data is None
    assert result.text == ""
    assert "part failure" in (result.detail or "")
    assert ("speaker diarization" in (result.detail or "")) == (
        failed_kind == "diarize"
    )
    assert reported == [result]
    jobs = client.jobs()
    assert {
        job.kind for job in jobs if is_cancel_requested(client.pool.paths, job.job_id)
    } == ({"diarize"} if failed_kind == "stt" else {"stt"})
    assert not list(client.pool.paths.results.iterdir())


@pytest.mark.parametrize(
    ("stt_status", "diarize_status", "expected", "source"),
    [
        ("error", "unavailable", "unavailable", "diarize"),
        ("unavailable", "error", "unavailable", "stt"),
        ("expired", "unavailable", "timeout", "stt"),
        ("unavailable", "expired", "timeout", "diarize"),
        ("expired", "error", "timeout", "stt"),
        ("error", "expired", "timeout", "diarize"),
    ],
)
def test_simultaneous_part_failures_choose_timeout_then_unavailable_then_error(
    client: ClientHarness,
    stt_status: ResultStatus,
    diarize_status: ResultStatus,
    expected: str,
    source: str,
) -> None:
    """Catch filesystem/job iteration order replacing the most significant ready failure."""
    peer = _result_peer(
        client,
        {
            "stt": ResultRecord("", stt_status, error_detail="stt detail"),
            "diarize": ResultRecord("", diarize_status, error_detail="diarize detail"),
        },
    )
    clock = PolicyClock()
    result = client.run(
        client.request(client.media(), diarization=DiarizeOptions()),
        spawn=peer,
        clock=clock,
        sleeper=clock.sleep,
    ).outcomes[0]
    assert result.status == expected
    assert f"{source} detail" in (result.detail or "")
    assert ("speaker diarization" in (result.detail or "")) == (source == "diarize")


@pytest.mark.parametrize("stt_finished", [False, True])
def test_paired_call_deadline_cancels_all_remaining_parts(
    client: ClientHarness, *, stt_finished: bool
) -> None:
    """Catch a call timing out only its primary job or forgetting an already completed part."""
    peer = _result_peer(
        client,
        {"stt": ResultRecord("", "ok", text="partial", data={"segments": []})}
        if stt_finished
        else {},
    )
    clock = PolicyClock()
    reported: list[InputOutcome] = []
    result = client.run(
        client.request(
            client.media(), diarization=DiarizeOptions(), client_timeout_s=5
        ),
        spawn=peer,
        clock=clock,
        sleeper=clock.sleep,
        poll_interval_s=1,
        on_outcome=reported.append,
    ).outcomes[0]
    assert clock() == 1005
    assert result.status == "timeout"
    assert result.text == ""
    assert result.data is None
    assert reported == [result]
    jobs = client.jobs()
    assert len(jobs) == 2
    assert {job.deadline_at for job in jobs} == {1005}
    assert {
        job.kind for job in jobs if is_cancel_requested(client.pool.paths, job.job_id)
    } == ({"diarize"} if stt_finished else {"stt", "diarize"})


def test_second_job_disk_failure_cancels_the_published_first_job(
    client: ClientHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch partial two-job submission leaking STT work or reporting a false successful input."""
    real_replace = os.replace

    def replace_file(source: str | Path, destination: str | Path) -> None:
        target = Path(destination)
        if target.parent == client.pool.paths.incoming:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
            if payload["kind"] == "diarize":
                raise OSError(errno.ENOSPC, "fixture disk full", str(target))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace_file)
    reported: list[InputOutcome] = []
    outcome = client.run(
        client.request(client.media(), diarization=DiarizeOptions()),
        on_outcome=reported.append,
    ).outcomes[0]
    assert outcome.status == "unavailable"
    assert "speaker diarization" in (outcome.detail or "")
    jobs = client.jobs()
    assert len(jobs) == 1
    assert jobs[0].kind == "stt"
    assert is_cancel_requested(client.pool.paths, jobs[0].job_id)
    assert client.spawned == 0
    assert reported == [outcome]


@pytest.mark.parametrize(
    ("window", "media_s", "expected"),
    [
        (MediaWindow(10.25, 100.0), 60.0, 49.75),
        (MediaWindow(10.25), 60.0, 49.75),
        (MediaWindow(0.0, 5.5), 60.0, 5.5),
        (MediaWindow(10.25, 2.5), None, 2.5),
        (MediaWindow(10.25), None, None),
    ],
)
def test_client_window_durations_apply_to_both_parts_before_submission(
    client: ClientHarness,
    window: MediaWindow,
    media_s: float | None,
    expected: float | None,
) -> None:
    """Catch full-media ETA reservations for cut jobs or different windows for sibling engines."""
    peer = _result_peer(
        client,
        {
            "stt": ResultRecord("", "empty", data={"segments": []}),
            "diarize": ResultRecord("", "ok", data={"turns": []}),
        },
    )
    result = client.run(
        client.request(client.media(), window=window, diarization=DiarizeOptions()),
        spawn=peer,
        probe_duration=lambda _path, _timeout: media_s,
    ).outcomes[0]
    assert result.status == "empty"
    jobs = client.jobs()
    assert len(jobs) == 2
    assert all(job.duration_s == expected and job.window == window for job in jobs)


@pytest.mark.parametrize("start", [60.0, 61.0])
def test_window_at_or_beyond_end_fails_before_any_part_is_written(
    client: ClientHarness, start: float
) -> None:
    """Catch empty off-media windows consuming pool capacity or publishing only one sibling."""
    reported: list[InputOutcome] = []
    outcome = client.run(
        client.request(
            client.media(), window=MediaWindow(start, 1.0), diarization=DiarizeOptions()
        ),
        probe_duration=lambda _path, _timeout: 60.0,
        on_outcome=reported.append,
    ).outcomes[0]
    assert outcome.status == "error"
    assert reported == [outcome]
    assert not client.jobs()
    assert client.spawned == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "invalid"},
        {"mode": "diarize", "language": "en"},
        {"mode": "diarize", "language": None, "want_segments": True},
        {
            "mode": "diarize",
            "language": None,
            "decode": WhisperDecodeOptions(vad_filter=False),
        },
        {"mode": "diarize", "language": None, "device": "cuda"},
        {"provider": "gigaam-ctc", "decode": WhisperDecodeOptions(beam_size=10)},
        {
            "provider": "gigaam-e2e-ctc",
            "decode": WhisperDecodeOptions(word_timestamps=False),
        },
    ],
)
def test_invalid_option_combinations_raise_without_queue_side_effects(
    client: ClientHarness, overrides: dict[str, object]
) -> None:
    """Catch incompatible options running expensive jobs with silently ignored settings."""
    with pytest.raises(ValueError):  # noqa: PT011 - validation type and zero side effects are the contract
        client.run(client.request(client.media(), **overrides))
    assert not client.jobs()
    assert client.spawned == 0


@pytest.mark.parametrize(
    ("bad_kind", "data"),
    [("stt", None), ("stt", {"segments": ["invalid"]}), ("diarize", {"turns": {}})],
)
def test_malformed_success_payload_is_an_explicit_input_error(
    client: ClientHarness, bad_kind: str, data: dict[str, object] | None
) -> None:
    """Catch a broken daemon payload being silently emitted as a complete transcript."""
    templates = {
        "stt": ResultRecord("", "ok", text="recognized", data={"segments": []}),
        "diarize": ResultRecord("", "ok", data={"turns": []}),
    }
    templates[bad_kind] = replace(templates[bad_kind], data=data)
    result = client.run(
        client.request(client.media(), diarization=DiarizeOptions()),
        spawn=_result_peer(client, templates),
    ).outcomes[0]
    assert result.status == "error"
    assert result.text == ""
    assert result.data is None
    assert ("segments" if bad_kind == "stt" else "turns") in (result.detail or "")


def test_one_failed_input_cancels_only_its_sibling_and_preserves_other_input(
    client: ClientHarness,
) -> None:
    """Catch a failed input cancelling another input or mixing sibling results in a batch."""
    segments = [{"start": 1.0, "end": 2.0, "text": "second input"}]

    def publish(paths: QueuePaths) -> None:
        write_heartbeat(paths, {"state": "running"})
        for job in client.jobs():
            if job.call_index == 0:
                if job.kind == "stt":
                    write_result(paths, ResultRecord(job.job_id, "error"))
                continue
            write_result(
                paths,
                ResultRecord(
                    job.job_id,
                    "ok",
                    text="second input" if job.kind == "stt" else "",
                    data={"segments": segments} if job.kind == "stt" else {"turns": []},
                ),
            )

    reported: list[InputOutcome] = []
    result = client.run(
        client.request(client.media(), client.media(), diarization=DiarizeOptions()),
        spawn=publish,
        on_outcome=reported.append,
    )
    assert [outcome.status for outcome in result.outcomes] == ["error", "ok"]
    assert result.outcomes[1].text == "second input"
    assert result.outcomes[1].data == {
        "segments": segments,
        "turns": [],
        "speakers": [None],
    }
    assert sorted(outcome.index for outcome in reported) == [0, 1]
    assert {
        (job.call_index, job.kind)
        for job in client.jobs()
        if is_cancel_requested(client.pool.paths, job.job_id)
    } == {(0, "diarize")}
