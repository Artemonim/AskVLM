"""Safe client/CLI scenarios with real v2 storage and fake model workers."""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

from core.stt_pool.client import ClientRequest, run_client
from core.stt_pool.queue import JobRecord, read_job_payload
from tests.stt_pool_supervisor_support import SupervisorHarness

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest

    from core.stt_pool.client import ClientOutcome
    from core.stt_pool.queue import QueuePaths


class ClientHarness:
    """Keep the client, queue, supervisor, protocol and worker entry connected."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.pool = SupervisorHarness(tmp_path)
        self.spawned = 0
        self._input_counter = 0
        self._threads: list[threading.Thread] = []
        self._failures: list[BaseException] = []
        config_path = tmp_path / "pool.json"
        config_path.write_text('{"gpu":{"max_workers":0}}', encoding="utf-8")
        monkeypatch.setenv("ASKVLM_EXTERNAL_QUEUE_DIR", str(self.pool.paths.root))
        monkeypatch.setenv("ASKVLM_STT_POOL_CONFIG", str(config_path))
        monkeypatch.setattr("core.stt_pool.client.spawn_supervisor", self.spawn)
        monkeypatch.setattr(
            "core.stt_pool.client._probe_with_ffprobe", lambda _path, _timeout: None
        )

    def spawn(self, paths: QueuePaths) -> None:
        """Launch a supervisor with fake models in real workers."""
        assert paths.root == self.pool.paths.root
        self.spawned += 1
        if self.pool.thread is None or not self.pool.thread.is_alive():
            self.pool.start()

    def media(self, name: str | None = None, **instructions: object) -> Path:
        """Create media read by the worker's fake recognizer."""
        self._input_counter += 1
        path = self.pool.engine_root / (name or f"input-{self._input_counter}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(instructions), encoding="utf-8")
        return path

    def request(self, *inputs: Path, **overrides: object) -> ClientRequest:
        """Build a CPU request without machine configuration or a real model factory."""
        fields = {
            "inputs": tuple(inputs),
            "provider": "whisper",
            "whisper_model": "small",
            "compute_type": "auto",
            "device": "cpu",
            "language": "ru",
            "priority": "normal",
            "max_parallel": None,
            "client_timeout_s": 300.0,
            **overrides,
        }
        return ClientRequest(**fields)

    def run(self, request: ClientRequest, **seams: object) -> ClientOutcome:
        """Drive the client with policy time and bounded polling."""

        def pause(delay: float) -> None:
            if self._failures:
                raise self._failures[0]
            threading.Event().wait(min(delay, 0.01))

        options = {
            "queue_root": self.pool.paths.root,
            "spawn": self.spawn,
            "clock": self.pool.clock,
            "sleeper": pause,
            "probe_duration": lambda _path, _timeout: None,
            "poll_interval_s": 0.01,
            **seams,
        }
        return run_client(request, **options)

    def jobs(self) -> list[JobRecord]:
        """Read the submitted records through the production queue parser."""
        jobs = [
            JobRecord.from_dict(payload, expected_job_id=path.stem)
            for path in self.pool.paths.incoming.glob("*.json")
            if (payload := read_job_payload(self.pool.paths, path.stem)) is not None
        ]
        return sorted(
            (job for job in jobs if job is not None), key=lambda job: job.call_index
        )

    def background(self, action: Callable[[], None]) -> threading.Thread:
        """Propagate gate failures to the test thread."""

        def invoke() -> None:
            try:
                action()
            except BaseException as exc:  # noqa: BLE001 - gate failures must reach the test thread, including interrupts
                self._failures.append(exc)

        thread = threading.Thread(target=invoke, daemon=True)
        self._threads.append(thread)
        thread.start()
        return thread

    def finish(self) -> None:
        """Release every gate and prove cleanup of all test-owned worker processes."""
        try:
            if self.pool.thread is not None:
                self.pool.finish(self.pool.code or 0)
        finally:
            self.pool.cleanup()
        for thread in self._threads:
            thread.join(timeout=30)
            assert not thread.is_alive()
        if self._failures:
            raise self._failures[0]


class PolicyClock:
    """Advance only on an explicitly requested probe or client polling pause."""

    def __init__(self, value: float = 1000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        """Read deterministic call time."""
        return self.value

    def sleep(self, seconds: float) -> None:
        """Consume call budget without a real wait."""
        self.value += seconds
