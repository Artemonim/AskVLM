"""Benchmark worker process: load one engine, warm up, and serve dispatched work.

The child is started by the parent as
``python -X utf8 tools/benchmark_stt_concurrency.py _child`` with the thread
environment (``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS``) already set, so torch
and CTranslate2 pick it up on their first import. The protocol is described in
:mod:`stt_concurrency.protocol`.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING, NoReturn, TextIO

from .engines import make_engine
from .protocol import EXIT_PARENT_GONE, ChildSpec, decode_message, encode_message

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .engines import SttEngine

# * Thread-related environment captured in the "loaded" message.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
_ERROR_TEXT_LIMIT = 2000


class _Channel:
    """Thread-safe protocol writer on a private duplicate of the original stdout.

    File descriptor 1 is redirected to stderr (the per-child log), so prints of
    third-party libraries can never corrupt the protocol stream.
    """

    def __init__(self) -> None:
        sys.stdout.flush()
        protocol_fd = os.dup(1)
        os.dup2(2, 1)
        self._stream: TextIO = os.fdopen(
            protocol_fd,
            "w",
            encoding="utf-8",
            newline="\n",
        )
        self._lock = threading.Lock()

    def send(self, message: Mapping[str, object]) -> None:
        """Write one message and flush.

        Args:
            message: Protocol message.

        """
        line = encode_message(message)
        with self._lock:
            try:
                self._stream.write(line)
                self._stream.flush()
            except (OSError, ValueError):
                # ! The parent closed the pipe: it is gone, so stop at once.
                os._exit(EXIT_PARENT_GONE)

    def close(self) -> None:
        """Close the protocol stream (best effort)."""
        with self._lock, contextlib.suppress(OSError, ValueError):
            self._stream.close()


class _WorkRouter:
    """Routes ``work``/``stop`` messages from stdin to per-slot queues."""

    def __init__(self, slots: int) -> None:
        self._queues: list[queue.Queue[list[tuple[int, int]]]] = [
            queue.Queue() for _ in range(slots)
        ]
        self._stopping = threading.Event()

    def read_stdin(self) -> None:
        """Read stdin until EOF; exit the process if the parent disappears."""
        for line in iter(sys.stdin.readline, ""):
            message = decode_message(line)
            if message is None:
                continue
            kind = message["type"]
            if kind == "work":
                slot = int(message["slot"])
                items = [int(item) for item in message["items"]]
                # * Without "clips" an item id is the clip index itself.
                clips = [int(clip) for clip in message.get("clips", items)]
                self._queues[slot].put(list(zip(items, clips, strict=True)))
            elif kind == "stop":
                self._stopping.set()
                for slot_queue in self._queues:
                    slot_queue.put([])
        if not self._stopping.is_set():
            # ! Parent is gone: never linger as an orphan holding a model.
            os._exit(EXIT_PARENT_GONE)

    def mark_stopping(self) -> None:
        """Mark a regular shutdown so that a later stdin EOF is expected."""
        self._stopping.set()

    def take(self, slot: int) -> list[tuple[int, int]]:
        """Block until work for *slot* arrives; an empty list means stop.

        Args:
            slot: Worker slot.

        Returns:
            ``(item id, clip index)`` pairs.

        """
        return self._queues[slot].get()


def _describe(exc: BaseException) -> str:
    """Return a bounded one-line description of *exc*."""
    return f"{type(exc).__name__}: {exc}"[:_ERROR_TEXT_LIMIT]


def _slot_loop(
    *,
    slot: int,
    spec: ChildSpec,
    engine: SttEngine,
    router: _WorkRouter,
    channel: _Channel,
) -> None:
    """Request work for *slot* until the parent answers with an empty list."""
    paths = {item.index: item.path for item in spec.items}
    while True:
        channel.send({"type": "want", "slot": slot, "count": spec.request_count})
        work = router.take(slot)
        if not work:
            return
        started = time.perf_counter()
        error: str | None = None
        try:
            texts = engine.transcribe([paths[clip] for _item, clip in work], slot=slot)
        except Exception as exc:  # noqa: BLE001 - item failures are benchmark data
            texts = [""] * len(work)
            error = _describe(exc)
            traceback.print_exc()
        proc_s = time.perf_counter() - started
        for (item, _clip), text in zip(work, texts, strict=True):
            channel.send(
                {
                    "type": "result",
                    "slot": slot,
                    "item": item,
                    "text": text,
                    "proc_s": proc_s,
                    "batch": len(work),
                    "error": error,
                },
            )


def _slot_main(
    *,
    slot: int,
    spec: ChildSpec,
    engine: SttEngine,
    router: _WorkRouter,
    channel: _Channel,
) -> None:
    """Run :func:`_slot_loop` and report an unexpected crash as ``fatal``."""
    try:
        _slot_loop(slot=slot, spec=spec, engine=engine, router=router, channel=channel)
    except Exception as exc:  # noqa: BLE001 - the parent must abort, not time out
        traceback.print_exc()
        channel.send(
            {
                "type": "fatal",
                "stage": f"slot-{slot}",
                "error": _describe(exc),
                "traceback": traceback.format_exc()[-_ERROR_TEXT_LIMIT:],
            },
        )


def _warmup(engine: SttEngine, spec: ChildSpec) -> dict[str, object]:
    """Run the primary warm-up alone, then one concurrent warm-up per slot.

    Args:
        engine: Loaded engine.
        spec: Child specification.

    Returns:
        Info of the primary warm-up.

    Raises:
        RuntimeError: When a concurrent warm-up fails.

    """
    info = engine.warmup(spec.warmup_path, slot=0, primary=True)
    slots = spec.config.slots
    if slots <= 1:
        return info
    errors: list[str] = []
    errors_lock = threading.Lock()

    def _one(slot: int) -> None:
        try:
            engine.warmup(spec.warmup_path, slot=slot, primary=False)
        except Exception as exc:  # noqa: BLE001 - collected and re-raised below
            with errors_lock:
                errors.append(f"slot {slot}: {_describe(exc)}")

    threads = [
        threading.Thread(target=_one, args=(slot,), name=f"warmup-{slot}")
        for slot in range(slots)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError("; ".join(errors))
    info["concurrent_warmup_slots"] = slots
    return info


def _serve(channel: _Channel) -> int:
    """Run the worker protocol on an open channel.

    Args:
        channel: Protocol writer.

    Returns:
        Process exit code.

    """
    stage = "spec"
    try:
        spec = ChildSpec.from_json(json.loads(sys.stdin.readline()))
        engine = make_engine(spec)
        stage = "load"
        started = time.perf_counter()
        info = engine.load()
        info["env"] = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
        channel.send(
            {
                "type": "loaded",
                "load_s": time.perf_counter() - started,
                "info": info,
                "pid": os.getpid(),
            },
        )
        stage = "warmup"
        started = time.perf_counter()
        warmup_info = _warmup(engine, spec)
        channel.send(
            {
                "type": "ready",
                "warmup_s": time.perf_counter() - started,
                "warmup": warmup_info,
                "cpu_s": time.process_time(),
            },
        )
    except Exception as exc:  # noqa: BLE001 - reported to the parent as fatal
        traceback.print_exc()
        channel.send(
            {
                "type": "fatal",
                "stage": stage,
                "error": _describe(exc),
                "traceback": traceback.format_exc()[-_ERROR_TEXT_LIMIT:],
            },
        )
        return 1
    router = _WorkRouter(spec.config.slots)
    reader = threading.Thread(
        target=router.read_stdin, name="stdin-router", daemon=True
    )
    reader.start()
    workers = [
        threading.Thread(
            target=_slot_main,
            kwargs={
                "slot": slot,
                "spec": spec,
                "engine": engine,
                "router": router,
                "channel": channel,
            },
            name=f"slot-{slot}",
        )
        for slot in range(spec.config.slots)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    router.mark_stopping()
    channel.send({"type": "done", "cpu_s": time.process_time()})
    engine.close()
    return 0


def run_child() -> NoReturn:
    """Entry point of the hidden child mode; never returns.

    The process ends with ``os._exit`` once the protocol stream is flushed: the
    stdin router is a daemon thread blocked in ``readline``, and a regular
    interpreter shutdown can abort while that thread holds the stdin buffer
    lock. The engine is already closed explicitly at that point.
    """
    # * Ctrl+C belongs to the parent, which terminates the children itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    channel = _Channel()
    code = 1
    try:
        code = _serve(channel)
    finally:
        channel.close()
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(OSError, ValueError):
                stream.flush()
        os._exit(code)
