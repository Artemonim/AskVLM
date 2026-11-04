"""Delegated cgroup v2 memory caps and the gated Linux worker bootstrap.

Only the managed Linux process owner creates these groups. Descriptor-relative
operations reject symlinks and never walk above the discovered service cgroup.
The bootstrap waits for enrollment before executing any worker code; it avoids
``preexec_fn``, which is unsafe in the multithreaded supervisor.
"""

from __future__ import annotations

import argparse
import errno
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.stt_pool.config import is_supervisor_managed

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

CGROUP_ROOT = Path("/sys/fs/cgroup")
SELF_CGROUP = Path("/proc/self/cgroup")
START_GATE = b"\0"
_WORKER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True, slots=True)
class MemoryEvents:
    """OOM kill counters of one worker cgroup, never of the whole service."""

    oom_kill: int
    oom_group_kill: int = 0

    def killed_since(self, baseline: MemoryEvents | None) -> bool:
        """Return whether a kill counter grew after a valid baseline."""
        return baseline is not None and (
            self.oom_kill > baseline.oom_kill
            or self.oom_group_kill > baseline.oom_group_kill
        )


def managed_linux() -> bool:
    """Return whether this process explicitly opts into managed Linux caps."""
    return sys.platform == "linux" and is_supervisor_managed()


def _required_flag(name: str) -> int:
    """Require a Linux open flag; never weaken symlink protection on other hosts."""
    value: object = getattr(os, name, None)
    if not isinstance(value, int):
        msg = f"cgroup v2 requires the Linux open flag {name}"
        raise OSError(errno.ENOSYS, msg)
    return value


def _file_flags(access: int) -> int:
    """Return control-file flags, including no-follow and close-on-exec."""
    return access | _required_flag("O_NOFOLLOW") | _required_flag("O_CLOEXEC")


def _directory_flags() -> int:
    """Return Linux directory flags that prohibit symlink traversal."""
    return _file_flags(os.O_RDONLY) | _required_flag("O_DIRECTORY")


def _open_directory(path: Path) -> int:
    """Open an absolute directory without following any path component."""
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        msg = f"unsafe cgroup directory: {path}"
        raise ValueError(msg)
    fd = os.open(path.anchor, _directory_flags())
    try:
        for part in path.parts[1:]:
            child = os.open(part, _directory_flags(), dir_fd=fd)
            os.close(fd)
            fd = child
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_at(directory: int, name: str) -> str:
    """Read a kernel-owned control file without following a symlink."""
    fd = os.open(name, _file_flags(os.O_RDONLY), dir_fd=directory)
    with os.fdopen(fd, encoding="ascii") as stream:
        return stream.read()


def _write_at(directory: int, name: str, value: str) -> None:
    """Write an existing kernel control file in one syscall, without creating it."""
    fd = os.open(name, _file_flags(os.O_WRONLY), dir_fd=directory)
    try:
        data = (value + "\n").encode("ascii")
        if os.write(fd, data) != len(data):
            msg = f"short write to cgroup control file {name}"
            raise OSError(errno.EIO, msg)
    finally:
        os.close(fd)


def _service_path() -> Path:
    """Find the enclosing service in the unified membership, rejecting escapes."""
    entries = [
        line[3:]
        for line in SELF_CGROUP.read_text(encoding="ascii").splitlines()
        if line.startswith("0::")
    ]
    if len(entries) != 1:
        msg = "managed workers require one cgroup v2 membership in /proc/self/cgroup"
        raise ValueError(msg)
    raw = entries[0]
    parts = raw.split("/")
    if not raw.startswith("/") or any(
        part in {"", ".", ".."} or "\\" in part or "\0" in part for part in parts[1:]
    ):
        msg = f"unsafe cgroup v2 membership: {raw!r}"
        raise ValueError(msg)
    services = [index for index, part in enumerate(parts) if part.endswith(".service")]
    if not services:
        msg = "managed workers require a visible delegated .service cgroup"
        raise ValueError(msg)
    return CGROUP_ROOT.joinpath(*parts[1 : services[-1] + 1])


def _memory_events(directory: int) -> MemoryEvents:
    """Parse nonnegative OOM kill counters; pressure alone is not kill evidence."""
    counters: dict[str, int] = {}
    for line in _read_at(directory, "memory.events").splitlines():
        name, raw = line.split()
        if name in counters or not raw.isascii() or not raw.isdecimal():
            msg = "invalid memory.events counter"
            raise ValueError(msg)
        counters[name] = int(raw)
    if "oom_kill" not in counters:
        msg = "memory.events has no oom_kill counter"
        raise ValueError(msg)
    return MemoryEvents(counters["oom_kill"], counters.get("oom_group_kill", 0))


def _remove_empty(parent: int, name: str, directory: int) -> None:
    """Remove only this unchanged, empty group; leave populated groups alone."""
    try:
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        owned = os.fstat(directory)
        if (current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino):
            logger.warning(
                "cgroup cleanup refused: directory identity changed: %s", name
            )
            return
        if _read_at(directory, "cgroup.procs").strip():
            logger.warning("cgroup cleanup deferred: group is populated: %s", name)
            return
        # * rmdir also refuses child groups or concurrent process enrollment.
        os.rmdir(name, dir_fd=parent)
    except OSError as exc:
        logger.warning(
            "cgroup cleanup left group in place: name=%s error=%s", name, exc
        )


def _enable_memory(service: int) -> None:
    """Enable memory only inside the delegated service, moving only this process.

    The v2 no-internal-process rule requires an empty service root. A private
    supervisor leaf holds this process until systemd removes the stopped unit.
    Other root processes are never moved, killed, or adopted.
    """
    if "memory" in _read_at(service, "cgroup.subtree_control").split():
        return
    if "memory" not in _read_at(service, "cgroup.controllers").split():
        msg = (
            "memory controller is not delegated to the service (Delegate=yes required)"
        )
        raise OSError(errno.ENODEV, msg)
    members = _read_at(service, "cgroup.procs").split()
    if members:
        if set(members) != {str(os.getpid())}:
            msg = (
                "delegated service root contains other processes; refusing to move them"
            )
            raise OSError(errno.EBUSY, msg)
        name = f"askvlm-supervisor-{uuid.uuid4().hex}"
        os.mkdir(name, mode=0o700, dir_fd=service)
        leaf = os.open(name, _directory_flags(), dir_fd=service)
        try:
            _write_at(leaf, "cgroup.procs", str(os.getpid()))
        except BaseException:
            _remove_empty(service, name, leaf)
            raise
        finally:
            os.close(leaf)
    if _read_at(service, "cgroup.procs").strip():
        msg = "delegated service root is not empty; cannot enable the memory controller"
        raise OSError(errno.EBUSY, msg)
    _write_at(service, "cgroup.subtree_control", "+memory")


class WorkerCgroup:
    """Own one fresh worker leaf and its control-file descriptors.

    No cgroup kill operation is used. Cleanup never moves processes, walks
    foreign groups, or recursively deletes directories.
    """

    def __init__(
        self,
        parent: int,
        directory: int,
        path: Path,
        baseline: MemoryEvents,
    ) -> None:
        self._parent = parent
        self._directory = directory
        self.path = path
        self.baseline = baseline
        self.closed = False

    @classmethod
    def create(cls, worker_id: str, limit_bytes: int) -> WorkerCgroup:
        """Create and cap a fresh leaf; fail before spawn if delegation is unusable."""
        if not managed_linux():
            msg = "cgroup v2 worker limits require managed Linux"
            raise OSError(errno.ENOSYS, msg)
        if _WORKER_ID.fullmatch(worker_id) is None or limit_bytes <= 0:
            msg = "invalid worker identifier or cgroup memory limit"
            raise ValueError(msg)
        path = _service_path()
        parent = _open_directory(path)
        directory: int | None = None
        name = f"askvlm-worker-{worker_id}-{uuid.uuid4().hex}"
        try:
            _enable_memory(parent)
            os.mkdir(name, mode=0o700, dir_fd=parent)
            directory = os.open(name, _directory_flags(), dir_fd=parent)
            _write_at(directory, "memory.max", str(limit_bytes))
            baseline = _memory_events(directory)
            return cls(parent, directory, path / name, baseline)
        except BaseException:
            if directory is not None:
                _remove_empty(parent, name, directory)
                os.close(directory)
            os.close(parent)
            raise

    def assign(self, pid: int) -> None:
        """Enroll the gated worker PID; its later children inherit this cap."""
        if self.closed or pid <= 0:
            msg = "cannot enroll a worker in a closed cgroup or with an invalid PID"
            raise ValueError(msg)
        _write_at(self._directory, "cgroup.procs", str(pid))

    def memory_events(self) -> MemoryEvents:
        """Read only this worker's hierarchical OOM kill counters."""
        if self.closed:
            msg = "worker cgroup is closed"
            raise ValueError(msg)
        return _memory_events(self._directory)

    def close(self) -> None:
        """Close descriptors and remove only the owned empty leaf; idempotent."""
        if self.closed:
            return
        self.closed = True
        try:
            _remove_empty(self._parent, self.path.name, self._directory)
        finally:
            os.close(self._directory)
            os.close(self._parent)


def _main(argv: Sequence[str] | None = None) -> int:
    """Wait for parent enrollment before exec, with no worker imports or forks."""
    parser = argparse.ArgumentParser(description="Internal gated STT worker bootstrap")
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="worker command after --"
    )
    args = parser.parse_args(argv)
    command: list[str] = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a worker command is required")
    if os.read(sys.stdin.fileno(), len(START_GATE)) != START_GATE:
        sys.stderr.write("worker bootstrap: cgroup enrollment was not confirmed\n")
        return 1
    try:
        os.execvpe(command[0], command, os.environ)  # noqa: S606 - supervisor-built argv
    except OSError as exc:
        sys.stderr.write(f"worker bootstrap: cannot execute worker: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
