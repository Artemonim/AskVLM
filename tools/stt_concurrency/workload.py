"""Workload discovery, transcript normalization, and small statistics helpers."""

from __future__ import annotations

import difflib
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

WAV_SUFFIX = ".wav"
# * The warm-up clip must fit one GigaAM window so the API parity check applies.
WARMUP_MAX_DURATION_S = 20.0


class WorkloadError(ValueError):
    """Raised when the workload cannot be built from the given paths."""


@dataclass(frozen=True, slots=True)
class AudioItem:
    """One unit of work: an audio file and its duration.

    Attributes:
        index: Position in the workload (stable across configurations).
        path: Absolute audio path.
        duration_s: Duration from the WAV header.

    """

    index: int
    path: Path
    duration_s: float

    def to_json(self) -> dict[str, object]:
        """Serialize for the child spec and result files."""
        return {
            "index": self.index,
            "path": str(self.path),
            "name": self.path.name,
            "duration_s": self.duration_s,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> AudioItem:
        """Rebuild an item serialized by :meth:`to_json`.

        Args:
            data: Serialized item.

        Returns:
            The item.

        """
        return cls(
            index=int(data["index"]),
            path=Path(str(data["path"])),
            duration_s=float(data["duration_s"]),
        )


def wav_duration_s(path: Path) -> float:
    """Return the duration of a PCM WAV file from its header.

    Args:
        path: WAV path.

    Returns:
        Duration in seconds.

    Raises:
        WorkloadError: When the header cannot be read.

    """
    try:
        with wave.open(str(path), "rb") as reader:
            rate = reader.getframerate()
            frames = reader.getnframes()
    except (wave.Error, EOFError, OSError) as exc:
        msg = f"Cannot read the WAV header of {path}: {exc}"
        raise WorkloadError(msg) from exc
    if rate <= 0:
        msg = f"WAV {path} reports an invalid sample rate {rate}"
        raise WorkloadError(msg)
    return frames / float(rate)


def discover_clips(clips_dir: Path, max_clips: int) -> list[AudioItem]:
    """Select clips deterministically: ``*.wav`` sorted by name, first N.

    Args:
        clips_dir: Directory with WAV clips.
        max_clips: Maximum clip count; ``0`` or less keeps all clips.

    Returns:
        Workload items.

    Raises:
        WorkloadError: When the directory is missing or has no WAV files.

    """
    if not clips_dir.is_dir():
        msg = f"Clips directory does not exist: {clips_dir}"
        raise WorkloadError(msg)
    paths = sorted(
        (
            path
            for path in clips_dir.iterdir()
            if path.is_file() and path.suffix.lower() == WAV_SUFFIX
        ),
        key=lambda path: path.name,
    )
    if max_clips > 0:
        paths = paths[:max_clips]
    if not paths:
        msg = f"No {WAV_SUFFIX} clips found in {clips_dir}"
        raise WorkloadError(msg)
    return [
        AudioItem(index=index, path=path.resolve(), duration_s=wav_duration_s(path))
        for index, path in enumerate(paths)
    ]


def load_long_files(paths: Sequence[Path]) -> list[AudioItem]:
    """Build the long-file workload in the given order.

    Args:
        paths: Long WAV files.

    Returns:
        Workload items.

    Raises:
        WorkloadError: When a file is missing or not a WAV.

    """
    items: list[AudioItem] = []
    for index, path in enumerate(paths):
        if not path.is_file():
            msg = f"Long file does not exist: {path}"
            raise WorkloadError(msg)
        if path.suffix.lower() != WAV_SUFFIX:
            msg = f"Long file must be a {WAV_SUFFIX} file: {path}"
            raise WorkloadError(msg)
        items.append(
            AudioItem(index=index, path=path.resolve(), duration_s=wav_duration_s(path))
        )
    return items


def pick_warmup(items: Sequence[AudioItem], override: Path | None) -> Path:
    """Return the warm-up clip: the override, else the first clip of <= 20 s.

    Args:
        items: Clip workload.
        override: Explicit warm-up clip.

    Returns:
        Absolute warm-up path.

    Raises:
        WorkloadError: When the override does not exist.

    """
    if override is not None:
        if not override.is_file():
            msg = f"Warm-up clip does not exist: {override}"
            raise WorkloadError(msg)
        return override.resolve()
    for item in items:
        if item.duration_s <= WARMUP_MAX_DURATION_S:
            return item.path
    return items[0].path


def normalize_text(text: str) -> str:
    """Normalize a transcript for comparison: casefold, drop punctuation.

    Args:
        text: Raw transcript.

    Returns:
        Lowercase alphanumeric words separated by single spaces.

    """
    kept = "".join(ch if ch.isalnum() else " " for ch in text.casefold())
    return " ".join(kept.split())


def similarity(left: str, right: str) -> float:
    """Return the character-level similarity ratio of two strings in ``[0, 1]``.

    Args:
        left: First string.
        right: Second string.

    Returns:
        ``difflib`` ratio; two empty strings are identical.

    """
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def percentile(values: Sequence[float], q: float) -> float | None:
    """Return the *q*-th percentile with linear interpolation.

    Args:
        values: Samples.
        q: Percentile in ``[0, 100]``.

    Returns:
        The percentile, or ``None`` for an empty sample.

    """
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass(frozen=True, slots=True)
class Occurrence:
    """One dispatched unit: a clip occurrence within a (possibly repeated) stream.

    Attributes:
        item_id: Unique id within the configuration (dispatch and result key).
        clip: The audio clip; ``clip.index`` identifies it across repeats.
        cycle: Zero-based pass over the stream's clip list.

    """

    item_id: int
    clip: AudioItem
    cycle: int


class WorkStream:
    """Endless cycle over a clip list with a minimum size and a minimum duration.

    Items are produced in clip order, wrapping around after the last clip, while
    fewer than ``min_items`` were produced or less than ``min_wall_s`` elapsed
    since the clock start, and never beyond ``max_items``. The first request that
    yields nothing closes the stream for good, so every worker slot sees the same
    end. A plain run over the clip list is ``min_items = max_items = len(clips)``
    with ``min_wall_s = 0``.
    """

    def __init__(
        self,
        clips: Sequence[AudioItem],
        *,
        min_items: int,
        min_wall_s: float,
        max_items: int,
        first_id: int = 0,
    ) -> None:
        """Create a stream.

        Args:
            clips: Clip list to cycle through (must not be empty).
            min_items: Items produced regardless of the elapsed time.
            min_wall_s: Keep producing until this much time elapsed.
            max_items: Hard cap of produced items.
            first_id: ``item_id`` of the first occurrence.

        Raises:
            WorkloadError: For an empty clip list.

        """
        if not clips:
            msg = "A work stream needs at least one clip"
            raise WorkloadError(msg)
        self._clips = tuple(clips)
        self._min_items = max(0, min_items)
        self._min_wall_s = max(0.0, min_wall_s)
        self._max_items = max(self._min_items, max_items)
        self._first_id = first_id
        self._produced: list[Occurrence] = []
        self.closed = False

    @property
    def clips(self) -> tuple[AudioItem, ...]:
        """The clip list of the stream."""
        return self._clips

    @property
    def produced(self) -> tuple[Occurrence, ...]:
        """Every occurrence produced so far, in order."""
        return tuple(self._produced)

    @property
    def min_items(self) -> int:
        """Items produced regardless of the elapsed time."""
        return self._min_items

    @property
    def min_wall_s(self) -> float:
        """Minimum measured time before the stream may close."""
        return self._min_wall_s

    def _occurrence(self, position: int) -> Occurrence:
        return Occurrence(
            item_id=self._first_id + position,
            clip=self._clips[position % len(self._clips)],
            cycle=position // len(self._clips),
        )

    def planned(self) -> tuple[Occurrence, ...]:
        """Return the produced occurrences plus those still owed to ``min_items``.

        An aborted configuration reports the owed ones as not processed.
        """
        owed = range(len(self._produced), self._min_items)
        return (*self._produced, *(self._occurrence(pos) for pos in owed))

    def minimum_met(self, elapsed_s: float) -> bool:
        """Return whether ``min_items`` were produced and ``min_wall_s`` elapsed.

        Args:
            elapsed_s: Seconds since the clock start.

        Returns:
            ``True`` once the stream's own minimum is reached.

        """
        return len(self._produced) >= self._min_items and elapsed_s >= self._min_wall_s

    def _may_produce(self, elapsed_s: float, *, hold_open: bool) -> bool:
        if len(self._produced) >= self._max_items:
            return False
        return hold_open or not self.minimum_met(elapsed_s)

    def take(
        self,
        count: int,
        elapsed_s: float,
        *,
        hold_open: bool = False,
    ) -> list[Occurrence]:
        """Produce up to *count* occurrences; an empty list closes the stream.

        Args:
            count: Requested items.
            elapsed_s: Seconds since the clock start.
            hold_open: Keep producing past the stream's own minimum (a combo
                feeds every side until all sides reached their minimum).

        Returns:
            The occurrences to dispatch.

        """
        if self.closed:
            return []
        batch: list[Occurrence] = []
        while len(batch) < max(1, count) and self._may_produce(
            elapsed_s,
            hold_open=hold_open,
        ):
            occurrence = self._occurrence(len(self._produced))
            self._produced.append(occurrence)
            batch.append(occurrence)
        if not batch:
            self.closed = True
        return batch
