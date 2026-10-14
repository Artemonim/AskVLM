"""Parent/child JSON-lines protocol and the child specification.

Parent -> child (child stdin), one JSON object per line:

* first line: the :class:`ChildSpec`;
* ``{"type": "work", "slot": k, "items": [i, ...], "clips": [c, ...]}`` -
  item ids with the clip index (``AudioItem.index``) of each item; without
  ``clips`` the item id is the clip index. An empty list stops slot k;
* ``{"type": "stop"}`` - stop every slot (graceful end of the configuration).

Child -> parent (a private duplicate of the child's original stdout):

* ``loaded`` (``load_s``, ``info``), ``ready`` (``warmup_s``, ``warmup``,
  ``cpu_s``), ``want`` (``slot``, ``count``), ``result`` (``slot``, ``item``,
  ``text``, ``proc_s``, ``batch``, ``error``), ``done`` (``cpu_s``), and
  ``fatal`` (``stage``, ``error``, ``traceback``).

The child closes the protocol when stdin reaches EOF before a ``stop``: that
means the parent is gone, so the child exits with :data:`EXIT_PARENT_GONE`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .configs import BenchConfig
from .workload import AudioItem

if TYPE_CHECKING:
    from collections.abc import Mapping

EXIT_PARENT_GONE = 3

MESSAGE_TYPES = frozenset(
    {"loaded", "ready", "want", "result", "done", "fatal"},
)


def encode_message(message: Mapping[str, object]) -> str:
    """Encode one protocol message as a newline-terminated JSON line.

    Args:
        message: JSON-serializable mapping.

    Returns:
        The encoded line.

    """
    return json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"


def decode_message(line: str) -> dict[str, Any] | None:
    """Decode a protocol line; return ``None`` for anything that is not a message.

    Args:
        line: Raw line.

    Returns:
        The message mapping, or ``None``.

    """
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("type"), str):
        return None
    return data


@dataclass(frozen=True, slots=True)
class ChildSpec:
    """Everything a child worker needs to load its engine and run work.

    Attributes:
        config: Configuration being measured.
        child_index: Index of this child within the configuration.
        items: Full workload (the parent dispatches indices into it).
        warmup_path: Untimed warm-up clip.
        language: Whisper language code.
        whisper_model: faster-whisper model name.
        whisper_options: Fixed Whisper decode options (``language`` included).
        whisper_chunk_s: Fixed chunk length of batched long-file Whisper.
        gigaam_cuda_streams: Give each GigaAM ``S`` thread its own CUDA stream.

    """

    config: BenchConfig
    child_index: int
    items: tuple[AudioItem, ...]
    warmup_path: Path
    language: str
    whisper_model: str
    whisper_options: dict[str, object]
    whisper_chunk_s: float
    gigaam_cuda_streams: bool

    @property
    def request_count(self) -> int:
        """Items a slot asks for at once: the batch size for clip batching."""
        if self.config.mode == "B" and not self.config.long:
            return self.config.batch_size
        return 1

    def to_json(self) -> dict[str, object]:
        """Serialize for the first stdin line of the child."""
        return {
            "config": self.config.to_json(),
            "child_index": self.child_index,
            "items": [item.to_json() for item in self.items],
            "warmup_path": str(self.warmup_path),
            "language": self.language,
            "whisper_model": self.whisper_model,
            "whisper_options": dict(self.whisper_options),
            "whisper_chunk_s": self.whisper_chunk_s,
            "gigaam_cuda_streams": self.gigaam_cuda_streams,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ChildSpec:
        """Rebuild a spec serialized by :meth:`to_json`.

        Args:
            data: Serialized spec.

        Returns:
            The spec.

        """
        return cls(
            config=BenchConfig.from_json(data["config"]),
            child_index=int(data["child_index"]),
            items=tuple(AudioItem.from_json(item) for item in data["items"]),
            warmup_path=Path(str(data["warmup_path"])),
            language=str(data["language"]),
            whisper_model=str(data["whisper_model"]),
            whisper_options=dict(data["whisper_options"]),
            whisper_chunk_s=float(data["whisper_chunk_s"]),
            gigaam_cuda_streams=bool(data["gigaam_cuda_streams"]),
        )
