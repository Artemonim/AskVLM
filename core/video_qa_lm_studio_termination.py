"""Normalize LM Studio and OpenAI-compatible termination metadata for Video QA."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

_SIGNAL_KEYS: frozenset[str] = frozenset({"stopReason", "stop_reason", "finish_reason"})
_MAX_WALK_DEPTH: int = 32
_CANCELLED_LITERALS: frozenset[str] = frozenset(
    {"usercancelled", "cancelled", "canceled", "aborted", "interrupted"},
)


class LlmTerminationKind(str, Enum):
    """High-level classification of why generation stopped."""

    CONTEXT_OVERFLOW = "context_overflow"
    OUTPUT_LENGTH_LIMIT = "output_length_limit"
    STOP_STRING = "stop_string"
    TOOL_CALLS = "tool_calls"
    USER_CANCELLED = "user_cancelled"
    NORMAL = "normal"
    UNKNOWN = "unknown"


_PRIORITY: tuple[LlmTerminationKind, ...] = (
    LlmTerminationKind.USER_CANCELLED,
    LlmTerminationKind.CONTEXT_OVERFLOW,
    LlmTerminationKind.TOOL_CALLS,
    LlmTerminationKind.STOP_STRING,
    LlmTerminationKind.OUTPUT_LENGTH_LIMIT,
    LlmTerminationKind.NORMAL,
    LlmTerminationKind.UNKNOWN,
)


def _priority_index(kind: LlmTerminationKind) -> int:
    return _PRIORITY.index(kind)


@dataclass(frozen=True, slots=True)
class LlmTerminationInfo:
    """Typed view of termination signals for downstream policy."""

    kind: LlmTerminationKind
    raw_signals: tuple[str, ...]

    @property
    def is_overflow(self) -> bool:
        """True when the context window limit was reached (input-side limit)."""
        return self.kind == LlmTerminationKind.CONTEXT_OVERFLOW

    @property
    def is_truncated(self) -> bool:
        """True when a length limit stopped generation (context or max output)."""
        return self.kind in (
            LlmTerminationKind.CONTEXT_OVERFLOW,
            LlmTerminationKind.OUTPUT_LENGTH_LIMIT,
        )

    @property
    def is_stop_string(self) -> bool:
        """True when a configured stop string ended generation."""
        return self.kind == LlmTerminationKind.STOP_STRING

    @property
    def is_tool_calls(self) -> bool:
        """True when the model ended to emit tool calls."""
        return self.kind == LlmTerminationKind.TOOL_CALLS

    @property
    def is_user_cancelled(self) -> bool:
        """True when inference was cancelled or aborted by the user or runtime."""
        return self.kind == LlmTerminationKind.USER_CANCELLED

    @property
    def is_normal_completion(self) -> bool:
        """True for natural completion (e.g. EOS / OpenAI finish_reason stop)."""
        return self.kind == LlmTerminationKind.NORMAL

    @property
    def is_unknown(self) -> bool:
        """True when no known termination signal was found."""
        return self.kind == LlmTerminationKind.UNKNOWN


def normalize_video_qa_llm_termination(payload: object) -> LlmTerminationInfo:
    """Extract and classify termination hints from LM Studio or OpenAI-shaped JSON.

    Walks nested dict/list structures and collects ``stopReason``, ``stop_reason``,
    and ``finish_reason`` string values. Does not perform network I/O.

    Args:
        payload: Parsed JSON (typically ``dict``) or any nested structure.

    Returns:
        ``LlmTerminationInfo`` with ``kind`` and convenience flags for policy code.

    """
    signals = _collect_signal_strings(payload)
    if not signals:
        return LlmTerminationInfo(kind=LlmTerminationKind.UNKNOWN, raw_signals=())

    kinds = [_classify_signal(s) for s in signals]
    merged = _merge_kinds(kinds)
    ordered = tuple(sorted(set(signals)))
    return LlmTerminationInfo(kind=merged, raw_signals=ordered)


def _merge_kinds(kinds: list[LlmTerminationKind]) -> LlmTerminationKind:
    known = [k for k in kinds if k != LlmTerminationKind.UNKNOWN]
    if not known:
        return LlmTerminationKind.UNKNOWN
    return min(known, key=_priority_index)


def _collect_signal_strings(obj: object, depth: int = 0) -> list[str]:
    out: list[str] = []
    if depth > _MAX_WALK_DEPTH:
        return out
    if isinstance(obj, Mapping):
        for key in _SIGNAL_KEYS:
            if key not in obj:
                continue
            val = obj[key]
            text = _stringify_signal_value(val)
            if text:
                out.append(text)
        for child in obj.values():
            out.extend(_collect_signal_strings(child, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_collect_signal_strings(item, depth + 1))
    return out


def _stringify_signal_value(val: object) -> str | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, str):
        cleaned = val.strip()
        return cleaned if cleaned else None
    if isinstance(val, int | float):
        return str(val).strip()
    return None


def _normalize_for_match(s: str) -> str:
    t = s.lower().strip()
    return t.replace("_", "").replace("-", "").replace(" ", "")


def _classify_signal(s: str) -> LlmTerminationKind:
    t_norm = _normalize_for_match(s)
    kind = LlmTerminationKind.UNKNOWN

    if t_norm in _CANCELLED_LITERALS or (
        "usercancel" in t_norm or t_norm.endswith(("cancelled", "canceled"))
    ):
        kind = LlmTerminationKind.USER_CANCELLED
    elif t_norm in {"contextlengthreached", "contextlength"}:
        kind = LlmTerminationKind.CONTEXT_OVERFLOW
    elif t_norm in {"toolcalls", "functioncall", "functioncalls"}:
        kind = LlmTerminationKind.TOOL_CALLS
    elif t_norm == "stopstringfound" or "stopstring" in t_norm:
        kind = LlmTerminationKind.STOP_STRING
    elif t_norm in {
        "maxpredictedtokensreached",
        "length",
        "maxtokens",
        "maxcompletiontokens",
    }:
        kind = LlmTerminationKind.OUTPUT_LENGTH_LIMIT
    elif t_norm == "stop" or t_norm in {"eos", "endoftext", "completed", "done"}:
        kind = LlmTerminationKind.NORMAL

    return kind
