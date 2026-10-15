"""Structured transcript and diarization output of the external STT contract.

The JSON layout (field names, key order and rounding) follows the hardsub
skill's ``transcribe.py`` segments and ``diarize.py`` turns, so the skill can
consume AskVLM output unchanged. The module depends on the standard library
and :mod:`core.stt_options` only, so the thin client and the supervisor can
merge results without loading any ML package.

Times in this module are media-relative: every producer passes the window
start as ``offset_s`` and :func:`media_time` is the only place that adds the
offset and rounds.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections.abc import Mapping
from typing import TYPE_CHECKING

from core.stt_options import SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# * Decimal places of every emitted time and score (as in the hardsub skill).
_ROUND_DIGITS = 3
# * Decimal places used to compare summed overlaps, so float noise never decides a tie.
_OVERLAP_DIGITS = 6

# * Fewest distinct voices among the transcript segments that make the text a
# * speaker-labelled dialog; with fewer voices the plain transcript stays
# * unprefixed (Architect's rule: «без префикса при 1 голосе»).
MIN_VOICES_FOR_SPEAKER_PREFIXES = 2


def media_time(t: float, offset_s: float) -> float:
    """Return the media-relative time of a window-relative timestamp.

    Args:
        t: Seconds from the start of the processed window.
        offset_s: Seconds from the media start to the window start.

    Returns:
        ``offset_s + t`` rounded to milliseconds.

    """
    return round(offset_s + t, _ROUND_DIGITS)


def _number(item: Mapping[str, object], key: str, where: str) -> float:
    """Return ``item[key]`` as a float.

    Args:
        item: A raw segment, word or turn.
        key: Field name.
        where: Description of *item* for the error message.

    Returns:
        The field converted to ``float``.

    Raises:
        ValueError: When the field is missing or not a real number.

    """
    value = item.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    msg = f"{where}: field {key!r} must be a number (got {value!r})"
    raise ValueError(msg)


def _word_json(raw: object, offset_s: float, where: str) -> dict[str, object]:
    """Return one word dict in the skill format.

    Raises:
        ValueError: When *raw* is not an object with numeric timings.

    """
    if isinstance(raw, Mapping):
        return {
            "word": str(raw.get("word", "")).strip(),
            "start": media_time(_number(raw, "start", where), offset_s),
            "end": media_time(_number(raw, "end", where), offset_s),
            "probability": round(_number(raw, "probability", where), _ROUND_DIGITS),
        }
    msg = f"{where}: expected an object (got {raw!r})"
    raise ValueError(msg)


def _words_json(
    raw_words: object, offset_s: float, where: str
) -> list[dict[str, object]]:
    """Return word dicts in the skill format.

    Raises:
        ValueError: When *raw_words* is not a list of word objects.

    """
    if isinstance(raw_words, (list, tuple)):
        return [
            _word_json(raw, offset_s, f"{where}, word {index}")
            for index, raw in enumerate(raw_words)
        ]
    msg = f"{where}: field 'words' must be a list (got {raw_words!r})"
    raise ValueError(msg)


def segments_json(
    raw_segments: Iterable[Mapping[str, object]], offset_s: float
) -> list[dict[str, object]]:
    """Convert wrapper segments to the skill's segment objects.

    Key order is ``start, end, text, no_speech_prob, avg_logprob, words``. The
    Whisper-only keys are emitted only when the raw segment carries them, so
    GigaAM segments become ``{start, end, text}`` (keys absent, not null).

    Args:
        raw_segments: Segment dicts with window-relative ``start``/``end`` and
            ``text``, plus optional ``no_speech_prob``, ``avg_logprob`` and
            ``words`` (``{word, start, end, probability}``).
        offset_s: Seconds from the media start to the window start.

    Returns:
        New segment dicts with media-relative times.

    Raises:
        ValueError: When a segment or word lacks a numeric field.

    """
    segments: list[dict[str, object]] = []
    for index, raw in enumerate(raw_segments):
        where = f"segment {index}"
        segment: dict[str, object] = {
            "start": media_time(_number(raw, "start", where), offset_s),
            "end": media_time(_number(raw, "end", where), offset_s),
            "text": str(raw.get("text") or "").strip(),
        }
        for key in ("no_speech_prob", "avg_logprob"):
            if raw.get(key) is not None:
                segment[key] = round(_number(raw, key, where), _ROUND_DIGITS)
        if raw.get("words") is not None:
            segment["words"] = _words_json(raw["words"], offset_s, where)
        segments.append(segment)
    return segments


def turns_json(
    raw_turns: Iterable[tuple[float, float, int]], offset_s: float
) -> list[dict[str, object]]:
    """Convert raw diarization turns to the skill's turn objects.

    The labels are renumbered: the first voice heard is ``speaker_00``, the next
    new voice ``speaker_01``, and so on, so the numbering is contiguous from
    zero in order of first appearance in time. The raw cluster indices of the
    diarizer (which need not be contiguous or start at zero) are not exposed.

    Args:
        raw_turns: ``(start, end, speaker_index)`` with window-relative times;
            the index only tells which turns share a voice.
        offset_s: Seconds from the media start to the window start.

    Returns:
        ``{start, end, speaker}`` dicts ordered by ``(start, end, raw speaker
        index)``, with speakers named ``speaker_00``, ``speaker_01``, ...

    """
    ordered = sorted(raw_turns, key=lambda turn: (turn[0], turn[1], int(turn[2])))
    labels: dict[int, str] = {}
    turns: list[dict[str, object]] = []
    for start, end, speaker in ordered:
        raw_index = int(speaker)
        # * A voice gets the next free number when it is first met in time order.
        label = labels.setdefault(raw_index, f"speaker_{len(labels):02d}")
        turns.append(
            {
                "start": media_time(start, offset_s),
                "end": media_time(end, offset_s),
                "speaker": label,
            }
        )
    return turns


def assign_speakers(
    segments: Sequence[Mapping[str, object]], turns: Sequence[Mapping[str, object]]
) -> list[str | None]:
    """Pick the speaker of every segment by the largest total turn overlap.

    The overlap of a speaker is summed over all of that speaker's turns with
    ``[start, end]`` of the segment. A segment without positive overlap gets
    ``None``; equal totals go to the lexicographically smallest label.

    Args:
        segments: Objects with ``start`` and ``end`` in the same time base as
            *turns*.
        turns: Objects with ``start``, ``end`` and ``speaker``.

    Returns:
        One label or ``None`` per segment, in segment order.

    Raises:
        ValueError: When a segment or turn lacks a numeric ``start``/``end``.

    """
    prepared = sorted(
        (
            (
                _number(turn, "start", f"turn {index}"),
                _number(turn, "end", f"turn {index}"),
                str(turn.get("speaker")),
            )
            for index, turn in enumerate(turns)
        ),
        key=lambda item: item[0],
    )
    starts = [item[0] for item in prepared]
    speakers: list[str | None] = []
    for index, segment in enumerate(segments):
        seg_start = _number(segment, "start", f"segment {index}")
        seg_end = _number(segment, "end", f"segment {index}")
        overlaps: dict[str, list[float]] = {}
        # * Turns starting at or after the segment end cannot overlap it.
        for turn_start, turn_end, label in prepared[: bisect_left(starts, seg_end)]:
            overlap = min(seg_end, turn_end) - max(seg_start, turn_start)
            if overlap > 0:
                overlaps.setdefault(label, []).append(overlap)
        totals = {
            label: total
            for label, parts in overlaps.items()
            if (total := round(math.fsum(parts), _OVERLAP_DIGITS)) > 0
        }
        speakers.append(
            min(totals, key=lambda label: (-totals[label], label)) if totals else None
        )
    return speakers


def voice_count(speakers_or_turns: Iterable[str | Mapping[str, object] | None]) -> int:
    """Count the distinct non-null speaker labels.

    Args:
        speakers_or_turns: Labels (``None`` for an unassigned segment) or
            objects with a ``speaker`` key (turns, or segments with speakers).

    Returns:
        The number of distinct labels.

    """
    labels: set[str] = set()
    for item in speakers_or_turns:
        label = item.get("speaker") if isinstance(item, Mapping) else item
        if label is not None:
            labels.add(str(label))
    return len(labels)


def speaker_text(texts: Sequence[str], speakers: Sequence[str | None]) -> str:
    """Join segment texts as speaker-prefixed lines.

    The caller applies this only when the transcript carries at least two
    distinct voices; with fewer voices the plain text stays unprefixed.

    Args:
        texts: Segment texts.
        speakers: The speaker of each segment, ``None`` when unassigned.

    Returns:
        One ``"{speaker}: {text}"`` line per segment with non-empty text
        (``"{text}"`` when the speaker is ``None``), joined with line feeds.

    Raises:
        ValueError: When *texts* and *speakers* differ in length.

    """
    if len(texts) != len(speakers):
        msg = (
            f"got {len(texts)} segment texts but {len(speakers)} speakers; "
            "the two lists must have the same length"
        )
        raise ValueError(msg)
    lines = [
        text if speaker is None else f"{speaker}: {text}"
        for text, speaker in zip(texts, speakers, strict=True)
        if text.strip()
    ]
    return "\n".join(lines)


def transcript_document(
    media: str,
    segments: Sequence[Mapping[str, object]],
    *,
    turns: Sequence[Mapping[str, object]] | None = None,
    speakers: Sequence[str | None] | None = None,
) -> dict[str, object]:
    """Build the transcript document of the skill's ``transcribe.py``.

    Layout: ``{"media", "sample_rate", "segments"}``. With diarization
    (``turns`` given) a top-level ``"turns"`` follows ``"segments"`` and every
    segment gains ``"speaker"`` (a label or ``None``) as its last key.

    Args:
        media: The input path as the user gave it.
        segments: Output of :func:`segments_json`.
        turns: Output of :func:`turns_json`; enables diarization fields.
        speakers: One label per segment; computed by :func:`assign_speakers`
            when omitted while ``turns`` is given.

    Returns:
        The JSON-serializable document; *segments* and *turns* are not changed.

    Raises:
        ValueError: When ``speakers`` is given without ``turns`` or has a
            different length than *segments*.

    """
    if turns is None:
        if speakers is not None:
            msg = "speakers can be attached only together with the diarization turns"
            raise ValueError(msg)
        return {
            "media": media,
            "sample_rate": SAMPLE_RATE,
            "segments": [dict(segment) for segment in segments],
        }
    labels = assign_speakers(segments, turns) if speakers is None else list(speakers)
    if len(labels) != len(segments):
        msg = f"got {len(segments)} segments but {len(labels)} speakers"
        raise ValueError(msg)
    return {
        "media": media,
        "sample_rate": SAMPLE_RATE,
        "segments": [
            {**segment, "speaker": label}
            for segment, label in zip(segments, labels, strict=True)
        ],
        "turns": [dict(turn) for turn in turns],
    }


def diarization_document(
    media: str, turns: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Build the diarization document of the skill's ``diarize.py``.

    Args:
        media: The input path as the user gave it.
        turns: Output of :func:`turns_json`.

    Returns:
        ``{"media", "turns"}``.

    """
    return {"media": media, "turns": [dict(turn) for turn in turns]}


def document_json(doc: Mapping[str, object]) -> str:
    """Serialize a document the way the skill's ``write_json`` does.

    Args:
        doc: A JSON-serializable document.

    Returns:
        UTF-8 friendly JSON with two-space indentation and no trailing newline.

    """
    return json.dumps(doc, ensure_ascii=False, indent=2)
