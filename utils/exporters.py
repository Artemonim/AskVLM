"""Export utilities for Documents to various formats.

Functions in this module convert a `Document` into TXT, SRT, VTT or JSON.
Includes SRT/VTT exporters with readability rules (CPS/durations/line length).
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from editing.text_model import Document


# * Readability constraints for subtitles
@dataclass
class SubtitleRules:
    """Constraints for subtitle cue formatting.

    Attributes:
        max_line_chars: Maximum characters per line.
        max_lines: Maximum number of lines per cue.
        min_duration: Minimum cue duration (seconds).
        max_duration: Maximum cue duration (seconds).
        max_cps: Maximum characters per second across a cue.

    """

    max_line_chars: int = 42
    max_lines: int = 2
    min_duration: float = 1.2
    max_duration: float = 6.0
    max_cps: float = 18.0


# * TXT exporter
def export_txt(doc: Document) -> str:
    """Return plain text for a `Document`."""
    return doc.get_full_text()


# * SRT exporter
def _format_ts_srt(seconds: float) -> str:
    ms = round(seconds * 1000)
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt(doc: Document) -> str:
    """Return SRT string for a `Document`."""
    lines: list[str] = []
    for i, seg in enumerate(doc.segments, start=1):
        start = _format_ts_srt(seg.start_time)
        end = _format_ts_srt(seg.end_time if seg.end_time > 0 else seg.start_time)
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        # Do not prefix with speaker if diarization is effectively absent (default id)
        if seg.speaker_id and seg.speaker_id != "speaker_1":
            lines.append(f"{seg.speaker_id}: {seg.text}".strip())
        else:
            lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def _chunk_words_by_length(words: list[str], max_len: int) -> list[list[str]]:
    """Split words into lines where each line's length is <= max_len.

    Preserves word boundaries; avoids creating empty lines.
    """
    lines: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for w in words:
        wlen = len(w)
        if current and current_len + 1 + wlen > max_len:
            lines.append(current)
            current = [w]
            current_len = wlen
        else:
            if current:
                current_len += 1 + wlen
            else:
                current_len = wlen
            current.append(w)
    if current:
        lines.append(current)
    return lines


def _split_text_into_cues(text: str, rules: SubtitleRules) -> list[list[str]]:
    """Split text into multiple cues (each as list of lines) given constraints.

    Strategy:
    - Break into words
    - Chunk into lines (<= max_line_chars)
    - Group up to max_lines lines per cue; if more remain, start a new cue
    """
    words = [w for w in text.strip().split() if w]
    if not words:
        return []
    line_chunks = _chunk_words_by_length(words, rules.max_line_chars)
    cues: list[list[str]] = []
    buf: list[str] = []
    for line_words in line_chunks:
        line = " ".join(line_words)
        buf.append(line)
        if len(buf) >= rules.max_lines:
            cues.append(buf)
            buf = []
    if buf:
        cues.append(buf)
    return cues


def _format_cue_lines(lines: Iterable[str]) -> str:
    return "\n".join(lines)


def export_srt_with_rules(doc: Document, rules: SubtitleRules | None = None) -> str:
    """Return SRT string for a `Document` applying readability rules.

    Heuristics:
    - Split each segment text into multiple cues if needed
    - Assign durations to cues proportionally to text length
    - Constrain durations to [min_duration, max_duration]
    - Ensure CPS <= max_cps by stretching within caps
    """
    rules = rules or SubtitleRules()
    blocks: list[tuple[float, float, str]] = []
    for seg in doc.segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        cues = _split_text_into_cues(text, rules)
        if not cues:
            continue
        # Compute base duration
        base_duration = max(0.0, (seg.end_time - seg.start_time))
        # Estimate durations if missing or zero using CPS
        char_counts = [sum(len(line) for line in cue) for cue in cues]
        # Initial per-cue durations based on cps (before scaling to fit base_duration)
        init_durs = [
            max(rules.min_duration, min(rules.max_duration, cc / rules.max_cps))
            for cc in char_counts
        ]
        total_init = sum(init_durs)
        if base_duration > 0 and total_init > 0:
            scale = base_duration / total_init
            durs = [
                max(rules.min_duration, min(rules.max_duration, d * scale))
                for d in init_durs
            ]
        else:
            durs = init_durs
        # Build time-aligned blocks within the segment window (or from seg.start_time if unknown)
        t = seg.start_time
        for cue_lines, dur_value in zip(cues, durs, strict=False):
            # Re-check CPS, stretch within caps
            cc = sum(len(line) for line in cue_lines)
            dur_adj = dur_value
            if dur_adj > 0 and (cc / max(dur_adj, 1e-6)) > rules.max_cps:
                dur_adj = min(
                    rules.max_duration, max(rules.min_duration, cc / rules.max_cps)
                )
            dur = dur_adj
            start = t
            end = t + dur if dur > 0 else t
            blocks.append((start, end, _format_cue_lines(cue_lines)))
            t = end

    # Emit SRT
    out: list[str] = []
    for idx, (start, end, cue_text) in enumerate(blocks, start=1):
        out.append(str(idx))
        out.append(f"{_format_ts_srt(start)} --> {_format_ts_srt(max(start, end))}")
        out.append(cue_text)
        out.append("")
    return "\n".join(out)


# * VTT exporter
def _format_ts_vtt(seconds: float) -> str:
    ms = round(seconds * 1000)
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def export_vtt(doc: Document) -> str:
    """Return WebVTT string for a `Document`."""
    out: list[str] = ["WEBVTT", ""]
    for seg in doc.segments:
        start = _format_ts_vtt(seg.start_time)
        end = _format_ts_vtt(seg.end_time if seg.end_time > 0 else seg.start_time)
        out.append(f"{start} --> {end}")
        out.append(f"{seg.speaker_id}: {seg.text}".strip())
        out.append("")
    return "\n".join(out)


# * JSON exporter
def export_json(doc: Document) -> dict[str, list[dict[str, object]]]:
    """Return JSON-serializable structure for a `Document`."""
    return {
        "segments": [
            {
                "speaker_id": seg.speaker_id,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
            }
            for seg in doc.segments
        ]
    }


EXPORTERS = {
    "txt": export_txt,
    "srt": export_srt,
    "vtt": export_vtt,
    "json": export_json,
}


def export_document(doc: Document, fmt: str, out_path: Path) -> Path:
    """Export `doc` to the given `fmt` and write into `out_path`."""
    fmt = fmt.lower()
    if fmt not in EXPORTERS:
        msg = f"Unknown export format: {fmt}"
        raise ValueError(msg)
    result = EXPORTERS[fmt](doc)
    if isinstance(result, str):
        out_path.write_text(result, encoding="utf-8")
    else:
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out_path
