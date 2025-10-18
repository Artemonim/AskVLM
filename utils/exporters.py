"""Export utilities for Documents to various formats.

Functions in this module convert a `Document` into TXT, SRT, VTT or JSON.
"""

import json
from pathlib import Path

from editing.text_model import Document


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
        lines.append(f"{seg.speaker_id}: {seg.text}".strip())
        lines.append("")
    return "\n".join(lines)


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
