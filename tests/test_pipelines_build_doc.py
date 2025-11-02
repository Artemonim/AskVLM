from collections import namedtuple

from core.pipelines import _build_document


def test_build_document_from_transcript_with_diarization_and_formatting() -> None:
    """_build_document assigns speakers by overlap and applies formatting when enabled."""
    Segment = namedtuple("Segment", ["start", "end", "speaker"])
    diar = [Segment(0.0, 1.5, "S1"), Segment(1.5, 3.0, "S2")]
    transcript = [
        {"start": 0.0, "end": 1.0, "text": " hello "},
        {"start": 2.0, "end": 2.5, "text": " world"},
    ]

    def fmt(s: str) -> str:
        return s.strip().capitalize()

    doc = _build_document(
        formatted_text="ignored",
        transcript_segments=transcript,
        diarization_segments=diar,
        enable_dialog_blocks=True,
        format_text_fn=fmt,
    )
    assert len(doc.segments) == 2
    assert doc.segments[0].speaker_id == "S1"
    assert doc.segments[0].text == "Hello"
    assert doc.segments[1].speaker_id == "S2"


def test_build_document_from_only_diarization() -> None:
    """When transcript is empty, diarization segments fill document with formatted_text."""
    Segment = namedtuple("Segment", ["start", "end", "speaker"])
    diar = [Segment(0.0, 1.0, "S1")]
    doc = _build_document(
        formatted_text="TEXT",
        transcript_segments=[],
        diarization_segments=diar,
        enable_dialog_blocks=False,
        format_text_fn=lambda s: s,
    )
    assert len(doc.segments) == 1
    assert doc.segments[0].text == "TEXT"


def test_build_document_fallback_single_segment() -> None:
    """With neither transcript nor diarization, fallback single segment is used."""
    doc = _build_document(
        formatted_text="T",
        transcript_segments=[],
        diarization_segments=[],
        enable_dialog_blocks=False,
        format_text_fn=lambda s: s,
    )
    assert len(doc.segments) == 1
    assert doc.segments[0].speaker_id == "speaker_1"
