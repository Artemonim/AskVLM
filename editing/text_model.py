from collections.abc import Iterable
from dataclasses import dataclass, field


# * Represent a segment of text with speaker and timing
@dataclass
class TextSegment:
    """A single piece of text with speaker id and timestamps."""

    speaker_id: str
    start_time: float
    end_time: float
    text: str


# * Document composed of multiple text segments
@dataclass
class Document:
    """A collection of `TextSegment` instances representing a transcript."""

    segments: list[TextSegment] = field(default_factory=list)
    # * When True, text exporters treat consecutive segments of the same speaker as dialog blocks
    dialog_blocks_enabled: bool = False

    def add_segment(self, segment: TextSegment) -> None:
        """Add a text segment to the document."""
        self.segments.append(segment)

    def get_full_text(self) -> str:
        """Return concatenated text of all segments (plain text).

        Rules:
        - If multiple distinct speakers are present, prefix each line with speaker id.
        - If exactly one speaker and it is "speaker_1", omit the prefix.
        - If exactly one non-default speaker, keep the prefix.
        - Use a single newline between segments (or dialog blocks when enabled).
        """
        if self.dialog_blocks_enabled:
            segments = merge_consecutive_segments_by_speaker(self.segments)
        else:
            segments = self.segments
        speakers = [s.speaker_id for s in segments if s.speaker_id]
        distinct = set(speakers)
        multiple = len(distinct) > 1
        parts: list[str] = []
        for segment in segments:
            need_prefix = multiple or (
                bool(segment.speaker_id) and segment.speaker_id != "speaker_1"
            )
            if need_prefix:
                parts.append(f"{segment.speaker_id}: {segment.text}")
            else:
                parts.append(segment.text)
        return "\n".join(parts)


def merge_consecutive_segments_by_speaker(
    segments: Iterable[TextSegment],
) -> list[TextSegment]:
    """Return a new list where adjacent segments of the same speaker are merged.

    The merged segment keeps the first segment start time, the last segment end time,
    and concatenates texts with a single space separating blocks.
    """
    merged: list[TextSegment] = []
    current: TextSegment | None = None
    for seg in segments:
        if current is None:
            current = TextSegment(
                speaker_id=seg.speaker_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
            )
            continue
        if seg.speaker_id == current.speaker_id:
            left = current.text.rstrip()
            right = seg.text.lstrip()
            if left and right:
                current.text = f"{left} {right}"
            elif right:
                current.text = right
            # * Preserve non-empty left when right is empty
            current.end_time = max(current.end_time, seg.end_time)
        else:
            merged.append(current)
            current = TextSegment(
                speaker_id=seg.speaker_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
            )
    if current is not None:
        merged.append(current)
    return merged
