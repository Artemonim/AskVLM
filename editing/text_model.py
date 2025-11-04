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

    def add_segment(self, segment: TextSegment) -> None:
        """Add a text segment to the document."""
        self.segments.append(segment)

    def get_full_text(self) -> str:
        """Return concatenated text of all segments (plain text).

        Rules:
        - If multiple distinct speakers are present, prefix each line with speaker id.
        - If exactly one speaker and it is "speaker_1", omit the prefix.
        - If exactly one non-default speaker, keep the prefix.
        - Use a single newline between segments.
        """
        speakers = [s.speaker_id for s in self.segments if s.speaker_id]
        distinct = set(speakers)
        multiple = len(distinct) > 1
        parts: list[str] = []
        for segment in self.segments:
            need_prefix = multiple or (
                bool(segment.speaker_id) and segment.speaker_id != "speaker_1"
            )
            if need_prefix:
                parts.append(f"{segment.speaker_id}: {segment.text}")
            else:
                parts.append(segment.text)
        return "\n".join(parts)
