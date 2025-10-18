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
        """Return concatenated text of all segments."""
        return "\n\n".join(
            f"{segment.speaker_id}: {segment.text}" for segment in self.segments
        )
