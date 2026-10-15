"""Per-request options of the external STT contract.

Covers the media window, the Whisper decoding overrides and the speaker
diarization settings, together with their JSON wire form and validation. The
module depends on the standard library only, so the thin client, the
supervisor and the worker processes validate and transport these options
without loading any ML package.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Final

# * Sample rate of the prepared audio (mono s16 WAV) for every STT/diarization model.
SAMPLE_RATE: Final = 16000

# * Beam sizes outside this range are rejected before any model work starts.
MIN_BEAM_SIZE: Final = 1
MAX_BEAM_SIZE: Final = 64

# * Diarization defaults of the hardsub skill (``diarize.py``).
DIARIZE_DEFAULT_THRESHOLD: Final = 0.5
DIARIZE_DEFAULT_MIN_DURATION_ON: Final = 0.3
DIARIZE_DEFAULT_MIN_DURATION_OFF: Final = 0.5

# * Decoding settings of ``LocalPipeline`` for Whisper. ``condition_on_previous_text``
# * is faster-whisper's own default: the pipeline does not pass it, so it is
# * forwarded only when a request disables it.
WHISPER_PIPELINE_DEFAULTS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "beam_size": 10,
        "vad_filter": True,
        "word_timestamps": True,
        "condition_on_previous_text": True,
    }
)


def _finite_number(name: str, value: object) -> float:
    """Return *value* as a finite float.

    Args:
        name: Option name used in the error message.
        value: Candidate value; booleans are not numbers.

    Returns:
        The value converted to ``float``.

    Raises:
        ValueError: When *value* is not a finite real number.

    """
    number: float | None = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:
            number = None
    if number is None or not math.isfinite(number):
        msg = f"{name} must be a finite number (got {value!r})"
        raise ValueError(msg)
    return number


def _optional_bool(name: str, value: object) -> bool | None:
    """Return *value* when it is ``None`` or a boolean.

    Raises:
        ValueError: When *value* is anything else.

    """
    if value is None or isinstance(value, bool):
        return value
    msg = f"{name} must be true or false (got {value!r})"
    raise ValueError(msg)


def _optional_int(
    name: str, value: object, *, minimum: int, maximum: int | None
) -> int | None:
    """Return *value* when it is ``None`` or an integer within the bounds.

    Args:
        name: Option name used in the error message.
        value: Candidate value; booleans are not integers.
        minimum: Smallest accepted value.
        maximum: Largest accepted value, or ``None`` for no upper bound.

    Returns:
        The validated integer, or ``None``.

    Raises:
        ValueError: When *value* is not an integer within the bounds.

    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        if value >= minimum and (maximum is None or value <= maximum):
            return value
        bounds = f">= {minimum}" if maximum is None else f"from {minimum} to {maximum}"
        msg = f"{name} must be {bounds} (got {value})"
    else:
        msg = f"{name} must be an integer (got {value!r})"
    raise ValueError(msg)


def _wire_mapping(
    label: str, data: object, allowed: frozenset[str]
) -> Mapping[str, object]:
    """Return *data* as a mapping with known keys only.

    Args:
        label: Option group name used in the error message.
        data: Decoded JSON value.
        allowed: Accepted keys.

    Returns:
        *data* typed as a mapping.

    Raises:
        ValueError: When *data* is not a JSON object or has unknown keys.

    """
    if isinstance(data, Mapping):
        unknown = sorted(str(key) for key in data if key not in allowed)
        if not unknown:
            return data
        msg = (
            f"unknown {label} option(s): {', '.join(unknown)}; "
            f"expected: {', '.join(sorted(allowed))}"
        )
    else:
        msg = f"{label} options must be a JSON object (got {data!r})"
    raise ValueError(msg)


@dataclass(frozen=True)
class MediaWindow:
    """A time window of the input media (seconds from the media start).

    Attributes:
        start_s: Window start, ``>= 0``.
        duration_s: Window length, ``> 0``, or ``None`` for "to the end".

    """

    start_s: float = 0.0
    duration_s: float | None = None

    def __post_init__(self) -> None:
        """Validate the window and normalize numbers to floats.

        Raises:
            ValueError: When the start is negative or not finite, or the
                duration is not a positive finite number.

        """
        start = _finite_number("start_s", self.start_s)
        if start < 0:
            msg = f"start_s must be >= 0 (got {self.start_s!r})"
            raise ValueError(msg)
        object.__setattr__(self, "start_s", 0.0 if start == 0 else start)
        if self.duration_s is not None:
            duration = _finite_number("duration_s", self.duration_s)
            if duration <= 0:
                msg = f"duration_s must be > 0 (got {self.duration_s!r})"
                raise ValueError(msg)
            object.__setattr__(self, "duration_s", duration)

    @property
    def is_default(self) -> bool:
        """Whether the window covers the whole media."""
        return self.start_s == 0 and self.duration_s is None

    def to_wire(self) -> dict[str, object]:
        """Return the JSON form ``{"start_s": ..., "duration_s": ...}``."""
        return {"start_s": self.start_s, "duration_s": self.duration_s}

    @classmethod
    def from_wire(cls, data: object) -> MediaWindow:
        """Build a window from its JSON form; missing keys take the defaults.

        Args:
            data: Decoded JSON object.

        Returns:
            The validated window.

        Raises:
            ValueError: When *data* is not an object with known keys, or a value
                has the wrong type or range.

        """
        mapping = _wire_mapping("window", data, frozenset({"start_s", "duration_s"}))
        duration = mapping.get("duration_s")
        return cls(
            start_s=_finite_number("start_s", mapping.get("start_s", 0.0)),
            duration_s=None
            if duration is None
            else _finite_number("duration_s", duration),
        )


@dataclass(frozen=True)
class WhisperDecodeOptions:
    """Whisper-only decoding overrides; ``None`` keeps the pipeline default.

    Attributes:
        beam_size: Beam search width, ``1..64``.
        vad_filter: Voice-activity filter of faster-whisper.
        condition_on_previous_text: Feed the previous text as the prompt.
        word_timestamps: Compute word-level timestamps.

    """

    beam_size: int | None = None
    vad_filter: bool | None = None
    condition_on_previous_text: bool | None = None
    word_timestamps: bool | None = None

    def __post_init__(self) -> None:
        """Validate the explicit values.

        Raises:
            ValueError: When ``beam_size`` is not an integer in ``1..64`` or a
                flag is not a boolean.

        """
        _optional_int(
            "beam_size", self.beam_size, minimum=MIN_BEAM_SIZE, maximum=MAX_BEAM_SIZE
        )
        _optional_bool("vad_filter", self.vad_filter)
        _optional_bool("condition_on_previous_text", self.condition_on_previous_text)
        _optional_bool("word_timestamps", self.word_timestamps)

    @property
    def is_default(self) -> bool:
        """Whether no option is set."""
        return all(getattr(self, field.name) is None for field in fields(self))

    def to_wire(self) -> dict[str, object]:
        """Return the JSON form: a dict of the options that are set."""
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }

    @classmethod
    def from_wire(cls, data: object) -> WhisperDecodeOptions:
        """Build the options from their JSON form.

        Args:
            data: Decoded JSON object; ``null`` values mean "not set".

        Returns:
            The validated options.

        Raises:
            ValueError: When *data* is not an object with known keys, or a value
                has the wrong type or range.

        """
        mapping = _wire_mapping(
            "decode", data, frozenset(field.name for field in fields(cls))
        )
        return cls(
            beam_size=_optional_int(
                "beam_size",
                mapping.get("beam_size"),
                minimum=MIN_BEAM_SIZE,
                maximum=MAX_BEAM_SIZE,
            ),
            vad_filter=_optional_bool("vad_filter", mapping.get("vad_filter")),
            condition_on_previous_text=_optional_bool(
                "condition_on_previous_text", mapping.get("condition_on_previous_text")
            ),
            word_timestamps=_optional_bool(
                "word_timestamps", mapping.get("word_timestamps")
            ),
        )

    def transcribe_overrides(self, defaults: Mapping[str, object]) -> dict[str, object]:
        """Return the explicit options that differ from *defaults*.

        Args:
            defaults: The decoding settings applied when a request sets nothing
                (see :data:`WHISPER_PIPELINE_DEFAULTS`).

        Returns:
            Keyword arguments to merge into the Whisper ``transcribe`` call;
            empty when every explicit value equals its default.

        """
        overrides: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and defaults.get(field.name) != value:
                overrides[field.name] = value
        return overrides


@dataclass(frozen=True)
class DiarizeOptions:
    """Speaker diarization settings (defaults of the hardsub skill).

    Attributes:
        num_speakers: Exact speaker count, or ``None`` to cluster by threshold.
        threshold: Clustering distance threshold, ``> 0`` (used when
            ``num_speakers`` is ``None``).
        min_duration_on: Shortest kept turn in seconds, ``>= 0``.
        min_duration_off: Shortest gap in seconds that splits two turns of one
            speaker, ``>= 0``.

    """

    num_speakers: int | None = None
    threshold: float = DIARIZE_DEFAULT_THRESHOLD
    min_duration_on: float = DIARIZE_DEFAULT_MIN_DURATION_ON
    min_duration_off: float = DIARIZE_DEFAULT_MIN_DURATION_OFF

    def __post_init__(self) -> None:
        """Validate the settings and normalize numbers to floats.

        Raises:
            ValueError: When a value has the wrong type or is out of range.

        """
        _optional_int("num_speakers", self.num_speakers, minimum=1, maximum=None)
        threshold = _finite_number("threshold", self.threshold)
        if threshold <= 0:
            msg = f"threshold must be > 0 (got {self.threshold!r})"
            raise ValueError(msg)
        object.__setattr__(self, "threshold", threshold)
        for name in ("min_duration_on", "min_duration_off"):
            value = _finite_number(name, getattr(self, name))
            if value < 0:
                msg = f"{name} must be >= 0 (got {getattr(self, name)!r})"
                raise ValueError(msg)
            object.__setattr__(self, name, value)

    def to_wire(self) -> dict[str, object]:
        """Return the JSON form with all four fields (``num_speakers`` may be null)."""
        return {
            "num_speakers": self.num_speakers,
            "threshold": self.threshold,
            "min_duration_on": self.min_duration_on,
            "min_duration_off": self.min_duration_off,
        }

    @classmethod
    def from_wire(cls, data: object) -> DiarizeOptions:
        """Build the settings from their JSON form; missing keys take the defaults.

        Args:
            data: Decoded JSON object.

        Returns:
            The validated settings.

        Raises:
            ValueError: When *data* is not an object with known keys, or a value
                has the wrong type or range.

        """
        mapping = _wire_mapping(
            "diarize", data, frozenset(field.name for field in fields(cls))
        )
        return cls(
            num_speakers=_optional_int(
                "num_speakers", mapping.get("num_speakers"), minimum=1, maximum=None
            ),
            threshold=_finite_number(
                "threshold", mapping.get("threshold", DIARIZE_DEFAULT_THRESHOLD)
            ),
            min_duration_on=_finite_number(
                "min_duration_on",
                mapping.get("min_duration_on", DIARIZE_DEFAULT_MIN_DURATION_ON),
            ),
            min_duration_off=_finite_number(
                "min_duration_off",
                mapping.get("min_duration_off", DIARIZE_DEFAULT_MIN_DURATION_OFF),
            ),
        )
