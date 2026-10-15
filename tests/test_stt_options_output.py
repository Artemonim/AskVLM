"""Dense validation and serialization rules of the structured STT contract."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pytest

from core.stt_options import (
    WHISPER_PIPELINE_DEFAULTS,
    DiarizeOptions,
    MediaWindow,
    WhisperDecodeOptions,
)
from core.stt_output import (
    assign_speakers,
    diarization_document,
    document_json,
    media_time,
    segments_json,
    speaker_text,
    transcript_document,
    turns_json,
    voice_count,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.mark.parametrize(
    ("option_type", "wire", "field"),
    [
        (MediaWindow, {"start_s": True}, "start_s"),
        (MediaWindow, {"start_s": -0.001}, "start_s"),
        (MediaWindow, {"start_s": float("nan")}, "start_s"),
        (MediaWindow, {"start_s": 10**400}, "start_s"),
        (MediaWindow, {"duration_s": 0}, "duration_s"),
        (MediaWindow, {"duration_s": float("inf")}, "duration_s"),
        (MediaWindow, {"duration_s": "1"}, "duration_s"),
        (MediaWindow, {"start": 1}, "start"),
        (WhisperDecodeOptions, {"beam_size": True}, "beam_size"),
        (WhisperDecodeOptions, {"beam_size": 0}, "beam_size"),
        (WhisperDecodeOptions, {"beam_size": 65}, "beam_size"),
        (WhisperDecodeOptions, {"beam_size": 1.0}, "beam_size"),
        (WhisperDecodeOptions, {"vad_filter": 0}, "vad_filter"),
        (WhisperDecodeOptions, {"word_timestamps": "false"}, "word_timestamps"),
        (
            WhisperDecodeOptions,
            {"condition_on_previous_text": []},
            "condition_on_previous_text",
        ),
        (WhisperDecodeOptions, {"temperature": 1}, "temperature"),
        (DiarizeOptions, {"num_speakers": False}, "num_speakers"),
        (DiarizeOptions, {"num_speakers": 0}, "num_speakers"),
        (DiarizeOptions, {"num_speakers": 1.5}, "num_speakers"),
        (DiarizeOptions, {"threshold": 0}, "threshold"),
        (DiarizeOptions, {"threshold": float("nan")}, "threshold"),
        (DiarizeOptions, {"min_duration_on": -0.1}, "min_duration_on"),
        (DiarizeOptions, {"min_duration_off": True}, "min_duration_off"),
        (DiarizeOptions, {"min_off": 0.1}, "min_off"),
    ],
)
def test_options_reject_invalid_types_ranges_and_unknown_wire_fields(
    option_type: type[MediaWindow | WhisperDecodeOptions | DiarizeOptions],
    wire: dict[str, object],
    field: str,
) -> None:
    """Catch invalid or misspelled options silently becoming model defaults."""
    with pytest.raises(ValueError) as error:  # noqa: PT011 - inspect option names
        option_type.from_wire(wire)
    assert field in str(error.value)


@pytest.mark.parametrize("wire", [None, [], "{}", True])
@pytest.mark.parametrize(
    "option_type", [MediaWindow, WhisperDecodeOptions, DiarizeOptions]
)
def test_options_require_wire_objects(
    option_type: type[MediaWindow | WhisperDecodeOptions | DiarizeOptions], wire: object
) -> None:
    """Catch non-object option packets reaching inference as valid requests."""
    with pytest.raises(ValueError):  # noqa: PT011 - error type is the wire contract
        option_type.from_wire(wire)


@pytest.mark.parametrize(
    "options",
    [
        MediaWindow(-0.0, None),
        MediaWindow(0.0001, 0.0001),
        MediaWindow(123, 10),
        WhisperDecodeOptions(
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False,
        ),
        WhisperDecodeOptions(
            beam_size=64,
            vad_filter=True,
            condition_on_previous_text=True,
            word_timestamps=True,
        ),
        DiarizeOptions(1, 0.0001, 0, 0),
        DiarizeOptions(None, 0.7, 0.2, 0.9),
    ],
)
def test_option_wire_preserves_valid_boundaries(
    options: MediaWindow | WhisperDecodeOptions | DiarizeOptions,
) -> None:
    """Catch lost false flags, small windows or clustering values on the wire."""
    decoded = json.loads(json.dumps(options.to_wire(), allow_nan=False))
    assert type(options).from_wire(decoded) == options


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ({}, {}),
        ({"beam_size": 10, "condition_on_previous_text": True}, {}),
        ({"beam_size": 1}, {"beam_size": 1}),
        (
            {"vad_filter": False, "word_timestamps": False},
            {"vad_filter": False, "word_timestamps": False},
        ),
        ({"condition_on_previous_text": False}, {"condition_on_previous_text": False}),
    ],
)
def test_decode_overrides_distinguish_explicit_defaults_from_omitted_options(
    wire: dict[str, object], expected: dict[str, object]
) -> None:
    """Catch dropped false overrides or lost explicit GigaAM-incompatible options."""
    options = WhisperDecodeOptions.from_wire(wire)
    assert options.is_default == (not wire)
    assert options.to_wire() == wire
    assert options.transcribe_overrides(WHISPER_PIPELINE_DEFAULTS) == expected


@pytest.mark.parametrize(
    ("time", "offset", "expected"),
    [(0.00051, 1.25, 1.251), (1.24949, 1.25, 2.499), (0.125, 10.0, 10.125)],
)
def test_media_time_adds_offset_before_millisecond_rounding(
    time: float, offset: float, expected: float
) -> None:
    """Catch premature timestamp rounding or duplicated window offsets."""
    assert media_time(time, offset) == expected


def test_whisper_segments_match_the_reference_json_formula() -> None:
    """Catch lost words/scores, wrong time bases or incompatible skill JSON fields."""
    raw: list[Mapping[str, object]] = [
        {
            "start": 0.00051,
            "end": 1.24949,
            "text": "  Привет  ",
            "no_speech_prob": 0.00051,
            "avg_logprob": -0.45651,
            "words": [
                {
                    "word": "  Привет ",
                    "start": 0.01251,
                    "end": 0.08649,
                    "probability": 0.99949,
                }
            ],
        }
    ]
    original = copy.deepcopy(raw)
    result = segments_json(raw, 1.25)
    assert result == [
        {
            "start": 1.251,
            "end": 2.499,
            "text": "Привет",
            "no_speech_prob": 0.001,
            "avg_logprob": -0.457,
            "words": [
                {"word": "Привет", "start": 1.263, "end": 1.336, "probability": 0.999}
            ],
        }
    ]
    assert list(result[0]) == [
        "start",
        "end",
        "text",
        "no_speech_prob",
        "avg_logprob",
        "words",
    ]
    assert raw == original


@pytest.mark.parametrize("whisper", [False, True])
def test_segment_json_preserves_provider_specific_empty_word_lists(
    *, whisper: bool
) -> None:
    """Catch spurious GigaAM Whisper fields or missing disabled Whisper words."""
    raw: dict[str, object] = {"start": 0.0, "end": 20.0, "text": "  речь  "}
    if whisper:
        raw.update(no_speech_prob=0.0, avg_logprob=-1.0, words=[])
    segment = segments_json([raw], 5.0)[0]
    assert segment["start"] == 5.0
    assert segment["end"] == 25.0
    assert segment["text"] == "речь"
    if whisper:
        assert segment["words"] == []
        assert set(segment) == {
            "start",
            "end",
            "text",
            "no_speech_prob",
            "avg_logprob",
            "words",
        }
    else:
        assert set(segment) == {"start", "end", "text"}


@pytest.mark.parametrize(
    ("interval", "turns", "expected"),
    [
        ((1.0, 2.0), [], None),
        ((1.0, 2.0), [(0.0, 1.0, 0), (2.0, 3.0, 1)], None),
        ((1.0, 1.0), [(0.0, 2.0, 0)], None),
        ((2.0, 1.0), [(0.0, 3.0, 0)], None),
        ((0.0, 4.0), [(0.0, 1.0, 0), (1.0, 2.5, 1), (3.0, 4.0, 0)], "speaker_00"),
        ((0.0, 2.0), [(1.0, 2.0, 1), (0.0, 1.0, 0)], "speaker_00"),
        ((0.0, 2.0), [(0.0, 1.0, 1), (1.0, 2.0, 0)], "speaker_00"),
        ((0.0, 2.0), [(0.0, 2.0, 1), (0.0, 1.0, 0)], "speaker_01"),
        ((0.0, 0.3), [(0.0, 0.1, 0), (0.1, 0.3, 0), (0.0, 0.3, 1)], "speaker_00"),
    ],
)
def test_speaker_assignment_sums_overlaps_and_resolves_ties_without_guessing_silence(
    interval: tuple[float, float],
    turns: list[tuple[float, float, int]],
    expected: str | None,
) -> None:
    """Catch longest-turn selection, float-noise ties or speakers for silence."""
    segment: dict[str, object] = {"start": interval[0], "end": interval[1]}
    assert assign_speakers([segment], turns_json(turns, 0.0)) == [expected]


def test_documents_attach_null_speakers_and_preserve_skill_order_and_unicode() -> None:
    """Catch stringified null speakers or incompatible skill JSON envelopes."""
    turns = turns_json([(2.0, 3.0, 1), (0.0, 1.0, 0)], 10.0)
    segments = segments_json(
        [
            {"start": 0.0, "end": 0.5, "text": "А"},  # noqa: RUF001 - intentional Cyrillic test data
            {"start": 1.2, "end": 1.8, "text": "Б"},
        ],
        10.0,
    )
    document = transcript_document("диалог.wav", segments, turns=turns)
    serialized = document_json(document)
    decoded = json.loads(serialized)
    assert list(decoded) == ["media", "sample_rate", "segments", "turns"]
    assert decoded["sample_rate"] == 16000
    assert [item["speaker"] for item in decoded["segments"]] == ["speaker_00", None]
    assert all(list(item)[-1] == "speaker" for item in decoded["segments"])
    assert [turn["start"] for turn in decoded["turns"]] == [10.0, 12.0]
    assert "диалог.wav" in serialized
    assert '\n  "media"' in serialized
    assert all("speaker" not in segment for segment in segments)
    assert list(diarization_document("диалог.wav", turns)) == ["media", "turns"]
    assert "turns" not in transcript_document("диалог.wav", segments)


def test_speaker_text_retains_unassigned_text_and_counts_assigned_voices() -> None:
    """Catch null voices counting toward prefixes or lost unassigned text."""
    labels = ["speaker_00", None, "speaker_01", "speaker_00"]
    assert voice_count(labels) == 2
    assert voice_count(["speaker_00", None, "speaker_00"]) == 1
    lines = speaker_text(["А", "Б", "В", "  "], labels).splitlines()  # noqa: RUF001 - intentional Cyrillic test data
    assert len(lines) == 3
    assert lines[0].split(": ", 1) == ["speaker_00", "А"]  # noqa: RUF001 - intentional Cyrillic test data
    assert lines[1] == "Б"
    assert lines[2].split(": ", 1) == ["speaker_01", "В"]  # noqa: RUF001 - intentional Cyrillic test data
    with pytest.raises(ValueError):  # noqa: PT011 - malformed parallel lists must fail
        speaker_text(["А"], [])  # noqa: RUF001 - intentional Cyrillic test data


@pytest.mark.parametrize("with_turns", [False, True])
def test_document_rejects_misaligned_explicit_speaker_lists(
    *, with_turns: bool
) -> None:
    """Catch misaligned speakers or lost diarization provenance."""
    with pytest.raises(ValueError):  # noqa: PT011 - the error kind carries the contract
        transcript_document(
            "a.wav", [], turns=[] if with_turns else None, speakers=[None]
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            [(2.0, 3.0, 8), (0.0, 1.0, 41), (4.0, 5.0, 41)],
            [
                {"start": 1.25, "end": 2.25, "speaker": "speaker_00"},
                {"start": 3.25, "end": 4.25, "speaker": "speaker_01"},
                {"start": 5.25, "end": 6.25, "speaker": "speaker_00"},
            ],
        ),
        (
            [(0.0, 2.0, 9), (0.0, 1.0, 23), (3.0, 4.0, 9), (0.0, 1.0, 7)],
            [
                {"start": 1.25, "end": 2.25, "speaker": "speaker_00"},
                {"start": 1.25, "end": 2.25, "speaker": "speaker_01"},
                {"start": 1.25, "end": 3.25, "speaker": "speaker_02"},
                {"start": 4.25, "end": 5.25, "speaker": "speaker_02"},
            ],
        ),
        ([], []),
    ],
)
def test_turn_labels_start_at_zero_follow_time_and_keep_voice_identity(
    raw: list[tuple[float, float, int]], expected: list[dict[str, object]]
) -> None:
    """Catch raw sherpa cluster IDs leaking or equal-start turns changing voice identity."""
    assert turns_json(raw, 1.25) == expected
    assert turns_json(reversed(raw), 1.25) == expected
