from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from core.video_qa_answer_bundle import (
    ANSWER_BUNDLE_SCHEMA_VERSION,
    VideoQAAnswerBundle,
    VideoQAEvidenceItem,
    answer_bundle_path_for_manifest,
    load_answer_bundle_from_json,
    save_answer_bundle_to_json,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_video_qa_answer_bundle_round_trip_dict() -> None:
    """to_dict / from_dict round-trip preserves the typed bundle."""
    bundle = VideoQAAnswerBundle(
        schema_version=ANSWER_BUNDLE_SCHEMA_VERSION,
        run_id="video-qa-run-1",
        created_at="2026-03-29T12:00:00Z",
        question="What is the speaker doing?",
        answer="They greet the audience.",
        evidence=(
            VideoQAEvidenceItem(
                transcript_quote="Hello everyone, welcome back.",
                t_start=1.5,
                t_end=4.25,
                frame_refs=("frames/chunk-0/mid.png",),
            ),
        ),
        is_uncertain=False,
        manifest_run_id="video-qa-run-1",
        uncertainty_note=None,
    )

    payload = json.loads(json.dumps(bundle.to_dict()))
    restored = VideoQAAnswerBundle.from_dict(payload)

    assert restored == bundle
    assert payload["evidence"][0]["t_start"] == 1.5
    assert payload["evidence"][0]["frame_refs"] == ["frames/chunk-0/mid.png"]


def test_video_qa_answer_bundle_optional_fields_omitted_in_json() -> None:
    """Minimal JSON without optional keys deserializes with None optional fields."""
    minimal = {
        "schema_version": ANSWER_BUNDLE_SCHEMA_VERSION,
        "run_id": "r2",
        "created_at": "2026-03-29T12:00:00Z",
        "question": "Q?",
        "answer": "A.",
        "evidence": [],
        "is_uncertain": True,
    }
    bundle = VideoQAAnswerBundle.from_dict(minimal)
    assert bundle.manifest_run_id is None
    assert bundle.uncertainty_note is None


def test_answer_bundle_path_for_manifest_sibling(tmp_path: Path) -> None:
    """Answer bundle path is a sibling next to the manifest path."""
    manifest = tmp_path / "nested" / "run.manifest.json"
    expected = tmp_path / "nested" / "run.manifest.answer.json"
    assert answer_bundle_path_for_manifest(manifest) == expected


def test_save_and_load_answer_bundle_json(tmp_path: Path) -> None:
    """save/load round-trip writes readable JSON and restores the bundle."""
    path = tmp_path / "out" / "bundle.json"
    bundle = VideoQAAnswerBundle(
        schema_version=ANSWER_BUNDLE_SCHEMA_VERSION,
        run_id="run-x",
        created_at="2026-03-29T12:00:00Z",
        question="Why?",
        answer="Because.",
        evidence=(
            VideoQAEvidenceItem(
                transcript_quote="…",
                t_start=0.0,
                t_end=1.0,
                frame_refs=("a.png", "b.png"),
            ),
        ),
        is_uncertain=True,
        manifest_run_id="run-x",
        uncertainty_note="Model expressed low confidence.",
    )
    save_answer_bundle_to_json(path, bundle)
    loaded = load_answer_bundle_from_json(path)
    assert loaded == bundle
    text = path.read_text(encoding="utf-8")
    assert "transcript_quote" in text
    assert "frame_refs" in text


def test_answer_bundle_rejects_wrong_schema_version() -> None:
    """from_dict raises when schema_version does not match."""
    bad = {
        "schema_version": 999,
        "run_id": "r",
        "created_at": "2026-03-29T12:00:00Z",
        "question": "q",
        "answer": "a",
        "evidence": [],
        "is_uncertain": False,
    }
    with pytest.raises(ValueError, match="schema version mismatch"):
        VideoQAAnswerBundle.from_dict(bad)
