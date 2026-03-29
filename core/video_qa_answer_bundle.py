"""Machine-readable Video QA final answer bundle (schema + JSON persistence)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from .video_qa_manifest import validate_manifest_timestamp

if TYPE_CHECKING:
    from pathlib import Path

ANSWER_BUNDLE_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class VideoQAEvidenceItem:
    """One grounded evidence block tied to transcript time and optional frames."""

    transcript_quote: str
    t_start: float
    t_end: float
    frame_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the evidence item into a JSON-compatible dictionary."""
        return {
            "transcript_quote": self.transcript_quote,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "frame_refs": list(self.frame_refs),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> VideoQAEvidenceItem:
        """Deserialize an evidence item from a JSON-compatible dictionary."""
        return cls(
            transcript_quote=_require_str(raw, "transcript_quote"),
            t_start=_require_number(raw, "t_start"),
            t_end=_require_number(raw, "t_end"),
            frame_refs=_require_string_tuple(raw, "frame_refs"),
        )


@dataclass(frozen=True, slots=True)
class VideoQAAnswerBundle:
    """Versioned final answer payload for a Video QA run (for export and replay)."""

    schema_version: int
    run_id: str
    created_at: str
    question: str
    answer: str
    evidence: tuple[VideoQAEvidenceItem, ...]
    is_uncertain: bool
    manifest_run_id: str | None = None
    uncertainty_note: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the answer bundle into a JSON-compatible dictionary."""
        _validate_answer_bundle_schema_version(self.schema_version)
        validate_manifest_timestamp(self.created_at)
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "question": self.question,
            "answer": self.answer,
            "evidence": [item.to_dict() for item in self.evidence],
            "is_uncertain": self.is_uncertain,
        }
        if self.manifest_run_id is not None:
            payload["manifest_run_id"] = self.manifest_run_id
        if self.uncertainty_note is not None:
            payload["uncertainty_note"] = self.uncertainty_note
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> VideoQAAnswerBundle:
        """Deserialize an answer bundle from a JSON-compatible dictionary."""
        schema_version = _require_int(raw, "schema_version")
        _validate_answer_bundle_schema_version(schema_version)

        evidence = tuple(
            VideoQAEvidenceItem.from_dict(_require_mapping(item, f"evidence[{index}]"))
            for index, item in enumerate(_require_sequence(raw, "evidence"))
        )

        return cls(
            schema_version=schema_version,
            run_id=_require_str(raw, "run_id"),
            created_at=validate_manifest_timestamp(_require_str(raw, "created_at")),
            question=_require_str(raw, "question"),
            answer=_require_str(raw, "answer"),
            evidence=evidence,
            is_uncertain=_require_bool(raw, "is_uncertain"),
            manifest_run_id=_require_optional_str(raw, "manifest_run_id"),
            uncertainty_note=_require_optional_str(raw, "uncertainty_note"),
        )


def answer_bundle_path_for_manifest(manifest_path: Path) -> Path:
    """Return a sibling JSON path for the answer bundle next to the manifest file."""
    return manifest_path.parent / f"{manifest_path.stem}.answer.json"


def save_answer_bundle_to_json(path: Path, bundle: VideoQAAnswerBundle) -> None:
    """Write the answer bundle to ``path`` as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def load_answer_bundle_from_json(path: Path) -> VideoQAAnswerBundle:
    """Load an answer bundle from a UTF-8 JSON file."""
    raw_text = path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        msg = "Answer bundle JSON root must be an object."
        raise TypeError(msg)
    return VideoQAAnswerBundle.from_dict(cast("Mapping[str, object]", payload))


def _validate_answer_bundle_schema_version(schema_version: int) -> None:
    if schema_version != ANSWER_BUNDLE_SCHEMA_VERSION:
        msg = (
            "Answer bundle schema version mismatch: "
            f"expected {ANSWER_BUNDLE_SCHEMA_VERSION}, got {schema_version}"
        )
        raise ValueError(msg)


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"Answer bundle field '{field_name}' must be an object."
        raise TypeError(msg)
    return cast("Mapping[str, object]", value)


def _require_sequence(raw: Mapping[str, object], field_name: str) -> Sequence[object]:
    value = raw.get(field_name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return cast("Sequence[object]", value)
    msg = f"Answer bundle field '{field_name}' must be an array."
    raise TypeError(msg)


def _require_string_tuple(
    raw: Mapping[str, object], field_name: str
) -> tuple[str, ...]:
    values = _require_sequence(raw, field_name)
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            msg = (
                f"Answer bundle field '{field_name}[{index}]' must be a string, "
                f"got {type(value).__name__}."
            )
            raise TypeError(msg)
        normalized.append(value)
    return tuple(normalized)


def _require_str(raw: Mapping[str, object], field_name: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str):
        msg = f"Answer bundle field '{field_name}' must be a string."
        raise TypeError(msg)
    return value


def _require_optional_str(raw: Mapping[str, object], field_name: str) -> str | None:
    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"Answer bundle field '{field_name}' must be a string or null."
        raise TypeError(msg)
    return value


def _require_bool(raw: Mapping[str, object], field_name: str) -> bool:
    value = raw.get(field_name)
    if not isinstance(value, bool):
        msg = f"Answer bundle field '{field_name}' must be a boolean."
        raise TypeError(msg)
    return value


def _require_int(raw: Mapping[str, object], field_name: str) -> int:
    value = raw.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"Answer bundle field '{field_name}' must be an integer."
        raise TypeError(msg)
    return value


def _require_number(raw: Mapping[str, object], field_name: str) -> float:
    value = raw.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"Answer bundle field '{field_name}' must be a number."
        raise TypeError(msg)
    return float(value)
