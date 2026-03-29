"""Versioned JSON manifest schema for Video QA runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast
from uuid import uuid4

from .video_qa_context import VideoQAAttachment

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from .video_qa_sources import LocalFileSource

SCHEMA_VERSION: Final[int] = 1

ChunkStatus = Literal["pending", "running", "completed", "failed"]
GraphNodeStatus = Literal["pending", "running", "completed", "failed"]
RunStatus = Literal["pending", "running", "completed", "failed"]
GraphNodeKind = Literal[
    "source_resolve",
    "attachment_prepare",
    "transcript_prepare",
    "chunk_plan",
    "frame_select",
    "llm_pass",
    "answer_aggregate",
]

_VALID_CHUNK_STATUSES: Final[frozenset[str]] = frozenset(
    {"pending", "running", "completed", "failed"}
)
_VALID_GRAPH_NODE_STATUSES: Final[frozenset[str]] = frozenset(
    {"pending", "running", "completed", "failed"}
)
_VALID_RUN_STATUSES: Final[frozenset[str]] = frozenset(
    {"pending", "running", "completed", "failed"}
)
_VALID_GRAPH_NODE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "source_resolve",
        "attachment_prepare",
        "transcript_prepare",
        "chunk_plan",
        "frame_select",
        "llm_pass",
        "answer_aggregate",
    }
)
_VALID_ATTACHMENT_TYPES: Final[frozenset[str]] = frozenset({"text", "code", "image"})


@dataclass(frozen=True, slots=True)
class VideoQASourceSnapshot:
    """Serializable snapshot of the resolved Video QA source."""

    path: Path
    name: str
    size_bytes: int
    suffix: str

    @classmethod
    def from_source(cls, source: LocalFileSource) -> VideoQASourceSnapshot:
        """Create a source snapshot from the normalized local file source."""
        return cls(
            path=source.path,
            name=source.name,
            size_bytes=source.size_bytes,
            suffix=source.suffix,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the source snapshot into a JSON-compatible dictionary."""
        return {
            "path": str(self.path),
            "name": self.name,
            "size_bytes": self.size_bytes,
            "suffix": self.suffix,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> VideoQASourceSnapshot:
        """Deserialize a source snapshot from a JSON-compatible dictionary."""
        return cls(
            path=Path(_require_str(raw, "path")),
            name=_require_str(raw, "name"),
            size_bytes=_require_int(raw, "size_bytes"),
            suffix=_require_str(raw, "suffix"),
        )


@dataclass(frozen=True, slots=True)
class VideoQAChunkRecord:
    """Serializable manifest record for one future or completed chunk."""

    chunk_id: str
    t_start: float
    t_end: float
    frames: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    status: ChunkStatus = "pending"
    attempts: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the chunk record into a JSON-compatible dictionary."""
        _validate_literal("status", self.status, _VALID_CHUNK_STATUSES)
        return {
            "chunk_id": self.chunk_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "frames": list(self.frames),
            "artifacts": list(self.artifacts),
            "status": self.status,
            "attempts": self.attempts,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> VideoQAChunkRecord:
        """Deserialize a chunk record from a JSON-compatible dictionary."""
        status = _require_str(raw, "status")
        _validate_literal("status", status, _VALID_CHUNK_STATUSES)
        return cls(
            chunk_id=_require_str(raw, "chunk_id"),
            t_start=_require_number(raw, "t_start"),
            t_end=_require_number(raw, "t_end"),
            frames=_require_string_tuple(raw, "frames"),
            artifacts=_require_string_tuple(raw, "artifacts"),
            status=cast("ChunkStatus", status),
            attempts=_require_int(raw, "attempts"),
            error=_require_optional_str(raw, "error"),
        )


@dataclass(frozen=True, slots=True)
class VideoQAGraphNode:
    """Serializable DAG node describing one orchestration step."""

    node_id: str
    kind: GraphNodeKind
    depends_on: tuple[str, ...] = ()
    status: GraphNodeStatus = "pending"
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the graph node into a JSON-compatible dictionary."""
        _validate_literal("kind", self.kind, _VALID_GRAPH_NODE_KINDS)
        _validate_literal("status", self.status, _VALID_GRAPH_NODE_STATUSES)
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "depends_on": list(self.depends_on),
            "status": self.status,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> VideoQAGraphNode:
        """Deserialize a graph node from a JSON-compatible dictionary."""
        kind = _require_str(raw, "kind")
        status = _require_str(raw, "status")
        _validate_literal("kind", kind, _VALID_GRAPH_NODE_KINDS)
        _validate_literal("status", status, _VALID_GRAPH_NODE_STATUSES)
        return cls(
            node_id=_require_str(raw, "node_id"),
            kind=cast("GraphNodeKind", kind),
            depends_on=_require_string_tuple(raw, "depends_on"),
            status=cast("GraphNodeStatus", status),
            note=_require_optional_str(raw, "note"),
        )


@dataclass(frozen=True, slots=True)
class VideoQARunManifest:
    """Serializable manifest for one Video QA preparation or run."""

    schema_version: int
    run_id: str
    created_at: str
    source: VideoQASourceSnapshot | None
    question: str
    attachments: tuple[VideoQAAttachment, ...]
    graph: tuple[VideoQAGraphNode, ...]
    chunks: tuple[VideoQAChunkRecord, ...]
    status: RunStatus = "pending"
    attempts: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the run manifest into a JSON-compatible dictionary."""
        _validate_schema_version(self.schema_version)
        validate_manifest_timestamp(self.created_at)
        _validate_literal("status", self.status, _VALID_RUN_STATUSES)
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "source": None if self.source is None else self.source.to_dict(),
            "question": self.question,
            "attachments": [
                _attachment_to_dict(attachment) for attachment in self.attachments
            ],
            "graph": [node.to_dict() for node in self.graph],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "status": self.status,
            "attempts": self.attempts,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> VideoQARunManifest:
        """Deserialize a run manifest from a JSON-compatible dictionary."""
        schema_version = _require_int(raw, "schema_version")
        _validate_schema_version(schema_version)

        source_raw = raw.get("source")
        source = (
            None
            if source_raw is None
            else VideoQASourceSnapshot.from_dict(_require_mapping(source_raw, "source"))
        )

        status = _require_str(raw, "status")
        _validate_literal("status", status, _VALID_RUN_STATUSES)

        attachments = tuple(
            _attachment_from_dict(_require_mapping(item, f"attachments[{index}]"))
            for index, item in enumerate(_require_sequence(raw, "attachments"))
        )
        graph = tuple(
            VideoQAGraphNode.from_dict(_require_mapping(item, f"graph[{index}]"))
            for index, item in enumerate(_require_sequence(raw, "graph"))
        )
        chunks = tuple(
            VideoQAChunkRecord.from_dict(_require_mapping(item, f"chunks[{index}]"))
            for index, item in enumerate(_require_sequence(raw, "chunks"))
        )

        return cls(
            schema_version=schema_version,
            run_id=_require_str(raw, "run_id"),
            created_at=validate_manifest_timestamp(_require_str(raw, "created_at")),
            source=source,
            question=_require_str(raw, "question"),
            attachments=attachments,
            graph=graph,
            chunks=chunks,
            status=cast("RunStatus", status),
            attempts=_require_int(raw, "attempts"),
            error=_require_optional_str(raw, "error"),
        )


def create_manifest_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp for a manifest record."""
    return (
        datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def create_video_qa_run_id() -> str:
    """Return a stable-looking unique identifier for a Video QA run."""
    return f"video-qa-{uuid4().hex}"


def validate_manifest_timestamp(value: str) -> str:
    """Validate an ISO 8601 timestamp string accepted by the manifest schema."""
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(candidate)
    except ValueError as exc:
        msg = f"Manifest field 'created_at' must be an ISO 8601 timestamp: {value}"
        raise ValueError(msg) from exc
    return value


def _attachment_to_dict(attachment: VideoQAAttachment) -> dict[str, object]:
    return {
        "name": attachment.name,
        "type": attachment.type,
        "size": attachment.size,
        "suffix": attachment.suffix,
        "language": attachment.language,
        "enabled": attachment.enabled,
    }


def _attachment_from_dict(raw: Mapping[str, object]) -> VideoQAAttachment:
    attachment_type = _require_str(raw, "type")
    _validate_literal("type", attachment_type, _VALID_ATTACHMENT_TYPES)
    return VideoQAAttachment(
        name=_require_str(raw, "name"),
        type=cast("Literal['text', 'code', 'image']", attachment_type),
        size=_require_int(raw, "size"),
        suffix=_require_str(raw, "suffix"),
        language=_require_optional_str(raw, "language"),
        enabled=_require_bool(raw, "enabled"),
    )


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"Manifest field '{field_name}' must be an object."
        raise TypeError(msg)
    return cast("Mapping[str, object]", value)


def _require_sequence(raw: Mapping[str, object], field_name: str) -> Sequence[object]:
    value = raw.get(field_name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return cast("Sequence[object]", value)
    msg = f"Manifest field '{field_name}' must be an array."
    raise TypeError(msg)


def _require_string_tuple(
    raw: Mapping[str, object], field_name: str
) -> tuple[str, ...]:
    values = _require_sequence(raw, field_name)
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            msg = (
                f"Manifest field '{field_name}[{index}]' must be a string, "
                f"got {type(value).__name__}."
            )
            raise TypeError(msg)
        normalized.append(value)
    return tuple(normalized)


def _require_str(raw: Mapping[str, object], field_name: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str):
        msg = f"Manifest field '{field_name}' must be a string."
        raise TypeError(msg)
    return value


def _require_optional_str(raw: Mapping[str, object], field_name: str) -> str | None:
    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"Manifest field '{field_name}' must be a string or null."
        raise TypeError(msg)
    return value


def _require_bool(raw: Mapping[str, object], field_name: str) -> bool:
    value = raw.get(field_name)
    if not isinstance(value, bool):
        msg = f"Manifest field '{field_name}' must be a boolean."
        raise TypeError(msg)
    return value


def _require_int(raw: Mapping[str, object], field_name: str) -> int:
    value = raw.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"Manifest field '{field_name}' must be an integer."
        raise TypeError(msg)
    return value


def _require_number(raw: Mapping[str, object], field_name: str) -> float:
    value = raw.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"Manifest field '{field_name}' must be a number."
        raise TypeError(msg)
    return float(value)


def _validate_literal(field_name: str, value: str, allowed: AbstractSet[str]) -> None:
    if value not in allowed:
        msg = f"Manifest field '{field_name}' has unsupported value: {value}"
        raise ValueError(msg)


def _validate_schema_version(schema_version: int) -> None:
    if schema_version != SCHEMA_VERSION:
        msg = (
            "Manifest schema version mismatch: "
            f"expected {SCHEMA_VERSION}, got {schema_version}"
        )
        raise ValueError(msg)
