"""Result files: fingerprints, resume decisions, and stale-result handling.

Every result document carries the ``fingerprint`` of the inputs that shape the
measurement (workload, decode options, models, CPU budget, manifest, schema).
On resume a result is reused only when its fingerprint matches the current one.
A result with another fingerprint is never reused: it is moved to
``<results dir>/stale/<key>__<old fingerprint>.json`` when the configuration is
measured again, and the summaries aggregate only current-fingerprint results.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from pathlib import Path

JsonDict = dict[str, Any]

STALE_DIRNAME = "stale"
FINGERPRINT_LENGTH = 16


def fingerprint_of(payload: Mapping[str, object]) -> str:
    """Return a stable short hash of a JSON-serializable payload.

    Args:
        payload: Inputs that define comparable measurements.

    Returns:
        Hex digest prefix.

    """
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def write_json(path: Path, document: Mapping[str, object]) -> None:
    """Write *document* atomically (temp file + replace).

    Args:
        path: Target file (its directory must exist).
        document: JSON-serializable mapping.

    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> JsonDict | None:
    """Read a JSON object; ``None`` when missing, unreadable, or not an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


class ResultStore:
    """One directory of ``<key>.json`` result files with a current fingerprint."""

    def __init__(self, directory: Path, fingerprint: str) -> None:
        """Create a store view (nothing is created on disk).

        Args:
            directory: Results directory.
            fingerprint: Fingerprint of the current inputs.

        """
        self.directory = directory
        self.fingerprint = fingerprint

    def path(self, key: str) -> Path:
        """Return the result path of *key*."""
        return self.directory / f"{key}.json"

    def load(self, key: str) -> JsonDict | None:
        """Return the stored document of *key* whatever its fingerprint."""
        return read_json(self.path(key))

    def reusable(
        self,
        key: str,
        *,
        statuses: Collection[str],
    ) -> JsonDict | None:
        """Return the stored document when it can stand for a new measurement.

        Args:
            key: Result key.
            statuses: Statuses that are final (reused without re-running).

        Returns:
            The document, or ``None`` when it is missing, has another
            fingerprint, or has a status outside *statuses*.

        """
        document = self.load(key)
        if document is None or document.get("fingerprint") != self.fingerprint:
            return None
        return document if document.get("status") in statuses else None

    def describe(self, key: str) -> str:
        """Describe the stored state of *key* for plans and logs."""
        document = self.load(key)
        if document is None:
            return "new"
        if document.get("fingerprint") != self.fingerprint:
            return "stale (other inputs)"
        return f"exists ({document.get('status')})"

    def save(self, key: str, document: JsonDict) -> None:
        """Store *document* under *key*, moving a stale predecessor aside.

        Args:
            key: Result key.
            document: Result document (its ``fingerprint`` is set here).

        """
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.path(key)
        previous = read_json(target)
        if previous is not None and previous.get("fingerprint") != self.fingerprint:
            stale_dir = self.directory / STALE_DIRNAME
            stale_dir.mkdir(exist_ok=True)
            old = str(previous.get("fingerprint") or "none")
            target.replace(stale_dir / f"{key}__{old}.json")
        document["fingerprint"] = self.fingerprint
        write_json(target, document)

    def all_documents(self) -> list[JsonDict]:
        """Return every stored document whatever its fingerprint."""
        if not self.directory.is_dir():
            return []
        documents = [read_json(path) for path in sorted(self.directory.glob("*.json"))]
        return [document for document in documents if document is not None]

    def current(self) -> list[JsonDict]:
        """Return every stored document with the current fingerprint."""
        return [
            document
            for document in self.all_documents()
            if document.get("fingerprint") == self.fingerprint
        ]


def latest_fingerprint(documents: Collection[Mapping[str, object]]) -> str | None:
    """Return the fingerprint of the most recently finished document, if any.

    Args:
        documents: Result or group documents.

    Returns:
        Fingerprint, or ``None`` when no document carries one.

    """
    stamped = [
        (str(d.get("finished_at") or ""), str(d["fingerprint"]))
        for d in documents
        if d.get("fingerprint")
    ]
    return max(stamped)[1] if stamped else None
