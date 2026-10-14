"""Transcript quality: WER against a reference manifest and agreement metrics.

The manifest is JSON Lines; every record names its audio file and references::

    {"id": "...", "wav": "<file name>", "duration_s": 9.1,
     "ref_raw": "...", "ref_fleurs_norm": "..."}

``ref_fleurs_norm`` is preferred when present. Hypotheses and references are
normalized the same way before scoring: lowercase, Cyrillic yo (U+0451) folded
to ye (U+0435), punctuation
removed (every non-alphanumeric character becomes a space), whitespace
collapsed. WER is a corpus WER: word edit distance summed over the scored items
divided by the summed reference word count.

Items of a result document are mappings with ``clip`` (clip index, falling back
to ``index``), ``name`` (audio file name), ``text`` and ``error``.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .workload import normalize_text, similarity

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

JsonDict = dict[str, Any]

_CYRILLIC_YO = chr(0x0451)
_CYRILLIC_YE = chr(0x0435)


class ManifestError(ValueError):
    """Raised for an unreadable or malformed reference manifest."""


def normalize_for_wer(text: str) -> list[str]:
    """Normalize *text* for WER and split it into words.

    Args:
        text: Hypothesis or reference text.

    Returns:
        Lowercase alphanumeric words with Cyrillic yo folded to ye.

    """
    folded = text.casefold().replace(_CYRILLIC_YO, _CYRILLIC_YE)
    kept = "".join(ch if ch.isalnum() else " " for ch in folded)
    return kept.split()


def word_errors(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return the word-level Levenshtein distance (sub + ins + del).

    Args:
        reference: Reference words.
        hypothesis: Hypothesis words.

    Returns:
        Minimum number of edits.

    """
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                ),
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class Manifest:
    """Reference transcripts keyed by audio file name.

    Attributes:
        path: Manifest path.
        sha256: Content hash (part of the result fingerprint).
        references: ``file name -> normalized reference words``.

    """

    path: Path
    sha256: str
    references: dict[str, tuple[str, ...]]

    def reference_for(self, name: str) -> tuple[str, ...] | None:
        """Return the normalized reference of an audio file name, if any."""
        return self.references.get(name)


def load_manifest(path: Path) -> Manifest:
    """Load a JSONL reference manifest.

    Args:
        path: Manifest path.

    Returns:
        The manifest.

    Raises:
        ManifestError: When the file cannot be read or a record is malformed.

    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        msg = f"Cannot read manifest {path}: {exc}"
        raise ManifestError(msg) from exc
    references: dict[str, tuple[str, ...]] = {}
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"{path}:{number}: invalid JSON ({exc})"
            raise ManifestError(msg) from exc
        if not isinstance(record, dict) or not isinstance(record.get("wav"), str):
            msg = f"{path}:{number}: record without a 'wav' file name"
            raise ManifestError(msg)
        text = record.get("ref_fleurs_norm") or record.get("ref_raw")
        if not isinstance(text, str):
            msg = f"{path}:{number}: record without 'ref_fleurs_norm' or 'ref_raw'"
            raise ManifestError(msg)
        references[Path(record["wav"]).name] = tuple(normalize_for_wer(text))
    if not references:
        msg = f"Manifest {path} has no records"
        raise ManifestError(msg)
    return Manifest(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        references=references,
    )


def _clip_of(item: Mapping[str, Any]) -> int:
    return int(item.get("clip", item["index"]))


def _ok_items(items: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [item for item in items if not item.get("error")]


def corpus_wer(
    items: Iterable[Mapping[str, Any]],
    manifest: Manifest,
) -> JsonDict:
    """Score successful items that have a reference.

    Args:
        items: Result items.
        manifest: Reference manifest.

    Returns:
        ``wer_percent`` (``None`` without scored words), ``ref_words``,
        ``scored_items`` and ``items_without_reference``.

    """
    errors = 0
    words = 0
    scored = 0
    missing = 0
    for item in _ok_items(items):
        reference = manifest.reference_for(str(item.get("name", "")))
        if reference is None:
            missing += 1
            continue
        errors += word_errors(reference, normalize_for_wer(str(item.get("text", ""))))
        words += len(reference)
        scored += 1
    return {
        "wer_percent": 100.0 * errors / words if words else None,
        "ref_words": words,
        "scored_items": scored,
        "items_without_reference": missing,
    }


def _first_texts(items: Iterable[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    """Return the first successful occurrence of every clip."""
    first: dict[int, Mapping[str, Any]] = {}
    for item in _ok_items(items):
        first.setdefault(_clip_of(item), item)
    return first


def compare_items(
    items: Sequence[Mapping[str, Any]],
    reference_items: Sequence[Mapping[str, Any]],
    manifest: Manifest | None,
) -> JsonDict:
    """Compare a result with a reference result clip by clip.

    Identity and similarity use every successful occurrence whose clip the
    reference transcribed. The WER delta uses the first occurrence of the clips
    both results transcribed, so different workload sizes stay comparable.

    Args:
        items: Result items.
        reference_items: Reference result items.
        manifest: Optional reference manifest for the WER delta.

    Returns:
        ``compared``, ``identical_percent``, ``similarity_mean``,
        ``similarity_min`` and, with a manifest, ``wer_common_percent``,
        ``wer_reference_common_percent`` and ``wer_delta_pp``.

    """
    reference = _first_texts(reference_items)
    identical = 0
    sims: list[float] = []
    for item in _ok_items(items):
        ref_item = reference.get(_clip_of(item))
        if ref_item is None:
            continue
        ours = normalize_text(str(item.get("text") or ""))
        theirs = normalize_text(str(ref_item.get("text") or ""))
        identical += int(ours == theirs)
        sims.append(similarity(ours, theirs))
    if not sims:
        return {"compared": 0}
    result: JsonDict = {
        "compared": len(sims),
        "identical_percent": 100.0 * identical / len(sims),
        "differing_percent": 100.0 * (len(sims) - identical) / len(sims),
        "similarity_mean": statistics.fmean(sims),
        "similarity_min": min(sims),
    }
    if manifest is not None:
        ours_first = _first_texts(items)
        common = sorted(set(ours_first) & set(reference))
        step_wer = corpus_wer([ours_first[c] for c in common], manifest)
        ref_wer = corpus_wer([reference[c] for c in common], manifest)
        if step_wer["wer_percent"] is not None and ref_wer["wer_percent"] is not None:
            result.update(
                wer_common_percent=step_wer["wer_percent"],
                wer_reference_common_percent=ref_wer["wer_percent"],
                wer_delta_pp=step_wer["wer_percent"] - ref_wer["wer_percent"],
                wer_common_items=len(common),
            )
    return result


def repeat_consistency(items: Sequence[Mapping[str, Any]]) -> float | None:
    """Return the share of repeated occurrences identical to the clip's first one.

    A deterministic engine returns the same text for the same clip; a lower
    value inside one configuration points at races between concurrent calls.

    Args:
        items: Result items (with repeats).

    Returns:
        Percentage, or ``None`` when no clip was repeated.

    """
    first = _first_texts(items)
    repeats = 0
    same = 0
    for item in _ok_items(items):
        anchor = first[_clip_of(item)]
        if anchor is item:
            continue
        repeats += 1
        same += int(
            normalize_text(str(item.get("text") or ""))
            == normalize_text(str(anchor.get("text") or "")),
        )
    return 100.0 * same / repeats if repeats else None
