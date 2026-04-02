"""LM Studio developer REST helpers (load/unload) for local multi-model Video QA runs."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Final, cast

logger = logging.getLogger(__name__)

DEFAULT_LM_STUDIO_LOAD_TIMEOUT_S: Final[float] = 600.0
DEFAULT_LM_STUDIO_UNLOAD_TIMEOUT_S: Final[float] = 120.0


class LMStudioRestError(Exception):
    """Raised when an LM Studio REST control call fails."""


def openai_chat_base_to_local_rest_root(openai_base_url: str) -> str | None:
    """Return ``http://host:port`` for LM Studio control API, or None if not a local /v1 base.

    Expects an OpenAI-compatible chat base such as ``http://127.0.0.1:1234/v1``.
    """
    raw = openai_base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    host = (parsed.hostname or "").lower()
    if host not in ("127.0.0.1", "localhost", "::1"):
        return None
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        return None
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or f"{host}:{parsed.port or 1234}"
    return f"{scheme}://{netloc}"


def _post_json(
    url: str,
    body: dict[str, Any],
    *,
    timeout: float,
    bearer: str | None,
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(  # noqa: S310
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
            raw = response.read()
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        msg = f"HTTP {exc.code} {url}: {err_body}"
        raise LMStudioRestError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"URL error {url}: {exc.reason}"
        raise LMStudioRestError(msg) from exc
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON from {url}"
        raise LMStudioRestError(msg) from exc
    if not isinstance(decoded, dict):
        msg = f"Expected JSON object from {url}"
        raise LMStudioRestError(msg)
    return decoded


def lm_studio_unload_model(
    rest_root: str,
    instance_id: str,
    *,
    timeout_s: float = DEFAULT_LM_STUDIO_UNLOAD_TIMEOUT_S,
    bearer: str | None = None,
) -> None:
    """POST ``/api/v1/models/unload``; ``instance_id`` is the LM Studio model instance id."""
    root = rest_root.rstrip("/")
    url = f"{root}/api/v1/models/unload"
    _post_json(url, {"instance_id": instance_id}, timeout=timeout_s, bearer=bearer)


def lm_studio_load_model(
    rest_root: str,
    model: str,
    *,
    context_length: int | None = None,
    timeout_s: float = DEFAULT_LM_STUDIO_LOAD_TIMEOUT_S,
    bearer: str | None = None,
) -> str:
    """POST ``/api/v1/models/load``; returns ``instance_id`` from the response."""
    root = rest_root.rstrip("/")
    url = f"{root}/api/v1/models/load"
    body: dict[str, Any] = {"model": model}
    if context_length is not None:
        body["context_length"] = int(context_length)
    result = _post_json(url, body, timeout=timeout_s, bearer=bearer)
    iid = result.get("instance_id")
    if not isinstance(iid, str) or not iid.strip():
        msg = f"LM Studio load response missing instance_id: {result!r}"
        raise LMStudioRestError(msg)
    return iid.strip()


def _rest_entry_is_llm(model_entry: dict[str, Any]) -> bool:
    """Return True when a ``/api/v1/models`` row is not clearly an embedding model."""
    raw_type = model_entry.get("type")
    if raw_type is None:
        return True
    lowered = str(raw_type).lower()
    return "embed" not in lowered


def _lm_studio_models_catalog_rows(decoded: object) -> list[object]:
    """Normalize LM Studio ``GET /api/v1/models`` JSON to a list of catalog rows."""
    if isinstance(decoded, dict):
        for key in ("data", "models"):
            candidate = decoded.get(key)
            if isinstance(candidate, list):
                return list(candidate)
        return []
    if isinstance(decoded, list):
        return list(decoded)
    return []


def _lm_studio_fetch_models_catalog_json(
    rest_root: str,
    *,
    bearer: str | None = None,
    timeout_s: float = 5.0,
) -> object | None:
    """Return decoded JSON from ``GET .../api/v1/models``, or None if unreachable."""
    root = rest_root.rstrip("/")
    url = f"{root}/api/v1/models"
    headers: dict[str, str] = {}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:  # noqa: S310
            raw = response.read()
    except OSError:
        return None
    try:
        return cast("object", json.loads(raw.decode("utf-8")))
    except json.JSONDecodeError:
        return None


def _unique_llm_loaded_instance_ids(decoded: object) -> list[str]:
    """Return distinct non-empty ``id`` strings from LLM ``loaded_instances`` rows."""
    seen: set[str] = set()
    ordered: list[str] = []
    for row in _lm_studio_models_catalog_rows(decoded):
        if not isinstance(row, dict) or not _rest_entry_is_llm(row):
            continue
        loaded = row.get("loaded_instances")
        if not isinstance(loaded, list):
            continue
        for inst in loaded:
            if not isinstance(inst, dict):
                continue
            iid = inst.get("id")
            if not isinstance(iid, str) or not iid.strip():
                continue
            sid = iid.strip()
            if sid in seen:
                continue
            seen.add(sid)
            ordered.append(sid)
    return ordered


def lm_studio_unload_all_llm_instances(
    rest_root: str,
    *,
    bearer: str | None = None,
    list_timeout_s: float = 5.0,
    unload_timeout_s: float = DEFAULT_LM_STUDIO_UNLOAD_TIMEOUT_S,
) -> int:
    """Unload every loaded LLM instance id from the catalog (best effort).

    Uses ``GET /api/v1/models`` then ``POST /api/v1/models/unload`` per instance. Failures are
    logged and ignored so callers can free VRAM before loading another stack on the same GPU.
    """
    decoded = _lm_studio_fetch_models_catalog_json(
        rest_root, bearer=bearer, timeout_s=list_timeout_s
    )
    if decoded is None:
        return 0
    n_ok = 0
    for sid in _unique_llm_loaded_instance_ids(decoded):
        try:
            lm_studio_unload_model(
                rest_root, sid, timeout_s=unload_timeout_s, bearer=bearer
            )
        except LMStudioRestError as exc:
            logger.warning("LM Studio unload failed for instance_id=%s: %s", sid, exc)
        else:
            n_ok += 1
    return n_ok


def lm_studio_llm_loaded_instance_count(
    rest_root: str,
    *,
    bearer: str | None = None,
    timeout_s: float = 5.0,
) -> int | None:
    """Return how many LLM instances are loaded (``GET /api/v1/models``), or None if unreachable.

    LM Studio returns catalog rows with ``loaded_instances`` per model; rows that look like
    embedding models are skipped. When instances expose ``id``, counts are deduplicated so the
    same loaded slot is not double-counted across multiple catalog rows.
    """
    decoded = _lm_studio_fetch_models_catalog_json(
        rest_root, bearer=bearer, timeout_s=timeout_s
    )
    if decoded is None:
        return None
    # * Prefer unique instance ids: the same loaded slot may appear under multiple catalog rows.
    instance_ids: set[str] = set()
    unkeyed = 0
    for row in _lm_studio_models_catalog_rows(decoded):
        if not isinstance(row, dict) or not _rest_entry_is_llm(row):
            continue
        loaded = row.get("loaded_instances")
        if not isinstance(loaded, list):
            continue
        for inst in loaded:
            if isinstance(inst, dict):
                iid = inst.get("id")
                if isinstance(iid, str) and iid.strip():
                    instance_ids.add(iid.strip())
                    continue
            unkeyed += 1
    return len(instance_ids) + unkeyed


__all__ = [
    "DEFAULT_LM_STUDIO_LOAD_TIMEOUT_S",
    "DEFAULT_LM_STUDIO_UNLOAD_TIMEOUT_S",
    "LMStudioRestError",
    "lm_studio_llm_loaded_instance_count",
    "lm_studio_load_model",
    "lm_studio_unload_all_llm_instances",
    "lm_studio_unload_model",
    "openai_chat_base_to_local_rest_root",
]
