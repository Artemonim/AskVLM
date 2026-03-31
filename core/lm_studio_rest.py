"""LM Studio developer REST helpers (load/unload) for local multi-model Video QA runs."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Final

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


__all__ = [
    "DEFAULT_LM_STUDIO_LOAD_TIMEOUT_S",
    "DEFAULT_LM_STUDIO_UNLOAD_TIMEOUT_S",
    "LMStudioRestError",
    "lm_studio_load_model",
    "lm_studio_unload_model",
    "openai_chat_base_to_local_rest_root",
]
