"""LM Studio HTTP client for Video QA multimodal and structured output interactions."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LMStudioResponse:
    """The normalized response from the LM Studio API."""

    content: str
    parsed_json: object | None
    used_fallback: bool
    finish_reason: str
    raw_response: dict[str, Any]


class LMStudioClientError(Exception):
    """Base exception for LM Studio client errors."""


def _guess_image_mime_type(path: Path) -> str:
    """Return the best-effort MIME type for an image path."""
    mime_type, _encoding = mimetypes.guess_type(str(path))
    if mime_type:
        return mime_type
    return "image/jpeg"


def _read_image_part(path: Path) -> dict[str, Any]:
    """Return one OpenAI-compatible image content part for `path`."""
    raw_bytes = path.read_bytes()
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    mime_type = _guess_image_mime_type(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def _build_payload(
    prompt: str,
    image_paths: Sequence[Path],
    json_schema: dict[str, Any] | None,
    model: str = "local-model",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Build the request payload for the chat completion endpoint."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(_read_image_part(image_path) for image_path in image_paths)

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "stream": False,
    }

    if json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "strict": True,
                "schema": json_schema,
            },
        }

    return payload


def _decode_json_body(body: bytes) -> dict[str, Any]:
    """Decode a JSON body returned by LM Studio."""
    try:
        raw = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON response from LM Studio: {exc}"
        raise LMStudioClientError(msg) from exc
    if not isinstance(raw, dict):
        msg = "LM Studio response root must be a JSON object."
        raise LMStudioClientError(msg)
    return raw


def _clean_json_candidate(text: str) -> str:
    """Strip common markdown code fences from a JSON candidate string."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate[3:]
        if candidate.lstrip().startswith("json"):
            candidate = candidate.lstrip()[4:]
    candidate = candidate.removesuffix("```")
    return candidate.strip()


def _parse_json_candidate(text: str) -> object | None:
    """Try to parse JSON from a text candidate and return None on failure."""
    candidate = _clean_json_candidate(text)
    if not candidate:
        return None
    try:
        return cast("object", json.loads(candidate))
    except json.JSONDecodeError:
        return None


def _extract_message_fields(raw_res: dict[str, Any]) -> tuple[str, str, str]:
    """Extract the first choice message content and finish metadata."""
    choices = raw_res.get("choices")
    if not isinstance(choices, list) or not choices:
        msg = "No choices returned in the LM Studio response."
        raise LMStudioClientError(msg)

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        msg = "LM Studio choice[0] must be a JSON object."
        raise LMStudioClientError(msg)

    message = first_choice.get("message")
    if not isinstance(message, dict):
        msg = "LM Studio choice[0].message must be a JSON object."
        raise LMStudioClientError(msg)

    content = message.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    reasoning_content = message.get("reasoning_content", "")
    if not isinstance(reasoning_content, str):
        reasoning_content = str(reasoning_content)
    finish_reason = first_choice.get("finish_reason", "unknown")
    if not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return content, reasoning_content, finish_reason


def _http_post_json(
    req: urllib.request.Request,
    *,
    timeout: float | None,
) -> dict[str, Any]:
    """POST ``req`` and decode JSON; ``timeout`` None omits the socket cap (blocking read)."""
    try:
        if timeout is None:
            with urllib.request.urlopen(req) as response:  # noqa: S310
                return _decode_json_body(response.read())
        with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
            return _decode_json_body(response.read())
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        msg = f"HTTP {exc.code}: {error_body}"
        raise LMStudioClientError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"URL Error: {exc.reason}"
        raise LMStudioClientError(msg) from exc


def request_chat_completion(
    base_url: str,
    prompt: str,
    image_paths: Sequence[Path | str] | None = None,
    json_schema: dict[str, Any] | None = None,
    model: str = "local-model",
    temperature: float = 0.0,
    timeout: float | None = None,
) -> LMStudioResponse:
    """Send a multimodal chat completion request to LM Studio.

    If `json_schema` is provided and the structured output path fails, the client
    retries once without the schema and attempts to parse JSON from the plain-text
    response. This keeps the caller usable even when the local build rejects
    structured output. When ``timeout`` is ``None``, the HTTP read has no fixed
    deadline so long local inference is not cut off mid-chunk.
    """
    normalized_image_paths = tuple(Path(path) for path in (image_paths or ()))
    url = f"{base_url.rstrip('/')}/chat/completions"

    def do_request(schema: dict[str, Any] | None) -> dict[str, Any]:
        payload = _build_payload(
            prompt, normalized_image_paths, schema, model, temperature
        )
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return _http_post_json(req, timeout=timeout)

    used_fallback = False

    try:
        raw_res = do_request(json_schema)
    except LMStudioClientError as exc:
        if json_schema is None:
            raise
        logger.warning(
            "Structured output request failed; retrying without schema. Error: %s",
            exc,
        )
        used_fallback = True
        raw_res = do_request(None)

    content, reasoning_content, finish_reason = _extract_message_fields(raw_res)
    if not content and reasoning_content:
        content = _clean_json_candidate(reasoning_content) or reasoning_content

    parsed_json = None
    if json_schema is not None:
        parsed_json = _parse_json_candidate(content)
        if parsed_json is None and reasoning_content and reasoning_content != content:
            parsed_json = _parse_json_candidate(reasoning_content)
        if parsed_json is None:
            logger.warning("Failed to parse JSON from the LM Studio response body.")

    return LMStudioResponse(
        content=content,
        parsed_json=parsed_json,
        used_fallback=used_fallback,
        finish_reason=finish_reason,
        raw_response=raw_res,
    )
