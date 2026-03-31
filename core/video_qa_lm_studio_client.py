"""LM Studio HTTP client for Video QA multimodal and structured output interactions."""

from __future__ import annotations

import base64
import contextlib
import http.client
import json
import logging
import mimetypes
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .pipelines import CancelledError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Build the request payload for the chat completion endpoint."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        if should_cancel and should_cancel():
            msg = "Canceled"
            raise CancelledError(msg)
        content.append(_read_image_part(image_path))

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
    """POST ``req`` and decode JSON.

    When ``timeout`` is ``None``, the socket read has no fixed deadline.
    """
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


def _http_post_json_cancellable(  # noqa: C901, PLR0915
    url: str,
    body: bytes,
    *,
    timeout: float | None,
    should_cancel: Callable[[], bool],
) -> dict[str, Any]:
    """POST JSON on a worker thread; close the socket when cancel is requested."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        msg = f"Unsupported URL scheme for LM Studio: {parsed.scheme!r}"
        raise LMStudioClientError(msg)
    host = parsed.hostname
    if not host:
        msg = "LM Studio URL is missing a host."
        raise LMStudioClientError(msg)
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    headers = {"Content-Type": "application/json"}

    conn_holder: list[
        http.client.HTTPConnection | http.client.HTTPSConnection | None
    ] = [None]
    worker_error: list[BaseException] = []
    response_status: list[int] = []
    response_body: list[bytes] = []

    def worker() -> None:
        try:
            conn: http.client.HTTPConnection | http.client.HTTPSConnection
            if parsed.scheme == "https":
                conn = http.client.HTTPSConnection(host, port, timeout=timeout)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=timeout)
            conn_holder[0] = conn
            conn.request("POST", path, body, headers)
            resp = conn.getresponse()
            status = int(resp.status)
            data = resp.read()
            response_status.append(status)
            response_body.append(data)
        except Exception as exc:  # noqa: BLE001
            # * Capture any transport failure so the parent can join cleanly.
            worker_error.append(exc)
        finally:
            c = conn_holder[0]
            if c is not None:
                with contextlib.suppress(Exception):
                    c.close()

    thread = threading.Thread(target=worker, name="askvlm-lmstudio-http", daemon=True)
    thread.start()
    poll_s = 0.05
    while thread.is_alive():
        if should_cancel():
            c = conn_holder[0]
            if c is not None:
                with contextlib.suppress(Exception):
                    c.close()
            thread.join(timeout=15.0)
            msg = "Canceled"
            raise CancelledError(msg)
        thread.join(timeout=poll_s)

    if worker_error:
        exc = worker_error[0]
        msg = f"LM Studio HTTP worker failed: {exc}"
        raise LMStudioClientError(msg) from exc
    if not response_status or not response_body:
        msg = "Empty HTTP response from LM Studio."
        raise LMStudioClientError(msg)
    status = response_status[0]
    raw = response_body[0]
    if status >= HTTPStatus.BAD_REQUEST:
        text = raw.decode("utf-8", errors="replace")
        msg = f"HTTP {status}: {text}"
        raise LMStudioClientError(msg)
    return _decode_json_body(raw)


def request_chat_completion(  # noqa: C901
    base_url: str,
    prompt: str,
    image_paths: Sequence[Path | str] | None = None,
    json_schema: dict[str, Any] | None = None,
    model: str = "local-model",
    temperature: float = 0.0,
    timeout: float | None = None,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> LMStudioResponse:
    """Send a multimodal chat completion request to LM Studio.

    If `json_schema` is provided and the structured output path fails, the client
    retries once without the schema and attempts to parse JSON from the plain-text
    response. This keeps the caller usable even when the local build rejects
    structured output. When ``timeout`` is ``None``, the HTTP read has no fixed
    deadline so long local inference is not cut off mid-chunk.

    When ``should_cancel`` is set, the client polls it while the HTTP call runs and
    closes the socket to stop waiting on LM Studio prefill/inference.
    """
    normalized_image_paths = tuple(Path(path) for path in (image_paths or ()))
    url = f"{base_url.rstrip('/')}/chat/completions"

    def do_request(schema: dict[str, Any] | None) -> dict[str, Any]:
        if should_cancel and should_cancel():
            msg = "Canceled"
            raise CancelledError(msg)
        payload = _build_payload(
            prompt,
            normalized_image_paths,
            schema,
            model,
            temperature,
            should_cancel=should_cancel,
        )
        data = json.dumps(payload).encode("utf-8")
        if should_cancel and should_cancel():
            msg = "Canceled"
            raise CancelledError(msg)
        if should_cancel is not None:
            return _http_post_json_cancellable(
                url,
                data,
                timeout=timeout,
                should_cancel=should_cancel,
            )
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
