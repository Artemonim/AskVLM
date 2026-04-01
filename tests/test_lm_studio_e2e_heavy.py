"""Heavy E2E: multimodal structured LM Studio request; skip if offline."""

from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from core.video_qa_lm_studio_client import LMStudioClientError, request_chat_completion
from utils.askvlm_defaults import get_default_video_qa_canonical_model_id

_ROOT = Path(__file__).resolve().parents[1]
_DOC_IMAGE = _ROOT / "doc/media/Multimodal GUI Design 01 - Архитектурная схема.png"
_LM_STUDIO_MODELS_URL = "http://127.0.0.1:1234/v1/models"
_LOG = logging.getLogger(__name__)


def _reasoning_tokens(raw: dict[str, Any]) -> int | None:
    """Return ``usage.completion_tokens_details.reasoning_tokens`` when present."""
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("reasoning_tokens")
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _lm_studio_http_reachable() -> bool:
    """Return True if LM Studio responds on the OpenAI-compatible models endpoint."""
    try:
        with urllib.request.urlopen(_LM_STUDIO_MODELS_URL, timeout=0.5) as resp:  # noqa: S310
            resp.read()
    except OSError:
        return False
    return True


@pytest.mark.heavy_ml
@pytest.mark.slow
@pytest.mark.integration
def test_heavy_lm_studio_multimodal_structured_doc_image() -> None:
    """Structured multimodal request; transport success only (no content checks)."""
    if not _lm_studio_http_reachable():
        pytest.skip("LM Studio not reachable on 127.0.0.1:1234")
    if not _DOC_IMAGE.is_file():
        pytest.skip(f"Fixture image missing: {_DOC_IMAGE}")

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "visible_elements": {"type": "array", "items": {"type": "string"}},
            "is_uncertain": {"type": "boolean"},
        },
        "required": ["answer", "visible_elements", "is_uncertain"],
        "additionalProperties": False,
    }
    prompt = "Что делает эта система?"
    model_id = get_default_video_qa_canonical_model_id()
    # * Multimodal + structured JSON can exceed 600s on local GPU; urllib raises TimeoutError otherwise.
    per_request_timeout_s = 1200.0
    started = time.perf_counter()
    try:
        response = request_chat_completion(
            base_url="http://localhost:1234/v1",
            prompt=prompt,
            image_paths=[_DOC_IMAGE],
            json_schema=schema,
            model=model_id,
            temperature=0.0,
            timeout=per_request_timeout_s,
            reasoning="off",
        )
    except LMStudioClientError as exc:
        pytest.fail(f"LM Studio request failed: {exc}")
    elapsed = time.perf_counter() - started
    rtok = _reasoning_tokens(response.raw_response)
    if rtok is not None:
        _LOG.warning(
            "LM Studio E2E multimodal structured (reasoning=off): %.1fs reasoning_tokens=%s (timeout=%ss)",
            elapsed,
            rtok,
            per_request_timeout_s,
        )
    else:
        _LOG.warning(
            "LM Studio E2E multimodal structured (reasoning=off): %.1fs (timeout=%ss)",
            elapsed,
            per_request_timeout_s,
        )
    assert isinstance(response.raw_response, dict)
    assert isinstance(response.finish_reason, str)
