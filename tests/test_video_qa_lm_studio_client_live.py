"""Opt-in live integration test against a real LM Studio server."""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING

import pytest

from core.video_qa_lm_studio_client import (
    LMStudioClientError,
    request_chat_completion,
)
from utils.askvlm_defaults import get_default_video_qa_canonical_model_id

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
@pytest.mark.heavy_ml
@pytest.mark.skipif(
    not os.getenv("ASKVLM_RUN_LIVE_LM_STUDIO"),
    reason="Live LM Studio test (set ASKVLM_RUN_LIVE_LM_STUDIO=1 to run)",
)
def test_live_lm_studio_multimodal_request(tmp_path: Path) -> None:
    """Test a live multimodal structured output request to LM Studio."""
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8C/7sAAAAASUVORK5CYII="
    )
    image_path = tmp_path / "probe.png"
    image_path.write_bytes(png_bytes)

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

    prompt = (
        "Describe the main subjects in this image and indicate if you are uncertain."
    )

    try:
        response = request_chat_completion(
            base_url="http://localhost:1234/v1",
            prompt=prompt,
            image_paths=[image_path],
            json_schema=schema,
            model=get_default_video_qa_canonical_model_id(),
            temperature=0.0,
            timeout=120.0,
        )
    except LMStudioClientError as exc:
        pytest.skip(f"LM Studio returned an error (model might not be loaded): {exc}")

    assert response.content is not None
    assert response.parsed_json is not None
    assert "answer" in response.parsed_json
    assert "visible_elements" in response.parsed_json
    assert "is_uncertain" in response.parsed_json
