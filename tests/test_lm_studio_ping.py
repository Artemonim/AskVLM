"""Lightweight LM Studio reachability via OpenAI-compatible HTTP (no inference)."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest


@pytest.mark.lm_studio_ping
def test_lm_studio_lists_models_via_openai_api() -> None:
    """GET ``/v1/models`` returns JSON when LM Studio is listening."""
    url = "http://127.0.0.1:1234/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as resp:  # noqa: S310
            raw = resp.read()
    except OSError:
        pytest.skip("LM Studio not reachable on 127.0.0.1:1234")
    data: Any = json.loads(raw.decode("utf-8"))
    assert isinstance(data, dict)
    assert "data" in data or data.get("object") == "list"
