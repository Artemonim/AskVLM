"""Tests for LM Studio REST control URL parsing."""

from __future__ import annotations

from core.lm_studio_rest import openai_chat_base_to_local_rest_root


def test_openai_base_maps_to_local_rest_root() -> None:
    """Local OpenAI-compatible /v1 base yields the LM Studio REST origin."""
    assert (
        openai_chat_base_to_local_rest_root("http://127.0.0.1:1234/v1")
        == "http://127.0.0.1:1234"
    )
    assert (
        openai_chat_base_to_local_rest_root("http://localhost:1234/v1/")
        == "http://localhost:1234"
    )


def test_openai_base_non_local_returns_none() -> None:
    """Non-loopback hosts are not treated as LM Studio control targets."""
    assert openai_chat_base_to_local_rest_root("https://openrouter.ai/api/v1") is None


def test_openai_base_wrong_path_returns_none() -> None:
    """Paths that are not …/v1 are ignored."""
    assert openai_chat_base_to_local_rest_root("http://127.0.0.1:1234/other") is None
