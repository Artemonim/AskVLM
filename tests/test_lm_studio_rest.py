"""Tests for LM Studio REST control URL parsing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.lm_studio_rest import (
    lm_studio_llm_loaded_instance_count,
    lm_studio_unload_all_llm_instances,
    openai_chat_base_to_local_rest_root,
)


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


@patch("core.lm_studio_rest.urllib.request.urlopen")
def test_lm_studio_llm_loaded_instance_count_skips_embedding_rows(
    mock_urlopen: MagicMock,
) -> None:
    """Embedding catalog rows do not add to the LLM loaded-instance total."""
    body = (
        b'{"data":['
        b'{"type":"llm","loaded_instances":[{"id":"a"}]},'
        b'{"type":"embedding","loaded_instances":[{"id":"e"}]}'
        b"]}"
    )
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value.read.return_value = body
    mock_urlopen.return_value = fake_cm
    assert lm_studio_llm_loaded_instance_count("http://127.0.0.1:1234") == 1


@patch("core.lm_studio_rest.lm_studio_unload_model")
@patch("core.lm_studio_rest.urllib.request.urlopen")
def test_lm_studio_unload_all_llm_instances_calls_unload_per_id(
    mock_urlopen: MagicMock,
    mock_unload: MagicMock,
) -> None:
    """``lm_studio_unload_all_llm_instances`` POSTs unload for each distinct instance id."""
    body = b'{"data":[{"type":"llm","loaded_instances":[{"id":"a"},{"id":"b"}]}]}'
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value.read.return_value = body
    mock_urlopen.return_value = fake_cm
    n = lm_studio_unload_all_llm_instances("http://127.0.0.1:1234")
    assert n == 2
    assert mock_unload.call_count == 2


@patch("core.lm_studio_rest.urllib.request.urlopen")
def test_lm_studio_llm_loaded_instance_count_dedupes_same_id_across_rows(
    mock_urlopen: MagicMock,
) -> None:
    """The same ``instance_id`` must not be double-counted across catalog rows."""
    body = (
        b'{"data":['
        b'{"type":"llm","loaded_instances":[{"id":"same"}]},'
        b'{"type":"llm","loaded_instances":[{"id":"same"}]}'
        b"]}"
    )
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value.read.return_value = body
    mock_urlopen.return_value = fake_cm
    assert lm_studio_llm_loaded_instance_count("http://127.0.0.1:1234") == 1


@patch("core.lm_studio_rest.urllib.request.urlopen")
def test_lm_studio_llm_loaded_instance_count_unreachable_returns_none(
    mock_urlopen: MagicMock,
) -> None:
    """Network failure yields None (caller decides whether to assert)."""
    mock_urlopen.side_effect = OSError("no server")
    assert lm_studio_llm_loaded_instance_count("http://127.0.0.1:1234") is None
