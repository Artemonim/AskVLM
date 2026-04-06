"""Unit tests for the LM Studio client."""

from __future__ import annotations

import json
import urllib.error
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from core.pipelines import CancelledError
from core.video_qa_lm_studio_client import (
    _build_payload,
    build_provider_reasoning_option,
    request_chat_completion,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_build_payload_plain_text() -> None:
    """Test payload construction with only text."""
    payload = _build_payload("Hello", [], None)
    assert payload["model"] == "local-model"
    assert payload["stream"] is False
    assert len(payload["messages"]) == 1
    content = payload["messages"][0]["content"]
    assert len(content) == 1
    assert content[0] == {"type": "text", "text": "Hello"}
    assert "response_format" not in payload


def test_build_payload_multimodal_png(tmp_path: Path) -> None:
    """Test payload construction with an image file path."""
    image_path = tmp_path / "probe.png"
    image_path.write_bytes(b"fake-png-bytes")

    payload = _build_payload("Hello", [image_path], None)
    content = payload["messages"][0]["content"]
    assert len(content) == 2
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_payload_json_schema() -> None:
    """Test payload construction with JSON schema."""
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    payload = _build_payload("Hello", [], schema)
    assert "response_format" in payload
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == schema


def test_build_payload_reasoning_effort() -> None:
    """OpenAI-style ``reasoning_effort`` is forwarded when set."""
    payload = _build_payload("Hello", [], None, reasoning_effort="minimal")
    assert payload["reasoning_effort"] == "minimal"


def test_build_payload_omits_reasoning_effort_when_unset() -> None:
    """Default payload does not include ``reasoning_effort``."""
    payload = _build_payload("Hello", [], None)
    assert "reasoning_effort" not in payload
    assert "reasoning" not in payload


def test_build_payload_lm_studio_reasoning() -> None:
    """LM Studio-style ``reasoning`` (e.g. ``off`` / ``on``) is forwarded when set."""
    payload = _build_payload("Hello", [], None, reasoning="off")
    assert payload["reasoning"] == "off"


def test_build_payload_openrouter_reasoning_object() -> None:
    """OpenRouter-style reasoning objects are forwarded without reshaping."""
    payload = _build_payload("Hello", [], None, reasoning={"effort": "low"})
    assert payload["reasoning"] == {"effort": "low"}


def test_build_provider_reasoning_option_uses_local_toggle_for_lm_studio() -> None:
    """Local OpenAI-compatible targets use LM Studio's on/off reasoning contract."""
    assert (
        build_provider_reasoning_option("http://127.0.0.1:1234/v1", enabled=False)
        == "off"
    )
    assert (
        build_provider_reasoning_option("http://127.0.0.1:1234/v1", enabled=True)
        == "on"
    )


def test_build_provider_reasoning_option_uses_object_for_openrouter() -> None:
    """OpenRouter targets use the documented object-based reasoning contract."""
    assert build_provider_reasoning_option(
        "https://openrouter.ai/api/v1",
        enabled=False,
    ) == {"effort": "none"}
    assert build_provider_reasoning_option(
        "https://openrouter.ai/api/v1",
        enabled=True,
    ) == {"effort": "low"}


@patch("core.video_qa_lm_studio_client.urllib.request.urlopen")
def test_request_chat_completion_success(mock_urlopen: MagicMock) -> None:
    """Test a successful request without fallback."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {"choices": [{"message": {"content": "World"}, "finish_reason": "stop"}]}
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    res = request_chat_completion("http://test", "Hello")
    assert res.content == "World"
    assert res.finish_reason == "stop"
    assert res.parsed_json is None
    assert res.used_fallback is False
    assert mock_urlopen.call_count == 1
    _args, kwargs = mock_urlopen.call_args
    assert "timeout" not in kwargs


@patch("core.video_qa_lm_studio_client.urllib.request.urlopen")
def test_request_chat_completion_sends_bearer_header(mock_urlopen: MagicMock) -> None:
    """Optional ``authorization_bearer`` adds an Authorization header on HTTP POST."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    request_chat_completion("http://test", "Hello", authorization_bearer="secret-token")
    req = mock_urlopen.call_args[0][0]
    headers = dict(req.header_items())
    assert headers.get("Authorization") == "Bearer secret-token"


@patch("core.video_qa_lm_studio_client.urllib.request.urlopen")
def test_request_chat_completion_json_schema_success(
    mock_urlopen: MagicMock,
) -> None:
    """Test a successful JSON schema request."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {
            "choices": [
                {"message": {"content": '{"answer": "yes"}'}, "finish_reason": "stop"}
            ]
        }
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    res = request_chat_completion("http://test", "Hello", json_schema=schema)
    assert res.content == '{"answer": "yes"}'
    assert res.parsed_json == {"answer": "yes"}
    assert res.used_fallback is False


@patch("core.video_qa_lm_studio_client.urllib.request.urlopen")
def test_request_chat_completion_fallback(
    mock_urlopen: MagicMock,
) -> None:
    """Test fallback when structured output fails with a non-400 HTTP error."""
    mock_error = urllib.error.HTTPError(
        url="http://test/chat/completions",
        code=422,
        msg="Unprocessable Entity",
        hdrs={},
        fp=MagicMock(),
    )
    mock_error.read.return_value = b'{"error": "Unsupported schema"}'

    mock_success = MagicMock()
    mock_success.read.return_value = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": '{"answer": "fallback_yes"}'},
                    "finish_reason": "stop",
                }
            ]
        }
    ).encode("utf-8")
    mock_success.__enter__.return_value = mock_success

    mock_urlopen.side_effect = [mock_error, mock_success]

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    res = request_chat_completion("http://test", "Hello", json_schema=schema)

    assert res.used_fallback is True
    assert res.parsed_json == {"answer": "fallback_yes"}
    assert mock_urlopen.call_count == 2


@patch("core.video_qa_lm_studio_client.urllib.request.urlopen")
def test_request_chat_completion_reasoning_content_json(
    mock_urlopen: MagicMock,
) -> None:
    """Test JSON extraction from reasoning content when message content is empty."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": '```json\n{"answer": "reasoning_yes"}\n```',
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    ).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    res = request_chat_completion("http://test", "Hello", json_schema=schema)

    assert res.content == '{"answer": "reasoning_yes"}'
    assert res.parsed_json == {"answer": "reasoning_yes"}
    assert res.used_fallback is False


def test_request_chat_completion_cancel_before_http() -> None:
    """``should_cancel`` can abort before any HTTP work starts."""
    with pytest.raises(CancelledError):
        request_chat_completion(
            "http://127.0.0.1:9/v1", "x", should_cancel=lambda: True
        )


def test_build_payload_cancel_between_images(tmp_path: Path) -> None:
    """``should_cancel`` is consulted before each embedded image is read."""
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p1.write_bytes(b"x")
    p2.write_bytes(b"y")
    calls = {"n": 0}

    def sc() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    with pytest.raises(CancelledError):
        _build_payload("t", [p1, p2], None, should_cancel=sc)
