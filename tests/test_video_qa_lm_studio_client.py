"""Unit tests for the LM Studio client."""

from __future__ import annotations

import json
import urllib.error
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from core.video_qa_lm_studio_client import (
    _build_payload,
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
