from __future__ import annotations

from core.video_qa_lm_studio_termination import (
    LlmTerminationInfo,
    LlmTerminationKind,
    normalize_video_qa_llm_termination,
)


def test_context_overflow_from_nested_stop_reason() -> None:
    """LM Studio may expose stopReason under nested stats-like objects."""
    payload = {
        "id": "pred-1",
        "stats": {"stopReason": "contextLengthReached"},
    }
    info = normalize_video_qa_llm_termination(payload)
    assert info.kind == LlmTerminationKind.CONTEXT_OVERFLOW
    assert info.is_overflow is True
    assert info.is_truncated is True
    assert info.is_unknown is False


def test_output_length_limit_from_max_predicted_tokens() -> None:
    """MaxPredictedTokensReached maps to output token limit, not context overflow."""
    payload = {"stopReason": "maxPredictedTokensReached"}
    info = normalize_video_qa_llm_termination(payload)
    assert info.kind == LlmTerminationKind.OUTPUT_LENGTH_LIMIT
    assert info.is_overflow is False
    assert info.is_truncated is True


def test_stop_string_termination() -> None:
    """StopStringFound is classified as stop-string termination."""
    payload = {"choices": [{"finish_reason": "stopStringFound"}]}
    info = normalize_video_qa_llm_termination(payload)
    assert info.kind == LlmTerminationKind.STOP_STRING
    assert info.is_stop_string is True


def test_tool_calls_termination() -> None:
    """OpenAI-style tool_calls finish_reason maps to tool-call termination."""
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {"tool_calls": []},
            }
        ]
    }
    info = normalize_video_qa_llm_termination(payload)
    assert info.kind == LlmTerminationKind.TOOL_CALLS
    assert info.is_tool_calls is True


def test_user_cancelled_signal() -> None:
    """User or runtime abort-style strings are treated as cancellation."""
    payload = {"stop_reason": "aborted"}
    info = normalize_video_qa_llm_termination(payload)
    assert info.kind == LlmTerminationKind.USER_CANCELLED
    assert info.is_user_cancelled is True


def test_normal_openai_stop_completion() -> None:
    """OpenAI finish_reason stop means natural completion."""
    payload = {"choices": [{"finish_reason": "stop"}]}
    info = normalize_video_qa_llm_termination(payload)
    assert info.kind == LlmTerminationKind.NORMAL
    assert info.is_normal_completion is True
    assert info.is_truncated is False


def test_empty_payload_is_unknown() -> None:
    """Missing or empty structures yield UNKNOWN with no raw signals."""
    for payload in ({}, [], None, {"choices": []}):
        info = normalize_video_qa_llm_termination(payload)
        assert info.kind == LlmTerminationKind.UNKNOWN
        assert info.raw_signals == ()
        assert info.is_unknown is True


def test_unrecognized_signal_is_unknown() -> None:
    """Arbitrary stopReason text that does not match known patterns stays UNKNOWN."""
    payload = {"stopReason": "custom_vendor_reason_v2"}
    info = normalize_video_qa_llm_termination(payload)
    assert info.kind == LlmTerminationKind.UNKNOWN
    assert info.raw_signals == ("custom_vendor_reason_v2",)


def test_priority_prefers_user_cancel_over_normal() -> None:
    """When multiple keys appear, higher-priority termination wins."""
    payload = {
        "finish_reason": "stop",
        "stats": {"stopReason": "cancelled"},
    }
    info = normalize_video_qa_llm_termination(payload)
    assert info.kind == LlmTerminationKind.USER_CANCELLED


def test_dataclass_flags_are_consistent() -> None:
    """LlmTerminationInfo exposes stable boolean helpers for policy code."""
    info = LlmTerminationInfo(
        kind=LlmTerminationKind.UNKNOWN,
        raw_signals=(),
    )
    assert info.is_overflow is False
    assert info.is_truncated is False
    assert info.is_stop_string is False
    assert info.is_tool_calls is False
    assert info.is_user_cancelled is False
    assert info.is_normal_completion is False
    assert info.is_unknown is True
