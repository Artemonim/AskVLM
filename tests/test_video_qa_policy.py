from __future__ import annotations

from core.video_qa_policy import (
    VideoQAUrlImportPolicy,
    default_video_qa_url_import_policy,
)


def test_default_policy_is_disabled_by_default() -> None:
    """Fresh policy objects reject URL import until explicitly enabled."""
    policy = VideoQAUrlImportPolicy()
    assert policy.enabled is False
    allowed, reason = policy.check("https://example.com/media.mp4")
    assert allowed is False
    assert "disabled" in reason.lower()

    factory = default_video_qa_url_import_policy()
    assert factory.enabled is False
    assert factory.is_allowed("http://example.com/") is False


def test_empty_url_is_rejected() -> None:
    """Whitespace-only input is treated as empty and never admitted."""
    policy = VideoQAUrlImportPolicy(enabled=True)
    for raw in ("", "   ", "\t\n"):
        allowed, reason = policy.check(raw)
        assert allowed is False
        assert "empty" in reason.lower()


def test_http_https_require_policy_enabled() -> None:
    """http(s) URLs succeed only when import is enabled and host is present."""
    url = "https://example.com/clip.mp4"
    disabled = VideoQAUrlImportPolicy(enabled=False)
    assert disabled.is_allowed(url) is False

    enabled = VideoQAUrlImportPolicy(enabled=True)
    assert enabled.is_allowed(url) is True
    assert enabled.is_allowed("http://cdn.example.com/x") is True


def test_scheme_and_host_validation() -> None:
    """Missing scheme, disallowed scheme, and missing host each fail clearly."""
    policy = VideoQAUrlImportPolicy(enabled=True)

    allowed, reason = policy.check("example.com/path")
    assert allowed is False
    assert "scheme" in reason.lower()

    allowed, reason = policy.check("ftp://files.example.com/bin")
    assert allowed is False
    assert "not allowed" in reason.lower()

    allowed, reason = policy.check("https://")
    assert allowed is False
    assert "host" in reason.lower()

    assert policy.is_allowed("https://Example.COM/path") is True


def test_file_scheme_blocked_by_default_allowed_when_flag_on() -> None:
    """file:// stays off unless allow_file_scheme is set alongside enabled."""
    file_url = "file:///tmp/sample.mp4"

    blocked = VideoQAUrlImportPolicy(enabled=True, allow_file_scheme=False)
    allowed, reason = blocked.check(file_url)
    assert allowed is False
    assert "blocked" in reason.lower()

    allowed_policy = VideoQAUrlImportPolicy(enabled=True, allow_file_scheme=True)
    assert allowed_policy.is_allowed(file_url) is True
    _ok, msg = allowed_policy.check(file_url)
    assert "allowed" in msg.lower()


def test_temp_file_policy_description_is_meaningful() -> None:
    """Staging behavior is described for operators and logs."""
    text = VideoQAUrlImportPolicy().temp_file_policy_description()
    assert isinstance(text, str)
    assert len(text.strip()) > 0
    assert "staged" in text.lower() or "staging" in text.lower()
    assert "cleanup" in text.lower()
    assert "explicit" in text.lower()
