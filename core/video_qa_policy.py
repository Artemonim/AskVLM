"""Backend-only URL import policy helpers for Video QA."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_VIDEO_QA_ALLOWED_URL_SCHEMES: tuple[str, ...] = ("http", "https")


@dataclass(frozen=True, slots=True)
class VideoQAUrlImportPolicy:
    """Describe how remote Video QA URLs are admitted and staged."""

    enabled: bool = False
    allowed_schemes: tuple[str, ...] = DEFAULT_VIDEO_QA_ALLOWED_URL_SCHEMES
    allow_file_scheme: bool = False
    temp_file_policy: str = "ephemeral"

    def check(self, raw_url: str) -> tuple[bool, str]:
        """Return a policy decision and a human-readable explanation."""
        cleaned = str(raw_url).strip()
        allowed = False
        reason = "URL import is allowed by policy."
        if not cleaned:
            reason = "URL is empty."
        elif not self.enabled:
            reason = "URL import is disabled."
        else:
            parsed = urlparse(cleaned)
            scheme = parsed.scheme.lower()
            allowed_schemes = tuple(s.lower() for s in self.allowed_schemes)
            if not scheme:
                reason = "URL is missing a scheme."
            elif scheme == "file":
                if self.allow_file_scheme:
                    allowed = True
                    reason = "file:// URLs are allowed by policy."
                else:
                    reason = "file:// URLs are blocked by policy."
            elif scheme not in allowed_schemes:
                reason = f"Scheme '{parsed.scheme}' is not allowed."
            elif not parsed.netloc:
                reason = "URL is missing a host."
            else:
                allowed = True
        return allowed, reason

    def is_allowed(self, raw_url: str) -> bool:
        """Return True when the URL is allowed by policy."""
        allowed, _reason = self.check(raw_url)
        return allowed

    def temp_file_policy_description(self) -> str:
        """Describe how temporary files are handled during URL import."""
        return (
            "Remote media is staged in a temporary file outside the orchestrator "
            "and removed after ingestion when the import completes."
        )


def default_video_qa_url_import_policy(
    *,
    enabled: bool = False,
) -> VideoQAUrlImportPolicy:
    """Return the default backend-only Video QA URL import policy."""
    return VideoQAUrlImportPolicy(enabled=enabled)
