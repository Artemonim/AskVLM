"""Load repository root defaults (canonical model ids, etc.)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_DEFAULTS_FILENAME = "askvlm.defaults.json"


@lru_cache(maxsize=1)
def _project_root() -> Path:
    """Return the repository root (parent of ``utils``)."""
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_askvlm_defaults() -> dict[str, Any]:
    """Load ``askvlm.defaults.json`` from the repo root, or return an empty dict."""
    path = _project_root() / _DEFAULTS_FILENAME
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def get_default_video_qa_canonical_model_id() -> str:
    """Return the configured Video QA canonical Hugging Face model id."""
    data = load_askvlm_defaults()
    vq = data.get("video_qa")
    if isinstance(vq, dict):
        mid = vq.get("canonical_model_id")
        if isinstance(mid, str) and mid.strip():
            return mid.strip()
    return "Qwen/Qwen3.5-35B-A3B"
