import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# * Resolve project directories


def _find_project_root(start: Path | None = None) -> Path:
    """Return the repository root by walking up from the given path.

    Prefers a directory containing `pyproject.toml` or a `.git` folder.
    Falls back to the parent of the `core/` package if markers are not found.
    """
    here = (start or Path(__file__)).resolve()
    for p in [here, *list(here.parents)]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # * Fallback: project root is parent of `core/`
    return here.parent


def get_project_cache_dir() -> Path:
    """Return the project-local cache directory for ML assets.

    The layout places all vendor caches under a single root:
    - huggingface → `<root>/.cache/huggingface`
    - torch       → `<root>/.cache/torch`
    - whisper     → `<root>/.cache/whisper`
    """
    root = _find_project_root()
    return root / ".cache"


def configure_ml_caches(cache_root: Path | None = None) -> Path:
    """Configure environment so ML libraries cache within the project.

    Args:
        cache_root: Optional explicit cache root. Defaults to `get_project_cache_dir()`.

    Returns:
        The cache root directory used.

    Note:
        Sets common environment variables used by Hugging Face, Torch, and others.
        Directories are created if missing.

    """
    cache_root = cache_root or get_project_cache_dir()
    hf_dir = cache_root / "huggingface"
    hf_hub_dir = hf_dir / "hub"
    torch_dir = cache_root / "torch"
    whisper_dir = cache_root / "whisper"

    # * Ensure directories exist
    for d in (hf_dir, hf_hub_dir, torch_dir, whisper_dir):
        d.mkdir(parents=True, exist_ok=True)

    # * Hugging Face caches
    os.environ.setdefault("HF_HOME", str(hf_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_hub_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_hub_dir))

    # * Torch cache (models/weights)
    os.environ.setdefault("TORCH_HOME", str(torch_dir))

    # * Generic XDG cache home (some libs respect this)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    # * Whisper-specific directory (used when passing download_root=None)
    os.environ.setdefault("WHISPER_CACHE", str(whisper_dir))

    return cache_root


# * Application settings model
class Settings(BaseModel):
    """Application settings model for the toolkit."""

    models_path: Path = Field(
        default_factory=lambda: get_project_cache_dir() / "models",
        description="Directory to store AI model files",
    )
    mode: Literal["local", "cloud"] = Field(
        default="local", description="Processing mode: 'local' or 'cloud'"
    )
    export_format: Literal["txt", "docx", "odt", "srt", "vtt", "md"] = Field(
        default="txt", description="Default export format"
    )
    gpu_memory: Literal["low", "high"] = Field(
        default="high", description="GPU memory usage for LLM models"
    )
    ui_language: str = Field(default="en", description="UI language code")
    yandex_api_key: str = Field(
        default="", description="Yandex IAM token (store securely)"
    )
    yandex_folder_id: str = Field(
        default="", description="Yandex folder ID for SpeechKit"
    )
    hf_token: str = Field(
        default="",
        description="Hugging Face token for gated models (read from HF_TOKEN env if empty)",
    )

    class Config:
        """Pydantic configuration for Settings model."""

        env_file = ".env"


def load_settings(path: Path) -> Settings:
    """Load settings from JSON file, or create defaults if not present."""
    if path.exists():
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        settings = Settings(**data)
        # * Allow environment override for hf_token
        if not settings.hf_token:
            env_token = os.getenv("HF_TOKEN", "")
            if env_token:
                settings.hf_token = env_token
        return settings
    settings = Settings()
    # * On first run, also populate hf_token from env if present
    try:
        env_token = os.getenv("HF_TOKEN", "")
        if env_token:
            settings.hf_token = env_token
    except Exception:  # noqa: BLE001,S110
        pass
    with path.open("w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2)
    return settings
