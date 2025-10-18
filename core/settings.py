import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# * Application settings model
class Settings(BaseModel):
    """Application settings model for the toolkit."""

    models_path: Path = Field(
        default_factory=lambda: Path.home() / ".mytranscriber" / "models",
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
            from os import getenv

            env_token = getenv("HF_TOKEN", "")
            if env_token:
                settings.hf_token = env_token
        return settings
    settings = Settings()
    # * On first run, also populate hf_token from env if present
    try:
        from os import getenv

        env_token = getenv("HF_TOKEN", "")
        if env_token:
            settings.hf_token = env_token
    except Exception:
        pass
    with path.open("w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2)
    return settings
