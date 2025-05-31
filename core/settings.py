import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# * Application settings model
class Settings(BaseModel):
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

    class Config:
        env_file = ".env"


def load_settings(path: Path) -> Settings:
    """Load settings from JSON file, or create defaults if not present."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Settings(**data)
    settings = Settings()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2)
    return settings
