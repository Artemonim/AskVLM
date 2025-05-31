# Placeholder for downloader.py
import json
from pathlib import Path
from typing import Dict, List, cast

# * Ensure models directory exists


def ensure_models_dir(models_path: Path) -> None:
    """Create models directory if it does not exist."""
    models_path.mkdir(parents=True, exist_ok=True)


# * Check for missing model files based on models.json configuration


def check_missing_models(models_path: Path, models_config: Dict[str, str]) -> List[str]:
    """Return list of model names that are not present in models_path."""
    missing = []
    for model_name, version in models_config.items():
        model_file = models_path / model_name
        if not model_file.exists():
            missing.append(model_name)
    return missing


# * Download a model from a given URL or repository


def download_model(model_name: str, url: str, models_path: Path) -> None:
    """Download model weights and save under models_path/model_name."""
    # ! Actual download logic (e.g., using requests or huggingface_hub) will be implemented in Phase 2
    raise NotImplementedError("Model download not yet implemented")


# * Load models configuration from JSON


def load_models_config(config_path: Path) -> Dict[str, str]:
    """Load models.json containing model names and versions."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return cast(Dict[str, str], json.load(f))
