import contextlib
import json
from pathlib import Path
from typing import cast

try:
    import requests
    from tqdm import tqdm
except ImportError:
    requests = None  # type: ignore[assignment]
    tqdm = None  # type: ignore[assignment, misc]

try:
    from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
except ImportError:
    snapshot_download = None


def ensure_models_dir(models_path: Path) -> None:
    """Create models directory if it does not exist."""
    models_path.mkdir(parents=True, exist_ok=True)


def check_missing_models(models_path: Path, models_config: dict[str, str]) -> list[str]:
    """Return list of model names that are not present in models_path."""
    missing = []
    for model_name in models_config:
        # Simple check: assume model_name corresponds to a directory or file
        # If it's a HF model like "systran/faster-whisper-small", we check for the dir
        name_clean = model_name.replace("/", "--")
        if (
            not (models_path / name_clean).exists()
            and not (models_path / model_name).exists()
        ):
            missing.append(model_name)
    return missing


def download_file(url: str, dest_path: Path) -> None:
    """Download a single file with progress bar."""
    if not requests or not tqdm:  # type: ignore[truthy-function]
        msg = "requests and tqdm are required for downloading."
        raise ImportError(msg)

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with (
        dest_path.open("wb") as f,
        tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


def download_model(model_name: str, source: str, models_path: Path) -> None:
    """Download a model from a given source."""
    # * Handle Hugging Face models (folder download)
    if source == "huggingface":
        if snapshot_download is None:
            return

        # Download to models_path/{model_name}
        # We'll download to models_path / model_name (sanitized)
        sanitized_name = model_name.replace("/", "--")
        target_dir = models_path / sanitized_name

        with contextlib.suppress(Exception):
            snapshot_download(
                repo_id=model_name, local_dir=target_dir, local_dir_use_symlinks=False
            )
        return

    # * Handle direct URL downloads (single file)
    if source.startswith("http"):
        # Assume it's a single file url
        filename = model_name.split("/")[-1]
        # If model_name doesn't look like a filename, use the last part of URL
        if "." not in filename:
            filename = source.split("/")[-1]

        target_path = models_path / filename
        try:
            download_file(source, target_path)
        except Exception:  # noqa: BLE001
            # Clean up partial file
            if target_path.exists():
                target_path.unlink()
        return


def load_models_config(config_path: Path) -> dict[str, str]:
    """Load models configuration from JSON file."""
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return cast("dict[str, str]", json.load(f))
