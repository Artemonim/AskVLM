from pathlib import Path
from typing import Any

import torch
import whisper


# * Wrapper around OpenAI Whisper model for transcription
class WhisperWrapper:
    """Handle loading and using the Whisper model for speech-to-text."""

    def __init__(
        self,
        model_name: str = "base",
        model_root: Path | None = None,
        device: str = "cuda",
    ) -> None:
        """Load Whisper model to specified device, downloading to model_root if provided."""
        download_root = str(model_root) if model_root else None
        # * Load model on GPU or CPU
        self.device = device if torch.cuda.is_available() else "cpu"
        try:
            self.model = whisper.load_model(
                model_name,
                download_root=download_root,
                device=self.device,
            )
        except Exception:
            # * Fallback to CPU if GPU load fails (e.g., OOM)
            self.device = "cpu"
            self.model = whisper.load_model(
                model_name,
                download_root=download_root,
                device=self.device,
            )

    def transcribe(
        self,
        audio_path: Path,
        **kwargs: object,
    ) -> str:
        """Transcribe audio file and return transcription text."""
        # * Cast kwargs to a concrete dict for the underlying API
        kw: dict[str, object] = dict(kwargs)
        result: dict[str, Any] = self.model.transcribe(str(audio_path), **kw)
        text: str = result.get("text", "")
        return text
