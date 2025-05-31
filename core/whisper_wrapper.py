from pathlib import Path
from typing import Optional, Any, Dict
import whisper
import torch


# * Wrapper around OpenAI Whisper model for transcription
class WhisperWrapper:
    """Handle loading and using the Whisper model for speech-to-text."""

    def __init__(
        self,
        model_name: str = "base",
        model_root: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        """Load Whisper model to specified device, downloading to model_root if provided."""
        if model_root:
            download_root = str(model_root)
        else:
            download_root = None
        # * Load model on GPU or CPU
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(
            model_name,
            download_root=download_root,
            device=self.device,
        )

    def transcribe(
        self,
        audio_path: Path,
        **kwargs: Any,
    ) -> str:
        """Transcribe audio file and return transcription text."""
        result: Dict[str, Any] = self.model.transcribe(str(audio_path), **kwargs)
        text: str = result.get("text", "")
        return text
