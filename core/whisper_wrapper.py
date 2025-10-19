import importlib
from pathlib import Path
from typing import Any


# * Wrapper around OpenAI Whisper model for transcription
class WhisperWrapper:
    """Handle loading and using the Whisper model for speech-to-text.

    This wrapper lazily imports heavy dependencies to keep GUI startup fast and
    working even when ML packages are not installed. Actual model loading
    happens on first `transcribe` call.
    """

    def __init__(
        self,
        model_name: str = "base",
        model_root: Path | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the wrapper; defer model loading until needed."""
        self.model_name = model_name
        self.model_root = model_root
        self.requested_device = device
        self.device = "cpu"
        self._model: Any | None = None

    def _ensure_model_loaded(self) -> None:
        """Load the Whisper model if not already loaded.

        Attempts GPU when available; gracefully falls back to CPU. If the
        `openai-whisper` package is not installed, raises a RuntimeError.
        """
        if self._model is not None:
            return

        # Lazy import dependencies
        try:
            whisper_mod = importlib.import_module("whisper")
        except ModuleNotFoundError as e:  # pragma: no cover
            msg = "openai-whisper is not installed"
            raise RuntimeError(msg) from e

        try:
            torch_mod = importlib.import_module("torch")
            cuda_available = (
                getattr(torch_mod, "cuda", None) is not None
                and torch_mod.cuda.is_available()
            )
        except ModuleNotFoundError:
            cuda_available = False

        if self.requested_device == "cuda" and not cuda_available:
            # * Enforce CUDA-only ML processing per project requirement
            msg = "CUDA is required for ML processing, but no compatible GPU is available."
            raise RuntimeError(msg)

        self.device = self.requested_device if cuda_available else "cpu"
        download_root = str(self.model_root) if self.model_root else None

        try:
            self._model = whisper_mod.load_model(
                self.model_name,
                download_root=download_root,
                device=self.device,
            )
        except Exception:  # noqa: BLE001
            # * Fallback to CPU if GPU load fails (e.g., OOM)
            self.device = "cpu"
            self._model = whisper_mod.load_model(
                self.model_name,
                download_root=download_root,
                device=self.device,
            )

    def transcribe(
        self,
        audio_path: Path,
        **kwargs: object,
    ) -> str:
        """Transcribe audio file and return transcription text."""
        self._ensure_model_loaded()
        model = self._model
        if model is None:
            msg = "Whisper model failed to load"
            raise RuntimeError(msg)
        # * Cast kwargs to a concrete dict for the underlying API
        kw: dict[str, object] = dict(kwargs)
        result: dict[str, Any] = model.transcribe(str(audio_path), **kw)
        text: str = result.get("text", "")
        return text
