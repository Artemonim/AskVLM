import importlib
import logging
import os
from typing import TYPE_CHECKING, Any

from .llm_prompts import build_text_formatting_prompt

if TYPE_CHECKING:
    from collections.abc import Callable


# * Wrapper for LLM-based text formatting (Mistral-3B or cloud LLM)
class LLMFormatter:
    """Format raw transcription text into punctuated, structured output."""

    def __init__(
        self,
        model_name: str = "gguf-q4_0",
        model_path: str | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize LLM formatter with optional llama.cpp local model.

        If llama-cpp-python is not installed or model not found, falls back to identity formatting.
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self._llm: Callable[..., dict[str, Any]] | None = None
        self._try_load_llm()

    def _try_load_llm(self) -> None:
        path = self.model_path
        if path is None:
            # allow env override
            path = os.getenv("LLM_GGUF_PATH")
        if not path:
            return
        try:
            llama_mod = importlib.import_module("llama_cpp")
            llama_class = llama_mod.Llama

            n_gpu_layers = 0
            try:
                torch_mod = importlib.import_module("torch")
                if (
                    getattr(torch_mod, "cuda", None) is not None
                    and torch_mod.cuda.is_available()
                    and self.device == "cuda"
                ):
                    n_gpu_layers = 20  # modest default
            except (ModuleNotFoundError, AttributeError):
                n_gpu_layers = 0
            self._llm = llama_class(
                model_path=path, n_ctx=2048, n_gpu_layers=n_gpu_layers
            )
        except (ModuleNotFoundError, OSError, ValueError) as e:
            logging.getLogger(__name__).debug("LLM load skipped: %s", e)
            self._llm = None

    def format_text(self, text: str) -> str:
        """Process text to correct punctuation and paragraphs. Falls back to identity."""
        if not text:
            return ""
        llm = self._llm
        if llm is None:
            return text
        try:
            prompt = build_text_formatting_prompt(text)
            out = llm(
                prompt,
                max_tokens=min(2048, max(256, len(text) // 2)),
                temperature=0.2,
                top_p=0.9,
                stop=["\n\nUser:", "\n\nAssistant:"],
            )
            choice = out.get("choices", [{}])[0]
            content = choice.get("text") or choice.get("message", {}).get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        except (KeyError, TypeError, ValueError) as e:
            logging.getLogger(__name__).debug("LLM format fallback: %s", e)
        return text

    def close(self) -> None:
        """Release the loaded LLM backend."""
        self._llm = None
