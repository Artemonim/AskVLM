from collections.abc import Generator
from contextlib import contextmanager

import torch


# * Guard to ensure only one heavy model is loaded on GPU at a time
class GPUResourceGuard:
    """Manage GPU memory allocation for AI models."""

    def __init__(self) -> None:
        self.current_model: str | None = None

    def acquire(self, model_name: str) -> None:
        """Acquire GPU resources for the specified model, unloading previous model if needed."""
        if self.current_model and self.current_model != model_name:
            # * Unload previous model
            try:
                del self.current_model
                torch.cuda.empty_cache()
            except (RuntimeError, AttributeError):
                pass
        self.current_model = model_name
        # ! VRAM checks with NVIDIA NVML will be added in Phase 2

    @contextmanager
    def model(self, model_name: str) -> Generator[None, None, None]:
        """Context manager to safely load a model and free resources afterwards."""
        self.acquire(model_name)
        try:
            yield
        finally:
            # * Release model resources
            try:
                del self.current_model
                torch.cuda.empty_cache()
            except (RuntimeError, AttributeError):
                pass
