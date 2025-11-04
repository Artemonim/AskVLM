import importlib


def test_cuda_is_available() -> None:
    """The application requires CUDA; tests must fail when CUDA is unavailable."""
    torch = importlib.import_module("torch")
    # Check cuda attribute presence separately to satisfy PT018
    assert getattr(torch, "cuda", None) is not None, "torch.cuda attribute is missing"
    assert torch.cuda.is_available(), (
        "CUDA is required by application policy; torch.cuda.is_available() is False"
    )
