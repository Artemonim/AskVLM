import importlib


def test_cuda_is_available() -> None:
    """The application requires CUDA; tests must fail when CUDA is unavailable."""
    torch = importlib.import_module("torch")
    assert getattr(torch, "cuda", None) is not None and torch.cuda.is_available(), (
        "CUDA is required by application policy; torch.cuda.is_available() is False"
    )
