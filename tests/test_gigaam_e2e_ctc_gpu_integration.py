"""Real-model check that GigaAM-v3 e2e CTC gives the same text on CUDA and CPU.

Excluded from ``-Fast`` (slow + heavy_ml). Skips itself when CUDA, the cached
GigaAM-v3 weights, or enough free VRAM are missing.
"""

from __future__ import annotations

import difflib
import re
from typing import TYPE_CHECKING

import pytest

from core.pipelines import LocalPipeline
from core.settings import get_project_cache_dir

if TYPE_CHECKING:
    from pathlib import Path

# * Same threshold ``--device auto`` uses: the model needs about 1.6 GiB of VRAM.
_MIN_FREE_VRAM_BYTES = int(2.5 * 1024**3)
_MIN_TEXT_SIMILARITY = 0.9
_VRAM_SLACK_BYTES = 128 * 1024**2


def _words(text: str) -> list[str]:
    """Lower-case word list without punctuation (fp16 autocast may move commas)."""
    return re.findall(r"\w+", text.lower())


def _transcribe(device: str, media: Path, work_dir: Path) -> tuple[str, str | None]:
    """Run the real pipeline once; return the text and the model's active device."""
    pipeline = LocalPipeline(
        engine="whisperx",
        stt_provider="gigaam-e2e-ctc",
        enable_diarization=False,
        enable_dialog_blocks=False,
        device=device,
    )
    try:
        text = pipeline.process(media, work_dir).get_full_text()
        active = getattr(pipeline.gigaam, "active_device", None)
    finally:
        pipeline.close(aggressive=False)
    return text, active


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.heavy_ml
@pytest.mark.xdist_group(name="ml_singleton")
def test_gigaam_e2e_cuda_text_matches_cpu_and_vram_is_released(
    tmp_path: Path, short_audio_fixture: Path
) -> None:
    """CUDA transcription equals the CPU one up to fp16 noise and frees its VRAM.

    Catches a GPU path that silently runs on CPU, produces different text, or
    keeps VRAM after ``close`` (which would starve the resident daemon).
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    hub = get_project_cache_dir() / "huggingface" / "hub"
    if not any(hub.glob("models--ai-sage--GigaAM-v3")):
        pytest.skip("GigaAM-v3 weights are not cached")
    free_bytes, _total = torch.cuda.mem_get_info()
    if free_bytes < _MIN_FREE_VRAM_BYTES:
        pytest.skip("not enough free VRAM for GigaAM-v3 on CUDA")

    baseline_allocated = torch.cuda.memory_allocated()
    cpu_text, cpu_device = _transcribe("cpu", short_audio_fixture, tmp_path / "cpu")
    cuda_text, cuda_device = _transcribe("cuda", short_audio_fixture, tmp_path / "gpu")

    assert cpu_device == "cpu"
    assert cuda_device == "cuda"
    assert _words(cpu_text)
    assert _words(cuda_text)
    similarity = difflib.SequenceMatcher(
        None, _words(cpu_text), _words(cuda_text)
    ).ratio()
    assert similarity >= _MIN_TEXT_SIMILARITY, (cpu_text, cuda_text)
    # * The fp32 weights alone are ~0.85 GiB; a few MiB of library workspaces
    # * (cuBLAS/cuFFT) may legitimately outlive the model.
    assert torch.cuda.memory_allocated() <= baseline_allocated + _VRAM_SLACK_BYTES
