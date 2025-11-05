from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for GPU variants"
)
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.gpu
def test_benchmark_two_small_variants_cli_real(tmp_path: Path) -> None:
    """Run real benchmark CLI on fixtures for two small variants and verify progression.

    This uses the CLI in a subprocess to mimic the real console run. It processes
    both unverified (videos) and verified (audio+ass) to reproduce the silent exit.
    We run two variants:
    - 2x small GPU [GPU int8]
    - 2x small GPU + 1x small CPU [GPU int8 | CPU int8]
    """
    fixtures = Path("tests/fixtures").resolve()
    assert fixtures.exists(), "Fixtures directory missing"

    out_dir = tmp_path / "bench_out"

    variants_val = (
        "2x small GPU [GPU int8], 2x small GPU + 1x small CPU [GPU int8 | CPU int8]"
    )

    cmd = [
        sys.executable,
        "tools/benchmark_stt.py",
        "--unverified",
        str(fixtures),
        "--verified",
        str(fixtures),
        "--output",
        str(out_dir),
        "--variants",
        variants_val,
        "--max-unverified",
        "2",
        "--max-verified",
        "2",
        "--clear",
    ]

    proc = subprocess.run(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path.cwd()),
        check=False,
    )

    assert proc.returncode == 0, f"CLI failed:\n{proc.stdout}"

    # Check markers
    rdir = out_dir / ".resume"
    assert rdir.exists(), f"Missing resume dir, stdout=\n{proc.stdout}"
    done = {p.stem for p in rdir.glob("*.done")}
    assert "2x_small_GPU_GPU_int8" in done, (
        f"First variant missing .done: {done}\n{proc.stdout}"
    )
    assert "2x_small_GPU___1x_small_CPU_GPU_int8___CPU_int8" in done, (
        f"Second variant missing .done: {done}\n{proc.stdout}"
    )

    # Check stdout progression
    out_text = proc.stdout
    assert "Run [1/2]: 2x small GPU [GPU int8]" in out_text, out_text
    assert "Variant completed: 2x small GPU [GPU int8]" in out_text, out_text
    assert "Run [2/2]: 2x small GPU + 1x small CPU [GPU int8 / CPU int8]" in out_text, (
        out_text
    )
    assert (
        "Variant completed: 2x small GPU + 1x small CPU [GPU int8 / CPU int8]"
        in out_text
    ), out_text
