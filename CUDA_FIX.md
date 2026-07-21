# CUDA Installation Issue - Quick Fix

## Problem
```
CUDA is required for ML processing, but no compatible GPU is available.
```

AskVLM `[ml]` needs **torch 2.10** with a CUDA wheel. Bare `pip install -e ".[ml]"` often leaves CPU `2.10.0`.

## Solution

### Automatic (Recommended)
```powershell
.\.venv\Scripts\Activate.ps1
.\run.ps1 -SkipLaunch -Fast
```

(`build.ps1` / `run.ps1` ensure torch 2.10+CUDA by default; opt out with `-SkipEnsureCUDA`.)

### Manual
```powershell
.\.venv\Scripts\Activate.ps1
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --extra-index-url https://pypi.org/simple
```

Fallback index if cu128 fails: `cu126` with the same `2.10.0+cu126` / `0.25.0+cu126` package names.

### Verify
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.10.0+cu128 True` (or `+cu126`).

## More Help
See `doc/CUDA_SETUP.md` for detailed troubleshooting and system requirements.

---

**Example Status After Fix:**
- ✅ PyTorch: 2.10.0+cu128 (CUDA-enabled)
- ✅ GPU: NVIDIA GeForce RTX XXXX (your GPU)
- ✅ VRAM: 8 GB+
- ✅ All checks passing
