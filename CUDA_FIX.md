# ⚡ CUDA Installation Issue - Quick Fix

## Problem
```
CUDA is required for ML processing, but no compatible GPU is available.
```

## Solution

### Automatic (Recommended)
```powershell
.\.venv\Scripts\Activate.ps1
.\build.ps1 -EnsureCUDA
```

### Manual
```powershell
.\.venv\Scripts\Activate.ps1
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Expected: `CUDA: True`

## More Help
See `doc/CUDA_SETUP.md` for detailed troubleshooting and system requirements.

---

**Example Status After Fix:**
- ✅ PyTorch: 2.9.0+cu128 (CUDA-enabled)
- ✅ GPU: NVIDIA GeForce RTX XXXX (your GPU)
- ✅ VRAM: 8 GB+
- ✅ All checks passing
