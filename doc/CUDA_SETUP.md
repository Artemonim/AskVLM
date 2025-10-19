# CUDA Setup Guide for Artemonim's Speech Kit

## Problem: PyTorch CPU-only Version

When running the application, you might encounter this error:

```
CUDA is required for ML processing, but no compatible GPU is available.
```

This happens when PyTorch is installed with CPU-only support instead of CUDA-enabled wheels.

## Root Cause

PyTorch provides different binary distributions:
- **CPU-only**: Runs on any system but cannot use GPU acceleration
- **CUDA-enabled**: Requires NVIDIA GPU but provides significant speedup

By default, `pip install torch` often installs the CPU-only version, especially if your system environment is not properly configured.

## Solution: Reinstall PyTorch with CUDA Support

### Quick Fix (Automatic)

Use the build script with the `-EnsureCUDA` flag:

```powershell
.\.venv\Scripts\Activate.ps1
.\build.ps1 -EnsureCUDA
```

This will:
1. Detect if CUDA is available
2. Try to install CUDA-enabled PyTorch wheels from multiple repositories (cu124, cu121, cu118)
3. Verify installation
4. Display success/failure status

### Manual Installation

Choose the CUDA version matching your system. To find your CUDA version:

```powershell
nvidia-smi
```

Look for "CUDA Capability Major/Minor version" or "CUDA Version" in the output.

#### Option 1: CUDA 12.1 (Recommended)

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Option 2: CUDA 12.4 (Latest)

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Option 3: CUDA 11.8 (Older Systems)

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verification

After installation, verify CUDA is properly configured:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 3070
```

## Troubleshooting

### Error: "nvidia-smi not found"

This means NVIDIA drivers are not installed or not in PATH.

**Solution:**
1. Download latest NVIDIA driver: https://www.nvidia.com/Download/driverDetails.aspx
2. Install the driver
3. Restart your computer
4. Run `nvidia-smi` again

### Error: "CUDA is available but ML still fails"

This might be due to VRAM limitations. Check available VRAM:

```powershell
python -c "import torch; print('VRAM available:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
```

The application requires:
- Whisper model: 1-4 GB depending on model size
- Diarization: 2-3 GB
- LLM formatter: 2-3 GB for 7B model

If VRAM is insufficient, use smaller models or reduce batch sizes.

### Error: "Mixed CUDA versions"

If you have multiple CUDA versions or toolkits installed, PyTorch might use the wrong one.

**Solution:**
```powershell
pip uninstall torch torchvision torchaudio
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
```

## System Information

- **Your GPU**: NVIDIA GeForce RTX 3070
- **VRAM**: 8 GB
- **Supported Models**: All Whisper models (tiny to large), PyAnnote, 7B LLM

## Advanced: System Requirements

| Requirement | Minimum | Recommended | Your System |
|---|---|---|---|
| GPU | GTX 960 (Maxwell) | RTX 2070+ | RTX 3070 ✅ |
| VRAM | 6 GB | 8-12 GB | 8 GB ✅ |
| CUDA Version | 11.8 | 12.1-12.4 | 12.1+ ✅ |
| Driver Version | 450+ | Latest | Check with nvidia-smi |

## References

- Official PyTorch Installation: https://pytorch.org/get-started/locally/
- PyTorch CUDA Support Matrix: https://pytorch.org/get-started/previous-versions/
- NVIDIA Driver Download: https://www.nvidia.com/Download/index.aspx
