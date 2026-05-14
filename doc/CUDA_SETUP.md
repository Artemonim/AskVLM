# CUDA Setup Guide for AskVLM

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

#### Option 1: CUDA 12.8 (Latest, Recommended for RTX 30/40 series)

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --extra-index-url https://pypi.org/simple
```

#### Option 2: CUDA 12.4

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 `
  --index-url https://download.pytorch.org/whl/cu124 `
  --extra-index-url https://pypi.org/simple
```

#### Option 3: CUDA 12.1

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 `
  --index-url https://download.pytorch.org/whl/cu121 `
  --extra-index-url https://pypi.org/simple
```

#### Option 4: CUDA 11.8 (Older Systems)

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install --no-cache-dir `
  torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 `
  --index-url https://download.pytorch.org/whl/cu118 `
  --extra-index-url https://pypi.org/simple
```

**Note:** CUDA 12.4+ wheels are backward compatible with CUDA 12.8 drivers. If you have CUDA 12.8 installed, using cu124 wheels will work fine, but cu128 provides the best compatibility.

## Verification

After installation, verify CUDA is properly configured:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX XXXX
```

## Troubleshooting

### Error: PyTorch installs CPU version despite using CUDA index

**Symptom:**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Results in `torch.__version__` showing `2.9.0+cpu` instead of `2.9.0+cu124`.

**Root Cause:**
When pip cannot find the exact CUDA wheel or encounters network issues (DNS resolution failures, SSL errors), it falls back to the CPU-only version from the default PyPI index.

**Solution:**
Explicitly specify the CUDA version suffix in the package name:

```powershell
# * Clear any cached wheels first
pip uninstall -y torch torchvision torchaudio
pip cache purge

# * Install with explicit CUDA version suffix
pip install --no-cache-dir `
  torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --extra-index-url https://pypi.org/simple
```

**Why this works:**
- Explicit version with `+cu128` suffix forces pip to look for CUDA-specific wheels only
- `--no-cache-dir` ensures fresh download without using potentially wrong cached wheels
- `--index-url` sets primary index to PyTorch CUDA repository
- `--extra-index-url` allows dependencies to be downloaded from PyPI

**Automated Fix:**
The `build.ps1 -EnsureCUDA` script now automatically uses this method with fallback to multiple CUDA versions (cu128 → cu124 → cu121 → cu118).

### Network Issues with PyTorch Downloads

**Symptoms:**
- `getaddrinfo failed` errors when downloading from `download.pytorch.org`
- `SSLEOFError` or TLS handshake failures
- Downloads work in browser but fail in pip

**Possible Causes:**
1. DNS resolution issues in Python's network stack
2. CloudFront serving different IPv4/IPv6 addresses
3. VPN or network configuration changes
4. Firewall or antivirus blocking pip connections

**Solutions:**

1. **Change DNS servers** (Windows Network Settings):
   - Set DNS to `1.1.1.1` (Cloudflare) or `8.8.8.8` (Google)
   - Run `ipconfig /flushdns` after changing

2. **Manual wheel download** (if network issues persist):
   ```powershell
   # Download .whl files in browser from:
   # https://download.pytorch.org/whl/cu128/torch/
   # https://download.pytorch.org/whl/cu128/torchvision/
   # https://download.pytorch.org/whl/cu128/torchaudio/
   
   # Look for cp311-cp311-win_amd64.whl files with +cu128 suffix
   # Example: torch-2.9.0+cu128-cp311-cp311-win_amd64.whl
   
   # Save to wheels/ directory, then install:
   pip install --no-deps .\wheels\torch-2.9.0+cu128-cp311-cp311-win_amd64.whl
   pip install --no-deps .\wheels\torchvision-0.24.0+cu128-cp311-cp311-win_amd64.whl
   pip install --no-deps .\wheels\torchaudio-2.9.0+cu128-cp311-cp311-win_amd64.whl
   ```

3. **Verify installation** after any method:
   ```powershell
   python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
   ```

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

To check your system's GPU and VRAM:

```powershell
python -c "import torch; props = torch.cuda.get_device_properties(0); print(f'GPU: {props.name}'); print(f'VRAM: {props.total_memory / 1e9:.1f} GB')"
```

## Advanced: System Requirements

| Requirement | Minimum | Recommended | RTX 30/40 series |
|---|---|---|---|
| GPU | GTX 960 (Maxwell) | RTX 2070+ | RTX 30/40 series |
| VRAM | 6 GB | 8-12 GB | 8+ GB |
| CUDA Version | 11.8 | 12.1-12.4 | 12.4+ |
| Driver Version | 450+ | Latest | Check with nvidia-smi |

## References

- Official PyTorch Installation: https://pytorch.org/get-started/locally/
- PyTorch CUDA Support Matrix: https://pytorch.org/get-started/previous-versions/
- NVIDIA Driver Download: https://www.nvidia.com/Download/index.aspx
