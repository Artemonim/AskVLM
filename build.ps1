# * Local CI thick wrapper (System Builder)
# * Responsibilities:
# * 1. Prepare Python environment (venv, python version)
# * 2. Install system/pip dependencies (requirements.txt, torch, cuda)
# * 3. Delegate to build.py for app-level logic (models, linting, testing)

param(
    [switch]$Help,
    [string]$Tool,
    [string[]]$Path,
    [switch]$Verbose,
    [switch]$Json,
    [switch]$NoFix,
    [switch]$SkipLaunch,
    [switch]$FastLaunch,
    [switch]$Fast,
    [switch]$RecreateVenv,
    # * ML + CUDA torch 2.10 are ensured by default on every build.ps1/run.ps1 entry.
    [switch]$SkipEnsureML,
    [switch]$SkipEnsureCUDA,
    # * Deprecated aliases kept so old docs/scripts keep working (default is already on).
    [switch]$EnsureML,
    [switch]$EnsureCUDA
)

function Show-Help {
    Write-Host "AskVLM build/runner" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\\build.ps1 [-Tool <name>] [-Path <paths...>] [-Verbose] [-Json] [-NoFix] [-SkipLaunch] [-FastLaunch] [-Fast] [-- <build.py args>]"
    Write-Host ""
    Write-Host "Flags:"
    Write-Host "  -Tool <name>      Run only one tool: ruff-format, ruff, compile, mypy, pyright, pytest, bandit, pip-audit"
    Write-Host "  -Path <paths...>  Target files/dirs (default: core, editing, utils, gui, tests)"
    Write-Host "  -Verbose          Verbose output"
    Write-Host "  -Json             JSON output"
    Write-Host "  -NoFix            Disable auto-fix phase"
    Write-Host "  -SkipLaunch       Run checks/tests only; do not launch the app"
    Write-Host "  -FastLaunch       Launch the app only; skip checks/tests"
    Write-Host "  -Fast             Skip slow and heavy ML tests (pytest)"
    Write-Host "  -SkipEnsureML     Skip auto-install of .[ml] (default: ensure ML deps)"
    Write-Host "  -SkipEnsureCUDA   Skip torch 2.10 CUDA repair (default: ensure CUDA stack)"
    Write-Host "  -Help             Show this help"
    Write-Host ""
    Write-Host "By default this script installs missing .[ml] deps and repairs torch to 2.10+CUDA (cu128/cu126)." -ForegroundColor Yellow
    Write-Host "Coverage gates: FAIL <65%, WARN <75% (applied post-pytest)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\\build.ps1 -Tool ruff"
    Write-Host "  .\\build.ps1 -Path core,editing"
    Write-Host "  .\\build.ps1 -FastLaunch"
    Write-Host "  .\\build.ps1 -SkipLaunch"
}

if ($Help) { Show-Help; exit 0 }

$dashdashIndex = $args.IndexOf("--")
$forward = @()
if ($dashdashIndex -ge 0) { $forward = $args[($dashdashIndex + 1)..($args.Length - 1)] }

# * 1. Venv Setup
$activate = Join-Path -Path ".venv" -ChildPath "Scripts/Activate.ps1"
if ($RecreateVenv -and (Test-Path ".venv")) {
    Write-Host "Recreating virtual environment..." -ForegroundColor Yellow
    try { Remove-Item -Recurse -Force ".venv" } catch {}
}
if (-not (Test-Path $activate)) {
    # Prefer Python 3.11 for this project
    $pyok = $false
    try { py -3.11 -V | Out-Null; if ($LASTEXITCODE -eq 0) { $pyok = $true } } catch {}
    if ($pyok) {
        py -3.11 -m venv .venv
    } else {
        if (Test-Path "venv/Scripts/python.exe") { & "venv/Scripts/python.exe" -m venv .venv }
        elseif (Test-Path "venv/ScriptS/python.exe") { & "venv/ScriptS/python.exe" -m venv .venv }
        else { python -m venv .venv }
    }
}
& $activate

# Ensure active venv Python is 3.11
try {
    $ver = & python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
    if ($ver -ne '3.11') {
        Write-Host "Active venv Python is $ver, recreating with Python 3.11..." -ForegroundColor Yellow
        try { Remove-Item -Recurse -Force ".venv" } catch {}
        if (Test-Path "venv/Scripts/python.exe") { & "venv/Scripts/python.exe" -m venv .venv }
        elseif (Test-Path "venv/ScriptS/python.exe") { & "venv/ScriptS/python.exe" -m venv .venv }
        elseif ($pyok) { py -3.11 -m venv .venv }
        else { Write-Error "Python 3.11 not found. Install it or provide venv with 3.11."; exit 1 }
        & $activate
        $ver = & python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
        if ($ver -ne '3.11') { Write-Error "Failed to activate Python 3.11 environment."; exit 1 }
    }
} catch {}

# * 2. Dependency Setup
# Helper: quick silent module existence check
function Test-Module {
    param([string]$Name)
    python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('$Name') else 1)" | Out-Null
    return $LASTEXITCODE -eq 0
}
# Helper: check numpy major version == 1
function Test-NumpyV1 {
    python -c "import sys;import numpy as np;sys.exit(0 if str(np.__version__).split('.')[0]=='1' else 1)" | Out-Null
    return $LASTEXITCODE -eq 0
}
# Helper: check CUDA availability
function Test-CUDA {
    python -c "import sys; import torch; ok = bool(getattr(torch,'cuda',None) and torch.cuda.is_available()); print('[CUDA CHECK] available=' + str(ok)); sys.exit(0 if ok else 1)" | Out-Null
    return $LASTEXITCODE -eq 0
}
# Helper: AskVLM ML stack needs torch 2.10.* with a working CUDA build
function Test-TorchMlStack {
    python -c "import sys; import torch; ver=str(getattr(torch,'__version__','')); cuda_ok=bool(getattr(torch,'cuda',None) and torch.cuda.is_available()); ok=cuda_ok and ver.startswith('2.10.'); print('[TORCH ML CHECK] version=' + ver + ' cuda=' + str(cuda_ok) + ' ok=' + str(ok)); sys.exit(0 if ok else 1)" | Out-Null
    return $LASTEXITCODE -eq 0
}

try {
    python -c "import importlib.util,sys;mods=['ruff','mypy','pytest','pytest_cov','pip_audit','tqdm','huggingface_hub'];sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing dev dependencies..." -ForegroundColor Yellow
        if (Test-Path "requirements-dev.txt") { python -m pip install -r requirements-dev.txt | Out-Null }
        if (Test-Path "requirements.txt") { python -m pip install -r requirements.txt | Out-Null }
        # Ensure tools present
        # * transformers 5 (via [ml] / GigaAM) needs huggingface_hub 1.x — do not pin <0.20.
        try { python -m pip install -q pip-audit pytest-cov tqdm huggingface_hub | Out-Null } catch {}
    }
} catch {}

# * ML + CUDA are on by default; -EnsureML/-EnsureCUDA remain accepted no-ops.
$doEnsureML = -not $SkipEnsureML
$doEnsureCUDA = -not $SkipEnsureCUDA
if ($EnsureML) { $doEnsureML = $true }
if ($EnsureCUDA) { $doEnsureCUDA = $true }

# Ensure ML deps exist (default on)
if ($doEnsureML) {
    try {
        # * whisperx PyPI package is optional (alignment); WhisperXWrapper uses faster-whisper.
        $needMl = (-not (Test-Module numpy)) -or (-not (Test-Module torch)) -or (-not (Test-Module whisper)) -or (-not (Test-Module faster_whisper)) -or (-not (Test-Module pyannote.audio)) -or (-not (Test-Module transformers)) -or (-not (Test-Module sentencepiece)) -or (-not (Test-Module hydra)) -or (-not (Test-Module omegaconf))
        if ($needMl) {
            Write-Host "Installing ML dependencies (extras: ml)..." -ForegroundColor Yellow
            try { python -m pip install -q -U pip wheel setuptools } catch {}
            # * numpy<2 for pyannote compatibility
            try { python -m pip install -q --only-binary=:all: "numpy<2.0" } catch {}
            python -m pip install -q -e .[ml]
        }
        if (-not (Test-NumpyV1)) {
            try { python -m pip install -q --only-binary=:all: "numpy<2.0" } catch {}
        }
    } catch {}
}

# Ensure CUDA-enabled torch 2.10 (Whisper GPU + GigaAM-compatible stack) — default on
# * Reinstall when CUDA is missing OR the installed torch is not 2.10.* (e.g. CPU
# * wheel pulled by bare pip / outdated 2.5.x+cu124 from older EnsureCUDA).
if ($doEnsureCUDA -and -not (Test-TorchMlStack)) {
    Write-Host "Attempting to install CUDA-enabled PyTorch 2.10 wheels..." -ForegroundColor Yellow
    $cudaConfigs = @(
        @{index="https://download.pytorch.org/whl/cu128"; torch="2.10.0+cu128"; vision="0.25.0+cu128"; audio="2.10.0+cu128"},
        @{index="https://download.pytorch.org/whl/cu126"; torch="2.10.0+cu126"; vision="0.25.0+cu126"; audio="2.10.0+cu126"}
    )
    try { python -m pip uninstall -y torch torchvision torchaudio | Out-Null } catch {}
    try { python -m pip cache purge | Out-Null } catch {}
    
    foreach ($cfg in $cudaConfigs) {
        $cudaTag = $cfg.index.Split('/')[-1]
        Write-Host ("Trying PyTorch {0} from: {1}" -f $cudaTag, $cfg.index) -ForegroundColor Yellow
        try {
            python -m pip install --no-cache-dir `
                --index-url $cfg.index `
                --extra-index-url https://pypi.org/simple `
                "torch==$($cfg.torch)" "torchvision==$($cfg.vision)" "torchaudio==$($cfg.audio)" | Out-Null
        } catch {
            Write-Host ("  Failed to install from {0}: {1}" -f $cudaTag, $_.Exception.Message) -ForegroundColor Red
        }
        if (Test-TorchMlStack) {
            Write-Host ("✓ Successfully installed PyTorch 2.10 with {0}" -f $cudaTag) -ForegroundColor Green
            break
        }
        try { python -m pip uninstall -y torch torchvision torchaudio | Out-Null } catch {}
    }
    if (-not (Test-TorchMlStack)) {
        Write-Error "CUDA torch 2.10 is not available via PyTorch after installation attempts (tried cu128, cu126)."
    }
}

# * 3. Handover to Python Build System
$pyArgs = @("build.py")
if ($Tool) { $pyArgs += "--tool"; $pyArgs += $Tool }
if ($Path) { $pyArgs += "--path"; $pyArgs += $Path }
if ($Verbose) { $pyArgs += "--verbose" }
if ($Json) { $pyArgs += "--json" }
if ($NoFix) { $pyArgs += "--no-fix" }
if ($SkipLaunch) { $pyArgs += "--skip-launch" }
if ($FastLaunch) { $pyArgs += "--fast-launch" }
if ($Fast) { $pyArgs += "--fast" }
if ($forward.Count -gt 0) { $pyArgs += $forward }

Write-Host "DEBUG: Running python with args: $($pyArgs -join ' ')" -ForegroundColor DarkGray
python @pyArgs
exit $LASTEXITCODE
