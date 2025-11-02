# * Local CI thick wrapper (with venv + simple help)
param(
    [switch]$Help,
    [string]$Tool,
    [string[]]$Path,
    [switch]$Verbose,
    [switch]$Json,
    [switch]$NoFix,
    [switch]$SkipLaunch,
    [switch]$FastLaunch,
    [switch]$RecreateVenv,
    [switch]$EnsureML,
    [switch]$EnsureCUDA
)

function Show-Help {
    Write-Host "Artemonim's Speech Kit build/runner" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\\build.ps1 [-Tool <name>] [-Path <paths...>] [-Verbose] [-Json] [-NoFix] [-SkipLaunch] [-FastLaunch] [-- <build.py args>]"
    Write-Host ""
    Write-Host "Flags:"
    Write-Host "  -Tool <name>      Run only one tool: ruff-format, ruff, compile, mypy, pyright, pytest, bandit, pip-audit"
    Write-Host "  -Path <paths...>  Target files/dirs (default: core, editing, utils, gui, tests)"
    Write-Host "  -Verbose          Verbose output"
    Write-Host "  -Json             JSON output"
    Write-Host "  -NoFix            Disable auto-fix phase"
    Write-Host "  -SkipLaunch       Run checks/tests only; do not launch the app"
    Write-Host "  -FastLaunch       Launch the app only; skip checks/tests"
    Write-Host "  -Help             Show this help"
    Write-Host ""
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

# * Ensure core dev deps exist
try {
    python -c "import importlib.util,sys;mods=['ruff','mypy','pytest','pytest_cov','pip_audit'];sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing dev dependencies..." -ForegroundColor Yellow
        if (Test-Path "requirements-dev.txt") { python -m pip install -r requirements-dev.txt | Out-Null }
        if (Test-Path "requirements.txt") { python -m pip install -r requirements.txt | Out-Null }
        # Ensure pip-audit present if not via requirements
        try { python -m pip install -q pip-audit | Out-Null } catch {}
        # Ensure pytest-cov present if not via requirements
        try { python -m pip install -q pytest-cov | Out-Null } catch {}
    }
} catch {}

# * Helper: quick silent module existence check
function Test-Module {
    param([string]$Name)
    python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('$Name') else 1)" | Out-Null
    return $LASTEXITCODE -eq 0
}

# * Helper: check numpy major version == 1
function Test-NumpyV1 {
    python -c "import sys;import numpy as np;sys.exit(0 if str(np.__version__).split('.')[0]=='1' else 1)" | Out-Null
    return $LASTEXITCODE -eq 0
}

# * Helper: check CUDA availability
function Test-CUDA {
    python -c "import sys; import torch; ok = bool(getattr(torch,'cuda',None) and torch.cuda.is_available()); print('[CUDA CHECK] available=' + str(ok)); sys.exit(0 if ok else 1)" | Out-Null
    return $LASTEXITCODE -eq 0
}

# ! Function to check and fix CUDA support
function Check-CUDA {
    Write-Host "🔍 Checking CUDA availability..." -ForegroundColor Cyan
    
    $cudaCheck = & python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>$null
    
    if ($cudaCheck -eq "CUDA") {
        Write-Host "✅ CUDA is available in PyTorch" -ForegroundColor Green
        return $true
    }
    else {
        Write-Host "❌ PyTorch CPU-only version detected" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "ℹ️  Your system has CUDA, but PyTorch CPU-only version is installed." -ForegroundColor Yellow
        Write-Host "To fix this, run:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Green
        Write-Host ""
        Write-Host "Or for other CUDA versions, visit: https://pytorch.org/get-started/locally/" -ForegroundColor Cyan
        Write-Host ""
        return $false
    }
}

# * Ensure ML deps exist (only when requested)
if ($EnsureML) {
    try {
        $needMl = (-not (Test-Module numpy)) -or (-not (Test-Module torch)) -or (-not (Test-Module whisper)) -or (-not (Test-Module faster_whisper)) -or (-not (Test-Module whisperx)) -or (-not (Test-Module pyannote.audio))
        if ($needMl) {
            Write-Host "Installing ML dependencies (extras: ml)..." -ForegroundColor Yellow
            try { python -m pip install -q -U pip wheel setuptools } catch {}
            # * numpy<2 for pyannote compatibility (skip if unavailable)
            try { python -m pip install -q --only-binary=:all: "numpy<2.0" } catch {}
            python -m pip install -q -e .[ml]
        }
        # * Enforce numpy<2 even if extras upgraded it
        if (-not (Test-NumpyV1)) {
            try { python -m pip install -q --only-binary=:all: "numpy<2.0" } catch {}
        }
    } catch {}
}

# * Ensure CUDA-enabled torch; attempt known CUDA indices if unavailable
if ($EnsureCUDA -and -not (Test-CUDA)) {
    Write-Host "Attempting to install CUDA-enabled PyTorch wheels..." -ForegroundColor Yellow
    
    # * Define CUDA versions to try with their explicit version specifiers
    # ! IMPORTANT: Must specify version with +cuXXX suffix to avoid CPU fallback
    $cudaConfigs = @(
        @{index="https://download.pytorch.org/whl/cu128"; torch="2.9.0+cu128"; vision="0.24.0+cu128"; audio="2.9.0+cu128"},
        @{index="https://download.pytorch.org/whl/cu124"; torch="2.6.0+cu124"; vision="0.21.0+cu124"; audio="2.6.0+cu124"},
        @{index="https://download.pytorch.org/whl/cu121"; torch="2.5.1+cu121"; vision="0.20.1+cu121"; audio="2.5.1+cu121"},
        @{index="https://download.pytorch.org/whl/cu118"; torch="2.5.1+cu118"; vision="0.20.1+cu118"; audio="2.5.1+cu118"}
    )
    
    # * Remove CPU-only installs and clear cache to avoid reuse of cached wheels
    try { python -m pip uninstall -y torch torchvision torchaudio | Out-Null } catch {}
    try { python -m pip cache purge | Out-Null } catch {}
    
    foreach ($cfg in $cudaConfigs) {
        $cudaTag = $cfg.index.Split('/')[-1]
        Write-Host ("Trying PyTorch {0} from: {1}" -f $cudaTag, $cfg.index) -ForegroundColor Yellow
        try {
            # * Explicitly specify version with CUDA suffix to prevent CPU fallback
            python -m pip install --no-cache-dir `
                --index-url $cfg.index `
                --extra-index-url https://pypi.org/simple `
                "torch==$($cfg.torch)" "torchvision==$($cfg.vision)" "torchaudio==$($cfg.audio)" | Out-Null
        } catch {
            Write-Host ("  Failed to install from {0}: {1}" -f $cudaTag, $_.Exception.Message) -ForegroundColor Red
        }
        if (Test-CUDA) {
            Write-Host ("✓ Successfully installed PyTorch with {0}" -f $cudaTag) -ForegroundColor Green
            break
        }
        # If still CPU, try uninstall again before next index
        try { python -m pip uninstall -y torch torchvision torchaudio | Out-Null } catch {}
    }
    if (-not (Test-CUDA)) {
        Write-Error "CUDA is not available via PyTorch after installation attempts."
    }
}

$cmd = "python build.py"
# * Add flags from parameters
if ($Tool) { $cmd = "{0} --tool {1}" -f $cmd, $Tool }
if ($Path) { $cmd = "{0} --path {1}" -f $cmd, ([string]::Join(' ', $Path)) }
if ($Verbose) { $cmd += " --verbose" }
if ($Json) { $cmd += " --json" }
if ($NoFix) { $cmd += " --no-fix" }
# * Launch control flags
if ($SkipLaunch) { $cmd += " --skip-launch" }
if ($FastLaunch) { $cmd += " --fast-launch" }
# * Add forwarded arguments
if ($forward.Count -gt 0) { $cmd = "{0} {1}" -f $cmd, ([string]::Join(' ', $forward)) }
Invoke-Expression $cmd
exit $LASTEXITCODE


