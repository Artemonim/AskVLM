# * Local CI thick wrapper (with venv + simple help)
param(
    [switch]$Help,
    [string]$Tool,
    [string[]]$Path,
    [switch]$Verbose,
    [switch]$Json,
    [switch]$NoFix
)

function Show-Help {
    Write-Host "Artemonim's Speech Kit build/runner" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\\build.ps1 [-Tool <name>] [-Path <paths...>] [-Verbose] [-Json] [-NoFix] [-- <build.py args>]"
    Write-Host ""
    Write-Host "Flags:"
    Write-Host "  -Tool <name>      Run only one tool: ruff-format, ruff, compile, mypy, pyright, pytest, bandit, pip-audit"
    Write-Host "  -Path <paths...>  Target files/dirs (default: core, editing, utils, gui, tests)"
    Write-Host "  -Verbose          Verbose output"
    Write-Host "  -Json             JSON output"
    Write-Host "  -NoFix            Disable auto-fix phase"
    Write-Host "  -Help             Show this help"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\\build.ps1 -Tool ruff"
    Write-Host "  .\\build.ps1 -Path core,editing"
}

if ($Help) { Show-Help; exit 0 }

$dashdashIndex = $args.IndexOf("--")
$forward = @()
if ($dashdashIndex -ge 0) { $forward = $args[($dashdashIndex + 1)..($args.Length - 1)] }

$activate = Join-Path -Path ".venv" -ChildPath "Scripts/Activate.ps1"
if (-not (Test-Path $activate)) { python -m venv .venv }
& $activate

# * Ensure core dev deps exist
try {
    python -c "import importlib.util,sys;mods=['ruff','mypy','pytest'];sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing dev dependencies..." -ForegroundColor Yellow
        if (Test-Path "requirements-dev.txt") { pip install -r requirements-dev.txt | Out-Null }
        if (Test-Path "requirements.txt") { pip install -r requirements.txt | Out-Null }
    }
} catch {}

$cmd = "python build.py"
# * Add flags from parameters
if ($Tool) { $cmd = "{0} --tool {1}" -f $cmd, $Tool }
if ($Path) { $cmd = "{0} --path {1}" -f $cmd, ([string]::Join(' ', $Path)) }
if ($Verbose) { $cmd += " --verbose" }
if ($Json) { $cmd += " --json" }
if ($NoFix) { $cmd += " --no-fix" }
# * Add forwarded arguments
if ($forward.Count -gt 0) { $cmd = "{0} {1}" -f $cmd, ([string]::Join(' ', $forward)) }
Invoke-Expression $cmd
exit $LASTEXITCODE


