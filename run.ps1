# * Local CI thin wrapper
param(
    [switch]$Help,
    [string]$Tool,
    [string[]]$Path,
    [switch]$Verbose,
    [switch]$Json,
    [switch]$NoFix,
    [switch]$SkipLaunch,
    [switch]$FastLaunch
)

function Show-Help {
    Write-Host "Artemonim's Speech Kit runner" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\run.ps1 [-Tool <name>] [-Path <paths...>] [-Verbose] [-Json] [-NoFix] [-SkipLaunch] [-FastLaunch] [-- <build.py args>]"
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
    Write-Host "Examples:"
    Write-Host "  .\run.ps1 -Tool ruff"
    Write-Host "  .\run.ps1 -Path core,editing"
    Write-Host "  .\run.ps1 -FastLaunch"
    Write-Host "  .\run.ps1 -SkipLaunch"
}

if ($Help) { Show-Help; exit 0 }

$dashdashIndex = $args.IndexOf("--")
$forward = @()
if ($dashdashIndex -ge 0) { $forward = $args[($dashdashIndex + 1)..($args.Length - 1)] }

$activate = Join-Path -Path ".venv" -ChildPath "Scripts/Activate.ps1"
if (-not (Test-Path $activate)) { python -m venv .venv }
& $activate

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


