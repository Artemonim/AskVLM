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
    Write-Host "Coverage gates: FAIL <65%, WARN <75% (applied post-pytest)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run.ps1 -Tool ruff"
    Write-Host "  .\run.ps1 -Path core,editing"
    Write-Host "  .\run.ps1 -FastLaunch"
    Write-Host "  .\run.ps1 -SkipLaunch"
}

if ($Help) { Show-Help; exit 0 }

# Forward known arguments to build.ps1
$buildParams = @{}
if ($Tool) { $buildParams['Tool'] = $Tool }
if ($Path) { $buildParams['Path'] = $Path }
if ($Verbose) { $buildParams['Verbose'] = $true }
if ($Json) { $buildParams['Json'] = $true }
if ($NoFix) { $buildParams['NoFix'] = $true }
if ($SkipLaunch) { $buildParams['SkipLaunch'] = $true }
if ($FastLaunch) { $buildParams['FastLaunch'] = $true }

# Handle remaining arguments (forwarded via --)
$dashdashIndex = $args.IndexOf("--")
if ($dashdashIndex -ge 0) {
    $forward = $args[($dashdashIndex + 1)..($args.Length - 1)]
    # Pass explicit '--' so build.ps1 detects them correctly
    & ./build.ps1 @buildParams -- $forward
} else {
    & ./build.ps1 @buildParams
}
exit $LASTEXITCODE
