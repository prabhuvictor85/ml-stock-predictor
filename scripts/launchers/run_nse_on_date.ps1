# ─────────────────────────────────────────────────────────────────────────────
#  run_nse_on_date.ps1 — Convenience wrapper to run NSE TradingView for a
#  specific historical date (backtest / what-if mode).
#
#  Usage:
#    .\scripts\launchers\run_nse_on_date.ps1 -Date 2024-03-01
#    .\scripts\launchers\run_nse_on_date.ps1 -Date 2024-03-01 -SkipTrain
#    .\scripts\launchers\run_nse_on_date.ps1 -Date 2024-03-01 -Mode reversal
#    .\scripts\launchers\run_nse_on_date.ps1 -Date 2024-03-01 -Folds 8 -Trials 25
# ─────────────────────────────────────────────────────────────────────────────

param(
    [Parameter(Mandatory=$true)]
    [string]$Date,

    [ValidateSet("all", "momentum", "reversal", "legacy")]
    [string]$Mode = "all",

    [int]$Folds = 5,
    [int]$Trials = 15,
    [switch]$SkipTrain
)

# Validate date format
if ($Date -notmatch '^\d{4}-\d{2}-\d{2}$') {
    Write-Host "ERROR: -Date must be YYYY-MM-DD format. Got: $Date" -ForegroundColor Red
    exit 1
}

$ScriptRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $ScriptRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $ScriptRoot "run_nse_tradingv_local.py"

if (-not (Test-Path $Python)) {
    Write-Host "ERROR: Python venv not found at $Python" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $Runner)) {
    Write-Host "ERROR: Runner not found at $Runner" -ForegroundColor Red
    exit 1
}

# Build arg list
$args = @(
    "-u", $Runner,
    "--mode",       $Mode,
    "--train_end",  $Date,
    "--n_folds",    $Folds,
    "--n_trials",   $Trials
)
if ($SkipTrain) { $args += "--skip_train" }

Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  NSE TradingView — as of $Date" -ForegroundColor Cyan
Write-Host "  Mode: $Mode | Folds: $Folds | Trials: $Trials | SkipTrain: $SkipTrain" -ForegroundColor Cyan
Write-Host "  Command: python $($args -join ' ')" -ForegroundColor DarkGray
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Set-Location $ScriptRoot
& $Python @args
exit $LASTEXITCODE
