# ── NSE ML Training Launcher ──────────────────────────────────────────────
# Double-click this file to run the full training pipeline.
# Prevents Windows from sleeping for the entire duration.

# Prevent sleep using Windows API
$code = @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$type = Add-Type -MemberDefinition $code -Name "WinAPI" -Namespace "Win32" -PassThru
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
$type::SetThreadExecutionState([uint32]"0x80000003") | Out-Null
Write-Host "Sleep prevention ENABLED" -ForegroundColor Green

# Change to project directory
Set-Location "C:\Victor\Project\ml-stock-predictor"

# Activate venv
& ".\.venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Starting NSE ML Pipeline..." -ForegroundColor Cyan
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
Write-Host ""

# Run the pipeline — edit args as needed
python run_nse_local.py --n_trials 0 --gpu

Write-Host ""
Write-Host "Finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow

# Restore sleep
$type::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
Write-Host "Sleep prevention DISABLED" -ForegroundColor Green

Write-Host ""
Write-Host "Press any key to close..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

