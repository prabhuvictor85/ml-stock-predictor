$process = Get-Process python | Where-Object { $_.MainWindowTitle -match "run_sp500_local" } -ErrorAction SilentlyContinue
# Get the process with highest CPU time, if it's the large running one
$p = Get-Process python | Sort-Object CPU -Descending | Select-Object -First 1

if ($p) {
    Write-Host "Waiting for panel builder (PID: $($p.Id)) to finish..."
    Wait-Process -Id $p.Id
    Write-Host "Panel builder finished."
}

Write-Host "--- Running Exp 0 Baseline ---"
.\.venv\Scripts\python.exe scripts/exp0_baseline.py | Tee-Object -FilePath phase0_results.txt

Write-Host "--- Running Exp 501 VIF Pruning ---"
.\.venv\Scripts\python.exe scripts/exp501_vif_pruning.py | Tee-Object -FilePath exp501_log.txt

Write-Host "--- Running Exp 502 SHAP Stability ---"
.\.venv\Scripts\python.exe scripts/exp502_shap_stability.py | Tee-Object -FilePath exp502_log.txt

Write-Host "--- Running Exp 503 Ablation Test ---"
.\.venv\Scripts\python.exe scripts/exp503_ablation_test.py | Tee-Object -FilePath exp503_log.txt

Write-Host "All Phase 5 experiments completed."

