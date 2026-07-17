#!/bin/bash
echo "--- Running Target Builder for Alpha ---"
python run_us_local_alpha.py --mode momentum --stop_after_targets
echo "--- Running Exp 0 Baseline ---"
python scripts/exp0_baseline.py alpha | tee phase0_results_alpha.txt
echo "--- Running Exp 501 VIF Pruning ---"
python scripts/exp501_vif_pruning.py alpha | tee exp501_log_alpha.txt
echo "--- Running Exp 502 SHAP Stability ---"
python scripts/exp502_shap_stability.py alpha | tee exp502_log_alpha.txt
echo "--- Running Exp 503 Ablation Test ---"
python scripts/exp503_ablation_test.py alpha | tee exp503_log_alpha.txt
echo "All Phase 5 Alpha experiments completed."

