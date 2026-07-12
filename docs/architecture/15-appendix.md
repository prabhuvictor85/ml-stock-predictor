[← Back to index](README.md)

# Appendix

### Configuration Files
- [`paths.yaml`](../../paths.yaml) — centralized filesystem paths, overridable via `ML_DATA_ROOT` / `ML_PROJECT_ROOT` / `ML_ARTEFACTS_ROOT` and per-list/per-source env vars.
- `pipeline/config/{base,sp500,nse,nasdaq}.py` — `MarketConfig` and market presets.
- [`signal_weights.yaml`](../../signal_weights.yaml) — signal blend weights.

### Model Parameters (representative defaults; see `optuna_study_meta.json` per trained artifact for the actual tuned values)

| Parameter | Default / Range | Tuned by |
|---|---|---|
| `objective` | `lambdarank` | fixed |
| `metric` / `ndcg_eval_at` | `ndcg` @ `[10]` | fixed |
| `feature_top_K` | Optuna-tuned | Optuna |
| `num_leaves`, `min_data_in_leaf`, `lambda_l1/l2`, `feature_fraction`, `bagging_fraction` | Optuna-tuned | Optuna |
| `n_trials` | 40 (lockbox-scale) | run config |
| `n_folds` | auto (12 typical fenced range) or explicit override | run config |
| `random_seed` | 42 | `MarketConfig` |
| `SSZ_VETO_THRESHOLD` | 0.6 | calibrated, self-monitored (`pipeline/gating.py`) |
| `ICT_BEAR_VETO_THRESHOLD` | 0.4 | calibrated, self-monitored (`pipeline/gating.py`) |
| `psi_alert_threshold` / `psi_retrain_threshold` | 0.20 / 0.25 | `MarketConfig` |
| `max_sector_weight` / `max_single_stock_weight` / `max_portfolio_beta` | 0.40 / 0.15 / 1.3 | `MarketConfig` |
| `profit_target_pct` / `stop_loss_pct` | 0.08 / 0.04 | `MarketConfig` |

### Example Payloads
See [Functional Design § API Contracts](04-functional-design.md#api-contracts) for sample watchlist and explanation JSON.

### Reference Links (in-repo)
- [PROTOCOL.md](../../PROTOCOL.md) — lockbox validation protocol, the researcher-degrees-of-freedom ledger, and pass/fail criteria.
- [CODE_EXPLANATION.md](../CODE_EXPLANATION.md) — module-by-module reference.
- [CV_EXPLANATION.md](../CV_EXPLANATION.md), [FEATURE_REFERENCE.md](../FEATURE_REFERENCE.md).
- [pipeline/gating.py](../../pipeline/gating.py) — quality gate source and calibration provenance.

### Mathematical Formulae (selected)

**LambdaRank gradient (conceptual):** for a pair of documents (stocks) $i, j$ with $i$ ranked correctly above $j$ in the label but the model scores them $s_i < s_j$, the pairwise loss gradient is scaled by $|\Delta \text{NDCG}_{ij}|$ — the change in NDCG that would result from swapping their positions — so pairs whose ordering matters most for the final list (i.e., near the top) get the largest gradient signal.

**Optuna objective:**
$$
\text{score} = \overline{\text{NDCG@10}} - 0.5 \cdot \sigma(\text{NDCG@10})
$$
across CV folds — a lower-confidence-bound preference for consistency over peak performance.

**PSI (Population Stability Index)** for a feature with training-baseline bins and current-period bins:
$$
\text{PSI} = \sum_{k} (\text{Actual}_k - \text{Expected}_k) \cdot \ln\left(\frac{\text{Actual}_k}{\text{Expected}_k}\right)
$$

**Camarilla H5 level (pivot family, per *Secrets of a Pivot Boss*):**
$$
H5 = \frac{H}{L} \cdot C
$$

---

**Previous:** [← 14 · Glossary](14-glossary.md) &nbsp;|&nbsp; **Back to:** [Index](README.md)

*End of Living Architecture Document. Update the relevant file in place as the system evolves — do not fork a "v2" copy; git history is the version history.*
