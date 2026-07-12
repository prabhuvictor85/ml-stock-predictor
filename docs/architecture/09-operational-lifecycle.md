[← Back to index](README.md)

# Operational Lifecycle

**Deployment:** manual/scripted invocation of runner scripts locally or on Hetzner; no continuous deployment of a "live service" — this is a batch job invoked on a schedule (weekly for inference, ad hoc for training/backtesting/lockbox runs).

**Monitoring:** `FeatureDriftMonitor` (PSI per feature), gate calibration self-check (structural veto rate), lockbox/walk-forward validator as the periodic "is the strategy still working" check.

**Logging:** `pipeline/utils/logging.py` — JSON-like structured logs to stdout via `get_logger(name)`.

**Alerting:** printed console alarms today (gate inactive, gate miscalibration >15%, PSI retrain trigger) — not yet wired to an external paging system (Slack/PagerDuty); a documented future-roadmap item if this moves toward more automated operation.

**Model Drift / Concept Drift / Data Drift:**
- **Data drift:** detected via PSI on `features_*` distributions vs. training baseline.
- **Concept drift** (the *relationship* between features and outcomes changing, not just feature distributions): the primary defense is the periodic lockbox/walk-forward re-validation, not a real-time concept-drift detector — an explicit design choice given the weekly (not streaming) cadence.
- **Model drift:** implicitly monitored via the same PSI mechanism plus manual review of watchlist quality and gate veto rates.

**Retraining Strategy:** `RetrainingScheduler` queues one pending job per market when the drift monitor's 20%-features-breach threshold fires; retraining itself re-runs the full training pipeline (fresh CV, HPO, feature selection) rather than incremental/online updates — appropriate given the weekly cadence and the need for every retrain to be independently walk-forward validated.

**Rollback:** artifacts are versioned implicitly by directory (`artefacts/{market}/`) and git commit; rolling back means restoring the prior artifact directory and recording the corresponding commit hash — no automated rollback tooling exists today (future-roadmap item).

**Disaster Recovery:** panel/artifact reproducibility depends on (a) the source CSV data being intact/backed up, and (b) the git history for code — no automated backup/restore procedure is documented; a gap worth closing before this moves toward production trading reliance.

**Versioning:** git commit hash + `optuna_study_meta.json` + `selected_features.txt` collectively define a reproducible "recipe version"; PROTOCOL.md's changelog (§3.1) is the closest thing to a formal recipe-version changelog today.

**A/B Testing / Canary Releases:** not applicable in the traditional live-traffic sense (no serving infrastructure); the closest analog is the **lockbox protocol** itself — a fenced, pre-registered comparison of a frozen recipe against a holdout period, and the tuning-era CV gates (e.g., MODEL_D pivot-only pre-registration in PROTOCOL.md §3.1) that must pass *before* a new feature family is even eligible for a lockbox look.

---

**Previous:** [← 08 · System Design](08-system-design.md) &nbsp;|&nbsp; **Next:** [10 · Explainability →](10-explainability.md)
