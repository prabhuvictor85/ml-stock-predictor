"""
FeatureDriftMonitor — PSI-based feature drift detection (§10.3).

PSI formula: PSI = Σ (actual% − expected%) × ln(actual% / expected%)
Using 10 equal-frequency bins computed at training time.

Alert threshold  : PSI > cfg.psi_alert_threshold  → log WARNING
Retrain trigger  : PSI > cfg.psi_retrain_threshold on > 20% of features → trigger RetrainingScheduler
Output           : monitoring/feature_drift.parquet
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.config.base import MarketConfig
from pipeline.utils.logging import get_logger

log = get_logger(__name__)

MONITORING_DIR = Path("monitoring")
N_PSI_BINS = 10
RETRAIN_FEATURE_FRACTION = 0.20   # 20% of features breaching retrain threshold triggers retrain


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: np.ndarray,
) -> float:
    """
    Compute PSI between expected and actual distributions.

    Parameters
    ----------
    expected : training-distribution array
    actual   : recent-window array
    bins     : bin edges from training distribution (equal-frequency)

    Returns
    -------
    PSI value (scalar)
    """
    def _bin_counts(arr: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr, bins=bin_edges)
        # Avoid zero counts
        counts = np.where(counts == 0, 1, counts)
        return counts / counts.sum()

    exp_pct = _bin_counts(expected, bins)
    act_pct = _bin_counts(actual, bins)

    # Clip to avoid log(0)
    exp_pct = np.clip(exp_pct, 1e-10, None)
    act_pct = np.clip(act_pct, 1e-10, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


class FeatureDriftMonitor:
    """
    Monitors feature distributions weekly.

    Parameters
    ----------
    cfg           : MarketConfig
    feature_cols  : list of feature column names to monitor
    """

    def __init__(self, cfg: MarketConfig, feature_cols: List[str]) -> None:
        self.cfg = cfg
        self.feature_cols = feature_cols
        self._training_bins: Dict[str, np.ndarray] = {}
        self._training_data: Dict[str, np.ndarray] = {}
        self._drift_records: List[dict] = []

    def fit_baseline(self, train_panel: pd.DataFrame,
                     baseline_months: int = 12) -> None:
        """
        Compute training-distribution bin edges (equal-frequency, 10 bins)
        from the RECENT tail of the training panel.

        Using the full 14-year history as baseline causes permanent PSI alerts
        during any single market regime (bull/bear) because the recent window
        always looks different from the long-run average.  Using only the most
        recent `baseline_months` months keeps the baseline representative of the
        current regime, so PSI only fires when that regime genuinely shifts.

        Parameters
        ----------
        train_panel     : full training panel (used for bin edges on all data)
        baseline_months : how many months of the most recent data to use as the
                          reference distribution (default 12 = last 1 year).
        """
        # PSI with 10 bins needs at most ~10k reference points.
        _MAX_DRIFT_SAMPLES = 10_000
        _rng = np.random.default_rng(42)

        # ── Restrict baseline to the most recent `baseline_months` of data ──
        dates = train_panel.index.get_level_values("date")
        max_date = dates.max()
        cutoff = max_date - pd.DateOffset(months=baseline_months)
        recent_panel = train_panel[dates >= cutoff]
        n_recent = len(recent_panel)
        log.info(f"Drift baseline: using {n_recent:,} rows from last {baseline_months} months "
                 f"({cutoff.date()} → {max_date.date()}) out of {len(train_panel):,} total.")

        if n_recent < 500:
            # Fallback: not enough recent data — use full panel
            log.warning("Drift baseline fallback: fewer than 500 recent rows, using full panel.")
            recent_panel = train_panel

        for feat in self.feature_cols:
            if feat not in recent_panel.columns:
                continue
            values = recent_panel[feat].dropna().values
            if len(values) < 20:
                continue
            # Equal-frequency bins (computed on full distribution for accuracy)
            percentiles = np.linspace(0, 100, N_PSI_BINS + 1)
            bin_edges = np.percentile(values, percentiles)
            # Ensure monotonically increasing bins
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 3:
                continue
            self._training_bins[feat] = bin_edges
            # Store a capped sample for PSI reference — 10k is sufficient for 10-bin PSI
            if len(values) > _MAX_DRIFT_SAMPLES:
                values = _rng.choice(values, size=_MAX_DRIFT_SAMPLES, replace=False)
            self._training_data[feat] = values
        log.info(f"Drift monitor baseline fitted for {len(self._training_bins)} features.")

    def compute_weekly_drift(
        self,
        current_panel: pd.DataFrame,
        reference_date: pd.Timestamp,
        lookback_weeks: int = 4,
    ) -> pd.DataFrame:
        """
        Compute PSI for the most recent `lookback_weeks`-week window.

        Parameters
        ----------
        current_panel  : panel slice for recent window (or full panel — we'll filter)
        reference_date : date to anchor the window end
        lookback_weeks : number of weeks to look back

        Returns
        -------
        DataFrame with columns: date, feature, psi, alert, retrain_flag
        """
        window_start = reference_date - pd.Timedelta(weeks=lookback_weeks)
        dates = current_panel.index.get_level_values("date")
        window_panel = current_panel[
            (dates >= window_start) & (dates <= reference_date)
        ]

        rows = []
        n_retrain = 0
        for feat in self.feature_cols:
            if feat not in window_panel.columns or feat not in self._training_bins:
                continue
            actual_vals = window_panel[feat].dropna().values
            if len(actual_vals) < 10:
                continue

            psi = compute_psi(
                self._training_data[feat],
                actual_vals,
                self._training_bins[feat],
            )
            alert = psi > self.cfg.psi_alert_threshold
            retrain_flag = psi > self.cfg.psi_retrain_threshold

            if alert:
                log.warning(f"Feature drift ALERT: feature='{feat}' PSI={psi:.4f} "
                             f"(threshold={self.cfg.psi_alert_threshold})")
            if retrain_flag:
                n_retrain += 1

            rows.append({
                "date": reference_date,
                "feature": feat,
                "psi": round(psi, 6),
                "alert": alert,
                "retrain_flag": retrain_flag,
            })

        df = pd.DataFrame(rows)
        self._drift_records.append(df)

        # Check global retrain trigger
        if len(rows) > 0:
            frac = n_retrain / len(rows)
            if frac > RETRAIN_FEATURE_FRACTION:
                log.warning(
                    f"RETRAIN TRIGGER: {frac:.1%} of features exceed PSI retrain threshold "
                    f"{self.cfg.psi_retrain_threshold}. Queuing retrain."
                )

        return df

    def save(self, output_dir: Path = MONITORING_DIR) -> None:
        """Append drift records to monitoring/feature_drift.parquet."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "feature_drift.parquet"

        if not self._drift_records:
            return

        new_df = pd.concat(self._drift_records, ignore_index=True)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
                subset=["date", "feature"], keep="last"
            )
            # keep="last" so a post-retrain PSI computation (against the fresh baseline)
            # overwrites the pre-retrain value for the same date.  Without this, the high
            # PSI from the inference pass is kept and the next step re-triggers a retrain.
        else:
            combined = new_df

        combined.to_parquet(path, index=False)
        log.info(f"Drift records saved to {path}")

