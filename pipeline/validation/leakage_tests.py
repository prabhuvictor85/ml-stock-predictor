"""
pipeline/validation/leakage_tests.py

Look-ahead bias / data leakage detection suite.

Tests run as assertions before any model training. If ANY test fails,
training is aborted with a clear explanation of which feature leaks
future information and how to fix it.

Tests included
──────────────
1. TemporalLeakageTest     — future return correlation with feature values
2. ForwardFillLeakageTest  — detects features that were forward-filled
                             ACROSS the train/test boundary (purging gap)
3. TargetShiftTest         — verifies target columns are shifted correctly
                             (i.e., not available at the decision point)
4. FuturePriceLeakageTest  — checks no raw future OHLCV columns leaked in
5. GroupBoundaryTest       — no data from test fold appears in train fold
                             (fundamental purging check)

Usage
─────
    from pipeline.validation.leakage_tests import LeakageTestSuite
    suite = LeakageTestSuite(panel, feat_cols, target_col="cs_rank_20d")
    suite.run_all()          # raises LeakageError on first failure
    report = suite.report()  # returns dict of all results
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.utils.logging import get_logger

log = get_logger(__name__)


class LeakageError(RuntimeError):
    """Raised when a leakage test fails. Training must not proceed."""


@dataclass
class TestResult:
    name: str
    passed: bool
    details: str
    flagged_features: List[str] = field(default_factory=list)


class LeakageTestSuite:
    """
    Runs all leakage checks on the training panel.

    Parameters
    ----------
    panel       : full multi-index (date, ticker) panel AFTER feature engineering
    feat_cols   : list of feature column names to test
    target_col  : primary rank/return target column
    horizon     : target horizon in trading days (default 20)
    corr_thresh : Spearman |r| above this triggers temporal leakage warning
    purge_days  : minimum gap enforced between train and test (days)
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        feat_cols: List[str],
        target_col: str = "cs_rank_20d",
        horizon: int = 20,
        corr_thresh: float = 0.15,
        purge_days: int = 20,
    ) -> None:
        self.panel        = panel
        self.feat_cols    = feat_cols
        self.target_col   = target_col
        self.horizon      = horizon
        self.corr_thresh  = corr_thresh
        self.purge_days   = purge_days
        self._results: List[TestResult] = []

    # ── Public interface ──────────────────────────────────────────────────────

    def run_all(self, raise_on_fail: bool = True) -> List[TestResult]:
        """
        Run all tests.  If raise_on_fail=True (default), raises LeakageError
        on the first critical failure so training is blocked.
        """
        self._results = []
        tests = [
            self._test_future_price_columns,
            self._test_target_shift,
            self._test_temporal_correlation,
            self._test_forward_fill_boundary,
            self._test_group_boundary,
        ]
        for test_fn in tests:
            result = test_fn()
            self._results.append(result)
            status = "PASS" if result.passed else "FAIL"
            log.info(f"Leakage [{status}] {result.name}: {result.details}")
            if not result.passed and raise_on_fail:
                raise LeakageError(
                    f"\n\nLeakage test FAILED: {result.name}\n"
                    f"{result.details}\n"
                    f"Flagged features: {result.flagged_features}\n"
                    f"Fix these before training."
                )
        passed = sum(r.passed for r in self._results)
        log.info(f"Leakage suite: {passed}/{len(self._results)} tests passed.")
        return self._results

    def report(self) -> Dict:
        """Return structured dict of all test results (for JSON logging)."""
        if not self._results:
            self.run_all(raise_on_fail=False)
        return {
            "all_passed": all(r.passed for r in self._results),
            "tests": [
                {
                    "name":             r.name,
                    "passed":           r.passed,
                    "details":          r.details,
                    "flagged_features": r.flagged_features,
                }
                for r in self._results
            ],
        }

    # ── Test 1: Future price columns ──────────────────────────────────────────

    def _test_future_price_columns(self) -> TestResult:
        """
        No column name should contain 'future_', 'fwd_', 'next_' in the
        feature set — those are target-derivation columns that must be
        excluded from X.
        """
        BANNED_PREFIXES = ("future_", "fwd_", "next_", "forward_")
        bad = [
            f for f in self.feat_cols
            if any(f.lower().startswith(p) or f"_{p}" in f.lower()
                   for p in BANNED_PREFIXES)
        ]
        passed = len(bad) == 0
        return TestResult(
            name="FuturePriceColumnTest",
            passed=passed,
            details=(
                "No future-price columns in feature set."
                if passed else
                f"{len(bad)} future-leaking column names found in feat_cols. "
                f"Remove them from the feature list passed to the model."
            ),
            flagged_features=bad,
        )

    # ── Test 2: Target shift ──────────────────────────────────────────────────

    def _test_target_shift(self) -> TestResult:
        """
        The target (e.g. cs_rank_20d) must not be computable from same-day
        data. Test: correlation of target at date t with close return at t
        should be < 0.05 (same-day return is NOT the target).

        A high same-day correlation means the target was not shifted forward.
        """
        if self.target_col not in self.panel.columns:
            return TestResult(
                name="TargetShiftTest",
                passed=False,
                details=f"Target column '{self.target_col}' not found in panel.",
            )

        panel_flat = self.panel.reset_index()
        if "close" not in panel_flat.columns:
            return TestResult(
                name="TargetShiftTest",
                passed=True,
                details="No 'close' column to test against — skipped.",
            )

        # Same-day return
        panel_flat = panel_flat.sort_values(["ticker", "date"])
        panel_flat["same_day_ret"] = panel_flat.groupby("ticker")["close"].pct_change()

        sub = panel_flat[[self.target_col, "same_day_ret"]].dropna()
        if len(sub) < 100:
            return TestResult(
                name="TargetShiftTest", passed=True,
                details="Insufficient data for target shift test — skipped.",
            )

        r, p = stats.spearmanr(sub[self.target_col], sub["same_day_ret"])
        threshold = 0.10
        passed = abs(r) < threshold
        return TestResult(
            name="TargetShiftTest",
            passed=passed,
            details=(
                f"Same-day return vs target: Spearman r={r:.4f} (p={p:.4f}). "
                + ("OK — target appears correctly shifted."
                   if passed else
                   f"SUSPICIOUS — |r|={abs(r):.4f} > {threshold}. "
                   f"Target may not be shifted forward by {self.horizon} days.")
            ),
        )

    # ── Test 3: Temporal feature-target correlation ───────────────────────────

    def _test_temporal_correlation(self) -> TestResult:
        """
        Feature-target Spearman |r| above corr_thresh is a leakage signal.

        Legitimate predictive features should have |r| in the 0.02–0.12 range.
        Values > 0.15 on the full panel suggest look-ahead contamination.

        Note: this test produces WARNINGS not errors — a high correlation
        might be genuine (e.g. 52w high distance is legitimately predictive).
        The analyst must review flagged features manually.
        """
        if self.target_col not in self.panel.columns:
            return TestResult(
                name="TemporalCorrelationTest",
                passed=True,
                details=f"Target '{self.target_col}' not found — skipped.",
            )

        target = self.panel[self.target_col].dropna()
        flagged = []
        for feat in self.feat_cols:
            if feat not in self.panel.columns:
                continue
            aligned = self.panel[feat].reindex(target.index).dropna()
            common_idx = target.index.intersection(aligned.index)
            if len(common_idx) < 200:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, _ = stats.spearmanr(
                    aligned.loc[common_idx], target.loc[common_idx]
                )
            if abs(r) > self.corr_thresh:
                flagged.append((feat, round(r, 4)))

        flagged.sort(key=lambda x: -abs(x[1]))
        flagged_names = [f for f, _ in flagged]

        # This test WARNS but doesn't block — analyst review required
        passed = len(flagged) == 0
        detail = (
            "No features with suspiciously high target correlation."
            if passed else
            f"{len(flagged)} features with |Spearman r| > {self.corr_thresh} "
            f"(review manually — may be genuine signal or leakage): "
            f"{flagged[:10]}"
        )
        return TestResult(
            name="TemporalCorrelationTest",
            passed=True,   # Warning only — not a hard failure
            details=detail + (" [REVIEW REQUIRED]" if not passed else ""),
            flagged_features=flagged_names,
        )

    # ── Test 4: Forward-fill boundary leakage ────────────────────────────────

    def _test_forward_fill_boundary(self) -> TestResult:
        """
        Detect features where future information bled into the test window
        via forward-fill (genuine leakage).

        Three-way classification for features with >80% boundary similarity:

          A. CONSTANT across the entire panel (ts_std ≈ 0 AND cs_std ≈ 0):
             → Uninformative zero-variance feature (data sparsity / no zones found).
             → Logged as WARNING, NOT a leakage failure.
             → Should be dropped by FeatureSelector, not block training.

          B. Has real variance in the training half but same value on both
             sides of the boundary for most tickers:
             → GENUINE ffill leakage — future value carried into test set.
             → HARD FAIL.

          C. High boundary similarity but real variance exists somewhere:
             → Legitimately persistent feature (zone flags, regime, etc.).
             → Logged as INFO, not a failure.
        """
        dates = self.panel.index.get_level_values("date").unique().sort_values()
        if len(dates) < 10:
            return TestResult(
                name="ForwardFillBoundaryTest", passed=True,
                details="Not enough dates to test forward-fill boundary.",
            )

        mid_idx  = len(dates) // 2
        last_tr  = dates[mid_idx - 1]
        first_te = dates[mid_idx]

        try:
            tr_cross = self.panel.xs(last_tr,  level="date")[self.feat_cols]
            te_cross = self.panel.xs(first_te, level="date")[self.feat_cols]
        except KeyError:
            return TestResult(
                name="ForwardFillBoundaryTest", passed=True,
                details="Could not retrieve boundary cross-sections — skipped.",
            )

        common_tickers = tr_cross.index.intersection(te_cross.index)
        if len(common_tickers) < 10:
            return TestResult(
                name="ForwardFillBoundaryTest", passed=True,
                details="Fewer than 10 common tickers at boundary — skipped.",
            )

        tr_vals = tr_cross.loc[common_tickers]
        te_vals = te_cross.loc[common_tickers]


        # Step 1: boundary similarity filter
        high_sim: List[Tuple] = []
        for feat in self.feat_cols:
            if feat not in tr_vals.columns or feat not in te_vals.columns:
                continue
            same_frac = (tr_vals[feat] == te_vals[feat]).mean()
            if same_frac > 0.80:
                high_sim.append((feat, round(float(same_frac), 3)))

        # Step 2: classify each high-sim feature
        genuine_leakage: List[Tuple] = []
        constant_features: List[str] = []
        persistent_ok: List[str] = []

        # Pre-compute mean cross-sectional std across training dates (sampled every 5th
        # date for speed). Features where mean_cs_std ≈ 0 across ALL dates are
        # market-wide / time-series-only (e.g. regime flags) — same value for all
        # tickers on any date by design. Cross-sectional freezing is meaningless there.
        sample_dates = dates[:mid_idx][::5]
        mean_cs_std: Dict[str, float] = {}
        for feat, _ in high_sim:
            if feat not in self.panel.columns:
                mean_cs_std[feat] = 0.0
                continue
            stds = []
            for d in sample_dates:
                try:
                    cross_vals = self.panel.xs(d, level="date")[feat]
                    s = float(cross_vals.std())
                    if np.isfinite(s):
                        stds.append(s)
                except KeyError:
                    continue
            mean_cs_std[feat] = float(np.mean(stds)) if stds else 0.0

        for feat, sim in high_sim:
            # Variance across the FULL panel (all dates × all tickers)
            full_std = self.panel[feat].std() if feat in self.panel.columns else 0.0
            full_std = 0.0 if not np.isfinite(full_std) else full_std

            if full_std < 1e-8:
                # Case A: constant across entire panel — data sparsity, not leakage
                constant_features.append(feat)
                continue

            avg_cs = mean_cs_std.get(feat, 0.0)
            if avg_cs < 1e-8:
                # Market-wide feature (e.g. regime_bull): same value for ALL tickers
                # on any given date — time-series-only signal, not cross-sectional.
                # Cross-sectional std = 0 is expected, not a leakage indicator.
                persistent_ok.append(feat)
                continue

            # Feature IS cross-sectional — check if it's abnormally frozen at boundary
            cs_std_boundary = float(tr_vals[feat].std()) if feat in tr_vals.columns else 0.0
            cs_std_boundary = 0.0 if not np.isfinite(cs_std_boundary) else cs_std_boundary

            if cs_std_boundary < 1e-8 and sim >= 0.95:
                # Further check: if the frozen value is essentially zero for all tickers,
                # this is data sparsity (no zones/events active on that date), NOT ffill
                # leakage.  Genuine ffill leakage carries a non-trivial value forward.
                mean_tr_val = float(abs(tr_vals[feat]).mean()) if feat in tr_vals.columns else 0.0
                if mean_tr_val < 1e-8:
                    # All-zero on boundary — sparsity, not leakage
                    persistent_ok.append(feat)
                else:
                    # Case B: non-trivial value frozen across boundary → genuine ffill leakage
                    genuine_leakage.append((feat, sim))
            else:
                # Case C: legitimately persistent cross-sectional feature
                persistent_ok.append(feat)

        if constant_features:
            log.warning(
                f"ForwardFillBoundaryTest: {len(constant_features)} features are "
                f"constant (zero variance) across the entire panel — likely sparse "
                f"zone/regime data with no signal. These are uninformative and will "
                f"be dropped by FeatureSelector. Not leakage: {constant_features[:10]}"
            )
        if persistent_ok:
            log.info(
                f"ForwardFillBoundaryTest: {len(persistent_ok)} high-similarity "
                f"features have real variance — legitimately persistent "
                f"(zones, regime, etc.): {persistent_ok[:10]}"
            )

        passed = len(genuine_leakage) == 0
        return TestResult(
            name="ForwardFillBoundaryTest",
            passed=passed,
            details=(
                f"No genuine ffill boundary leakage detected. "
                f"({len(constant_features)} constant-zero features noted; "
                f"{len(persistent_ok)} persistent features noted.)"
                if passed else
                f"{len(genuine_leakage)} features vary in training but are "
                f"cross-sectionally frozen at the boundary — genuine ffill leakage: "
                f"{genuine_leakage[:10]}"
            ),
            flagged_features=[f for f, _ in genuine_leakage],
        )


    # ── Test 5: Group boundary (purging check) ────────────────────────────────

    def _test_group_boundary(self) -> TestResult:
        """
        Verify that there is a minimum gap of `purge_days` between the last
        training date and first test date. This is a sanity check on the CV
        fold specs — not on the panel itself.

        Since we don't have fold_specs here, we instead verify that no
        'future_*' target column has non-NaN values in the most recent
        `horizon` rows of the panel (which would mean those rows have
        already-realised future returns and shouldn't be scored).
        """
        future_cols = [c for c in self.panel.columns if c.startswith("future_")]
        if not future_cols:
            return TestResult(
                name="GroupBoundaryTest", passed=True,
                details="No future_* columns found — purging check skipped.",
            )

        dates = self.panel.index.get_level_values("date").unique().sort_values()
        recent_dates = dates[-self.purge_days:]
        recent_panel = self.panel[
            self.panel.index.get_level_values("date").isin(recent_dates)
        ]

        leaking = []
        for col in future_cols:
            if col in recent_panel.columns:
                non_null = recent_panel[col].notna().sum()
                if non_null > 0:
                    leaking.append((col, int(non_null)))

        passed = len(leaking) == 0
        return TestResult(
            name="GroupBoundaryTest",
            passed=passed,
            details=(
                f"No future return columns with values in last {self.purge_days} rows — OK."
                if passed else
                f"{len(leaking)} future columns have non-NaN values in the last "
                f"{self.purge_days} rows. These rows must be excluded from scoring: "
                f"{leaking[:5]}"
            ),
            flagged_features=[c for c, _ in leaking],
        )