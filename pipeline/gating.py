"""
Momentum-bull quality gate — shared by all run scripts.

Removes model-driven longs that are technically broken (NKE / CPRT
false-positive filter):
  • under heavy overhead supply        → ssz_htf_score > 0.6   (CPRT-type)
  • overhead bearish ICT structure     → ict_bear_htf_score > 0.4
    (bear OB/BB/FVG composite across 5 TFs — complementary engine to SSZ:
     measured on 22k momentum-universe rows, 0.4 vetoes 4.1% of which
     3.8% are NOT caught by the ssz prong; 0.3 too broad, 0.5 inert)
  • broken/declining trend stack       → NOT(price>sma50 AND sma50↑ AND sma200↛↓) (NKE-type)
  • ADX direction owned by bears       → −DI > +DI

Applies to momentum BULL candidates only. Bear/reversal untouched.
scores_detail still records full ungated scores for transparency.

Calibration provenance: the STRUCTURAL prongs (ssz>0.6, ict_bear>0.4) were
calibrated 2026-06 on the US large/mid universe at ~5% combined prevalence —
prevalence-calibrated, not outcome-validated. The veto rate is self-monitored
each run; >15% means universe composition or feature distributions have
shifted: re-run the dose-response sweep (scripts/diagnostics/
diag_zone_overlap.py + threshold sweep) before trusting the lists. The NSE
universe was NOT part of the calibration sample — watch the printed veto rate
on the first NSE runs; the alarm will flag gross miscalibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SSZ_VETO_THRESHOLD       = 0.6
ICT_BEAR_VETO_THRESHOLD  = 0.4
STRUCTURAL_RATE_BASELINE = 0.05
STRUCTURAL_RATE_ALARM    = 0.15


def momentum_bull_quality_gate(
    cross_wl: pd.DataFrame,
    mode: str,
    feature_prefix: str,
) -> pd.Series:
    """
    Return a boolean keep-mask aligned to cross_wl.index.

    For mode != "momentum" the gate is a no-op (all True). Missing feature
    columns disable their prong individually; if every prong's columns are
    missing the gate is inactive and a loud warning is printed.
    """
    if mode != "momentum":
        return pd.Series(True, index=cross_wl.index)

    def _gc(col):
        full = f"{feature_prefix}{col}"
        return cross_wl[full].fillna(0.0).values.astype(float) if full in cross_wl.columns else None

    _ssz   = _gc("ssz_htf_score")
    _ictb  = _gc("ict_bear_htf_score")
    _pvs50 = _gc("price_vs_sma50")
    _s50   = _gc("sma50_slope_5")
    _s200  = _gc("sma200_slope_10")
    _pdi   = _gc("plus_di")
    _mdi   = _gc("minus_di")

    _keep = np.ones(len(cross_wl), dtype=bool)
    if _ssz is not None:                                    # CPRT: under supply
        _keep &= ~(_ssz > SSZ_VETO_THRESHOLD)
    if _ictb is not None:                                   # overhead bear OB/BB/FVG
        _keep &= ~(_ictb > ICT_BEAR_VETO_THRESHOLD)
    if _pvs50 is not None and _s50 is not None and _s200 is not None:  # NKE: broken stack
        _keep &= (_pvs50 > 0.0) & (_s50 > 0.0) & (_s200 >= 0.0)
    if _pdi is not None and _mdi is not None:               # bears control ADX
        _keep &= ~(_mdi > _pdi)

    _gate = pd.Series(_keep, index=cross_wl.index)
    _n_rej = int((~_gate).sum())
    if _n_rej:
        print(f"  [{mode}] bull quality gate: rejected {_n_rej} of {len(cross_wl)} "
              f"candidates (ssz-supply / ict-bear-structure / broken-trend / bearish-ADX)")

    # ── Calibration self-check ─────────────────────────────────────────
    # Only the structural prongs are monitored. The trend/ADX prongs are
    # deliberately regime-dependent (they reject most of the universe in a
    # bear market — correct, not drift).
    if _ssz is None and _ictb is None and _pvs50 is None and _pdi is None:
        print(f"  [{mode}] *** GATE INACTIVE: none of the gate's feature columns "
              f"were found in the panel — quality gate is NOT filtering. "
              f"Check feature names in engineer.py vs pipeline/gating.py.")
    else:
        _struct_rej = np.zeros(len(cross_wl), dtype=bool)
        if _ssz is not None:
            _struct_rej |= (_ssz > SSZ_VETO_THRESHOLD)
        if _ictb is not None:
            _struct_rej |= (_ictb > ICT_BEAR_VETO_THRESHOLD)
        _struct_rate = float(_struct_rej.mean()) if len(cross_wl) else 0.0
        print(f"  [{mode}] gate calibration: structural veto rate "
              f"{_struct_rate:.1%} (baseline ~{STRUCTURAL_RATE_BASELINE:.0%}, "
              f"alarm >{STRUCTURAL_RATE_ALARM:.0%})")
        if _struct_rate > STRUCTURAL_RATE_ALARM:
            print(f"  [{mode}] *** GATE CALIBRATION ALARM: structural veto rate "
                  f"{_struct_rate:.0%} is 3x the calibrated baseline — universe "
                  f"composition has shifted. Re-run the threshold sweep before "
                  f"trusting this watchlist.")

    return _gate
