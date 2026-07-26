---
domain: ict
branch: alpha-research-v2
commit: 31e52ed
generated: 2026-07-26
verify-before-trusting: true
---
# Domain: ICT / SMC (Pine Reference Engine)

## Core Objective
Own the Inner Circle Trader / Smart Money Concepts reference implementation — the TradingView Pine engine — and the parity tooling that checks whether the Python feature engine fires where Pine fires. Pine is the specification; Python (`pipeline/features/ict_features.py`, owned by Features) is the consumer that must maintain parity.

## Responsibilities
- The strict Pine engine (order blocks, breaker blocks, FVGs, BSL/SSL sweeps, structure gates).
- Engine documentation, bug reports, and feedback history.
- Bar-by-bar parity verification between Pine and Python.

## Key Files
- `ict/ICT_Strict_Engine_v2_fixed.pine` — the reference engine. (UNCOMMITTED CHANGES)
- `ict/ICT_Strict_Engine_CLAUDE.md`, `ict/ICT_Strict_Engine_DEV_CLAUDE.md` (UNCOMMITTED CHANGES) — engine spec / dev notes.
- `ict/ICT_Engine_BugReport_2026-07-05.md`, `ict/ICT_feedback.md` — defect and review history.
- `ict/pine_export_block.txt` — Pine block that emits `py_*` series for export.
- `ict/parity/` — parity workspace (currently empty).
- `scripts/tools/ict_pine_parity.py` (UNTRACKED) — reads a TradingView chart-export CSV (OHLCV + `py_*` series), re-runs the Python engine on those exact bars, compares each primitive.
- `scripts/tools/ict_signal_diagnostic.py` — signal-level diagnostics.

## Public Interfaces
- The parity contract is the anti-corruption layer between this domain and Features: `ict_features.py` may evolve only if parity counts stay acceptable.
- Downstream consumers of ICT outputs: `features_ict_*` columns and the gate prong `ict_bear_htf_score` (Portfolio).

## Dependencies
None internal (reference domain). External: TradingView/Pine runtime for the engine; the parity script uses pandas + the Python ICT engine.

## Architecture / Design Rules
- Ubiquitous language: OB (order block), BB (breaker), FVG (fair value gap), BSL/SSL (buy/sell-side liquidity), BOS/CHoCH, *strict mode*, *legacy mode*.
- Key documented divergence (from `ict_pine_parity.py` docstring): the Python engine and the Pine v2 engine were written against DIFFERENT reference indicators, and production runs `implementation_mode="legacy"` with every institutional gate disabled, while Pine defaults have them all on. Parity must therefore be measured by counts on identical bars, not argued from code reading.
- Invariant: parity checks run on exported bars — re-running on those exact bars is the method, not a convenience.

## Current State
- Pine engine v2 exists with active uncommitted edits; bug-report cycle documented (2026-07-05).
- Parity tooling written (`ict_pine_parity.py`) but UNTRACKED; `ict/parity/` holds no results yet — no committed parity verdict exists.
- ICT is NOT disabled project-wide. Commit `c584fba` ("disable ICT feature for Phase 5") only added an opt-in `--feature_set` CLI flag to `run_us_local_alpha.py`; the default remains `"all"` (ICT included) and the flag never existed in `run_sp500_local.py` — the script actually used for SP500/NDX/SP400/600 runs, which always trains on every `features_ict_*` column. Verified 2026-07-26: a full `run_sp500_local.py` momentum run had 10 of its top-20 SHAP features be ICT-derived.
- 2026-07-26: added a narrower `--feature_set ob_fvg` mode (both `run_sp500_local.py` and `run_us_local_alpha.py`) that restricts model training to Order Blocks + Fair Value Gaps only (`ICT_OB_FVG_PREFIXES`/`ICT_OB_FVG_EXACT_COLS` in `pipeline/features/engineer.py`) while leaving `skip_ict=False` so `ict_bull_htf_score`/`ict_bear_htf_score` stay computed for `pipeline/gating.py`'s bull-quality veto. Not yet wired into `run_nse_local.py` / `run_nse_tradingv_local.py` (neither script has any `--feature_set` plumbing at all).

## Known Limitations
- No quantified parity result yet; the legacy-vs-strict configuration gap means production ICT signals may not match the documented Pine behavior.
- `ict/parity/` is empty — the workflow's output half is missing.

## Pending TODOs
1. Run `ict_pine_parity.py` on at least one exported symbol/timeframe; commit script + results into `ict/parity/`.
2. Decide (and document) legacy vs strict mode for any post-Phase-5 ICT re-enable.

## Future Improvements
Automate export-CSV ingestion for a small parity panel of symbols; regression-run parity when either engine changes.

## Testing Status
- Python side: `tests/test_ict_features.py` (uncommitted changes). Pine side: no automated tests — verification is the parity script by design. Not executed this session.

## Notes for Future AI Sessions
Treat the Pine file as the spec: fix Python to match Pine, or explicitly document a deliberate divergence — never silently drift. Before using ICT-derived features or gate prongs, check the Phase 5 disable state and the parity status. Uncommitted `.pine`/doc changes here need committing or explaining.
