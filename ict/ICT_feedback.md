# ICT Strict Engine — Feedback Log

Each testing round goes in its own section. Tester fills this in, developer reads it.

---

## How to use (Tester)

At the end of your testing session, ask Claude:
> "Summarize all issues we found today as a structured bug report for the developer"

Paste the output below under a new round heading.

---

## Round 1 — 2026-07-05

### ISSUE 1: BSL cascades on same liquidity pool
- OBSERVED: 3 BSL labels fire within ~5 bars all reaching the same price ceiling
- EXPECTED: One pool = one sweep event until the pool is retired / a new one forms at a distinct level
- Tester's suspected cause: `lastSwingHigh` re-anchoring
- **Dev finding (root cause corrected):** `BSL` labels come from the pool engine (`bslSwept`, line 225), NOT `lastSwingHigh` (that only drives `bullBosState`). Real cause: after the Lifecycle #1 fix, shallow sweeps leave the pool active, so it re-fired `bslSwept` on every consecutive poke bar with no cooldown (and inflated `sweep_cnt`).
- **STATUS: FIXED** — per-pool sweep cooldown (`>= nSwing` bars between counted events) via the previously write-only `p_last_swp`.

### ISSUE 2: BSL fires on insignificant micro-pivots (no real stop cluster)
- OBSERVED: single small candle tagged BSL where the swept high had no liquidity behind it
- EXPECTED: only fire against multi-touched or calendar-aligned (PWH/PMH) levels
- Tester's suspected cause: `pivotLen=3` confirming micro-tops (note: default `nSwing`=5, not 3)
- **Dev finding:** a pool is created from any single pivot (`p_touches=1`) with no significance filter before it can be swept.
- **STATUS: FIXED** — sweep now requires a significant pool: `p_touches >= 2` OR aligned to PWH/PMH within `htfConfluenceTolAtr`. New toggle `sweepRequireSignificance` (default on) as escape hatch.

**No new tunable thresholds** — reused `nSwing` (cooldown) and `htfConfluenceTolAtr` (alignment). Both fixes apply symmetrically to BSL and SSL.

### Issue 2 — re-report (still seeing micro-pivot BSL)
- Tester re-observed a young micro-pivot firing BSL; requested criteria (a) age OR (b) multi-touch OR (c) calendar.
- **Dev finding:** (a) is an ADMIT criterion — it can't suppress a *young* pivot (it fails age too), so it's not the missing filter. Real leak was in (c): the calendar tolerance used `htfConfluenceTolAtr` (1.0 ATR ≈ a full daily range), so a micro-pivot merely *near* the W/M high qualified.
- **STATUS: FIXED (leak tightened)** — calendar-alignment tolerance for the significance GATE changed from `htfConfluenceTolAtr` (1.0 ATR) → `mergeProx` (0.25 ATR, "at the level"). Zero new DOF.
- **ACTION FOR TESTER:** confirm the updated `.pine` is actually live — Settings → "Signal Logic" → the checkbox **"BSL/SSL requires significant pool"** must exist and be ON. If it's absent, the previous fix never loaded (which alone explains the re-report).

### Issue 2 — over-suppression correction (criterion (a) added)
- OBSERVED (chart review): a legit deeper single-touch swing low was swept (wick below, close back above) but got NO SSL — the significance gate was too strict.
- **Dev correction:** I previously declined criterion (a) "N bars old," arguing it can't suppress a *young* pivot. That was right about suppression but MISSED the reverse: `touches>=2 OR calendar` alone wrongly rejects legitimate **single-touch swing lows/highs that have rested** long enough for stops to build. Criterion (a) re-admits exactly those.
- **STATUS: FIXED** — added `rested = (bar_index - p_first_t) >= minRestBars` as a third OR branch. A single-touch, unaligned FRESH micro-pivot (age ~ nSwing) still fails; a rested one (>= `minRestBars`, default 20) now qualifies. Repurposed the dead `sweepHalfLife` input into `minRestBars` (net zero new inputs; removed dead code).
- **DOF note:** the 20-bar rest threshold is a new *effective* researcher DOF (logged) — set on reasoning ("~one liquidity time-scale for stops to accumulate"), not tuned to a target signal count.

---

## Round 2 — [Date]

_Tester: paste Claude's bug report summary here_

---
