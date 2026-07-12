# ICT Strict Engine — Bug Report
**Session date:** 2026-07-05  
**Tester:** victor.prabhu  
**File under test:** `ICT_Strict_Engine_v2_fixed.pine`  
**Mode:** TradingView Replay, daily chart

---

## BUG-001 — BSL cascades on the same liquidity pool

**Severity:** High  
**Test:** Related to Test 9 (Liquidity Creation)

**Observed:**  
3 BSL labels fire within ~5 bars, all with wicks reaching the same horizontal price ceiling (confirmed by zooming in — all three ▼ triangles sit at the same dotted resistance line).

**Expected:**  
One liquidity pool at a given price level = one sweep event. Subsequent touches of the same level should not re-fire BSL until the pool is formally retired and a new one forms at a distinctly different price.

**Root cause:**  
`lastSwingHigh` re-anchors to each new micro-pivot before the prior sweep window closes. As price makes marginally higher micro-highs within a tight cluster, each one becomes a new `lastSwingHigh`, generating repeated BSL triggers against essentially the same level.

**Fix direction:**  
After BSL fires, freeze `lastSwingHigh` until a clean BOS in either direction resets state — OR require a minimum price distance (e.g. 0.5 × ATR) between consecutive BSL events before a new one is eligible.

---

## BUG-002 — BSL fires on insignificant micro-pivots (no real stop cluster)

**Severity:** High  
**Test:** Related to Test 9 (Liquidity Creation)

**Observed:**  
A single small red candle is tagged BSL where the swept "high" was a micro-pivot formed only 1–2 bars earlier. No recognizable stop cluster exists at that level. The candle body is bearish (opened near the top, closed lower) — not a wick-and-reject pattern.

**Expected:**  
BSL should only fire against swing highs that represent real stop clusters:  
- At least N bars old (minimum age), OR  
- Touched/tested more than once (multi-touch pool), OR  
- Aligned with a calendar level (PWH / PMH / PDH)

A 1-bar-old micro-top with no prior tests is not a liquidity pool.

**Root cause:**  
`pivotLen = 3` confirms pivots with only 3 bars each side — very fast. `lastSwingHigh` reflects minor tops with no meaningful liquidity behind them, making BSL hypersensitive.

**Fix direction:**  
Add a minimum age filter: require `lastSwingHigh` to be at least 5–8 bars old before it qualifies as a BSL target. Optionally add a multi-touch requirement or calendar-level alignment bonus (already partially implemented in the scoring system — wire it into the sweep gate).

---

## BUG-003 — CH− mislabeled, should be CH+ after established bearish structure

**Severity:** Medium  
**Test:** Related to Test structure (BOS / CHoCH)

**Observed:**  
CH− label fires on or after a strong green candle that breaks upward, following:  
- 2 consecutive BOS− events (bearish structure firmly established)  
- 3 SSL sweeps with bullish rejections (accumulation pattern)

**Expected:**  
Current structure at that point is **bearish** (2× BOS− confirmed). The first bullish structural break against a bearish trend = **CH+** (Change of Character bullish — potential reversal signal). CH− would only be correct if the system still believed structure was bullish, which contradicts the two visible BOS− labels.

**Possible cause:**  
The `mss` variable or structure direction tracker may not have updated to bearish yet due to pivot confirmation lag (5-bar delay from `pivotLen = 5` each side). The system may still read structure as bullish at that bar, causing it to label the downmove as CH− rather than the subsequent upmove as CH+. The 5-bar lag means BOS− confirms late, and by the time it does, the reversal candle has already been labelled incorrectly.

**Fix direction:**  
Cross-check the bar index of CH label assignment against the bar index of the most recent confirmed BOS. If two BOS− are confirmed before this bar, CH must be CH+. Consider whether pivot lag is causing structure direction to be stale at the time of label assignment.

---

## BUG-004 — Bear OB not invalidated after multiple closes above its level

**Severity:** High  
**Test:** Test 2 (Bear OB persistence)

**Observed:**  
Orange Bear OB horizontal line persists on the chart through:  
- 2× BOS+ events, both with candle bodies closing clearly above the OB line  
- A large bullish rally taking price well above the line  

The line remains visible and active despite repeated invalidation-eligible closes.

**Expected:**  
Per the ICT rules in the session doc: *"Invalidated when price closes through the body (not just wicks)"*. The orange line is the `close` of the bull OB candle (top of the body = `dBodyMax`). The first BOS+ close above this level should have deleted the line.

**Three likely causes for developer to investigate:**

1. **Wrong boundary in invalidation check** — the loop may compare `close > bearOB_high` (the wick top) instead of `close > bearOB_close` (the body top / the orange line). This sets the bar too high.

2. **Array deletion silently failing** — Bear OBs live in the `_bObLn` line array. If the for-loop iterates in the wrong direction or has an off-by-one error, the deletion call is never reached for that element.

3. **`recentBearBos` gate accidentally blocking invalidation** — if the invalidation path is gated behind `recentBearBos`, once BOS+ fires and resets `recentBearBos` to false, the OB becomes "orphaned" — the invalidation condition can never run because the gate is closed.

**Reproduce:**  
Load any chart with a visible Bear OB (orange line). Step forward in Replay until a BOS+ candle closes above the line. Confirm whether the line disappears on that bar. If it does not, the invalidation logic is not firing.

---

## Summary Table

| ID | Issue | Severity | ICT Rule Violated | Dev status |
|---|---|---|---|---|
| BUG-001 | BSL cascades on same pool | High | One pool = one sweep | ✅ FIXED — per-pool sweep cooldown (`p_last_swp`, `>= nSwing` bars). Root cause was the pool engine (`bslSwept`), NOT `lastSwingHigh`. |
| BUG-002 | BSL fires on micro-pivots with no stop cluster | High | BSL requires real liquidity pool | ✅ FIXED — significance gate: multi-touch OR PWH/PMH-aligned OR rested `>= minRestBars`. `pivotLen` is `nSwing`=5 not 3. |
| BUG-003 | CH− mislabeled, should be CH+ | Medium | CHoCH direction must match established structure | ✅ FIXED (by retirement) — the `CH+/CH-` seen are the LEGACY engine (`bullChoch = bullBosEvt and mssPrev<=0`), a ±1-sign "first opposite BOS" that re-emits on chop — exactly the flaw the Stage-1 strict engine replaces. Legacy labels now gated behind `showLegacyStructure` (default OFF); the strict `sCH+?`/`RVSL+` engine (default ON) is the source of truth. |
| BUG-004 | Bear OB not invalidated after closes above | High | OB invalidated on close through body | ⚠️ CODE VERIFIED CORRECT + clarity fix — invalidation already uses `close > dBodyMax` (body top, line 946/1035), matching the requested rule; none of the 3 hypothesized causes exist. Changed invalidation to `line.delete()` (was fade-to-stub) so a violated OB unambiguously disappears. **Need a specific repro bar** to confirm a genuine defect vs. the orange PWH/PWL stepline being mistaken for an OB. |

---

*Generated by Claude Code in testing-only mode. No Pine Script files were modified.*
