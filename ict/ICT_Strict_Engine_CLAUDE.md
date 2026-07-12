# ICT Strict Engine — Claude Session Context

## IMPORTANT — Read This First (Claude Instructions)

This is a **testing-only session**. Your role is strictly:
- Help the tester understand what they see on the chart
- Explain ICT concepts and what each label/zone means
- Identify whether observed behavior matches expected behavior
- Report findings clearly so they can be passed back to the developer

**You must NOT:**
- Edit, modify, or rewrite any Pine Script code
- Edit, modify, or rewrite any Python code
- Suggest code changes directly in the file
- Use the Edit, Write, or Bash tools on any `.pine` or `.py` file

If you identify a bug or issue, **describe it clearly in plain English** so it can be reported to the developer. Do not fix it yourself.

If the tester asks you to make a code change, respond:
> "I'm in testing-only mode for this session. I can document the issue clearly for the developer — here's what I found: [description]"

---

## Project Overview

This is a TradingView Pine Script v5 strategy that implements ICT (Inner Circle Trader) / Smart Money Concepts for visual analysis and automated signal generation. It is companion code to a Python ML stock predictor at `C:\Victor\Project\ml-stock-predictor` — both implement the same ICT logic, Pine for visual/manual trading, Python as ML features.

**Primary file**: `C:\Users\victor.prabhu\Documents\ICT_Strict_Engine_v2_fixed.pine`

---

## ICT Concepts Implemented

### Order Blocks (OB)
- **Bull OB**: Last bearish candle before a bullish impulse. Key level = `open` of bearish candle = `dBodyMax`
- **Bear OB**: Last bullish candle before a bearish impulse. Key level = `close` of bullish candle = `dBodyMax`
- Invalidated when price **closes through the body** (not just wicks)
- Only form when accompanied by a BOS (`bullObKeep = bullObRaw and recentBullBos`)
- Drawn as **horizontal lines** using `line.new()` — never `plot()`

### Fair Value Gaps (FVG)
- **Bull FVG**: 3-bar gap where `low[0] > high[2]` — displayed as blue box
- **Bear FVG**: 3-bar gap where `high[0] < low[2]` — displayed as red box
- Invalidated at **midpoint** — deleted immediately (`box.delete()`), no grey fade
- No BOS gate at formation — gate applied at signal time only

### Breaker Blocks (BK)
- **Bull BK**: Former bear OB violated by close above its top → flips to support
- **Bear BK**: Former bull OB violated by close below its bottom → flips to resistance
- Drawn as **horizontal lines** (two lines: top and bottom of the former OB body)
- Only shown when macro regime supports it (see Macro Filter below)

### Liquidity (BSL/SSL)
- **BSL** (Buy-Side Liquidity): Pool above swing highs — swept when wick above + close below
- **SSL** (Sell-Side Liquidity): Pool below swing lows — swept when wick below + close above
- SSL in **discount** = institutional accumulation (long setup)
- SSL in **premium** = distribution trap (bearish continuation expected)

### BOS / CHoCH
- **BOS+/BOS-**: Break of Structure — close above/below last confirmed swing
- **CH+/CH-**: Change of Character — BOS against the current structure direction
- Pivot confirmation lag: `nSwing=5` bars each side → BOS confirms 5 bars after the actual break

---

## Architecture

### Visualization — Array-Based (critical)
All zones use arrays so multiple valid zones persist simultaneously:
```pine
var float[] _bObH  = array.new_float(0)   // bull OB highs
var float[] _bObL  = array.new_float(0)   // bull OB lows
var line[]  _bObLn = array.new_line(0)    // bull OB lines
```
- `plot()` is **never used** for zones — it creates diagonal connecting lines
- `line.new()` requires `x1 ≠ x2` to be horizontal: use `bar_index - 1` for x1
- Pine v5 for-loop: `for i = array.size() - 1 to 0` — do NOT add `by -1` (causes runtime error)

### Signal Logic — Two Modes
```pine
// Legacy mode (OB + FVG confluence)
legacyLongSignal  = bullObNear and bullFvgSweepConf and recentBullBos and bullConfirm

// Strict mode (BK continuation) — active when strictBreakerContinuationOnly = true
strictLongSignal  = bullBkActive and recentBullBos and inDiscount and bullConfirm and not macroIsBearish
strictShortSignal = bearBkActive and recentBearBos and inPremium  and bearConfirm and not macroIsBullish
```

**Key design decisions:**
- `recentBullBos` replaced `mss > 0` — BK triggers 5 bars before BOS confirms due to pivot lag, so `mss` is still -1 when BK fires
- Sweep gate (`recentSslSwept`) removed from strict signals — BK trigger already implies a prior sweep structurally

---

## Macro Trend Filter

Prevents counter-trend signals during sustained trends:

```pine
var int bearBosCount = 0
var int bullBosCount = 0
if bearBosEvt
    bearBosCount += 1
    bullBosCount := math.max(0, bullBosCount - 1)  // decay, not hard-reset
if bullBosEvt
    bullBosCount += 1
    bearBosCount := math.max(0, bearBosCount - 1)

macroIsBearish = bearBosCount >= 2   // 2+ consecutive bear BOS = downtrend
macroIsBullish = bullBosCount >= 2   // 2+ consecutive bull BOS = uptrend
```

**Decay not hard-reset**: One weak opposite BOS subtracts 1 from the counter rather than zeroing it. This prevents a dead-cat bounce from instantly clearing a strong downtrend signal.

**Gated by macro filter:**
- `strictLongSignal` — blocked when `macroIsBearish`
- `strictShortSignal` — blocked when `macroIsBullish`
- `BK+` label and lines — suppressed when `macroIsBearish`
- `BK-` label and lines — suppressed when `macroIsBullish`

---

## Known Open Issues

(none currently — see Test 8 below for the hybrid dealing-range fix)

---

## Testing Methodology

Testing is done via **TradingView Replay mode** on daily charts. Standard test sequence:
1. Load script in TradingView
2. Open a daily chart of any liquid stock (SPY, AAPL, etc.)
3. Use Replay to step through bars
4. Check: do OB lines appear correctly? Do FVG boxes delete at midpoint? Do BK lines stay horizontal?
5. Screenshot any unexpected behavior and share

**Diagnostic table** (top-right of chart) shows:
- Long/Short signal counts
- BK trigger counts
- Filter rejection counts (Long Fail MSS, Long Fail P/D, Long Fail Sweep)

---

## ICT Trading Rules Implemented (for reference)

1. **SSL in discount + BOS+ = long setup** (accumulation)
2. **BSL in premium + BOS- = short setup** (distribution)
3. **After SSL sweep**: enter long on BK+ confirmation in FVG — stop below SSL low, target BSL
4. **After BSL sweep**: enter short on BK- confirmation in FVG — stop above BSL high, target SSL
5. **Failed setup**: if CH- prints after SSL sweep, the long failed — exit immediately, look for short
6. **Macro filter**: if 2+ consecutive BOS- with no recovery → no longs regardless of BK+
7. **Overhead supply**: stacked bear FVGs from a waterfall = resistance ladder — take partial profits at each box

---

## Python ML Counterpart

**File**: `C:\Victor\Project\ml-stock-predictor\pipeline\features\ict_features.py`

Differences from Pine:
- No hard gates — everything is a **feature column** for the ML model to weight
- Macro trend added as features (not gates): `ict_bear_bos_streak`, `ict_bull_bos_streak`, `ict_macro_regime`
- Zone Priority Deduplication: BB > OB > FVG (Pine shows all simultaneously)
- Numba JIT on liquidity engine for speed across 500+ stocks
- 9 liquidity features per side vs Pine's simple nearest-pool distance

---

## Bug Reporting Format (for testers)

When something looks wrong, Claude will help you document it in this format so it can be passed to the developer:

```
ISSUE: [short title]
CHART: [symbol, timeframe, approximate date range]
OBSERVED: [what you see on the chart]
EXPECTED: [what should have happened based on the ICT rules above]
REPRODUCE: [steps — which bars to look at, what sequence of events]
SCREENSHOT: [attach if possible]
```

Example:
```
ISSUE: BK+ label appearing during sustained downtrend
CHART: AAPL daily, Jan-Mar 2026
OBSERVED: BK+ label fires on Jan 15 after 3 consecutive BOS-
EXPECTED: BK+ should be suppressed when macroIsBearish = true
REPRODUCE: Load replay, advance to Jan 15 — BK+ visible with no BOS+ in prior 10 bars
```

---

## What to Test

Work through these scenarios in TradingView Replay:

### Test 1 — Bull OB persistence + expiry
- Find a bullish impulse candle that followed a bearish candle
- Confirm a blue horizontal line appears at the open of the bearish candle
- Advance bars — line should persist until price closes BELOW the body of that bearish candle
- At violation: line should fade, not disappear instantly
- **Expiry test**: if the OB line is still present after `expiryBars` (default 63) bars with NO price violation, it should also fade — the line should NOT stay bright indefinitely

### Test 2 — Bear OB persistence
- Find a bearish impulse candle that followed a bullish candle
- Confirm an orange horizontal line appears at the close of the bullish candle
- Line should persist until price closes ABOVE the body of that bullish candle

### Test 3 — Bull FVG box
- Find 3 bars where `low[0] > high[2]` (gap between bar-2's high and bar-0's low)
- Blue box should appear spanning that gap
- Box should delete when price closes BELOW the midpoint of the gap
- Box should NOT fade grey — it should disappear immediately
- **Expiry test**: if the box is still present after 63 bars with no midpoint violation, it should also be deleted

### Test 4 — Bear FVG box
- Same as Test 3 but for downward gaps
- Red box, deleted when price closes ABOVE the midpoint

### Test 5 — No false BK+ in downtrend
- Find a section with 2+ consecutive BOS- labels
- Confirm NO BK+ labels appear during this stretch
- If BK+ appears, that is a bug to report

### Test 6 — Macro filter working
- Find a waterfall section (sustained drop with multiple BOS-)
- Confirm no long entry signals fire during this section
- The diagnostic table (top-right) should show high "Long Fail" counts

### Test 7 — Valid BK+ in uptrend
- Find a section with BOS+ firing
- Confirm BK+ does appear here when a bear OB is violated
- Confirm a long signal eventually fires after BK+ + BOS+ + price in lower half of range

### Test 8 — Hybrid dealing range (premium/discount)
- **Trend case**: find a sustained trend (several consecutive same-direction BOS). Confirm `inDiscount`/`inPremium` no longer reads as constantly discount (or constantly premium) on nearly every bar — it should track a real, stable range, not flip-flop with every tiny local pivot.
- **Consolidation case**: find a sideways/ranging section with no clean break of the visible top/bottom band. Confirm Bull BK / Bear BK triggers are noticeably less frequent than before — premium/discount (and therefore `strictLongSignal`/`strictShortSignal`) should stay anchored to the real range, not flip on every small interior oscillation.
- **Breakout case**: find a bar where price makes a genuine new high/low beyond the recent ~100-bar range. Confirm the dealing range updates immediately on that bar (no multi-bar lag) — this is the "hybrid" part: a real new extreme should override the old rolling window right away.

### Test 9 — Liquidity Creation (EQH/EQL, PWH/PWL/PMH/PML, internal/external)
- **EQH/EQL**: find two swing highs (or lows) sitting close together in price. Confirm an "EQH"/"EQL" label appears the moment the *second* one forms (not the first).
- **PWH/PWL/PMH/PML**: confirm two orange step-lines (prior week high/low) and two fuchsia step-lines (prior month high/low) are visible, each flat within its period and jumping to a new level at the week/month boundary. Confirm they do NOT update mid-week/mid-month (no repaint) — only at the boundary, using the now-completed prior period's value.
- **Internal vs external**: check the diagnostic table's "Nearest BSL/SSL Pool" rows. Find a pool sitting at the obvious edge of the visible chart range — should read "External." Find a minor pool well inside the range — should read "Internal."

### Test 11 — Liquidity Scoring
- Find a pool that's been touched multiple times (look for an "EQH"/"EQL" label, or check `p_touches` implicitly via repeated swing highs/lows at the same level) and has gone a long time without being swept. Confirm its Score (in "Nearest BSL/SSL Pool") is noticeably higher than a fresh, single-touch pool nearby.
- Find a pool whose level lines up closely with one of the PWH/PWL/PMH/PML step-lines. Confirm its score jumps by roughly 20 points compared to a similar pool with no such alignment.
- Confirm External pools score at least 10 points higher than otherwise-similar Internal pools.

### Test 12 — Liquidity Lifecycle
- **Partial sweep survives**: find a shallow wick-through-and-reject (small penetration past a BSL/SSL level). Confirm the pool does NOT disappear — it should still be there on the next bar, with its sweep counted.
- **Full raid retires**: find a deep, displaced wick-through-and-reject. Confirm the pool DOES disappear immediately — that's a decisive event, no survival.
- **Sustained acceptance**: find a single-bar close beyond a pool's level that gets reclaimed the very next bar (a whipsaw). Confirm the pool survived that single bar — it should only retire if price closes beyond it for `acceptanceBars` (default 2) bars IN A ROW.
- **Stage progression**: pick one still-active pool and watch its "Nearest BSL/SSL Pool" table stage (Resting → Partial Sweep → Multiple Sweeps → Full Raid) update as you step through Replay.
- **Converted (heuristic, approximate)**: find a sweep followed shortly by a same-direction BK trigger. Confirm the stage reads "Converted (BK)." Treat this one loosely — it's a same-time-window heuristic, not a guaranteed causal link, so a false positive/negative here isn't necessarily a bug worth reporting.

### Test 13 — External reviewer fixes (maxSweeps, BOS-gated acceptance, score penalty, external sub-category)
- **maxSweeps**: find a pool swept many times shallowly (never a Full Raid). Confirm it eventually retires once sweep count hits `maxSweeps` (default 4), even without ever crossing the Full Raid depth threshold.
- **BOS-gated acceptance**: find a clean break beyond a pool's level that holds for `acceptanceBars`+ bars but with NO confirming BOS in that direction. Confirm the pool does NOT retire yet — it should keep waiting. Then find a case where BOS does confirm shortly after — confirm it retires then.
- **Score penalty**: compare two pools with similar touch count, but one swept several times (survived as partial sweeps) and one never swept. Confirm the swept one scores lower.
- **External sub-category**: find an external pool that also lines up with a PWH/PWL/PMH/PML step-line. Confirm the table shows "External (PWH)" etc., not just "External." Find one at the range edge with no calendar alignment — confirm it shows "External (range-edge)."

### Test 14 — Selection reweighting, PDH, edge-merge, BK Confidence
- **PDH/PDL**: confirm a new gray step-line pair (previous day high/low) is visible alongside the orange (week) and fuchsia (month) ones.
- **Weighted selection**: find a case where a strong, multi-touch internal pool sits much closer to price than a weak, single-touch external pool. Confirm "Nearest BSL/SSL Pool" now shows the STRONG internal one, not the external one — the old nearest-distance-only logic would have shown whichever was geometrically closest regardless of strength.
- **Edge-based merge**: watch a cluster of 3+ touches form. Confirm a new pivot merges in based on whether it's within the cluster's [min, max] band (plus tolerance), not whether it's close to the cluster's running average — should feel less "drifty" than before, especially once the cluster has a few touches spread across a wider band.
- **BK Confidence**: find a Bull/Bear BK Trigger that followed a deep sweep (close to or beyond `fullRaidAtrThresh`). Confirm "Bull/Bear BK Confidence" in the table reads high (near 100). Find one following only a shallow sweep — confirm confidence reads low. Confirm BOTH still produce a BK trigger/label regardless of confidence — confidence must NOT gate whether the BK fires.

### Test 15 — Raw BK trigger markers
- Confirm small plain triangles (blue below bars for Bull BK, purple above bars for Bear BK) appear on the chart at EVERY entry in the "Bull/Bear BK Trigger Dates" table lists — including ones during a counter-trend macro regime where the labeled "BK+"/"BK-" text marker is suppressed. The raw triangle should still show even when the gated label doesn't.
- Confirm the COUNT of triangles visible over any stretch matches "Bull BK Triggers"/"Bear BK Triggers" in the table for that same stretch.

### Test 16 — OB deferred BOS confirmation (first-reversal-off-a-swing-low fix)
- Find a clean reversal: a sharp decline (or rally) into an SSL/BSL sweep, followed by a reversal candle, followed several bars later by a confirming BOS+/BOS-. Confirm a blue (Bull) or orange (Bear) OB line now appears anchored at the reversal-causing candle itself — not at the later bar where BOS confirmed, and not missing entirely (which was the bug).
- Confirm the OB line's age/expiry counts from the ORIGINAL candle, not the confirmation bar — i.e. it shouldn't get a "fresh" `expiryBars`-bar lifespan starting from when BOS happened to confirm.
- Confirm continuation-pullback OBs (where a BOS already happened recently, e.g. mid-trend) still register immediately as before — this fix should be purely additive, not change the existing fast-path behavior.
- If a BOS never confirms within `obBosConfirmWindow` (default 20 bars) of a raw OB candle, confirm the OB correctly never appears (the candidate should expire, not linger forever).

### Test 17 — Sweep cooldown + significance gate (Round 1 fixes)
- **No cascade (Issue 1)**: find a level that price pokes above (wick over, close below) on several consecutive bars. Confirm only ONE "BSL" triangle prints for that probe, not one per bar — a repeat sweep of the same pool is suppressed for `nSwing` (default 5) bars after the first. Mirror for "SSL" on a level poked from below.
- **Significance (Issue 2)**: find a lone, **fresh** (recently formed), single-touch swing high — not multi-touched, not on a weekly/monthly step-line, and less than `minRestBars` (default 20) bars old — that gets wicked through and rejected. With `sweepRequireSignificance` ON (default), confirm NO "BSL" triangle fires there. Then confirm a sweep DOES fire for any of the three qualifiers: (b) a multi-touched "EQH" cluster, (c) a level on a PWH/PMH line, OR (a) a **rested** single-touch swing whose original pivot is >= `minRestBars` bars in the past.
- **Rested re-admit (over-suppression fix)**: find a clean, single-touch swing low that has sat untouched for well over `minRestBars` bars, then gets swept (wick below, close back above). Confirm SSL now fires — the "rested = real liquidity" case. Mirror for BSL. This is the fix for legit deeper swings that were previously being wrongly suppressed.
- **Escape hatch**: toggle `sweepRequireSignificance` OFF and confirm the micro-pivot sweeps re-appear (proves the gate, not some unrelated change, is what suppressed them). Leave it ON for normal testing.
- **Retire still works**: confirm a DEEP raid (past `fullRaidAtrThresh`) still retires the pool even if it's a single-touch/insignificant one or lands inside the cooldown window — the significance/cooldown gate only affects the visible sweep EVENT, not pool retirement.

### Test 18 — Structure labels + OB invalidation (BUG-003 / BUG-004)
- **Legacy labels off (BUG-003)**: by default you should now see ONLY the strict engine's `sCH+?`/`RVSL+`/`sBOS+`/`sBOS-` labels — NOT the old `CH+`/`CH-`/`BOS+`/`BOS-`. Confirm the misleading legacy CH-/CH+ are gone. To bring the legacy layer back for comparison, enable **"Show LEGACY retail BOS+/BOS-/CH+/CH-"** in Signal Logic. Judge structure correctness by the strict labels only.
- **OB deletes on invalidation (BUG-004)**: find a Bear OB (orange line), step forward until a candle CLOSES above the line (above the body top). Confirm the line now DISAPPEARS cleanly on that bar (previously it faded to a faint stub). Mirror for a Bull OB (blue line) on a close below its body bottom. An OB that dies by AGE (`expiryBars`) instead still fades to a faint marker — that difference is intentional (invalidated = gone, expired = faded).
- **Don't confuse OB with calendar lines**: the orange PWH/PWL **step-lines** are NOT OBs and are correctly never removed by a close through them — they redraw each week. If an "orange line persisting through a rally" is a stepline, that's expected, not an OB bug.

### Test 19 — Rearchitected Order Block engine (displacement-origin)
*The OB engine was fully redesigned: an OB is now the ORIGIN of a confirmed institutional displacement, not an engulfing candle.*
- **OB only after a confirmed reversal**: a new OB (blue = bull / orange = bear line) should now appear ONLY at/after a strict `RVSL+`/`RVSL-` confirmation — never on a random engulfing candle mid-range. Confirm you no longer see OBs form without a preceding sweep + strict reversal.
- **OB sits at the origin, not the reaction**: after an `RVSL+`, the blue OB line should anchor to the **last down candle before the up-move began** (the base of the leg), NOT the big bullish candle that caused the reversal. Mirror for bear.
- **Discount/Premium**: bullish OBs should mostly originate in the lower half of the range (discount), bearish in the upper half (premium).
- **Multiple OBs coexist**: forming a new OB should NOT erase an older still-valid one (they're now held in an array). Several blue/orange lines can be active at once.
- **Touch ≠ death**: price tapping into an OB and closing back out should NOT invalidate it (that's mitigation). Only a **close beyond the far side** removes the drawn line.
- **Backward-compat sanity**: BK+/BK- triggers, entries, and the diagnostics table should still behave — the refactor preserved those interfaces.
- ⚠️ **First load = compile check**: this is a large refactor with new PineScript types. If TradingView shows a compile error on paste, copy the exact error text back to the developer.

---

## Conversation Style Preference

- Use **Feynman technique**: analogy first, build from first principles, never lead with jargon
- When explaining ICT concepts: use plain English, then the technical term
- When reviewing charts: identify the sequence of events left-to-right, then give the human action steps
- When fixing code: explain the root cause before showing the fix
- Short concise responses preferred — no padding
