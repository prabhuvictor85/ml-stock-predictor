"""
draw_scoring_pipeline.py — render the composite ML scoring pipeline as a PNG.
Usage:  python scripts/tools/draw_scoring_pipeline.py
Output: scripts/tools/scoring_pipeline.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "scoring_pipeline.png"

# ── colour palette ────────────────────────────────────────────────────────────
C = dict(
    bg       = "#0f1117",
    header   = "#1e2130",
    feat     = "#1a2744",
    feat_bd  = "#3b5bdb",
    model    = "#1a3344",
    model_bd = "#1971c2",
    zone     = "#1a3322",
    zone_bd  = "#2f9e44",
    blend    = "#2a1f3d",
    blend_bd = "#9c36b5",
    wl       = "#3d2a1a",
    wl_bd    = "#e67700",
    pure     = "#2a1a1a",
    pure_bd  = "#c92a2a",
    text     = "#e8eaf0",
    dim      = "#8892a4",
    arrow    = "#4dabf7",
    bull_g   = "#40c057",
    bear_r   = "#fa5252",
    acc      = "#ffd43b",
)

fig, ax = plt.subplots(figsize=(16, 20))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 16)
ax.set_ylim(0, 20)
ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────
def box(x, y, w, h, fc, ec, radius=0.25, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=1.5, alpha=alpha,
                       zorder=3)
    ax.add_patch(p)
    return p

def txt(x, y, s, size=9, color=None, ha="center", va="center",
        bold=False, mono=False):
    color = color or C["text"]
    family = "monospace" if mono else "sans-serif"
    weight = "bold" if bold else "normal"
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontfamily=family, fontweight=weight, zorder=5)

def arrow(x1, y1, x2, y2, color=None, lw=1.8, style="->"):
    color = color or C["arrow"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0"),
                zorder=4)

def divider(x, y, w, color):
    ax.plot([x, x+w], [y, y], color=color, lw=0.8, ls="--", zorder=4, alpha=0.5)

# ── Title ─────────────────────────────────────────────────────────────────────
box(0.3, 19.0, 15.4, 0.75, C["header"], "#4dabf7", radius=0.3)
txt(8, 19.38, "ML Stock Scoring Pipeline", size=14, bold=True, color="#4dabf7")
txt(8, 19.1,  "LightGBM Ranker  ×  SDZ/SSZ Zone Composite  →  Final Watchlist",
    size=9, color=C["dim"])

# ════════════════════════════════════════════════════════════════════════════════
# 1. FEATURES
# ════════════════════════════════════════════════════════════════════════════════
box(0.5, 16.8, 15, 1.95, C["feat"], C["feat_bd"])
txt(8, 18.55, "RAW FEATURES  (150+ columns per ticker — latest date cross-section)",
    size=9, bold=True, color="#74c0fc")

feat_groups = [
    ("ICT Zones\nOB · BB · FVG", 1.4),
    ("HTF Bias\nBull/Bear\nScore", 3.3),
    ("Trend\nADX · SMA\nDirection", 5.2),
    ("Momentum\n52w high\nRSI", 7.1),
    ("Volume\nATR\nVol ratio", 9.0),
    ("Zone\nSDZ · SSZ\nHTF score", 10.9),
    ("Sector\nETF\nScore", 12.8),
    ("Target\nLabel\n(train only)", 14.5),
]
for label, cx in feat_groups:
    bx = cx - 0.75
    box(bx, 17.0, 1.5, 1.35, "#0d1525", C["feat_bd"], radius=0.2, alpha=0.7)
    txt(cx, 17.67, label, size=7.5, color=C["text"], mono=False)

arrow(8, 16.8, 8, 16.35, color=C["arrow"])

# ════════════════════════════════════════════════════════════════════════════════
# 2. LGBM MODEL
# ════════════════════════════════════════════════════════════════════════════════
box(2.0, 14.4, 8.5, 1.8, C["model"], C["model_bd"])
txt(6.25, 16.0,  "LightGBM RANKER  (LGBMRanker)", size=10, bold=True, color="#74c0fc")
txt(6.25, 15.7,  "objective: rank:ndcg  ·  walk-forward folds  ·  Optuna-tuned hyperparams",
    size=8, color=C["dim"])
txt(6.25, 15.42, "predict(X)  →  raw ranking score per ticker", size=8.5,
    color=C["text"], mono=True)
txt(6.25, 15.12, "\"Is stock A ranked higher than B?\" — not absolute probability",
    size=8, color=C["dim"])
txt(6.25, 14.75, "features_* columns only  ·  missing values filled → 0",
    size=7.5, color=C["dim"])

arrow(6.25, 14.4, 6.25, 13.95, color=C["arrow"])

# ════════════════════════════════════════════════════════════════════════════════
# 3. ENSEMBLE BLEND
# ════════════════════════════════════════════════════════════════════════════════
box(2.0, 12.3, 8.5, 1.55, C["model"], "#0ea5e9", alpha=0.85)
txt(6.25, 13.66, "ENSEMBLE BLEND", size=10, bold=True, color="#7dd3fc")
txt(6.25, 13.35, "lgbm_rank  =  rank01( lgbm_raw )", size=8.5, color=C["text"], mono=True)
txt(6.25, 13.05, "ivol_tilt   =  1 / hist_vol_20d   (low-vol stocks weighted up)",
    size=8, color=C["dim"])
divider(2.2, 12.85, 8.1, "#0ea5e9")
txt(6.25, 12.62, "model_score  =  0.90 × lgbm_rank  +  0.10 × rank01(ivol_tilt)",
    size=9, bold=True, color=C["acc"], mono=True)

# two arrows down from ensemble: one to Final Blend, one to zone branch
arrow(6.25, 12.3, 6.25, 11.85, color=C["arrow"])

# ════════════════════════════════════════════════════════════════════════════════
# 4. ZONE COMPOSITE (right branch)
# ════════════════════════════════════════════════════════════════════════════════
# branch line from ensemble right side
ax.annotate("", xy=(12.8, 13.07), xytext=(10.5, 13.07),
            arrowprops=dict(arrowstyle="-", color=C["arrow"], lw=1.5), zorder=4)
ax.annotate("", xy=(12.8, 11.5), xytext=(12.8, 13.07),
            arrowprops=dict(arrowstyle="->", color=C["arrow"], lw=1.5), zorder=4)

box(11.0, 9.9, 4.5, 1.55, C["zone"], C["zone_bd"])
txt(13.25, 11.25, "ZONE COMPOSITE", size=9, bold=True, color="#69db7c")
txt(13.25, 10.97, "SDZ/SSZ signals only", size=8, color=C["dim"])
divider(11.1, 10.78, 4.3, C["zone_bd"])

# bull signals
bull_sigs = [
    ("sdz_htf_score", "×3.0", C["bull_g"]),
    ("sdz_1wk",       "×2.0", C["bull_g"]),
    ("sdz_1mo",       "×2.5", C["bull_g"]),
    ("sdz_3mo",       "×3.0", C["bull_g"]),
    ("sdz_1y",        "×3.5", C["bull_g"]),
    ("sector_etf_bull","×1.0", "#a9e34b"),
]
row_y = 10.63
for sig, wt, col in bull_sigs:
    txt(12.1, row_y, f"▸ {sig}", size=7, color=col, ha="left", mono=True)
    txt(15.2, row_y, wt,         size=7, color=C["acc"], ha="right", mono=True)
    row_y -= 0.12

# arrow down from zone to final blend
arrow(13.25, 9.9, 10.5, 9.15, color=C["zone_bd"])

# ════════════════════════════════════════════════════════════════════════════════
# 5. FINAL BLEND
# ════════════════════════════════════════════════════════════════════════════════
box(1.0, 7.9, 12.0, 1.95, C["blend"], C["blend_bd"])
txt(7.0, 9.66, "FINAL BLEND", size=10, bold=True, color="#cc5de8")
divider(1.1, 9.4, 11.8, C["blend_bd"])

# pureml column
box(1.2, 7.98, 5.5, 1.3, "#1a0a2e", C["pure_bd"], radius=0.2)
txt(3.95, 9.15, "pureml  (model only)", size=8.5, bold=True, color="#ff6b6b")
txt(3.95, 8.85, "bull = norm( model_score )",      size=8, color=C["text"], mono=True)
txt(3.95, 8.6,  "bear = norm( 1 − model_score )", size=8, color=C["text"], mono=True)
txt(3.95, 8.2,  "weight:  100% model  ·  0% zones", size=7.5, color=C["dim"])

# composite column
box(7.1, 7.98, 5.7, 1.3, "#1f0e35", C["blend_bd"], radius=0.2)
txt(9.95, 9.15, "composite  (model + zones)", size=8.5, bold=True, color="#cc5de8")
txt(9.95, 8.85, "bull = norm( 0.85×norm(model) + 0.15×bull_zone )",
    size=7.5, color=C["text"], mono=True)
txt(9.95, 8.6,  "bear = norm( 0.85×norm(1−model) + 0.15×bear_zone )",
    size=7.5, color=C["text"], mono=True)
txt(9.95, 8.2,  "weight:  85% model  ·  15% zones", size=7.5, color=C["dim"])

arrow(7.0, 7.9, 7.0, 7.35, color=C["arrow"])

# ════════════════════════════════════════════════════════════════════════════════
# 6. RANKING
# ════════════════════════════════════════════════════════════════════════════════
box(2.5, 6.2, 9.0, 1.05, "#1a1a2e", "#6741d9")
txt(7.0, 6.93, "CROSS-SECTIONAL RANKING   (all ~1 550 tickers)", size=9,
    bold=True, color="#9775fa")
txt(7.0, 6.55, "sort by bull_score  →  bull_rank    |    sort by bear_score  →  bear_rank",
    size=8.5, color=C["text"], mono=True)

arrow(7.0, 6.2, 7.0, 5.75, color=C["arrow"])

# ════════════════════════════════════════════════════════════════════════════════
# 7. WATCHLIST FILTER
# ════════════════════════════════════════════════════════════════════════════════
box(1.0, 4.05, 5.6, 1.6, C["wl"], C["wl_bd"])
txt(3.8, 5.45, "MOMENTUM watchlist", size=9, bold=True, color="#ffa94d")
txt(3.8, 5.15, "within 15% of 52-week high", size=8, color=C["text"])
txt(3.8, 4.88, "continuation / breakout plays", size=7.5, color=C["dim"])
divider(1.1, 4.72, 5.4, C["wl_bd"])
txt(3.8, 4.5,  "top-30 bull  +  top-30 bear", size=8, color=C["acc"], bold=True)
txt(3.8, 4.25, "split into Large / Mid / Small", size=7.5, color=C["dim"])

box(8.4, 4.05, 5.6, 1.6, C["wl"], "#c2255c")
txt(11.2, 5.45, "REVERSAL watchlist", size=9, bold=True, color="#ff8787")
txt(11.2, 5.15, "20%+ below 52-week high", size=8, color=C["text"])
txt(11.2, 4.88, "demand zone bounce plays", size=7.5, color=C["dim"])
divider(8.5, 4.72, 5.4, "#c2255c")
txt(11.2, 4.5,  "top-30 bull  +  top-30 bear", size=8, color=C["acc"], bold=True)
txt(11.2, 4.25, "split into Large / Mid / Small", size=7.5, color=C["dim"])

# branch arrows
ax.annotate("", xy=(3.8,  5.65), xytext=(5.2,  5.75),
            arrowprops=dict(arrowstyle="-", color=C["arrow"], lw=1.4), zorder=4)
ax.annotate("", xy=(11.2, 5.65), xytext=(8.8,  5.75),
            arrowprops=dict(arrowstyle="-", color=C["arrow"], lw=1.4), zorder=4)
ax.plot([5.2, 8.8], [5.75, 5.75], color=C["arrow"], lw=1.4, zorder=4)
ax.annotate("", xy=(3.8,  5.65), xytext=(3.8,  5.75),
            arrowprops=dict(arrowstyle="->", color=C["arrow"], lw=1.4), zorder=4)
ax.annotate("", xy=(11.2, 5.65), xytext=(11.2, 5.75),
            arrowprops=dict(arrowstyle="->", color=C["arrow"], lw=1.4), zorder=4)

# ════════════════════════════════════════════════════════════════════════════════
# 8. OUTPUT
# ════════════════════════════════════════════════════════════════════════════════
arrow(3.8,  4.05, 3.8,  3.55, color=C["bull_g"])
arrow(11.2, 4.05, 11.2, 3.55, color=C["bear_r"])

box(0.5, 2.5, 6.5, 1.0, "#0a2318", C["bull_g"], radius=0.25)
txt(3.75, 3.2,  "BULL watchlist  (scores_detail + watchlist_*.csv)", size=8,
    bold=True, color=C["bull_g"])
txt(3.75, 2.82, "model_score  ·  composite_score  ·  bull_rank", size=7.5,
    color=C["text"], mono=True)

box(9.0, 2.5, 6.5, 1.0, "#2a0a0a", C["bear_r"], radius=0.25)
txt(12.25, 3.2,  "BEAR watchlist  (scores_detail + watchlist_*.csv)", size=8,
    bold=True, color=C["bear_r"])
txt(12.25, 2.82, "bear_model_score  ·  bear_composite_score  ·  bear_rank", size=7.5,
    color=C["text"], mono=True)

# ── footnote ─────────────────────────────────────────────────────────────────
box(0.3, 0.15, 15.4, 2.2, C["header"], "#343a4f", radius=0.25)
txt(8, 2.1, "NOTES", size=8.5, bold=True, color=C["dim"])
notes = [
    "▸  norm() = min-max normalisation cross-sectionally across all ~1 550 tickers",
    "▸  rank01() = rank each value, rescale to [0, 1]",
    "▸  Bear score = 1 − model_score  (same model, inverted).  No separate bear model.",
    "▸  pureml variant ignores zones entirely (weights: model=1.0, composite=0.0)",
    "▸  signal_weights.yaml is the single config file — edit weights without code changes",
    "▸  Zone signals (SDZ/SSZ) are the ONLY composite inputs; ICT/ADX/momentum live inside the model",
]
for i, n in enumerate(notes):
    txt(0.6, 1.82 - i*0.265, n, size=7.5, color=C["dim"], ha="left")

plt.tight_layout(pad=0)
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print(f"Saved: {OUT}")
