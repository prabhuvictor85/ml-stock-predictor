"""
preflight_nse_tv.py  —  Pre-flight check before running run_nse_tradingv_local.py
"""
import sys, os
from pathlib import Path
import pandas as pd

OK   = "  [OK]  "
WARN = "  [WARN]"
FAIL = "  [FAIL]"

issues  = []
warnings = []

def chk(label, condition, detail="", fatal=True):
    if condition:
        print(f"{OK}  {label}  {detail}")
    else:
        tag = FAIL if fatal else WARN
        print(f"{tag}  {label}  {detail}")
        (issues if fatal else warnings).append(label)

print("=" * 65)
print("  NSE TradingView Training — Pre-Flight Check")
print("=" * 65)

# ── 1. Dependencies ────────────────────────────────────────────────────────────
print("\n[1] Python dependencies")
import importlib
for pkg, min_ver in [("lightgbm","4.0"), ("optuna","3.0"), ("scipy","1.0"),
                     ("sklearn","1.0"), ("pandas","2.0"), ("numpy","1.0"),
                     ("joblib","1.0"), ("tqdm","4.0"), ("requests","2.0")]:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", "?")
        chk(f"{pkg:<15} {v}", True)
    except ImportError:
        chk(pkg, False, "NOT INSTALLED")

# ── 2. Pipeline modules ────────────────────────────────────────────────────────
print("\n[2] Pipeline modules")
os.chdir(r"C:\Victor\Project\ml-stock-predictor")
sys.path.insert(0, ".")
for mod in ["pipeline.config.nse",
            "pipeline.features.engineer",
            "pipeline.targets.builder",
            "pipeline.models.lgbm_ranker",
            "pipeline.models.ensemble",
            "pipeline.validation.cv"]:
    try:
        importlib.import_module(mod)
        chk(mod, True)
    except Exception as e:
        chk(mod, False, str(e)[:60])

# ── 3. Key files ───────────────────────────────────────────────────────────────
print("\n[3] Key input files")
files = {
    "Constituent CSV" : Path(r"C:\Victor\Learning_charts\stock_lists\constituents_nse_tradingv.csv"),
    "Cap tiers CSV"   : Path(r"C:\Victor\Learning_charts\stock_lists\nse_cap_tiers.csv"),
    "TV data dir"     : Path(r"C:\Victor\Learning_charts\stock_data\tradingview"),
}
for label, p in files.items():
    chk(label, p.exists(), str(p))

# Constituent CSV columns
csv = Path(r"C:\Victor\Learning_charts\stock_lists\constituents_nse_tradingv.csv")
if csv.exists():
    df = pd.read_csv(csv)
    chk("Constituent cols (Symbol, TV_ticker)", "Symbol" in df.columns and "TV_ticker" in df.columns,
        f"{len(df)} rows, cols={list(df.columns)}")

# Cap tiers
ct = Path(r"C:\Victor\Learning_charts\stock_lists\nse_cap_tiers.csv")
if ct.exists():
    df_ct = pd.read_csv(ct)
    dist = df_ct["cap_tier"].value_counts().to_dict()
    chk("Cap tiers (2517 rows)", len(df_ct) == 2517, str(dist))
    chk("Cap tier col exists",   "cap_tier" in df_ct.columns)

# ── 4. TV data files ───────────────────────────────────────────────────────────
print("\n[4] TradingView data files")
tv_dir = Path(r"C:\Victor\Learning_charts\stock_data\tradingview")
if tv_dir.exists():
    files_tv = list(tv_dir.glob("NSE_*_1D_TV_div_adj.csv"))
    chk(f"TV files found", len(files_tv) >= 2000, f"{len(files_tv)} files")

    # Check a sample file format (need ts and c columns)
    sample = files_tv[100] if len(files_tv) > 100 else files_tv[0]
    df_s = pd.read_csv(sample, nrows=3)
    chk("TV file format (ts, o, h, l, c, v)",
        all(col in df_s.columns for col in ["ts","o","h","l","c","v"]),
        f"cols={list(df_s.columns)}")

    # How many have >= 500 rows
    import random
    sample_files = random.sample(files_tv, min(200, len(files_tv)))
    ok_count = sum(1 for f in sample_files if sum(1 for _ in open(f)) > 500)
    pct = ok_count / len(sample_files) * 100
    chk(f">=500 rows (sample check)", pct >= 60,
        f"{pct:.0f}% of sampled files pass (need >60%)", fatal=False)

# ── 5. Benchmark file ──────────────────────────────────────────────────────────
print("\n[5] Benchmark")
# Check what benchmark the script expects
import subprocess
res = subprocess.run(
    ["python", "-c",
     "import sys; sys.path.insert(0,'.'); "
     "from run_nse_tradingv_local import BENCHMARK_FILE; print(BENCHMARK_FILE)"],
    capture_output=True, text=True, cwd=r"C:\Victor\Project\ml-stock-predictor"
)
bm_path = res.stdout.strip()
if bm_path:
    chk("Benchmark file", Path(bm_path).exists(), bm_path, fatal=False)
else:
    print(f"{WARN}  Could not determine benchmark path: {res.stderr[:80]}")

# ── 6. Output / artefacts directories writeable ────────────────────────────────
print("\n[6] Output directories (will be created if missing)")
for d in [Path("output/nse_tradingv"), Path("artefacts/nse_tradingv"),
          Path("artefacts/nse_tradingv/momentum"), Path("artefacts/nse_tradingv/reversal")]:
    try:
        d.mkdir(parents=True, exist_ok=True)
        chk(str(d), True, "exists / created")
    except Exception as e:
        chk(str(d), False, str(e))

# ── 7. Disk space ──────────────────────────────────────────────────────────────
print("\n[7] System resources")
import shutil, subprocess
disk = shutil.disk_usage(r"C:\\")
free_gb = disk.free / 1e9
chk(f"Disk space (C:)", free_gb >= 5, f"{free_gb:.1f} GB free (need >= 5 GB)", fatal=False)

# RAM via wmic
try:
    res = subprocess.run(["wmic","OS","get","FreePhysicalMemory","/Value"],
                         capture_output=True, text=True)
    free_kb = int([l for l in res.stdout.splitlines() if "FreePhysicalMemory" in l][0].split("=")[1])
    free_ram_gb = free_kb / 1e6
    chk(f"Available RAM", free_ram_gb >= 8, f"{free_ram_gb:.1f} GB free (need >= 8 GB)", fatal=False)
except Exception:
    print(f"{WARN}  Could not check RAM")

# ── 8. No stale lock files ─────────────────────────────────────────────────────
print("\n[8] Stale artefacts")
ckpt = Path("artefacts/nse_tradingv/checkpoints")
if ckpt.exists():
    ckpts = list(ckpt.glob("*.pkl"))
    if ckpts:
        print(f"{WARN}  Old checkpoints found: {len(ckpts)} files in {ckpt}")
        print(f"        These will be reused. Delete them if you want a completely clean run.")
        warnings.append("old checkpoints")
    else:
        chk("No stale checkpoints", True)
else:
    chk("No stale checkpoints", True, "(dir doesn't exist yet)")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
if not issues:
    print(f"  RESULT: READY TO RUN  ({len(warnings)} warning(s))")
    for w in warnings:
        print(f"    Warning: {w}")
    print()
    print("  Command:")
    print("  python run_nse_tradingv_local.py --mode all --n_folds 8 --n_trials 25")
else:
    print(f"  RESULT: NOT READY — {len(issues)} issue(s) must be fixed:")
    for i in issues:
        print(f"    FAIL: {i}")
print("=" * 65)
