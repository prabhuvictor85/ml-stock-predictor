"""Checkpoint manifest guard.

The panel checkpoints (panel_features.pkl / panel_targets.pkl) used to be
trusted on pure file-existence: nothing checked whether the pickle matched the
CURRENT feature code, env recipe gates, or price data. That cut both ways —
silent stale reuse (a run "resumes" onto last week's feature recipe) or
paranoid manual deletion (a 2h rebuild that wasn't needed).

A manifest written beside the checkpoint records what the panel was built
from; on load, mismatch => the checkpoint is IGNORED with a loud reason and
the panel rebuilds. The guard can only ever force a rebuild you'd have wanted
anyway — it never widens what gets trusted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Env vars that change the panel recipe: a checkpoint built under different
# values is a DIFFERENT panel even if the code and data are unchanged.
_RECIPE_ENV = [
    "PHASE4_FEATURES", "PIVOT_FEATURES",
    "TARGET_HORIZONS", "TARGET_TWAP_WINDOW", "TARGET_DIAGNOSTICS",
]

# Source whose changes invalidate a features/targets checkpoint.
_CODE_ROOTS = ["features", "targets"]


def _code_fingerprint() -> str:
    base = Path(__file__).resolve().parent.parent  # pipeline/
    h = hashlib.sha256()
    for root in _CODE_ROOTS:
        for p in sorted((base / root).rglob("*.py")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _data_fingerprint(data_dir) -> dict:
    """Cheap freshness proxy: CSV count + newest mtime. A re-download changes
    mtimes; an added/delisted ticker changes the count."""
    try:
        entries = [e for e in os.scandir(data_dir) if e.name.endswith(".csv")]
    except OSError:
        return {"n_csv": -1, "max_mtime": 0}
    return {
        "n_csv": len(entries),
        "max_mtime": int(max((e.stat().st_mtime for e in entries), default=0)),
    }


def compute_manifest(data_dir) -> dict:
    return {
        "code_sha":  _code_fingerprint(),
        "data":      _data_fingerprint(data_dir),
        "env":       {k: os.environ.get(k, "") for k in _RECIPE_ENV},
    }


def manifest_ok(manifest_path, current: dict) -> tuple[bool, str]:
    """(True, "") when the stored manifest matches `current`; else a reason."""
    p = Path(manifest_path)
    if not p.exists():
        return False, "no manifest recorded for this checkpoint (pre-guard build)"
    try:
        stored = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return False, f"manifest unreadable ({e})"
    for key in ("code_sha", "data", "env"):
        if stored.get(key) != current.get(key):
            return False, (f"{key} changed since checkpoint was built "
                           f"(stored={stored.get(key)!r} now={current.get(key)!r})")
    return True, ""


def write_manifest(manifest_path, current: dict) -> None:
    Path(manifest_path).write_text(json.dumps(current, indent=2))
