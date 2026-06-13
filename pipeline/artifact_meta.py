"""
Artifact metadata sidecar — answers "how was this model trained?" without
unpickling anything or attribute-sniffing (the nan_native_ workaround).

Written as artefact_meta.json next to ensemble.pkl by every training run.
Pure-stdlib, best-effort: metadata failure must never fail a training run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Bump when the artifact contract changes (what is pickled, how it must be
# loaded, or how scores must be interpreted).
ARTIFACT_SCHEMA_VERSION = 1


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _lib_versions() -> dict:
    versions = {}
    for lib in ("lightgbm", "pandas", "numpy", "sklearn"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[lib] = "not-installed"
    return versions


def write_artifact_meta(
    art_dir: Path,
    *,
    mode: str,
    final_features: List[str],
    panel: pd.DataFrame,
    nan_native: bool = True,
    extra: Optional[dict] = None,
) -> Optional[Path]:
    """
    Write artefact_meta.json into art_dir. Returns the path, or None on
    failure (never raises — metadata must not break training).

    feature_hash is sha256 of the newline-joined selected feature list:
    two artifact sets with the same hash were trained on the same feature
    schema, regardless of pickle internals.
    """
    try:
        dates = panel.index.get_level_values("date")
        tickers = panel.index.get_level_values("ticker")
        feature_blob = "\n".join(final_features)
        meta = {
            "schema_version":  ARTIFACT_SCHEMA_VERSION,
            "created_utc":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit":      _git_commit(),
            "mode":            mode,
            "nan_native":      nan_native,
            "feature_count":   len(final_features),
            "feature_hash":    hashlib.sha256(feature_blob.encode()).hexdigest()[:16],
            "train_data": {
                "date_min":    str(dates.min().date()),
                "date_max":    str(dates.max().date()),
                "n_rows":      int(len(panel)),
                "n_tickers":   int(tickers.nunique()),
            },
            "lib_versions":    _lib_versions(),
        }
        if extra:
            meta.update(extra)
        path = Path(art_dir) / "artefact_meta.json"
        path.write_text(json.dumps(meta, indent=2))
        return path
    except Exception as e:
        print(f"      [warn] Could not write artefact_meta.json: {e}")
        return None
