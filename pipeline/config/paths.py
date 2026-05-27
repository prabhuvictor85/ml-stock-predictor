"""
paths.py — Centralized path resolution for the ML stock predictor pipeline.

Loads paths from paths.yaml at the project root, with optional environment
variable overrides. Use this module instead of hardcoding absolute paths
in any script.

Usage
─────
    from pipeline.config.paths import PATHS

    panel_dir = PATHS.stock_data.nse_tv
    cap_tiers = PATHS.stock_lists.nse_cap_tiers

Environment variable overrides take precedence over the YAML file. See
paths.yaml for the full list of supported env vars.

Resolution order:
  1. Environment variable (if set)
  2. paths.yaml entry (with {data_root}/{project_root} substitution)
  3. Built-in default (Windows convention — only used if paths.yaml is missing)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


# ── Built-in fallback defaults (used only if paths.yaml is missing) ──────────
_FALLBACK_DATA_ROOT    = "C:/Victor/Learning_charts"
_FALLBACK_PROJECT_ROOT = "C:/Victor/Project/ml-stock-predictor"


# ── Env var → YAML key mapping ───────────────────────────────────────────────
_ENV_OVERRIDES = {
    "ML_DATA_ROOT":                 ("data_root",),
    "ML_PROJECT_ROOT":              ("project_root",),
    "ML_STOCK_LIST_NSE_LOCAL":      ("stock_lists", "nse_local"),
    "ML_STOCK_LIST_NSE_TV":         ("stock_lists", "nse_tv"),
    "ML_STOCK_LIST_NSE_CAP_TIERS":  ("stock_lists", "nse_cap_tiers"),
    "ML_STOCK_LIST_US_COMBINED":    ("stock_lists", "us_combined"),
    "ML_STOCK_LIST_LISTS_DIR":      ("stock_lists", "lists_dir"),
    "ML_STOCK_DATA_NSE_LOCAL":      ("stock_data", "nse_local"),
    "ML_STOCK_DATA_NSE_TV":         ("stock_data", "nse_tv"),
    "ML_STOCK_DATA_US":             ("stock_data", "us"),
    "ML_STOCK_DATA_US_ALT":         ("stock_data", "us_alt"),
}


@dataclass(frozen=True)
class _StockLists:
    nse_local:     Path
    nse_tv:        Path
    nse_cap_tiers: Path
    us_combined:   Path
    lists_dir:     Path


@dataclass(frozen=True)
class _StockData:
    nse_local: Path
    nse_tv:    Path
    us:        Path
    us_alt:    Path


@dataclass(frozen=True)
class _Paths:
    data_root:    Path
    project_root: Path
    stock_lists:  _StockLists
    stock_data:   _StockData


def _find_yaml() -> Optional[Path]:
    """Locate paths.yaml. Walks up from this file until found."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "paths.yaml"
        if candidate.exists():
            return candidate
    return None


def _set_nested(d: dict, keys: tuple, value: str) -> None:
    """Set d[keys[0]][keys[1]]... = value, creating intermediates as needed."""
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _substitute(template: str, data_root: str, project_root: str) -> str:
    """Substitute {data_root} and {project_root} placeholders in path strings."""
    return template.format(data_root=data_root, project_root=project_root)


def _load() -> _Paths:
    """Load paths from YAML + env vars and return a frozen _Paths dataclass."""
    yaml_path = _find_yaml()
    if yaml_path and yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # Apply env var overrides (highest priority)
    for env_var, key_path in _ENV_OVERRIDES.items():
        val = os.environ.get(env_var)
        if val:
            _set_nested(cfg, key_path, val)

    # Resolve roots first (needed for {data_root} / {project_root} substitution)
    data_root    = cfg.get("data_root",    _FALLBACK_DATA_ROOT)
    project_root = cfg.get("project_root", _FALLBACK_PROJECT_ROOT)

    def _resolve(template: str) -> Path:
        return Path(_substitute(template, data_root, project_root)).resolve()

    sl = cfg.get("stock_lists", {})
    sd = cfg.get("stock_data",  {})

    return _Paths(
        data_root    = Path(data_root).resolve(),
        project_root = Path(project_root).resolve(),
        stock_lists  = _StockLists(
            nse_local     = _resolve(sl.get("nse_local",     "{data_root}/stock_lists/constituentsi.csv")),
            nse_tv        = _resolve(sl.get("nse_tv",        "{data_root}/stock_lists/constituents_nse_tradingv.csv")),
            nse_cap_tiers = _resolve(sl.get("nse_cap_tiers", "{data_root}/stock_lists/nse_cap_tiers.csv")),
            us_combined   = _resolve(sl.get("us_combined",   "{data_root}/stock_lists/constituents_us_combined.csv")),
            lists_dir     = _resolve(sl.get("lists_dir",     "{data_root}/stock_lists")),
        ),
        stock_data = _StockData(
            nse_local = _resolve(sd.get("nse_local", "{data_root}/stock_data")),
            nse_tv    = _resolve(sd.get("nse_tv",    "{data_root}/stock_data/tradingview")),
            us        = _resolve(sd.get("us",        "{data_root}/stock_data/us_stocks")),
            us_alt    = _resolve(sd.get("us_alt",    "{data_root}/us_data")),
        ),
    )


# Singleton — imported once at module load
PATHS: _Paths = _load()
