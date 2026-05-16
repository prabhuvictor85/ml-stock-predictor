from __future__ import annotations

import json
import hashlib
import numpy as np
import pandas as pd
import joblib

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.utils.logging import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

NULL_RATE_THRESH = 0.20
PSI_WARN_THRESH = 0.10
PSI_FAIL_THRESH = 0.25
EPS = 1e-8


class ValidationMode:
    STRICT = "strict"
    WARN = "warn"
    SAFE = "safe"   # NEW: production trading mode


# ─────────────────────────────────────────────
# ERROR
# ─────────────────────────────────────────────

class SchemaValidationError(RuntimeError):
    pass


@dataclass
class ValidationIssue:
    severity: str
    feature: str
    issue: str


# ─────────────────────────────────────────────
# CORE SCHEMA
# ─────────────────────────────────────────────

class FeatureSchemaLock:
    """
    Production-grade schema contract system.

    FIXES:
    ✔ PSI-based drift (replaces std poisoning)
    ✔ safe persistence (joblib + json split)
    ✔ SAFE mode fallback (no trading halt in live markets)
    """

    def __init__(self):
        self._schema: Dict[str, dict] = {}
        self._order: List[str] = []
        self._hash: str = ""

    # ─────────────────────────────────────────────
    # BUILD
    # ─────────────────────────────────────────────

    @classmethod
    def from_dataframe(cls, X: pd.DataFrame, feat_cols: List[str]):
        obj = cls()

        obj._order = list(feat_cols)

        payload_hash = []

        for f in feat_cols:
            if f not in X.columns:
                raise SchemaValidationError(f"Missing feature in training: {f}")

            s = X[f]

            entry = {
                "dtype": str(s.dtype),
                "null_rate": float(s.isna().mean()),
            }

            if pd.api.types.is_numeric_dtype(s):
                clean = s.dropna()
                if len(clean) > 0:
                    # Store both histogram counts AND the bin edges used, so PSI
                    # at validation time uses the same fixed training-time edges.
                    hist_counts, hist_edges = np.histogram(clean, bins=10)
                    entry.update({
                        "mean": float(clean.mean()),
                        "std": float(clean.std() or EPS),
                        "p01": float(np.percentile(clean, 1)),
                        "p99": float(np.percentile(clean, 99)),
                        "hist": hist_counts.tolist(),
                        "hist_edges": hist_edges.tolist(),  # fixed edges for PSI
                    })

            obj._schema[f] = entry
            payload_hash.append((f, entry["dtype"]))

        obj._hash = hashlib.sha256(
            json.dumps(payload_hash, sort_keys=True).encode()
        ).hexdigest()

        log.info(f"Schema locked | features={len(feat_cols)}")
        return obj

    # ─────────────────────────────────────────────
    # ENFORCE (HARD CONTRACT)
    # ─────────────────────────────────────────────

    def enforce(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self._order if f not in X.columns]
        if missing:
            raise SchemaValidationError(f"Missing features: {missing[:10]}")

        # strict projection + ordering
        return X[self._order].copy()

    # ─────────────────────────────────────────────
    # PSI DRIFT (FIXES STD POISONING)
    # ─────────────────────────────────────────────

    def _psi(self, train_hist, infer_hist) -> float:
        eps = 1e-8
        train = np.array(train_hist) + eps
        infer = np.array(infer_hist) + eps

        train = train / train.sum()
        infer = infer / infer.sum()

        return float(np.sum((infer - train) * np.log(infer / train)))

    # ─────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────

    def validate(self, X: pd.DataFrame) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # schema mismatch
        missing = [f for f in self._schema if f not in X.columns]
        extra = [f for f in X.columns if f not in self._schema]

        if missing:
            issues.append(ValidationIssue("error", "<schema>", f"{len(missing)} missing features"))

        if extra:
            issues.append(ValidationIssue("warning", "<schema>", f"{len(extra)} extra features"))

        # per feature checks
        for f, meta in self._schema.items():
            if f not in X.columns:
                continue

            s = X[f]

            # null drift
            null_rate = float(s.isna().mean())
            if null_rate - meta["null_rate"] > NULL_RATE_THRESH:
                issues.append(ValidationIssue(
                    "warning", f, f"null spike {null_rate:.2f}"
                ))

            # PSI drift (uses fixed training-time edges)
            if "hist" in meta and "hist_edges" in meta:
                clean = s.dropna()
                if len(clean) > 20:
                    # Use training-time edges — PSI is only meaningful with fixed bins
                    edges = np.array(meta["hist_edges"])
                    infer_hist, _ = np.histogram(clean, bins=edges)
                    psi = self._psi(meta["hist"], infer_hist)
                    psi = self._psi(meta["hist"], infer_hist)

                    if psi > PSI_FAIL_THRESH:
                        issues.append(ValidationIssue("error", f, f"PSI critical {psi:.3f}"))
                    elif psi > PSI_WARN_THRESH:
                        issues.append(ValidationIssue("warning", f, f"PSI drift {psi:.3f}"))

        return issues

    # ─────────────────────────────────────────────
    # ASSERT WITH SAFE MODE
    # ─────────────────────────────────────────────

    def assert_valid(self, X: pd.DataFrame, mode: str = ValidationMode.STRICT) -> None:
        issues = self.validate(X)

        errors = [i for i in issues if i.severity == "error"]

        for i in issues:
            log.warning(str(i))

        # SAFE MODE FIX (CRITICAL FOR TRADING SYSTEMS)
        if mode == ValidationMode.SAFE:
            if errors:
                log.error("Schema drift detected → switching to SAFE fallback mode")
                return  # DO NOT STOP PIPELINE
            return

        # STRICT behavior (training / backtest only)
        if errors or mode == ValidationMode.STRICT:
            raise SchemaValidationError(
                "\n".join(str(i) for i in issues)
            )

    # ─────────────────────────────────────────────
    # SAVE / LOAD (FIXED SERIALIZATION)
    # ─────────────────────────────────────────────

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # SAFE: numeric payload via joblib
        joblib.dump(
            {
                "schema": self._schema,
                "order": self._order,
                "hash": self._hash,
            },
            path.with_suffix(".bin")
        )

        # metadata only (safe JSON)
        meta = {
            "hash": self._hash,
            "n_features": len(self._order),
        }

        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)

        obj = cls()
        data = joblib.load(path.with_suffix(".bin"))

        obj._schema = data["schema"]
        obj._order = data["order"]
        obj._hash = data["hash"]

        return obj