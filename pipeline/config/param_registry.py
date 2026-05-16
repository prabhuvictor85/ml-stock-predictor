"""
pipeline/config/param_registry.py

Versioned parameter registry.

Every set of hyperparameters used in training is stored with:
  - a version hash (SHA-256 of the param dict)
  - timestamp
  - source ('optuna', 'manual', 'default')
  - best validation metric at time of registration

This means you can always reproduce any historical run and compare
parameter generations without guessing which params produced which model.

Usage
─────
    from pipeline.config.param_registry import ParamRegistry

    registry = ParamRegistry("artefacts/param_registry.json")

    # After Optuna
    version = registry.register(
        params=study.best_params,
        source="optuna",
        metric_name="ndcg@10",
        metric_value=study.best_value,
        notes=f"n_trials={n_trials}, n_folds={n_folds}",
    )
    print(f"Registered as version: {version}")

    # At inference — load the best known params
    best = registry.best(metric_name="ndcg@10")
    ranker = LGBMRanker(params=best["params"])

    # Load a specific version
    params_v3 = registry.get(version="abc12345")["params"]
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.utils.logging import get_logger

log = get_logger(__name__)

# Params that should NOT be part of the version hash
# (they change per-run without affecting model behaviour)
_EXCLUDE_FROM_HASH = {"seed", "random_seed", "verbosity", "n_jobs"}


class ParamRegistry:
    """
    JSON-backed versioned parameter store.

    Thread-safe for single-process use (read-modify-write with file lock
    is not implemented — add fcntl locking if running parallel HPO workers
    that all write to the same registry).

    Parameters
    ----------
    path : path to the JSON registry file (created if absent)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: Dict[str, Any] = self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def register(
        self,
        params: Dict[str, Any],
        source: str = "manual",
        metric_name: str = "ndcg@10",
        metric_value: float = 0.0,
        feature_schema_hash: Optional[str] = None,
        notes: str = "",
    ) -> str:
        """
        Register a parameter set. Returns the version hash string.

        If an identical param set was already registered, returns the
        existing version hash without creating a duplicate entry.
        """
        version = self._hash(params)
        if version in self._data["versions"]:
            log.info(f"ParamRegistry: params already registered as version {version[:8]}")
            return version

        entry = {
            "version":             version,
            "short":               version[:8],
            "registered_at":       datetime.utcnow().isoformat(),
            "source":              source,
            "metric_name":         metric_name,
            "metric_value":        round(float(metric_value), 6),
            "feature_schema_hash": feature_schema_hash or "",
            "notes":               notes,
            "params":              {k: self._serialise(v) for k, v in params.items()},
        }
        self._data["versions"][version] = entry
        self._data["latest"] = version
        self._save()
        log.info(f"ParamRegistry: registered version {version[:8]}  "
                 f"({metric_name}={metric_value:.4f}, source={source})")
        return version

    def get(self, version: str) -> Dict[str, Any]:
        """
        Retrieve a registered parameter set by version hash (full or short).

        Raises KeyError if not found.
        """
        # Allow 8-char short lookup
        if len(version) == 8:
            matches = [v for v in self._data["versions"] if v.startswith(version)]
            if len(matches) == 1:
                version = matches[0]
            elif len(matches) == 0:
                raise KeyError(f"No version starting with '{version}' in registry.")
            else:
                raise KeyError(f"Ambiguous short version '{version}' — {len(matches)} matches.")
        if version not in self._data["versions"]:
            raise KeyError(f"Version '{version}' not in registry.")
        entry = self._data["versions"][version].copy()
        entry["params"] = self._deserialise_params(entry["params"])
        return entry

    def best(self, metric_name: str = "ndcg@10", higher_is_better: bool = True) -> Dict[str, Any]:
        """Return the registered entry with the best metric value."""
        candidates = [
            v for v in self._data["versions"].values()
            if v.get("metric_name") == metric_name
        ]
        if not candidates:
            raise ValueError(f"No registered versions with metric '{metric_name}'.")
        best_entry = sorted(
            candidates,
            key=lambda x: x["metric_value"],
            reverse=higher_is_better,
        )[0]
        result = best_entry.copy()
        result["params"] = self._deserialise_params(result["params"])
        log.info(f"ParamRegistry.best: version {result['short']}  "
                 f"{metric_name}={result['metric_value']:.4f}  "
                 f"(registered {result['registered_at'][:10]})")
        return result

    def latest(self) -> Dict[str, Any]:
        """Return the most recently registered entry."""
        latest_version = self._data.get("latest")
        if not latest_version:
            raise ValueError("Registry is empty.")
        return self.get(latest_version)

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all versions as a list, newest first."""
        versions = list(self._data["versions"].values())
        versions.sort(key=lambda x: x["registered_at"], reverse=True)
        return [
            {
                "short":        v["short"],
                "registered_at":v["registered_at"][:19],
                "source":       v["source"],
                "metric":       f"{v['metric_name']}={v['metric_value']:.4f}",
                "notes":        v["notes"][:60],
            }
            for v in versions
        ]

    def compare(self, v1: str, v2: str) -> Dict[str, Any]:
        """Show param diff between two versions."""
        e1 = self.get(v1)
        e2 = self.get(v2)
        p1, p2 = e1["params"], e2["params"]
        all_keys = set(p1) | set(p2)
        diffs = {}
        for k in sorted(all_keys):
            val1 = p1.get(k, "<missing>")
            val2 = p2.get(k, "<missing>")
            if val1 != val2:
                diffs[k] = {"v1": val1, "v2": val2}
        return {
            "v1":          e1["short"],
            "v2":          e2["short"],
            "metric_diff": round(e2["metric_value"] - e1["metric_value"], 6),
            "param_diffs": diffs,
        }

    def delete(self, version: str, confirm: bool = False) -> None:
        """Delete a version. Requires confirm=True to prevent accidents."""
        if not confirm:
            raise RuntimeError("Pass confirm=True to delete a registered version.")
        entry = self.get(version)
        full_version = entry["version"]
        del self._data["versions"][full_version]
        if self._data.get("latest") == full_version:
            remaining = list(self._data["versions"])
            self._data["latest"] = remaining[-1] if remaining else None
        self._save()
        log.info(f"ParamRegistry: deleted version {full_version[:8]}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                log.warning(f"ParamRegistry: corrupt file at {self.path} — starting fresh.")
        return {"versions": {}, "latest": None}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    def _hash(self, params: Dict[str, Any]) -> str:
        """SHA-256 of the serialised param dict (excluding noise keys)."""
        filtered = {
            k: self._serialise(v)
            for k, v in sorted(params.items())
            if k not in _EXCLUDE_FROM_HASH
        }
        raw = json.dumps(filtered, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _serialise(v: Any) -> Any:
        """Make a value JSON-serialisable."""
        if isinstance(v, (int, float, str, bool, type(None))):
            return v
        return str(v)

    @staticmethod
    def _deserialise_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to coerce string-serialised numbers back to native types."""
        result = {}
        for k, v in params.items():
            if isinstance(v, str):
                for cast in (int, float):
                    try:
                        result[k] = cast(v)
                        break
                    except (ValueError, TypeError):
                        pass
                else:
                    result[k] = v
            else:
                result[k] = v
        return result