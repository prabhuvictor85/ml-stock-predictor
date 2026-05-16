# pipeline/calibration/probability_calibration.py

from __future__ import annotations

import json
import numpy as np
import joblib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from sklearn.isotonic import IsotonicRegression
from pipeline.utils.logging import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────
# CONFIG / POLICY LAYER
# ─────────────────────────────────────────────
ECE_REJECT_THRESHOLD = 0.05
N_BINS = 10
EPS = 1e-6
MIN_SLOPE = 1e-4


# ─────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────
@dataclass
class CalibrationMetadata:
    model_version: str
    raw_ece: float
    calibrated_ece: float
    n_samples: int
    fit_timestamp: str


# ─────────────────────────────────────────────
# VALIDATION LAYER
# ─────────────────────────────────────────────
class CalibrationValidation:

    @staticmethod
    def check_inputs(probs, labels):
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels)

        if len(probs) != len(labels):
            raise ValueError("shape mismatch")

        if not np.all(np.isfinite(probs)):
            raise ValueError("NaN/Inf in probabilities")

        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("labels must be binary (0/1)")

        return probs, labels


# ─────────────────────────────────────────────
# CORE CALIBRATION ENGINE
# ─────────────────────────────────────────────
class ProbabilityCalibrator:

    def __init__(self, model_version: str = "v1"):
        self.iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self.model_version = model_version
        self.fitted = False

        # extrapolation params
        self.x_min = self.x_max = None
        self.y_min = self.y_max = None
        self.slope_l = self.slope_r = None

        self.meta: Optional[CalibrationMetadata] = None

    # ── FIT ───────────────────────────────────
    def fit(self, probs, labels):

        probs, labels = CalibrationValidation.check_inputs(probs, labels)

        self.iso.fit(probs, labels)
        self._compute_slopes()

        self.fitted = True   # must be True before calling transform()
        cal = self.transform(probs)

        raw_ece = self._ece(probs, labels)
        cal_ece = self._ece(cal, labels)

        if cal_ece > ECE_REJECT_THRESHOLD:
            raise ValueError(
                f"Calibration rejected: ECE={cal_ece:.4f}"
            )

        self.meta = CalibrationMetadata(
            model_version=self.model_version,
            raw_ece=raw_ece,
            calibrated_ece=cal_ece,
            n_samples=len(probs),
            fit_timestamp=datetime.utcnow().isoformat(),
        )

        log.info(f"Calibration OK | {raw_ece:.4f} → {cal_ece:.4f}")
        return self

    # ── TRANSFORM ─────────────────────────────
    def transform(self, probs):

        if not self.fitted:
            raise RuntimeError("Calibrator not fitted")

        probs = np.asarray(probs, dtype=float)

        y = self.iso.transform(probs)

        out = y.copy()

        # LEFT extrapolation
        left = probs < self.x_min
        out[left] = self.y_min + self.slope_l * (probs[left] - self.x_min)

        # RIGHT extrapolation
        right = probs > self.x_max
        out[right] = self.y_max + self.slope_r * (probs[right] - self.x_max)

        return np.clip(out, 0, 1)

    # ── VALIDATE AND APPLY ────────────────────
    def validate_and_apply(self, probs, labels):
        """
        Apply calibration; if the resulting ECE exceeds ECE_REJECT_THRESHOLD,
        log a warning and return the raw probs instead of the calibrated ones.
        """
        cal = self.transform(probs)
        cal_ece = self._ece(cal, np.asarray(labels, dtype=float))
        if cal_ece > ECE_REJECT_THRESHOLD:
            log.warning(
                f"validate_and_apply: calibrated ECE={cal_ece:.4f} exceeds "
                f"threshold {ECE_REJECT_THRESHOLD} — returning raw probabilities."
            )
            return np.asarray(probs, dtype=float)
        return cal

    # ── ECE ───────────────────────────────────
    def _ece(self, probs, labels):

        bins = np.linspace(0, 1, N_BINS + 1)
        ece = 0.0
        n = len(probs)

        for i in range(N_BINS):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            if mask.sum() == 0:
                continue

            acc = labels[mask].mean()
            conf = probs[mask].mean()
            ece += (mask.sum() / n) * abs(acc - conf)

        return float(ece)

    # ── SLOPE CONTROL (TAIL SAFETY) ───────────
    def _compute_slopes(self):
        xs = self.iso.X_thresholds_
        ys = self.iso.y_thresholds_

        self.x_min, self.x_max = xs[0], xs[-1]
        self.y_min, self.y_max = ys[0], ys[-1]

        self.slope_l = max((ys[1]-ys[0])/(xs[1]-xs[0]+EPS), MIN_SLOPE)
        self.slope_r = max((ys[-1]-ys[-2])/(xs[-1]-xs[-2]+EPS), MIN_SLOPE)

    # ── SAVE / LOAD ───────────────────────────
    def save(self, path: Path):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        joblib.dump(self.iso, path / "iso.joblib")

        meta = {
            "model_version": self.model_version,
            "slopes": {
                "l": self.slope_l,
                "r": self.slope_r
            },
            "meta": self.meta.__dict__ if self.meta else None
        }

        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path):
        obj = cls()
        obj.iso = joblib.load(path / "iso.joblib")
        meta = json.loads((path / "meta.json").read_text())

        obj.model_version = meta["model_version"]
        obj.slope_l = meta["slopes"]["l"]
        obj.slope_r = meta["slopes"]["r"]
        obj.fitted = True

        return obj