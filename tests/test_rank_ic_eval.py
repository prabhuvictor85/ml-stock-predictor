"""rank_ic_eval feval: exact equivalence with the per-group scipy spearmanr
reference (the implementation it replaced), plus the fail-loud contract for
ungrouped datasets."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest
from scipy.stats import spearmanr

from pipeline.models.lgbm_ranker import rank_ic_eval


def _dataset(labels: np.ndarray, groups: list[int] | None) -> lgb.Dataset:
    X = np.random.default_rng(0).normal(size=(len(labels), 3))
    d = lgb.Dataset(X, label=labels, group=groups, free_raw_data=False)
    d.construct()
    return d


def _scipy_reference(preds, labels, groups) -> float:
    """The original per-group loop this feval replaced."""
    ics, ptr = [], 0
    for g in groups:
        p, l = preds[ptr:ptr + g], labels[ptr:ptr + g]
        ptr += g
        if len(p) > 1 and np.std(p) > 1e-9:
            ic, _ = spearmanr(p, l)
            if not np.isnan(ic):
                ics.append(float(ic))
    return float(np.mean(ics)) if ics else 0.0


def test_matches_scipy_on_tie_heavy_binned_labels():
    rng    = np.random.default_rng(7)
    groups = [50, 120, 5, 300, 2]
    n      = sum(groups)
    labels = rng.integers(0, 100, n).astype(float)   # binned labels → heavy ties
    preds  = rng.normal(size=n)

    name, val, higher_better = rank_ic_eval(preds, _dataset(labels, groups))

    assert name == "rank_ic_binned"
    assert higher_better is True
    assert np.isclose(val, _scipy_reference(preds, labels, groups), atol=1e-12)


def test_degenerate_groups_excluded_like_scipy():
    groups = [4, 3, 2]
    labels = np.array([0, 1, 2, 3,  5, 5, 5,  1, 2], dtype=float)  # grp2: const labels
    preds  = np.array([.1, .4, .2, .9,  .3, .1, .2,  .7, .7])      # grp3: const preds

    _, val, _ = rank_ic_eval(preds, _dataset(labels, groups))

    assert np.isclose(val, _scipy_reference(preds, labels, groups), atol=1e-12)


def test_ungrouped_dataset_fails_loud():
    labels = np.arange(6, dtype=float)
    d = _dataset(labels, groups=None)
    with pytest.raises(ValueError, match="grouped"):
        rank_ic_eval(np.arange(6, dtype=float), d)
