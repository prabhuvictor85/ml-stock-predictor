"""Smoke tests for paths.py — config loader behavior."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_paths_module_imports():
    """The module must load without error."""
    from pipeline.config.paths import PATHS
    assert PATHS is not None


def test_paths_have_expected_attrs():
    """All declared paths exist on the PATHS singleton."""
    from pipeline.config.paths import PATHS
    # Roots
    assert isinstance(PATHS.data_root, Path)
    assert isinstance(PATHS.project_root, Path)
    # Stock lists
    for attr in ["nse_local", "nse_tv", "nse_cap_tiers", "us_combined", "lists_dir"]:
        assert hasattr(PATHS.stock_lists, attr), f"Missing PATHS.stock_lists.{attr}"
        assert isinstance(getattr(PATHS.stock_lists, attr), Path)
    # Stock data
    for attr in ["nse_local", "nse_tv", "us", "us_alt"]:
        assert hasattr(PATHS.stock_data, attr), f"Missing PATHS.stock_data.{attr}"
        assert isinstance(getattr(PATHS.stock_data, attr), Path)


def test_paths_yaml_substitution():
    """Templates like {data_root}/... must be substituted."""
    from pipeline.config.paths import PATHS
    # If substitution didn't happen, paths would still contain literal "{data_root}"
    assert "{data_root}" not in str(PATHS.stock_lists.nse_tv)
    assert "{project_root}" not in str(PATHS.project_root)


def test_paths_env_var_override(monkeypatch, tmp_path):
    """Setting ML_DATA_ROOT should override the YAML value."""
    custom_root = tmp_path / "fake_data"
    monkeypatch.setenv("ML_DATA_ROOT", str(custom_root))
    # Force re-import to pick up the env var
    import importlib
    from pipeline.config import paths as paths_mod
    importlib.reload(paths_mod)
    assert paths_mod.PATHS.data_root == custom_root.resolve()
    # Cleanup: reload again without env var so other tests see the YAML values
    monkeypatch.delenv("ML_DATA_ROOT", raising=False)
    importlib.reload(paths_mod)
