"""Checkpoint manifest guard: trusted only when code + recipe env + data match."""
from __future__ import annotations

from pipeline.utils.checkpoint_manifest import (
    compute_manifest, manifest_ok, write_manifest,
)


def test_roundtrip_matches(tmp_path, monkeypatch):
    monkeypatch.delenv("PHASE4_FEATURES", raising=False)
    data = tmp_path / "csvs"; data.mkdir()
    (data / "AAA-1d.csv").write_text("Date,Close\n2020-01-01,1\n")
    mf = compute_manifest(data)
    path = tmp_path / "ckpt_manifest.json"
    write_manifest(path, mf)
    ok, reason = manifest_ok(path, compute_manifest(data))
    assert ok, reason


def test_missing_manifest_rejected(tmp_path):
    ok, reason = manifest_ok(tmp_path / "nope.json", compute_manifest(tmp_path))
    assert not ok and "no manifest" in reason


def test_env_gate_change_invalidates(tmp_path, monkeypatch):
    monkeypatch.delenv("PHASE4_FEATURES", raising=False)
    path = tmp_path / "ckpt_manifest.json"
    write_manifest(path, compute_manifest(tmp_path))
    monkeypatch.setenv("PHASE4_FEATURES", "1")   # recipe gate flipped
    ok, reason = manifest_ok(path, compute_manifest(tmp_path))
    assert not ok and "env" in reason


def test_data_refresh_invalidates(tmp_path):
    data = tmp_path / "csvs"; data.mkdir()
    (data / "AAA-1d.csv").write_text("Date,Close\n2020-01-01,1\n")
    path = tmp_path / "ckpt_manifest.json"
    write_manifest(path, compute_manifest(data))
    (data / "BBB-1d.csv").write_text("Date,Close\n2020-01-01,2\n")  # new ticker
    ok, reason = manifest_ok(path, compute_manifest(data))
    assert not ok and "data" in reason
