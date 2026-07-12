"""
Regression tests for the 3 bugs we fixed in save_outputs across all run_*.py:

  1. bull_scores_orig is in the return dict from score_and_rank
     (NameError would crash at output phase otherwise)
  2. _render_side passes score_override for the bull side
     (otherwise bull cards show portfolio weight 0.083 not the score)
  3. Bear --explain output does NOT double-invert the bear model score
     (otherwise it displays the bull-bias raw score mislabeled as bearish)

These tests use static text inspection — no pipeline run needed.
They catch only the SHAPE of the bugs, not behavioral regression in zone math.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_RUN_SCRIPTS = ["run_nse_local.py", "run_nse_tradingv_local.py", "run_sp500_local.py"]


@pytest.fixture(params=_RUN_SCRIPTS)
def runner_text(request, project_root):
    p = project_root / request.param
    if not p.exists():
        pytest.skip(f"{p} not present in this checkout")
    return request.param, p.read_text(encoding="utf-8")


def test_bull_scores_orig_in_return_dict(runner_text):
    """Bug C1: bull_scores_orig must appear inside a dict literal in score_and_rank."""
    name, src = runner_text
    # It must appear at least twice: once as variable assignment, once in the return dict
    occurrences = re.findall(r'\bbull_scores_orig\b', src)
    assert len(occurrences) >= 3, (
        f"{name}: bull_scores_orig appears only {len(occurrences)} times — "
        "expected at least 3 (definition, return dict, save_outputs)"
    )
    # The literal "bull_scores_orig": bull_scores_orig pattern (return dict entry)
    assert re.search(r'"bull_scores_orig":\s*bull_scores_orig', src), (
        f"{name}: missing 'bull_scores_orig' key in score_and_rank return dict"
    )


def test_render_side_bull_has_score_override(runner_text):
    """Bug C2: _render_side(bull_weights, "bull", score_override=...) must include override."""
    name, src = runner_text
    # The bull call must pass score_override
    pattern = r'_render_side\(\s*bull_weights\s*,\s*["\']bull["\']\s*,\s*score_override\s*='
    assert re.search(pattern, src), (
        f"{name}: _render_side(bull_weights, 'bull') is missing score_override — "
        "bull console cards will show portfolio weight instead of model score"
    )


def test_bear_explain_no_double_inversion(runner_text):
    """
    Bug C3: --explain bear card must use m_score_b * 100, NOT (1-m_score_b) * 100.
    _scoring_detail already inverts (1 - raw_m) for bear; a second inversion would
    display the bull-bias score mislabeled as 'bearish'.
    """
    name, src = runner_text
    # Look for the bear-side print line
    bad_pattern = r'Model \(bearish\).*\(1\s*-\s*m_score_b\)'
    assert not re.search(bad_pattern, src), (
        f"{name}: --explain bear card still double-inverts (1 - m_score_b). "
        "It should print m_score_b * 100 since _scoring_detail already inverted."
    )
    # Affirmative: the correct form is present
    good_pattern = r'Model \(bearish\)\s*:\s*\{m_score_b\s*\*\s*100'
    assert re.search(good_pattern, src), (
        f"{name}: --explain bear card does not use m_score_b*100 — fix Bug #3"
    )


def test_save_outputs_extracts_bull_scores(runner_text):
    """save_outputs() must extract bull_scores_orig from the result dict."""
    name, src = runner_text
    # The pattern: bull_scores_orig = result["bull_scores_orig"]
    pattern = r'bull_scores_orig\s*=\s*result\[\s*["\']bull_scores_orig["\']\s*\]'
    assert re.search(pattern, src), (
        f"{name}: save_outputs() does not extract bull_scores_orig from result dict — "
        "this would NameError at runtime when building per-tier or main CSVs"
    )
