"""Tests for analysis.stay_probability.compute_stay_probability."""

import numpy as np
import pandas as pd
import pytest

from two_step_stress.analysis.stay_probability import compute_stay_probability


def _cell(result: pd.DataFrame, reward: int, transition: str, block_type: str | None = None):
    mask = (result["reward"] == reward) & (result["transition"] == transition)
    if block_type is not None:
        mask &= result["block_type"] == block_type
    return result[mask].iloc[0]


def test_basic_cells_and_counts():
    # One no-load block of 8 completed trials; 7 within-block pairs.
    df = pd.DataFrame(
        {
            "trial_global": [1, 2, 3, 4, 5, 6, 7, 8],
            "block": [1] * 8,
            "block_type": ["no_load"] * 8,
            "trial_in_block": [1, 2, 3, 4, 5, 6, 7, 8],
            "stage1_choice": [0, 0, 1, 1, 0, 1, 0, 0],
            "stage2_choice": [0, 1, 0, 1, 0, 1, 0, 1],
            "reward": [1, 1, 0, 0, 1, 0, 1, 1],
            "transition": ["common", "rare", "common", "rare",
                           "common", "rare", "common", "rare"],
        }
    )
    res = compute_stay_probability(df)

    assert res["n_trials"].sum() == 7
    assert _cell(res, 1, "common")["n_trials"] == 3
    assert _cell(res, 1, "common")["stay_prob"] == pytest.approx(2 / 3)
    assert _cell(res, 1, "rare")["n_trials"] == 1
    assert _cell(res, 1, "rare")["stay_prob"] == pytest.approx(0.0)
    assert _cell(res, 0, "common")["n_trials"] == 1
    assert _cell(res, 0, "common")["stay_prob"] == pytest.approx(1.0)
    assert _cell(res, 0, "rare")["n_trials"] == 2
    assert _cell(res, 0, "rare")["stay_prob"] == pytest.approx(0.0)

    for col in ["reward", "transition", "stay_prob", "n_trials", "se", "ci_lower", "ci_upper"]:
        assert col in res.columns


def test_block_boundary_pairs_excluded():
    # block 1: tib 1-3, block 2: tib 1-2. The 3->1 cross-block pair must NOT count.
    df = pd.DataFrame(
        {
            "trial_global": [1, 2, 3, 4, 5],
            "block": [1, 1, 1, 2, 2],
            "block_type": ["no_load"] * 5,
            "trial_in_block": [1, 2, 3, 1, 2],
            "stage1_choice": [0, 0, 1, 1, 1],
            "stage2_choice": [0, 0, 0, 0, 0],
            "reward": [1, 1, 1, 0, 0],
            "transition": ["common"] * 5,
        }
    )
    res = compute_stay_probability(df)
    # Valid pairs: block1 (1->2),(2->3) and block2 (1->2) = 3 total (not 4).
    assert res["n_trials"].sum() == 3


def test_timeout_trials_skipped():
    # tib2 timed out (NaN choices). Only the (tib3 -> tib4) pair is valid.
    df = pd.DataFrame(
        {
            "trial_global": [1, 2, 3, 4],
            "block": [1, 1, 1, 1],
            "block_type": ["no_load"] * 4,
            "trial_in_block": [1, 2, 3, 4],
            "stage1_choice": [0, np.nan, 0, 0],
            "stage2_choice": [0, np.nan, 0, 0],
            "reward": [1, np.nan, 1, 1],
            "transition": ["common", None, "common", "common"],
        }
    )
    res = compute_stay_probability(df)
    assert res["n_trials"].sum() == 1
    assert _cell(res, 1, "common")["n_trials"] == 1
    assert _cell(res, 1, "common")["stay_prob"] == pytest.approx(1.0)


def test_split_by_block_type():
    df = pd.DataFrame(
        {
            "trial_global": [1, 2, 3, 4, 5, 6],
            "block": [1, 1, 1, 2, 2, 2],
            "block_type": ["no_load", "no_load", "no_load", "load", "load", "load"],
            "trial_in_block": [1, 2, 3, 1, 2, 3],
            "stage1_choice": [0, 0, 1, 1, 1, 0],
            "stage2_choice": [0, 0, 0, 0, 0, 0],
            "reward": [1, 1, 0, 1, 0, 1],
            "transition": ["common", "rare", "common", "common", "rare", "common"],
        }
    )
    res = compute_stay_probability(df, split_by_block_type=True)

    assert "block_type" in res.columns
    assert res["n_trials"].sum() == 4

    # no_load: (1->2) reward1/common stay=1; (2->3) reward1/rare stay=0
    assert _cell(res, 1, "common", "no_load")["n_trials"] == 1
    assert _cell(res, 1, "common", "no_load")["stay_prob"] == pytest.approx(1.0)
    assert _cell(res, 1, "rare", "no_load")["stay_prob"] == pytest.approx(0.0)

    # load: (4->5) reward1/common stay=1; (5->6) reward0/rare stay=0
    assert _cell(res, 1, "common", "load")["stay_prob"] == pytest.approx(1.0)
    assert _cell(res, 0, "rare", "load")["stay_prob"] == pytest.approx(0.0)

    # An empty cell is still present, with n_trials 0 and NaN stay_prob.
    empty = _cell(res, 0, "common", "no_load")
    assert empty["n_trials"] == 0
    assert np.isnan(empty["stay_prob"])
