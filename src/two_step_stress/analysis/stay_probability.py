"""Stay-probability analysis: the headline 2 x 2 (reward x transition).

A "stay" is repeating the Stage-1 choice on trial ``i + 1`` after trial ``i``.
A trial-pair ``(i, i+1)`` only contributes if both trials are completed and
they are genuinely adjacent within the same block — pairs that cross a block
boundary, or that straddle a timed-out trial, are excluded.  Adjacency is
checked by requiring the same ``block`` and ``trial_in_block`` differing by
exactly 1 (a dropped/timed-out trial leaves a gap > 1, so it is skipped).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

_REWARDS: tuple[int, ...] = (0, 1)
_TRANSITIONS: tuple[str, ...] = ("common", "rare")
_BLOCK_TYPES: tuple[str, ...] = ("load", "no_load")


def _bootstrap_ci(
    values: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Return the 2.5/97.5 percentile bootstrap CI of the mean of ``values``."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    means = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_stay_probability(
    df: pd.DataFrame,
    split_by_block_type: bool = False,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute stay probabilities by previous reward x transition.

    For each valid trial-pair, ``stayed = (stage1_choice[i+1] == stage1_choice[i])``
    is grouped by trial ``i``'s ``reward`` (0/1) and ``transition``
    (common/rare); optionally also by ``block_type`` (load/no_load).  The
    returned frame contains a row for every cell of the design (cells with no
    data get ``stay_prob = NaN`` and ``n_trials = 0``).

    Parameters
    ----------
    df : pandas.DataFrame
        Trials to analyse — typically the output of
        :func:`io.clean_for_stay_analysis`.  Must contain ``trial_global``,
        ``block``, ``trial_in_block``, ``stage1_choice``, ``reward``,
        ``transition`` (and ``block_type`` if splitting).
    split_by_block_type : bool, optional
        If True, additionally group by ``block_type``.
    n_boot : int, optional
        Bootstrap resamples for the confidence interval (default 1000).
    seed : int, optional
        Seed for the bootstrap RNG, for reproducibility (default 0).

    Returns
    -------
    pandas.DataFrame
        Long-format with columns ``reward``, ``transition``,
        [``block_type``], ``stay_prob``, ``n_trials``, ``se``, ``ci_lower``,
        ``ci_upper``.
    """
    rng = np.random.default_rng(seed)

    work = df.sort_values("trial_global").reset_index(drop=True).copy()
    work["next_choice"] = work["stage1_choice"].shift(-1)
    work["next_block"] = work["block"].shift(-1)
    work["next_tib"] = work["trial_in_block"].shift(-1)

    valid = (
        work["stage1_choice"].notna()
        & work["next_choice"].notna()
        & (work["next_block"] == work["block"])
        & (work["next_tib"] == work["trial_in_block"] + 1)
    )
    pairs = work[valid].copy()
    pairs["stayed"] = (pairs["next_choice"] == pairs["stage1_choice"]).astype(int)
    if len(pairs):
        pairs["reward"] = pairs["reward"].astype(int)

    block_levels = _BLOCK_TYPES if split_by_block_type else (None,)
    records: list[dict] = []
    for reward, transition, block_type in itertools.product(
        _REWARDS, _TRANSITIONS, block_levels
    ):
        mask = (pairs["reward"] == reward) & (pairs["transition"] == transition)
        if split_by_block_type:
            mask &= pairs["block_type"] == block_type
        vals = pairs.loc[mask, "stayed"].to_numpy()
        n = len(vals)
        if n > 0:
            p = float(vals.mean())
            se = float(np.sqrt(p * (1.0 - p) / n))
            lo, hi = _bootstrap_ci(vals, n_boot, rng)
        else:
            p = se = lo = hi = float("nan")

        rec = {"reward": reward, "transition": transition}
        if split_by_block_type:
            rec["block_type"] = block_type
        rec.update({"stay_prob": p, "n_trials": n, "se": se, "ci_lower": lo, "ci_upper": hi})
        records.append(rec)

    columns = ["reward", "transition"]
    if split_by_block_type:
        columns.append("block_type")
    columns += ["stay_prob", "n_trials", "se", "ci_lower", "ci_upper"]
    return pd.DataFrame(records)[columns]
