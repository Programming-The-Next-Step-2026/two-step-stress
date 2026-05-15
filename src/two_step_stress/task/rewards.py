"""Drifting reward-probability random walks for the two-step task.

Four independent Gaussian random walks, one per Stage-2 arm, with reflecting
boundaries.  All parameters are imported from ``config.py``; nothing is
hard-coded here.
"""

import numpy as np

from two_step_stress.config import (
    REWARD_WALK_MAX,
    REWARD_WALK_MIN,
    REWARD_WALK_SIGMA,
)

# Number of reward arms: 2 Stage-2 states × 2 actions each.
N_ARMS: int = 4


def initialise_reward_probs(rng: np.random.Generator) -> np.ndarray:
    """Draw four starting reward probabilities uniformly within the walk bounds.

    Each arm is initialised independently and uniformly from
    ``[REWARD_WALK_MIN, REWARD_WALK_MAX]``.  Pass the participant-level RNG so
    that starting positions are reproducible from the logged seed.

    Parameters
    ----------
    rng : numpy.random.Generator
        Caller-owned generator (from ``numpy.random.default_rng``).

    Returns
    -------
    probs : numpy.ndarray, shape (4,)
        Initial reward probabilities for arms 0–3, all in
        ``[REWARD_WALK_MIN, REWARD_WALK_MAX]``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> p = initialise_reward_probs(rng)
    >>> p.shape
    (4,)
    >>> bool(np.all(p >= 0.25) and np.all(p <= 0.75))
    True
    """
    return rng.uniform(REWARD_WALK_MIN, REWARD_WALK_MAX, size=N_ARMS)


def step_reward_probs(
    probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advance all four reward probabilities by one Gaussian step.

    Each arm moves independently: a zero-mean Gaussian with SD
    ``REWARD_WALK_SIGMA`` is added, then reflecting boundaries at
    ``REWARD_WALK_MIN`` and ``REWARD_WALK_MAX`` are applied.  Reflection
    (folding) rather than clipping preserves the stationary distribution of
    the walk near the boundaries.

    The reflecting rule for a single step:

    * If the proposed value falls below ``REWARD_WALK_MIN``, it is folded
      back: ``new = 2 * REWARD_WALK_MIN − proposed``.
    * If it exceeds ``REWARD_WALK_MAX``, it is folded back:
      ``new = 2 * REWARD_WALK_MAX − proposed``.

    With ``REWARD_WALK_SIGMA = 0.025`` and a boundary range of 0.50, the
    probability of needing more than one fold per step is negligible
    (< 3 × 10⁻²³); a single fold is applied for correctness and clarity.

    Parameters
    ----------
    probs : numpy.ndarray, shape (4,)
        Current reward probabilities, all in
        ``[REWARD_WALK_MIN, REWARD_WALK_MAX]``.
    rng : numpy.random.Generator
        Caller-owned generator (from ``numpy.random.default_rng``).  Advanced
        by exactly one call to ``rng.normal`` (4 values drawn at once).

    Returns
    -------
    new_probs : numpy.ndarray, shape (4,)
        Updated reward probabilities after one Gaussian step, guaranteed to
        lie within ``[REWARD_WALK_MIN, REWARD_WALK_MAX]``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> p0 = initialise_reward_probs(rng)
    >>> p1 = step_reward_probs(p0, rng)
    >>> p1.shape
    (4,)
    >>> bool(np.all(p1 >= 0.25) and np.all(p1 <= 0.75))
    True
    >>> # Boundaries are respected over a long walk.
    >>> rng2 = np.random.default_rng(7)
    >>> p = initialise_reward_probs(rng2)
    >>> for _ in range(10_000):
    ...     p = step_reward_probs(p, rng2)
    >>> bool(np.all(p >= 0.25) and np.all(p <= 0.75))
    True
    """
    noise = rng.normal(loc=0.0, scale=REWARD_WALK_SIGMA, size=N_ARMS)
    proposed = probs + noise

    # Reflect off lower boundary.
    too_low = proposed < REWARD_WALK_MIN
    proposed[too_low] = 2.0 * REWARD_WALK_MIN - proposed[too_low]

    # Reflect off upper boundary.
    too_high = proposed > REWARD_WALK_MAX
    proposed[too_high] = 2.0 * REWARD_WALK_MAX - proposed[too_high]

    return proposed
