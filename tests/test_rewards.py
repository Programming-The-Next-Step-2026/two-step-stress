"""Unit tests for two_step_stress.task.rewards."""

import numpy as np
import pytest

from two_step_stress.config import REWARD_WALK_MAX, REWARD_WALK_MIN, REWARD_WALK_SIGMA
from two_step_stress.task.rewards import initialise_reward_probs, step_reward_probs


# ---------------------------------------------------------------------------
# initialise_reward_probs
# ---------------------------------------------------------------------------

def test_initialise_shape():
    rng = np.random.default_rng(0)
    probs = initialise_reward_probs(rng)
    assert probs.shape == (4,)


def test_initialise_within_bounds():
    rng = np.random.default_rng(42)
    probs = initialise_reward_probs(rng)
    assert np.all(probs >= REWARD_WALK_MIN), f"Some probs below min: {probs}"
    assert np.all(probs <= REWARD_WALK_MAX), f"Some probs above max: {probs}"


def test_initialise_within_bounds_multiple_seeds():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        probs = initialise_reward_probs(rng)
        assert np.all(probs >= REWARD_WALK_MIN)
        assert np.all(probs <= REWARD_WALK_MAX)


# ---------------------------------------------------------------------------
# step_reward_probs — output shape
# ---------------------------------------------------------------------------

def test_step_shape():
    rng = np.random.default_rng(1)
    probs = initialise_reward_probs(rng)
    new_probs = step_reward_probs(probs, rng)
    assert new_probs.shape == (4,)


# ---------------------------------------------------------------------------
# step_reward_probs — boundaries respected over long walk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 7, 99, 2024])
def test_long_walk_stays_within_bounds(seed):
    rng = np.random.default_rng(seed)
    probs = initialise_reward_probs(rng)
    for _ in range(10_000):
        probs = step_reward_probs(probs, rng)
        assert np.all(probs >= REWARD_WALK_MIN), f"step went below min at seed={seed}"
        assert np.all(probs <= REWARD_WALK_MAX), f"step went above max at seed={seed}"


# ---------------------------------------------------------------------------
# step_reward_probs — mean absolute step size consistent with sigma
# ---------------------------------------------------------------------------

def test_mean_abs_step_size_consistent_with_sigma():
    """Mean |step| per arm should be roughly sigma * sqrt(2/pi) for a folded normal."""
    n_steps = 10_000
    rng = np.random.default_rng(5)
    probs = initialise_reward_probs(rng)

    abs_steps = []
    for _ in range(n_steps):
        prev = probs.copy()
        probs = step_reward_probs(probs, rng)
        abs_steps.append(np.abs(probs - prev))

    mean_abs_step = np.mean(abs_steps)
    # Expected mean |x| for N(0, sigma) is sigma * sqrt(2/pi) ≈ 0.798 * sigma.
    # Near boundaries the walk folds, compressing steps slightly.
    # Use a loose tolerance: [0.4 * sigma, 1.2 * sigma].
    assert mean_abs_step >= 0.4 * REWARD_WALK_SIGMA, (
        f"Mean |step| {mean_abs_step:.5f} unexpectedly small vs sigma={REWARD_WALK_SIGMA}"
    )
    assert mean_abs_step <= 1.2 * REWARD_WALK_SIGMA, (
        f"Mean |step| {mean_abs_step:.5f} unexpectedly large vs sigma={REWARD_WALK_SIGMA}"
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility_initialise():
    rng_a = np.random.default_rng(77)
    probs_a = initialise_reward_probs(rng_a)

    rng_b = np.random.default_rng(77)
    probs_b = initialise_reward_probs(rng_b)

    np.testing.assert_array_equal(probs_a, probs_b)


def test_reproducibility_walk_sequence():
    n_steps = 100

    rng_a = np.random.default_rng(55)
    p_a = initialise_reward_probs(rng_a)
    seq_a = []
    for _ in range(n_steps):
        p_a = step_reward_probs(p_a, rng_a)
        seq_a.append(p_a.copy())

    rng_b = np.random.default_rng(55)
    p_b = initialise_reward_probs(rng_b)
    seq_b = []
    for _ in range(n_steps):
        p_b = step_reward_probs(p_b, rng_b)
        seq_b.append(p_b.copy())

    for i, (a, b) in enumerate(zip(seq_a, seq_b)):
        np.testing.assert_array_equal(a, b, err_msg=f"Diverged at step {i}")


def test_different_seeds_produce_different_walks():
    rng_a = np.random.default_rng(1)
    p_a = initialise_reward_probs(rng_a)
    for _ in range(50):
        p_a = step_reward_probs(p_a, rng_a)

    rng_b = np.random.default_rng(2)
    p_b = initialise_reward_probs(rng_b)
    for _ in range(50):
        p_b = step_reward_probs(p_b, rng_b)

    assert not np.array_equal(p_a, p_b)
