"""Unit tests for two_step_stress.task.nback."""

import numpy as np
import pytest

from two_step_stress.config import NBACK_CONSONANTS
from two_step_stress.task.nback import generate_letter_stream, score_nback_response


# ---------------------------------------------------------------------------
# generate_letter_stream — structural properties
# ---------------------------------------------------------------------------

def test_returns_correct_length():
    rng = np.random.default_rng(0)
    letters, is_match = generate_letter_stream(10, 0.33, rng)
    assert len(letters) == 10
    assert len(is_match) == 10


@pytest.mark.parametrize("n_trials", [1, 5, 50, 200])
def test_length_matches_n_trials(n_trials):
    rng = np.random.default_rng(1)
    letters, is_match = generate_letter_stream(n_trials, 0.33, rng)
    assert len(letters) == n_trials
    assert len(is_match) == n_trials


def test_all_letters_in_consonant_pool():
    rng = np.random.default_rng(2)
    letters, _ = generate_letter_stream(200, 0.33, rng)
    consonant_set = set(NBACK_CONSONANTS)
    assert all(l in consonant_set for l in letters)


def test_first_is_match_always_false():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        _, is_match = generate_letter_stream(20, 0.33, rng)
        assert is_match[0] is False


# ---------------------------------------------------------------------------
# generate_letter_stream — match/non-match letter correctness
# ---------------------------------------------------------------------------

def test_match_trial_repeats_previous_letter():
    rng = np.random.default_rng(3)
    letters, is_match = generate_letter_stream(500, 0.5, rng)
    for i in range(1, len(letters)):
        if is_match[i]:
            assert letters[i] == letters[i - 1], (
                f"Match at position {i}: expected {letters[i-1]!r}, got {letters[i]!r}"
            )


def test_non_match_trial_differs_from_previous_letter():
    rng = np.random.default_rng(4)
    letters, is_match = generate_letter_stream(500, 0.5, rng)
    for i in range(1, len(letters)):
        if not is_match[i]:
            assert letters[i] != letters[i - 1], (
                f"Non-match at position {i}: got same letter {letters[i]!r} as previous"
            )


# ---------------------------------------------------------------------------
# generate_letter_stream — empirical match rate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_rate", [0.2, 0.33, 0.5])
def test_empirical_match_rate_close_to_target(target_rate):
    n = 5_000
    rng = np.random.default_rng(5)
    _, is_match = generate_letter_stream(n, target_rate, rng)
    # Exclude position 0 (always False) from the rate calculation.
    empirical = sum(is_match[1:]) / (n - 1)
    assert abs(empirical - target_rate) < 0.02, (
        f"target={target_rate}, empirical={empirical:.4f}, diff={abs(empirical-target_rate):.4f}"
    )


# ---------------------------------------------------------------------------
# generate_letter_stream — reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_produces_same_stream():
    rng_a = np.random.default_rng(99)
    letters_a, is_match_a = generate_letter_stream(100, 0.33, rng_a)

    rng_b = np.random.default_rng(99)
    letters_b, is_match_b = generate_letter_stream(100, 0.33, rng_b)

    assert letters_a == letters_b
    assert is_match_a == is_match_b


def test_different_seeds_produce_different_streams():
    rng_a = np.random.default_rng(10)
    letters_a, _ = generate_letter_stream(100, 0.33, rng_a)

    rng_b = np.random.default_rng(11)
    letters_b, _ = generate_letter_stream(100, 0.33, rng_b)

    assert letters_a != letters_b


# ---------------------------------------------------------------------------
# score_nback_response — all four signal-detection cases
# ---------------------------------------------------------------------------

def test_hit():
    is_correct, response_type = score_nback_response(True, "z", "z", "m")
    assert is_correct is True
    assert response_type == "hit"


def test_miss():
    is_correct, response_type = score_nback_response(True, "m", "z", "m")
    assert is_correct is False
    assert response_type == "miss"


def test_false_alarm():
    is_correct, response_type = score_nback_response(False, "z", "z", "m")
    assert is_correct is False
    assert response_type == "false_alarm"


def test_correct_rejection():
    is_correct, response_type = score_nback_response(False, "m", "z", "m")
    assert is_correct is True
    assert response_type == "correct_rejection"


# ---------------------------------------------------------------------------
# score_nback_response — invalid key raises ValueError
# ---------------------------------------------------------------------------

def test_unrecognised_key_raises_value_error():
    with pytest.raises(ValueError):
        score_nback_response(True, "x", "z", "m")


def test_unrecognised_key_error_message_contains_key():
    with pytest.raises(ValueError, match="x"):
        score_nback_response(False, "x", "z", "m")
