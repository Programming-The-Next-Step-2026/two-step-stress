"""Unit tests for two_step_stress.task.transitions."""

import numpy as np
import pytest

from two_step_stress.config import P_COMMON
from two_step_stress.task.transitions import apply_transition, build_transition_mapping


# ---------------------------------------------------------------------------
# build_transition_mapping
# ---------------------------------------------------------------------------

def test_build_mapping_even_parity():
    assert build_transition_mapping(0) == {0: 0, 1: 1}
    assert build_transition_mapping(2) == {0: 0, 1: 1}
    assert build_transition_mapping(100) == {0: 0, 1: 1}


def test_build_mapping_odd_parity():
    assert build_transition_mapping(1) == {0: 1, 1: 0}
    assert build_transition_mapping(3) == {0: 1, 1: 0}
    assert build_transition_mapping(99) == {0: 1, 1: 0}


def test_build_mapping_keys_are_0_and_1():
    for cb_id in (0, 1):
        mapping = build_transition_mapping(cb_id)
        assert set(mapping.keys()) == {0, 1}
        assert set(mapping.values()) == {0, 1}


# ---------------------------------------------------------------------------
# apply_transition — output validity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage1_choice", [0, 1])
@pytest.mark.parametrize("cb_id", [0, 1])
def test_apply_transition_valid_output(stage1_choice, cb_id):
    rng = np.random.default_rng(42)
    mapping = build_transition_mapping(cb_id)
    for _ in range(20):
        state, kind = apply_transition(stage1_choice, mapping, rng)
        assert state in (0, 1), f"stage2_state must be 0 or 1, got {state}"
        assert kind in ("common", "rare"), f"transition_type must be 'common' or 'rare', got {kind}"


# ---------------------------------------------------------------------------
# apply_transition — empirical common rate
# ---------------------------------------------------------------------------

def test_empirical_common_rate_close_to_p_common():
    n = 5_000
    rng = np.random.default_rng(0)
    mapping = build_transition_mapping(0)
    results = [apply_transition(0, mapping, rng) for _ in range(n)]
    common_rate = sum(kind == "common" for _, kind in results) / n
    assert abs(common_rate - P_COMMON) < 0.02, (
        f"Empirical common rate {common_rate:.4f} deviates from P_COMMON={P_COMMON} by more than 0.02"
    )


def test_empirical_common_rate_action_1():
    n = 5_000
    rng = np.random.default_rng(7)
    mapping = build_transition_mapping(1)
    results = [apply_transition(1, mapping, rng) for _ in range(n)]
    common_rate = sum(kind == "common" for _, kind in results) / n
    assert abs(common_rate - P_COMMON) < 0.02


# ---------------------------------------------------------------------------
# apply_transition — reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility_same_seed_same_sequence():
    mapping = build_transition_mapping(0)

    rng_a = np.random.default_rng(123)
    seq_a = [apply_transition(0, mapping, rng_a) for _ in range(50)]

    rng_b = np.random.default_rng(123)
    seq_b = [apply_transition(0, mapping, rng_b) for _ in range(50)]

    assert seq_a == seq_b


def test_reproducibility_different_seeds_differ():
    mapping = build_transition_mapping(0)

    rng_a = np.random.default_rng(1)
    seq_a = [apply_transition(0, mapping, rng_a) for _ in range(50)]

    rng_b = np.random.default_rng(2)
    seq_b = [apply_transition(0, mapping, rng_b) for _ in range(50)]

    assert seq_a != seq_b


# ---------------------------------------------------------------------------
# apply_transition — invalid choice raises KeyError
# ---------------------------------------------------------------------------

def test_invalid_stage1_choice_raises_key_error():
    rng = np.random.default_rng(0)
    mapping = build_transition_mapping(0)
    with pytest.raises(KeyError):
        apply_transition(2, mapping, rng)


def test_invalid_stage1_choice_negative_raises_key_error():
    rng = np.random.default_rng(0)
    mapping = build_transition_mapping(0)
    with pytest.raises(KeyError):
        apply_transition(-1, mapping, rng)
