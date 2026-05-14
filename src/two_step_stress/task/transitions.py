"""Stage-1 → Stage-2 transition logic for the two-step task."""

import numpy as np

from two_step_stress.config import P_COMMON, P_RARE

# A transition mapping is a dict[int, int] whose keys are Stage-1 actions
# (0 or 1) and whose values are the *commonly* associated Stage-2 states
# (0 or 1).  The rare destination is always the other state.
TransitionMapping = dict[int, int]


def build_transition_mapping(counterbalance_id: int) -> TransitionMapping:
    """Return the fixed Stage-1 → Stage-2 common-transition mapping.

    There are exactly two possible mappings; which one a participant receives
    is determined by ``counterbalance_id % 2``.  Pass the participant number
    (or any integer) to counterbalance across the sample.

    Parameters
    ----------
    counterbalance_id : int
        Any integer used to select between the two mappings.  Even values
        give mapping A; odd values give mapping B.

    Returns
    -------
    TransitionMapping
        ``{0: common_state_for_action_0, 1: common_state_for_action_1}``

    Examples
    --------
    >>> build_transition_mapping(0)
    {0: 0, 1: 1}
    >>> build_transition_mapping(1)
    {0: 1, 1: 0}
    >>> build_transition_mapping(2)   # even → same as 0
    {0: 0, 1: 1}
    """
    if counterbalance_id % 2 == 0:
        return {0: 0, 1: 1}   # action 0 → state 0 (common), action 1 → state 1 (common)
    else:
        return {0: 1, 1: 0}   # action 0 → state 1 (common), action 1 → state 0 (common)


def apply_transition(
    stage1_choice: int,
    mapping: TransitionMapping,
    rng: np.random.Generator,
) -> tuple[int, str]:
    """Sample the Stage-2 state reached after a Stage-1 choice.

    With probability ``P_COMMON`` the participant lands in the state that is
    *commonly* associated with their Stage-1 action (per ``mapping``); with
    probability ``P_RARE`` they land in the other state.  The draw uses a
    single Bernoulli sample from ``rng`` so the caller's RNG stream is
    advanced by exactly one step.

    Parameters
    ----------
    stage1_choice : int
        The Stage-1 action taken: ``0`` (left) or ``1`` (right).
    mapping : TransitionMapping
        Fixed per-participant dict mapping each Stage-1 action to its
        *common* Stage-2 destination.  Construct with
        :func:`build_transition_mapping`.
    rng : numpy.random.Generator
        Caller-owned random generator.  Must be a ``numpy.random.Generator``
        (i.e. created via ``numpy.random.default_rng``), not the legacy
        ``numpy.random`` module interface.

    Returns
    -------
    stage2_state : int
        The Stage-2 state reached: ``0`` or ``1``.
    transition_type : str
        ``"common"`` if the common transition fired, ``"rare"`` otherwise.

    Raises
    ------
    KeyError
        If ``stage1_choice`` is not a key in ``mapping``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> mapping = build_transition_mapping(0)   # {0: 0, 1: 1}
    >>> state, kind = apply_transition(0, mapping, rng)
    >>> kind in ("common", "rare")
    True
    >>> # Over many draws the common rate should be close to P_COMMON
    >>> rng2 = np.random.default_rng(0)
    >>> results = [apply_transition(0, mapping, rng2) for _ in range(10_000)]
    >>> common_rate = sum(k == "common" for _, k in results) / 10_000
    >>> abs(common_rate - 0.7) < 0.02
    True
    """
    common_state: int = mapping[stage1_choice]
    rare_state: int = 1 - common_state

    is_common: bool = rng.random() < P_COMMON
    stage2_state = common_state if is_common else rare_state
    transition_type = "common" if is_common else "rare"

    return stage2_state, transition_type
