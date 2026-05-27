"""1-back letter-stream generation and response scoring for the two-step task."""

import numpy as np

from two_step_stress.config import NBACK_CONSONANTS


def generate_letter_stream(
    n_trials: int,
    target_match_rate: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[bool]]:
    """Generate a sequence of letters for the 1-back task.

    The first letter is a random draw from ``NBACK_CONSONANTS`` with no
    match status (``is_match=False``).  Each subsequent letter is either a
    *match* (same as the immediately preceding letter, drawn with probability
    ``target_match_rate``) or a *non-match* (drawn uniformly from the
    remaining consonants, excluding the previous letter).

    The RNG is advanced by exactly ``n_trials`` steps: one draw per position
    (the Bernoulli match/non-match decision and the letter selection are both
    made from ``rng.choice``).

    Parameters
    ----------
    n_trials : int
        Number of letters to generate (one per trial).  Must be ≥ 1.
    target_match_rate : float
        Probability in ``(0, 1)`` that any given letter (from index 1 onward)
        is a match to the previous letter.
    rng : numpy.random.Generator
        Caller-owned generator (from ``numpy.random.default_rng``).

    Returns
    -------
    letters : list[str]
        Sequence of consonant letters of length ``n_trials``.
    is_match : list[bool]
        Parallel boolean list; ``is_match[0]`` is always ``False`` (no
        previous letter); ``is_match[i]`` is ``True`` when
        ``letters[i] == letters[i - 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> letters, is_match = generate_letter_stream(6, 0.33, rng)
    >>> len(letters) == 6 and len(is_match) == 6
    True
    >>> is_match[0]
    False
    >>> all(l in NBACK_CONSONANTS for l in letters)
    True
    >>> # A match trial must repeat the previous letter exactly.
    >>> all(letters[i] == letters[i - 1] for i in range(1, 6) if is_match[i])
    True
    >>> # A non-match trial must differ from the previous letter.
    >>> all(letters[i] != letters[i - 1] for i in range(1, 6) if not is_match[i])
    True
    """
    consonants = list(NBACK_CONSONANTS)
    n_consonants = len(consonants)

    letters: list[str] = []
    is_match: list[bool] = []

    # First letter: free draw, no match possible.
    letters.append(consonants[int(rng.integers(n_consonants))])
    is_match.append(False)

    for i in range(1, n_trials):
        prev = letters[i - 1]
        if rng.random() < target_match_rate:
            letters.append(prev)
            is_match.append(True)
        else:
            non_matches = [c for c in consonants if c != prev]
            letters.append(non_matches[int(rng.integers(len(non_matches)))])
            is_match.append(False)

    return letters, is_match


def score_nback_response(
    is_match: bool,
    response_key: str,
    key_match: str,
    key_no_match: str,
) -> tuple[bool, str]:
    """Classify a 1-back response against the ground-truth match status.

    Uses standard signal-detection terminology: a *hit* is a correct match
    response; a *miss* is a failure to respond to a match; a *false alarm* is
    a match response on a non-match trial; a *correct rejection* is a correct
    non-match response.

    Parameters
    ----------
    is_match : bool
        Whether the current letter matches the immediately preceding letter.
    response_key : str
        The key the participant pressed (e.g. ``"z"`` or ``"m"``).
    key_match : str
        The configured key for "match" responses (``KEY_NBACK_MATCH`` from
        ``config.py``).
    key_no_match : str
        The configured key for "no match" responses (``KEY_NBACK_NO_MATCH``
        from ``config.py``).

    Returns
    -------
    is_correct : bool
        ``True`` if the response was correct.
    response_type : str
        One of ``"hit"``, ``"miss"``, ``"false_alarm"``,
        ``"correct_rejection"``.

    Raises
    ------
    ValueError
        If ``response_key`` is neither ``key_match`` nor ``key_no_match``.

    Examples
    --------
    >>> score_nback_response(True, "z", key_match="z", key_no_match="m")
    (True, 'hit')
    >>> score_nback_response(True, "m", key_match="z", key_no_match="m")
    (False, 'miss')
    >>> score_nback_response(False, "z", key_match="z", key_no_match="m")
    (False, 'false_alarm')
    >>> score_nback_response(False, "m", key_match="z", key_no_match="m")
    (True, 'correct_rejection')
    """
    if response_key not in (key_match, key_no_match):
        raise ValueError(
            f"response_key {response_key!r} is not key_match {key_match!r} "
            f"or key_no_match {key_no_match!r}"
        )

    responded_match: bool = response_key == key_match

    if is_match and responded_match:
        return True, "hit"
    if is_match and not responded_match:
        return False, "miss"
    if not is_match and responded_match:
        return False, "false_alarm"
    return True, "correct_rejection"
