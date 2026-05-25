"""One full two-step trial: presentation, response collection, and a log row.

:func:`run_trial` runs a single trial (load or no-load), frame-locked to
``win.flip()``, and returns a plain ``dict`` keyed by
``io.data_logging.TRIAL_COLUMNS`` (minus ``participant_id``, which the caller
adds before logging).

Engine call order (fixed, for RNG reproducibility) on a COMPLETED trial
-----------------------------------------------------------------------
1. ``apply_transition(stage1_choice, mapping, rng)``     — 1 Bernoulli draw
2. ``reward = rng.random() < reward_probs[stage2_state * 2 + stage2_choice]``
3. ``step_reward_probs(reward_probs, rng)``              — advances the 4 walks

The walk is stepped ONLY on completed trials; on a timeout/skip it is frozen
(plan §9 C6 / decision Q6).  The walk is advanced **in place** (``reward_probs``
is mutated) so the caller threads the same array into the next trial; the
trial-start probabilities are snapshotted into the row *before* the step.

N-back vs choice ordering in load blocks (plan §9 C2)
-----------------------------------------------------
During the Stage-1 window both the choice (``f``/``j``) and the 1-back response
(``z``/``m``) are collected, on separate keys.  The transition reveal must not
begin until BOTH have resolved:

* **Case A** — choice first, n-back pending: keep listening for ``z``/``m``
  until it arrives or the 2000 ms cap; then reveal (n-back logged NA if it
  never comes).
* **Case B** — n-back first, choice pending: keep listening for ``f``/``j``
  until it arrives or the cap; no choice by the cap → timeout/skip.
* **Case C** — both pressed: reveal at ``max(t_choice, t_nback)``.
* **Case D** — neither by the cap → timeout/skip.

Equivalently, the Stage-1 loop breaks when
``choice is not None and (not load or nback is not None)``.  No-load blocks
listen for ``f``/``j`` only and break on the choice.  Load-block behaviour is
active whenever ``nback_letter`` is not ``None``.

A missing CHOICE (Cases B/D, or a Stage-2 timeout) shows "Too slow!" for
1000 ms, logs the missing fields as NA, and returns without stepping the walk.
A missing n-back is silent and logged NA (plan §3 / Q7).
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from psychopy import core, event, visual

from two_step_stress.config import (
    FEEDBACK_MS,
    ITI_MS,
    KEY_LEFT,
    KEY_NBACK_MATCH,
    KEY_NBACK_NO_MATCH,
    KEY_RIGHT,
    NBACK_LETTER_MS,
    STAGE1_WINDOW_MS,
    STAGE2_WINDOW_MS,
    TRANSITION_REVEAL_MS,
)
from two_step_stress.task import screens
from two_step_stress.task.nback import score_nback_response
from two_step_stress.task.rewards import step_reward_probs
from two_step_stress.task.transitions import TransitionMapping, apply_transition

logger = logging.getLogger(__name__)

ESCAPE_KEY: str = "escape"
# Map the two choice keys to Stage-1/Stage-2 action indices (0 = left, 1 = right).
CHOICE_KEYS: dict[str, int] = {KEY_LEFT: 0, KEY_RIGHT: 1}
NBACK_KEYS: tuple[str, str] = (KEY_NBACK_MATCH, KEY_NBACK_NO_MATCH)


def _ms_to_frames(ms: int, frame_rate: float) -> int:
    """Convert a duration in milliseconds to an integer number of frames."""
    return max(1, round(ms / 1000.0 * frame_rate))


def _abort_if_escape(keys: list[str]) -> None:
    """Raise ``KeyboardInterrupt`` if Escape is among ``keys``."""
    if ESCAPE_KEY in keys:
        raise KeyboardInterrupt("Escape pressed during trial")


def _hold(win: visual.Window, draw_fn, n_frames: int) -> None:
    """Draw a static screen for ``n_frames`` flips, aborting on Escape.

    ``draw_fn`` is a zero-arg callable that draws the screen and flips (one of
    the ``screens.draw_*`` helpers, wrapped in a lambda).
    """
    for _ in range(n_frames):
        draw_fn()
        _abort_if_escape(event.getKeys(keyList=[ESCAPE_KEY]))


def _collect_stage1(
    win: visual.Window,
    stims: screens.SessionStimuli,
    frame_rate: float,
    load: bool,
) -> tuple[int | None, float | None, str | None, float | None]:
    """Collect the Stage-1 choice (+ 1-back response in load blocks).

    Implements the plan §9 C2 ordering rule.  Returns
    ``(choice, choice_rt, nback_key, nback_rt)`` with ``None`` for anything not
    pressed within the window.  The caller must have set the n-back letter text
    (via :func:`screens.set_nback_letter`) before calling when ``load`` is True.
    """
    cap_frames = _ms_to_frames(STAGE1_WINDOW_MS, frame_rate)
    letter_frames = _ms_to_frames(NBACK_LETTER_MS, frame_rate)
    key_list = [KEY_LEFT, KEY_RIGHT, ESCAPE_KEY]
    if load:
        key_list += [KEY_NBACK_MATCH, KEY_NBACK_NO_MATCH]

    choice: int | None = None
    choice_rt: float | None = None
    nback_key: str | None = None
    nback_rt: float | None = None

    clock = core.Clock()
    event.clearEvents(eventType="keyboard")
    for frame in range(cap_frames):
        show_letter = load and frame < letter_frames
        screens.draw_stage1(
            win, stims.stage1, stims.counter,
            letter=stims.nback_letter, show_letter=show_letter,
        )
        if frame == 0:
            clock.reset()  # RT measured from the Stage-1 onset flip

        for name, t in event.getKeys(keyList=key_list, timeStamped=clock):
            if name == ESCAPE_KEY:
                raise KeyboardInterrupt("Escape pressed during Stage-1")
            if name in CHOICE_KEYS and choice is None:
                choice, choice_rt = CHOICE_KEYS[name], t
            elif load and name in NBACK_KEYS and nback_key is None:
                nback_key, nback_rt = name, t

        # Reveal only once the choice AND (in load) the n-back have resolved.
        if choice is not None and (not load or nback_key is not None):
            break

    return choice, choice_rt, nback_key, nback_rt


def _collect_stage2(
    win: visual.Window,
    stims: screens.SessionStimuli,
    frame_rate: float,
    stage2_state: int,
) -> tuple[int | None, float | None]:
    """Collect the Stage-2 alien choice. Returns ``(choice, choice_rt)``."""
    cap_frames = _ms_to_frames(STAGE2_WINDOW_MS, frame_rate)
    key_list = [KEY_LEFT, KEY_RIGHT, ESCAPE_KEY]

    choice: int | None = None
    choice_rt: float | None = None

    clock = core.Clock()
    event.clearEvents(eventType="keyboard")
    for frame in range(cap_frames):
        screens.draw_stage2(win, stims.stage2, stage2_state, stims.counter)
        if frame == 0:
            clock.reset()  # RT measured from the Stage-2 onset flip

        for name, t in event.getKeys(keyList=key_list, timeStamped=clock):
            if name == ESCAPE_KEY:
                raise KeyboardInterrupt("Escape pressed during Stage-2")
            if name in CHOICE_KEYS and choice is None:
                choice, choice_rt = CHOICE_KEYS[name], t

        if choice is not None:
            break

    return choice, choice_rt


def run_trial(
    win: visual.Window,
    stims: screens.SessionStimuli,
    rng: np.random.Generator,
    mapping: TransitionMapping,
    reward_probs: np.ndarray,
    frame_rate: float,
    *,
    block: int,
    block_type: str,
    trial_in_block: int,
    trial_global: int,
    nback_letter: str | None = None,
    nback_is_match: bool | None = None,
) -> dict:
    """Run one two-step trial and return its log row.

    Load-block behaviour (1-back letter + ``z``/``m`` collection) is active
    whenever ``nback_letter`` is not ``None``.

    Parameters
    ----------
    win : psychopy.visual.Window
        The open window.
    stims : screens.SessionStimuli
        Pre-built stimuli (see :func:`screens.build_session_stimuli`).
    rng : numpy.random.Generator
        The participant RNG.  Advanced by ``apply_transition`` (1 draw), the
        reward Bernoulli (1 draw), and ``step_reward_probs`` (1 draw) — in that
        order — on a completed trial.
    mapping : TransitionMapping
        The participant's fixed Stage-1 → Stage-2 common-transition mapping.
    reward_probs : numpy.ndarray, shape (4,)
        Reward probabilities at trial start.  **Mutated in place** by one walk
        step on a completed trial; left unchanged on a timeout/skip.
    frame_rate : float
        Measured refresh rate (Hz); fixed durations are rounded to whole frames.
    block, block_type, trial_in_block, trial_global
        Trial metadata for the log row (``block_type`` is ``load`` / ``no_load``
        / ``practice``; it is a label only — load mechanics key off
        ``nback_letter``).
    nback_letter : str or None
        The 1-back letter for this trial (load blocks only); ``None`` in no-load.
    nback_is_match : bool or None
        Whether ``nback_letter`` matches the previous trial's letter.

    Returns
    -------
    dict
        A row keyed by ``TRIAL_COLUMNS`` (without ``participant_id``); missing
        values are ``None`` (logged as ``NA``).

    Raises
    ------
    KeyboardInterrupt
        If Escape is pressed during any flip (caught by ``run_experiment``).
    """
    load = nback_letter is not None
    row: dict = {
        "block": block,
        "block_type": block_type,
        "trial_in_block": trial_in_block,
        "trial_global": trial_global,
        "stage1_choice": None,
        "stage1_rt": None,
        "transition": None,
        "stage2_state": None,
        "stage2_choice": None,
        "stage2_rt": None,
        "reward": None,
        "reward_probs": [float(p) for p in reward_probs],  # snapshot at trial start
        "nback_letter": nback_letter,
        "nback_is_match": nback_is_match,
        "nback_response": None,
        "nback_correct": None,
        "nback_rt": None,
        "timestamp": datetime.now().isoformat(),
    }
    logger.debug(
        "trial %d (block %d, %s, load=%s) start", trial_global, block, block_type, load
    )

    # 1. Inter-trial interval (fixation).
    _hold(
        win,
        lambda: screens.draw_fixation(win, stims.fixation, stims.counter),
        _ms_to_frames(ITI_MS, frame_rate),
    )

    # 2. Stage-1 choice (+ 1-back overlay in load blocks).
    if load:
        screens.set_nback_letter(stims.nback_letter, nback_letter)
    choice, choice_rt, nback_key, nback_rt = _collect_stage1(
        win, stims, frame_rate, load
    )

    # Score and log the n-back response if one was made (even on a skip trial).
    if load and nback_key is not None:
        nback_correct, nback_type = score_nback_response(
            nback_is_match, nback_key, KEY_NBACK_MATCH, KEY_NBACK_NO_MATCH
        )
        row["nback_response"] = nback_key
        row["nback_correct"] = nback_correct
        row["nback_rt"] = nback_rt
        logger.debug("  n-back: %s -> %s (correct=%s)", nback_key, nback_type, nback_correct)

    # Missing Stage-1 choice → "Too slow!", freeze the walk, skip the trial.
    if choice is None:
        _hold(
            win,
            lambda: screens.draw_too_slow(win, stims.too_slow, stims.counter),
            _ms_to_frames(FEEDBACK_MS, frame_rate),
        )
        logger.debug("  trial %d skipped: no Stage-1 choice", trial_global)
        return row

    row["stage1_choice"] = choice
    row["stage1_rt"] = choice_rt

    # 3. Transition (engine call 1).
    stage2_state, transition_type = apply_transition(choice, mapping, rng)
    row["transition"] = transition_type
    row["stage2_state"] = stage2_state

    # 4. Transition reveal (planet arrival).
    _hold(
        win,
        lambda: screens.draw_transition(win, stims.transition, stage2_state, stims.counter),
        _ms_to_frames(TRANSITION_REVEAL_MS, frame_rate),
    )

    # 5. Stage-2 choice.
    stage2_choice, stage2_rt = _collect_stage2(win, stims, frame_rate, stage2_state)

    # Missing Stage-2 choice → "Too slow!", freeze the walk, skip the trial.
    if stage2_choice is None:
        _hold(
            win,
            lambda: screens.draw_too_slow(win, stims.too_slow, stims.counter),
            _ms_to_frames(FEEDBACK_MS, frame_rate),
        )
        logger.debug("  trial %d skipped: no Stage-2 choice", trial_global)
        return row

    row["stage2_choice"] = stage2_choice
    row["stage2_rt"] = stage2_rt

    # 6. Reward (engine call 2) — flat arm index, matches scripts/sanity_check.py.
    arm = stage2_state * 2 + stage2_choice
    reward = int(rng.random() < float(reward_probs[arm]))
    row["reward"] = reward

    # 7. Reward feedback.
    _hold(
        win,
        lambda: screens.draw_reward(win, stims.reward, bool(reward), stims.counter),
        _ms_to_frames(FEEDBACK_MS, frame_rate),
    )

    # 8. Advance the reward walk (engine call 3) — completed trial only; in place.
    reward_probs[:] = step_reward_probs(reward_probs, rng)

    logger.debug(
        "  trial %d done: s1=%d t=%s s2=%d a2=%d r=%d",
        trial_global, choice, transition_type, stage2_state, stage2_choice, reward,
    )
    return row
