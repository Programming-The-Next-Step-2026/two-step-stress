"""Participant-facing screens and the practice loop for the two-step task.

All screens reuse the ``screens.MessageScreen`` infrastructure
(``screens.set_message`` + ``screens.show_message``) and wait for a keypress.
Escape on any screen raises ``KeyboardInterrupt`` (caught by
``run_experiment``); the only exception is the end screen, where Escape is a
normal exit.

Screen flow over a session (driven by ``run_experiment``):
    welcome → instructions → practice → [banner → block → break] × 4 → end

The ``frame_rate`` argument on the message screens is unused (they are not
frame-locked); it is kept for a uniform call signature across the screen API
and is used for real by :func:`run_practice`, which runs trials.
"""

from __future__ import annotations

import logging

import numpy as np
from psychopy import event, visual

from two_step_stress.config import (
    KEY_LEFT,
    KEY_NBACK_MATCH,
    KEY_NBACK_NO_MATCH,
    KEY_RIGHT,
    N_BLOCKS,
    N_PRACTICE_LOAD,
    N_PRACTICE_NO_LOAD,
    N_TRIALS_PRACTICE,
    NBACK_MATCH_RATE,
)
from two_step_stress.task import screens
from two_step_stress.task.nback import generate_letter_stream
from two_step_stress.task.trial import run_trial
from two_step_stress.task.transitions import TransitionMapping
from two_step_stress.io.data_logging import TrialWriter

logger = logging.getLogger(__name__)

_SPACE: str = "space"
_BACKSPACE: str = "backspace"
_ESCAPE: str = "escape"

CONTINUE_HINT: str = "Press SPACE to continue"

WELCOME_TEXT: str = (
    "Welcome, and thank you for taking part!\n\n"
    "You're about to pilot spaceships to distant planets and meet aliens who "
    "sometimes share treasure. Your job is simply to collect as much treasure "
    "as you can.\n\n"
    "The whole session takes about 20 minutes, with short breaks along the way."
)

# Short instruction pages (advance on SPACE, go back on BACKSPACE).  Keys are
# pulled from config so the on-screen text always matches the live mapping.
INSTRUCTION_PAGES: tuple[str, ...] = (
    # 1. Cover story.
    "On each trip you'll first choose one of two spaceships.\n\n"
    "Each spaceship flies you to a planet, where two aliens live. Choose an "
    "alien, and it may reward you with treasure.",
    # 2. Choice keys.
    f"You make every choice with two keys:\n\n"
    f"'{KEY_LEFT.upper()}' chooses the LEFT option, '{KEY_RIGHT.upper()}' "
    f"chooses the RIGHT option.\n\n"
    f"It's the same for picking a spaceship and for picking an alien. Try to "
    f"answer before the time runs out.",
    # 3. Common vs rare (no numbers).
    "Each spaceship has a home planet it USUALLY flies to.\n\n"
    "But space travel is unpredictable — every so often a spaceship ends up at "
    "the OTHER planet instead. It's worth learning where each ship usually goes.",
    # 4. Drifting rewards.
    "The aliens' generosity slowly drifts over time.\n\n"
    "An alien that shares treasure often now may turn stingy later, and a "
    "stingy one may come good. Keep an eye on who's paying out lately.",
    # 5. N-back rule + keys (taught to everyone).
    f"In some sections, a letter will briefly flash as you start each trip.\n\n"
    f"Your extra job: decide whether it's the SAME as the letter from the "
    f"trip just before it.\n\n"
    f"Press '{KEY_NBACK_MATCH.upper()}' if it matches the previous letter, or "
    f"'{KEY_NBACK_NO_MATCH.upper()}' if it doesn't — while you choose your "
    f"spaceship. We'll warn you when these sections are coming.",
)


def _wait(keys: list[str]) -> str:
    """Block until one of ``keys`` (or Escape) is pressed; return the key.

    Raises ``KeyboardInterrupt`` on Escape.  ``waitKeys`` clears the buffer
    first, so a keypress left over from the previous screen won't skip ahead.
    """
    pressed = event.waitKeys(keyList=[*keys, _ESCAPE])
    if _ESCAPE in pressed:
        raise KeyboardInterrupt("Escape pressed on an instruction screen")
    return pressed[0]


def show_welcome(win: visual.Window, stims: screens.SessionStimuli, frame_rate: float) -> None:
    """Show the one-screen welcome and wait for SPACE.

    Parameters
    ----------
    win : psychopy.visual.Window
        The open window.
    stims : screens.SessionStimuli
        Pre-built stimuli (the reusable message screen is used).
    frame_rate : float
        Measured refresh rate; unused here (kept for a uniform screen API).

    Returns
    -------
    None
    """
    logger.info("Screen: welcome")
    screens.set_message(stims.message, WELCOME_TEXT, "Press SPACE to begin")
    screens.show_message(win, stims.message)
    _wait([_SPACE])


def show_instructions(
    win: visual.Window, stims: screens.SessionStimuli, frame_rate: float
) -> None:
    """Show the multi-page instructions; SPACE advances, BACKSPACE goes back.

    Parameters
    ----------
    win : psychopy.visual.Window
        The open window.
    stims : screens.SessionStimuli
        Pre-built stimuli (the reusable message screen is used).
    frame_rate : float
        Measured refresh rate; unused here (kept for a uniform screen API).

    Returns
    -------
    None
    """
    logger.info("Screen: instructions (%d pages)", len(INSTRUCTION_PAGES))
    idx = 0
    while idx < len(INSTRUCTION_PAGES):
        hint = "SPACE: next" + ("    BACKSPACE: back" if idx > 0 else "")
        screens.set_message(stims.message, INSTRUCTION_PAGES[idx], hint)
        screens.show_message(win, stims.message)
        if _wait([_SPACE, _BACKSPACE]) == _BACKSPACE:
            idx = max(0, idx - 1)
        else:
            idx += 1


def run_practice(
    win: visual.Window,
    stims: screens.SessionStimuli,
    rng: np.random.Generator,
    mapping: TransitionMapping,
    reward_probs: np.ndarray,
    frame_rate: float,
    writer: TrialWriter,
    participant_id: str,
) -> None:
    """Run the 12-trial practice block (8 no-load, then 4 load).

    The ``reward_probs`` array passed here is the single, continuous session
    reward walk (initialised once in ``run_experiment`` before practice and
    threaded through practice and all main blocks): practice trials advance it
    in place, exactly like main trials.  Practice does, however, use its OWN
    1-back letter stream (generated below) so it doesn't consume the main
    blocks' letter streams.  All practice trials are logged with ``block=0`` /
    ``block_type="practice"`` and show explicit COMMON/RARE transition feedback.

    Parameters
    ----------
    win : psychopy.visual.Window
        The open window.
    stims : screens.SessionStimuli
        Pre-built stimuli.
    rng : numpy.random.Generator
        The participant RNG (shared with the main session).
    mapping : TransitionMapping
        The participant's Stage-1 → Stage-2 common-transition mapping.
    reward_probs : numpy.ndarray, shape (4,)
        The continuous session reward walk; advanced in place by each trial.
    frame_rate : float
        Measured refresh rate, passed through to :func:`run_trial`.
    writer : TrialWriter
        Open CSV writer; practice rows are written with ``participant_id`` set.
    participant_id : str
        Participant identifier, stamped onto each logged row.

    Returns
    -------
    None
    """
    logger.info(
        "Practice block start (%d trials: %d no-load + %d load)",
        N_TRIALS_PRACTICE, N_PRACTICE_NO_LOAD, N_PRACTICE_LOAD,
    )

    # Practice's own letter stream (load-practice trials only).
    letters, is_match = generate_letter_stream(N_PRACTICE_LOAD, NBACK_MATCH_RATE, rng)

    screens.set_message(
        stims.message,
        "Let's try a few practice trips first.\n\n"
        "This time we'll tell you which planet each spaceship reached, so you "
        "can get a feel for how it works.",
        "Press SPACE to start practice",
    )
    screens.show_message(win, stims.message)
    _wait([_SPACE])

    def _practice_trial(trial_num: int, letter: str | None, match: bool | None) -> None:
        row = run_trial(
            win, stims, rng, mapping, reward_probs, frame_rate,
            block=0,
            block_type="practice",
            trial_in_block=trial_num,
            trial_global=trial_num,
            nback_letter=letter,
            nback_is_match=match,
            show_transition_label=True,
        )
        row["participant_id"] = participant_id
        writer.write_row(row)

    for i in range(N_PRACTICE_NO_LOAD):
        _practice_trial(i + 1, None, None)
    for j in range(N_PRACTICE_LOAD):
        _practice_trial(N_PRACTICE_NO_LOAD + j + 1, letters[j], is_match[j])

    logger.info("Practice block complete")

    screens.set_message(
        stims.message, "Ready?\n\nThe real game begins now.", "Press SPACE to start"
    )
    screens.show_message(win, stims.message)
    _wait([_SPACE])


def show_block_banner(
    win: visual.Window,
    stims: screens.SessionStimuli,
    frame_rate: float,
    block_num: int,
    block_type: str,
) -> None:
    """Show the pre-block banner naming the upcoming condition; advance on SPACE.

    Parameters
    ----------
    win : psychopy.visual.Window
        The open window.
    stims : screens.SessionStimuli
        Pre-built stimuli (the reusable message screen is used).
    frame_rate : float
        Measured refresh rate; unused here (kept for a uniform screen API).
    block_num : int
        The upcoming block number (1-based).
    block_type : str
        ``"load"`` or ``"no_load"``; selects the banner wording.

    Returns
    -------
    None
    """
    logger.info("Screen: block %d banner (%s)", block_num, block_type)
    if block_type == "load":
        desc = (
            "Heads up: in this section a letter flashes at the start of each "
            f"trip. While choosing your spaceship, press '{KEY_NBACK_MATCH.upper()}' "
            f"if it matches the previous letter or '{KEY_NBACK_NO_MATCH.upper()}' "
            "if it doesn't."
        )
    else:
        desc = "In this section, just focus on spaceships and aliens — no letters to track."
    screens.set_message(
        stims.message, f"Section {block_num} of {N_BLOCKS}\n\n{desc}", "Press SPACE to begin"
    )
    screens.show_message(win, stims.message)
    _wait([_SPACE])


def show_break(
    win: visual.Window, stims: screens.SessionStimuli, frame_rate: float, block_num: int
) -> None:
    """Show the between-block break screen; advance on SPACE.

    Parameters
    ----------
    win : psychopy.visual.Window
        The open window.
    stims : screens.SessionStimuli
        Pre-built stimuli (the reusable message screen is used).
    frame_rate : float
        Measured refresh rate; unused here (kept for a uniform screen API).
    block_num : int
        The block just completed (1-based).

    Returns
    -------
    None
    """
    logger.info("Screen: break after block %d", block_num)
    screens.set_message(
        stims.message,
        f"Section {block_num} of {N_BLOCKS} complete.\n\n"
        "Feel free to take a short break.\n\n"
        "When you're ready, carry on.",
        CONTINUE_HINT,
    )
    screens.show_message(win, stims.message)
    _wait([_SPACE])


def show_end(
    win: visual.Window,
    stims: screens.SessionStimuli,
    frame_rate: float,
    total_treasure: int,
) -> None:
    """Show the debrief / end screen; exit on SPACE or ESCAPE (no abort).

    Parameters
    ----------
    win : psychopy.visual.Window
        The open window.
    stims : screens.SessionStimuli
        Pre-built stimuli (the reusable message screen is used).
    frame_rate : float
        Measured refresh rate; unused here (kept for a uniform screen API).
    total_treasure : int
        The participant's final treasure total, shown on screen.

    Returns
    -------
    None
    """
    logger.info("Screen: end (total_treasure=%d)", total_treasure)
    screens.set_message(
        stims.message,
        "All done — thank you!\n\n"
        f"You collected {total_treasure} treasure in total.\n\n"
        "Please let the experimenter know you've finished.",
        "Press SPACE or ESCAPE to exit",
    )
    screens.show_message(win, stims.message)
    event.waitKeys(keyList=[_SPACE, _ESCAPE])
