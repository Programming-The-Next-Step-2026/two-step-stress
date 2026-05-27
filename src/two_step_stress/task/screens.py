"""PsychoPy stimulus builders and draw helpers for the two-step task.

This module owns *only* the visual layer (per docs/psychopy_plan.md §1):

* **Builders** (``build_*``) construct ``visual.*`` objects once, given a
  ``win``.  Call them at session start.
* **Draw helpers** (``draw_*`` / ``show_*``) take those objects, draw them, and
  flip the window.  Each returns the ``win.flip()`` timestamp so the caller can
  align timing.

It does **not** own timing, key collection, or logging — those live in
``trial.py`` / ``run_experiment.py``.

Cover story (Decker et al., 2016): spaceships → planets → aliens → treasure.
Stimuli are **abstract geometric shapes** (filled rectangles, circles, and a
triangular rocket "nose"), as in most published Daw/Otto two-step tasks —
PsychoPy's ``TextStim`` cannot render colour emoji reliably (it draws only basic
glyph outlines).

PNG-swappability
----------------
Each cover-story object is built through a small factory: :func:`_rocket`
(spaceship), :func:`_alien`, :func:`_planet`, and :func:`_disc` (reward
feedback).  To switch to image assets later, change those factories to return
``visual.ImageStim`` objects — the containers, builders, and draw helpers all
stay unchanged.

Layout assumes the window is created with ``units="height"`` and a black
background.
"""

from __future__ import annotations

from dataclasses import dataclass

from psychopy import visual

# --- colours ------------------------------------------------------------------
TEXT_COLOR: str = "white"
TOO_SLOW_COLOR: str = "#E05050"
LEFT_COLOR: str = "#4C9BE8"   # Stage-1 left spaceship / left alien (blue)
RIGHT_COLOR: str = "#E8A33D"  # Stage-1 right spaceship / right alien (amber)
NOSE_COLOR: str = "white"     # rocket nose triangle
SHIP_NEUTRAL_COLOR: str = "#C8CDD2"  # transition-reveal ship (choice-agnostic)
ALIEN_LABEL_COLOR: str = "white"
REWARD_COLOR: str = "#3FB04F"     # rewarded feedback disc (green)
NO_REWARD_COLOR: str = "#7A7A7A"  # no-reward feedback disc (grey)
# Stage-2 planet colours, keyed by stage2_state — distinct hues (blue vs red).
PLANET_COLORS: dict[int, str] = {0: "#3F6FB0", 1: "#B0473F"}

# --- units & layout (height units; no magic numbers in the draw helpers) ------
UNITS: str = "height"

# Rockets (Stage-1 spaceships and the transition-reveal ship).
SHIP_W: float = 0.16
SHIP_H: float = 0.22
NOSE_H: float = 0.07

# Planets.
PLANET_RADIUS_BACKDROP: float = 0.42  # Stage-2 backdrop circle
PLANET_RADIUS_REVEAL: float = 0.28    # transition-reveal circle
PLANET_EDGES: int = 64                # smooth circle outline

# Aliens.
ALIEN_W: float = 0.16
ALIEN_H: float = 0.20
ALIEN_LABEL_HEIGHT: float = 0.09

# Reward-feedback discs.
REWARD_RADIUS: float = 0.16
REWARD_TEXT_HEIGHT: float = 0.10

# Transition-reveal label (practice only: "COMMON" / "RARE" across the planet).
REVEAL_LABEL_HEIGHT: float = 0.07

# Text sizes.
FIX_HEIGHT: float = 0.06
NBACK_HEIGHT: float = 0.16
TEXT_HEIGHT: float = 0.045
HINT_HEIGHT: float = 0.035
COUNTER_HEIGHT: float = 0.04
WRAP_WIDTH: float = 1.30

# Positions.
LEFT_POS: tuple[float, float] = (-0.33, 0.0)
RIGHT_POS: tuple[float, float] = (0.33, 0.0)
LETTER_POS: tuple[float, float] = (0.0, 0.34)
COUNTER_POS: tuple[float, float] = (0.0, -0.45)
MESSAGE_POS: tuple[float, float] = (0.0, 0.06)
HINT_POS: tuple[float, float] = (0.0, -0.40)
REVEAL_PLANET_POS: tuple[float, float] = (0.0, -0.06)
REVEAL_SHIP_POS: tuple[float, float] = (0.0, 0.26)

DEFAULT_HINT: str = "Press SPACE to continue"


# --- stimulus containers ------------------------------------------------------
@dataclass
class Stage1Stims:
    """Two colour-coded rockets (body + nose) for the Stage-1 choice."""

    left_body: visual.Rect
    left_nose: visual.ShapeStim
    right_body: visual.Rect
    right_nose: visual.ShapeStim


@dataclass
class TransitionStims:
    """Chosen rocket arriving at the (colour-coded) destination planet.

    ``label`` carries the optional "COMMON"/"RARE" practice feedback drawn
    across the planet (empty/hidden in the main task).
    """

    planet: visual.Circle
    ship_body: visual.Rect
    ship_nose: visual.ShapeStim
    label: visual.TextStim


@dataclass
class Stage2Stims:
    """Planet-coloured backdrop circle plus its two labelled aliens."""

    planet: visual.Circle
    left_alien: visual.Rect
    left_label: visual.TextStim
    right_alien: visual.Rect
    right_label: visual.TextStim


@dataclass
class RewardStims:
    """Reward (green disc + “+1”) vs no-reward (grey disc + “0”) feedback."""

    reward_disc: visual.Circle
    plus_one: visual.TextStim
    no_reward_disc: visual.Circle
    zero: visual.TextStim


@dataclass
class MessageScreen:
    """A centred block of text plus a press-to-continue hint.

    Reused for the welcome, instruction pages, block-start banner, between-block
    break, and end screens — set ``.main.text`` / ``.hint.text`` per use via
    :func:`set_message`.
    """

    main: visual.TextStim
    hint: visual.TextStim


@dataclass
class SessionStimuli:
    """All reusable stimuli, built once at session start."""

    fixation: visual.TextStim
    stage1: Stage1Stims
    transition: TransitionStims
    stage2: Stage2Stims
    reward: RewardStims
    too_slow: visual.TextStim
    nback_letter: visual.TextStim
    counter: visual.TextStim
    message: MessageScreen


# --- low-level factories (the PNG-swap points) -------------------------------
def _text(
    win: visual.Window,
    text: str,
    height: float,
    pos: tuple[float, float],
    color: str = TEXT_COLOR,
    wrap: float | None = None,
) -> visual.TextStim:
    """Build a plain text stimulus in the default (non-emoji) font."""
    return visual.TextStim(
        win,
        text=text,
        height=height,
        pos=pos,
        color=color,
        units=UNITS,
        wrapWidth=wrap,
        alignText="center",
        anchorHoriz="center",
    )


def _rocket(
    win: visual.Window,
    color: str,
    pos: tuple[float, float],
) -> tuple[visual.Rect, visual.ShapeStim]:
    """Build a rocket: a filled body rectangle plus a white nose triangle.

    Swap this for a ``visual.ImageStim`` of a spaceship to use art assets.
    """
    body = visual.Rect(
        win, width=SHIP_W, height=SHIP_H, pos=pos, units=UNITS,
        fillColor=color, lineColor=None,
    )
    nose = visual.ShapeStim(
        win,
        vertices=[(-SHIP_W / 2, 0.0), (SHIP_W / 2, 0.0), (0.0, NOSE_H)],
        pos=(pos[0], pos[1] + SHIP_H / 2),
        units=UNITS,
        fillColor=NOSE_COLOR,
        lineColor=NOSE_COLOR,
    )
    return body, nose


def _alien(
    win: visual.Window,
    color: str,
    label: str,
    pos: tuple[float, float],
) -> tuple[visual.Rect, visual.TextStim]:
    """Build an alien: a colour-coded box with a white identifying label."""
    box = visual.Rect(
        win, width=ALIEN_W, height=ALIEN_H, pos=pos, units=UNITS,
        fillColor=color, lineColor=None,
    )
    label_stim = _text(win, label, ALIEN_LABEL_HEIGHT, pos, color=ALIEN_LABEL_COLOR)
    return box, label_stim


def _planet(
    win: visual.Window,
    radius: float,
    pos: tuple[float, float] = (0.0, 0.0),
) -> visual.Circle:
    """Build a planet circle (fill colour is set per state in the draw helper)."""
    return visual.Circle(
        win, radius=radius, edges=PLANET_EDGES, pos=pos, units=UNITS,
        fillColor=PLANET_COLORS[0], lineColor=None,
    )


def _disc(
    win: visual.Window,
    radius: float,
    color: str,
    pos: tuple[float, float] = (0.0, 0.0),
) -> visual.Circle:
    """Build a solid feedback disc."""
    return visual.Circle(
        win, radius=radius, edges=PLANET_EDGES, pos=pos, units=UNITS,
        fillColor=color, lineColor=None,
    )


# --- builders -----------------------------------------------------------------
def build_fixation(win: visual.Window) -> visual.TextStim:
    """Build the central fixation cross shown during the inter-trial interval.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.

    Returns
    -------
    psychopy.visual.TextStim
        A "+" stimulus centred on screen.
    """
    return _text(win, "+", FIX_HEIGHT, (0.0, 0.0))


def build_stage1(win: visual.Window) -> Stage1Stims:
    """Build the two colour-coded Stage-1 rockets (left and right).

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.

    Returns
    -------
    Stage1Stims
        The left/right rocket bodies (blue/amber) and their nose triangles.
    """
    left_body, left_nose = _rocket(win, LEFT_COLOR, LEFT_POS)
    right_body, right_nose = _rocket(win, RIGHT_COLOR, RIGHT_POS)
    return Stage1Stims(
        left_body=left_body,
        left_nose=left_nose,
        right_body=right_body,
        right_nose=right_nose,
    )


def build_transition(win: visual.Window) -> TransitionStims:
    """Build the transition-reveal stimuli: a neutral rocket and a planet.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.

    Returns
    -------
    TransitionStims
        The destination planet circle, the arriving rocket (body + nose), and
        the (initially empty) COMMON/RARE practice label.
    """
    ship_body, ship_nose = _rocket(win, SHIP_NEUTRAL_COLOR, REVEAL_SHIP_POS)
    return TransitionStims(
        planet=_planet(win, PLANET_RADIUS_REVEAL, REVEAL_PLANET_POS),
        ship_body=ship_body,
        ship_nose=ship_nose,
        label=_text(win, "", REVEAL_LABEL_HEIGHT, REVEAL_PLANET_POS),
    )


def build_stage2(win: visual.Window) -> Stage2Stims:
    """Build the Stage-2 stimuli: a planet backdrop and its two aliens.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.

    Returns
    -------
    Stage2Stims
        The backdrop planet circle plus the left/right alien boxes and their
        "L"/"R" labels.
    """
    left_alien, left_label = _alien(win, LEFT_COLOR, "L", LEFT_POS)
    right_alien, right_label = _alien(win, RIGHT_COLOR, "R", RIGHT_POS)
    return Stage2Stims(
        planet=_planet(win, PLANET_RADIUS_BACKDROP, (0.0, 0.0)),
        left_alien=left_alien,
        left_label=left_label,
        right_alien=right_alien,
        right_label=right_label,
    )


def build_reward_feedback(win: visual.Window) -> RewardStims:
    """Build the reward and no-reward feedback stimuli.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.

    Returns
    -------
    RewardStims
        The green "+1" reward disc and the grey "0" no-reward disc.
    """
    return RewardStims(
        reward_disc=_disc(win, REWARD_RADIUS, REWARD_COLOR),
        plus_one=_text(win, "+1", REWARD_TEXT_HEIGHT, (0.0, 0.0)),
        no_reward_disc=_disc(win, REWARD_RADIUS, NO_REWARD_COLOR),
        zero=_text(win, "0", REWARD_TEXT_HEIGHT, (0.0, 0.0)),
    )


def build_too_slow(win: visual.Window) -> visual.TextStim:
    """Build the 'Too slow!' timeout message stimulus.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.

    Returns
    -------
    psychopy.visual.TextStim
        The red 'Too slow!' message.
    """
    return _text(win, "Too slow!", TEXT_HEIGHT * 1.4, (0.0, 0.0), color=TOO_SLOW_COLOR)


def build_nback_letter(win: visual.Window) -> visual.TextStim:
    """Build the 1-back letter stimulus (text set per trial).

    The glyph uses the default (non-emoji) font; set its text each trial via
    :func:`set_nback_letter`.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.

    Returns
    -------
    psychopy.visual.TextStim
        An (initially empty) large letter, positioned above the rockets.
    """
    return _text(win, "", NBACK_HEIGHT, LETTER_POS)


def build_treasure_counter(win: visual.Window) -> visual.TextStim:
    """Build the running treasure counter shown at the bottom of the screen.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.

    Returns
    -------
    psychopy.visual.TextStim
        A "Treasure: 0" counter; update it via :func:`update_treasure_counter`.
    """
    return _text(win, "Treasure: 0", COUNTER_HEIGHT, COUNTER_POS)


def build_message_screen(win: visual.Window) -> MessageScreen:
    """Build a reusable text page (welcome / instructions / banner / break / end).

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.

    Returns
    -------
    MessageScreen
        A centred body text plus a press-to-continue hint; set their text via
        :func:`set_message`.
    """
    return MessageScreen(
        main=_text(win, "", TEXT_HEIGHT, MESSAGE_POS, wrap=WRAP_WIDTH),
        hint=_text(win, DEFAULT_HINT, HINT_HEIGHT, HINT_POS),
    )


def build_session_stimuli(win: visual.Window) -> SessionStimuli:
    """Build every reusable stimulus once, at session start.

    Convenience aggregate over the individual ``build_*`` functions; the
    returned container is threaded through the trial loop.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.

    Returns
    -------
    SessionStimuli
        All stimuli needed for a session.
    """
    return SessionStimuli(
        fixation=build_fixation(win),
        stage1=build_stage1(win),
        transition=build_transition(win),
        stage2=build_stage2(win),
        reward=build_reward_feedback(win),
        too_slow=build_too_slow(win),
        nback_letter=build_nback_letter(win),
        counter=build_treasure_counter(win),
        message=build_message_screen(win),
    )


# --- text/state setters (formatting owned here, not in the trial loop) -------
def set_nback_letter(letter: visual.TextStim, char: str) -> None:
    """Set the displayed 1-back letter.

    Parameters
    ----------
    letter : psychopy.visual.TextStim
        The n-back letter stimulus (from :func:`build_nback_letter`).
    char : str
        The letter to display this trial.

    Returns
    -------
    None
    """
    letter.text = char


def update_treasure_counter(counter: visual.TextStim, total: int) -> None:
    """Update the running treasure total shown in the counter.

    Parameters
    ----------
    counter : psychopy.visual.TextStim
        The treasure counter stimulus.
    total : int
        The cumulative treasure to display.

    Returns
    -------
    None
    """
    counter.text = f"Treasure: {total}"


def set_message(
    screen: MessageScreen,
    main_text: str,
    hint_text: str = DEFAULT_HINT,
) -> None:
    """Set the body and hint text of a reusable message screen.

    Parameters
    ----------
    screen : MessageScreen
        The message screen to update.
    main_text : str
        The body text.
    hint_text : str, optional
        The press-to-continue hint (default :data:`DEFAULT_HINT`).

    Returns
    -------
    None
    """
    screen.main.text = main_text
    screen.hint.text = hint_text


# --- draw helpers (draw + flip; no timing or key collection) -----------------
def draw_fixation(
    win: visual.Window,
    fixation: visual.TextStim,
    counter: visual.TextStim,
) -> float:
    """Draw the ITI fixation cross and treasure counter, then flip.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.
    fixation : psychopy.visual.TextStim
        The fixation cross.
    counter : psychopy.visual.TextStim
        The treasure counter.

    Returns
    -------
    float
        The ``win.flip()`` timestamp of this frame.
    """
    fixation.draw()
    counter.draw()
    return win.flip()


def draw_stage1(
    win: visual.Window,
    stims: Stage1Stims,
    counter: visual.TextStim,
    letter: visual.TextStim | None = None,
    show_letter: bool = False,
) -> float:
    """Draw the Stage-1 rockets (and optionally the 1-back letter), then flip.

    The caller decides whether the letter is visible this frame (it is shown
    only for the first 500 ms of load-block Stage-1).

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.
    stims : Stage1Stims
        The Stage-1 rocket stimuli.
    counter : psychopy.visual.TextStim
        The treasure counter.
    letter : psychopy.visual.TextStim or None, optional
        The 1-back letter stimulus (load blocks only).
    show_letter : bool, optional
        Whether to draw ``letter`` this frame.

    Returns
    -------
    float
        The ``win.flip()`` timestamp of this frame.
    """
    stims.left_body.draw()
    stims.left_nose.draw()
    stims.right_body.draw()
    stims.right_nose.draw()
    if show_letter and letter is not None:
        letter.draw()
    counter.draw()
    return win.flip()


def draw_transition(
    win: visual.Window,
    stims: TransitionStims,
    stage2_state: int,
    counter: visual.TextStim,
    label_text: str | None = None,
) -> float:
    """Draw the transition reveal (planet backdrop + arriving ship), then flip.

    If ``label_text`` is given (practice only), it is drawn across the planet
    as explicit "COMMON"/"RARE" feedback.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.
    stims : TransitionStims
        The transition-reveal stimuli.
    stage2_state : int
        The reached Stage-2 state (0/1); selects the planet colour.
    counter : psychopy.visual.TextStim
        The treasure counter.
    label_text : str or None, optional
        "COMMON"/"RARE" practice label; omitted in the main task.

    Returns
    -------
    float
        The ``win.flip()`` timestamp of this frame.
    """
    stims.planet.fillColor = PLANET_COLORS[stage2_state]
    stims.planet.draw()
    stims.ship_body.draw()
    stims.ship_nose.draw()
    if label_text is not None:
        stims.label.text = label_text
        stims.label.draw()
    counter.draw()
    return win.flip()


def draw_stage2(
    win: visual.Window,
    stims: Stage2Stims,
    stage2_state: int,
    counter: visual.TextStim,
) -> float:
    """Draw the Stage-2 aliens on their planet-coloured backdrop, then flip.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.
    stims : Stage2Stims
        The Stage-2 stimuli.
    stage2_state : int
        The reached Stage-2 state (0/1); selects the planet colour.
    counter : psychopy.visual.TextStim
        The treasure counter.

    Returns
    -------
    float
        The ``win.flip()`` timestamp of this frame.
    """
    stims.planet.fillColor = PLANET_COLORS[stage2_state]
    stims.planet.draw()
    stims.left_alien.draw()
    stims.left_label.draw()
    stims.right_alien.draw()
    stims.right_label.draw()
    counter.draw()
    return win.flip()


def draw_reward(
    win: visual.Window,
    stims: RewardStims,
    rewarded: bool,
    counter: visual.TextStim,
) -> float:
    """Draw reward (green disc + "+1") or no-reward (grey disc + "0"), then flip.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.
    stims : RewardStims
        The feedback stimuli.
    rewarded : bool
        Whether this trial was rewarded.
    counter : psychopy.visual.TextStim
        The treasure counter.

    Returns
    -------
    float
        The ``win.flip()`` timestamp of this frame.
    """
    if rewarded:
        stims.reward_disc.draw()
        stims.plus_one.draw()
    else:
        stims.no_reward_disc.draw()
        stims.zero.draw()
    counter.draw()
    return win.flip()


def draw_too_slow(
    win: visual.Window,
    too_slow: visual.TextStim,
    counter: visual.TextStim,
) -> float:
    """Draw the 'Too slow!' timeout message and counter, then flip.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.
    too_slow : psychopy.visual.TextStim
        The 'Too slow!' message.
    counter : psychopy.visual.TextStim
        The treasure counter.

    Returns
    -------
    float
        The ``win.flip()`` timestamp of this frame.
    """
    too_slow.draw()
    counter.draw()
    return win.flip()


def show_message(win: visual.Window, screen: MessageScreen) -> float:
    """Draw a message screen (welcome / instructions / banner / break / end), then flip.

    Parameters
    ----------
    win : psychopy.visual.Window
        Target window.
    screen : MessageScreen
        The message screen to display.

    Returns
    -------
    float
        The ``win.flip()`` timestamp of this frame.
    """
    screen.main.draw()
    screen.hint.draw()
    return win.flip()
