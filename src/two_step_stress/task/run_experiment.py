"""Entry point: set up a session, run the two-step task, and tear down cleanly.

Run with::

    python -m two_step_stress.task.run_experiment [--participant ID]
        [--seed N] [--no-practice] [--windowed]

Responsibilities (plan §1): subject info (CLI/GUI), RNG seeding, refresh-rate
calibration, building the reward-walk lifecycle, the block→trial loops, and
clean teardown (flush CSV + write session JSON + close window) on normal exit
or Escape.

RNG consumption order (fixed, for reproducibility from the logged seed)
-----------------------------------------------------------------------
1. ``initialise_reward_probs(rng)``                          — once, before practice
2. practice (unless --no-practice): its own letter stream, then per-trial
   ``apply_transition`` / reward / ``step_reward_probs``
3. for each main block in order: if a load block, ``generate_letter_stream``
   for that block; then per-trial ``apply_transition`` / reward / step

The single ``reward_probs`` array is threaded through practice and all four main
blocks (a continuous walk; plan §2).  Changing the flags (e.g. --no-practice)
changes how much of the stream is consumed, so the same seed reproduces a run
only with the same flags.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import re
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as pkg_version

import numpy as np
import psychopy
from psychopy import core, gui, visual

from two_step_stress.config import (
    BLOCK_ORDERS,
    KEY_LEFT,
    KEY_NBACK_MATCH,
    KEY_NBACK_NO_MATCH,
    KEY_RIGHT,
    N_BLOCKS,
    N_TRIALS_PER_BLOCK,
    NBACK_MATCH_RATE,
)
from two_step_stress.io.data_logging import TrialWriter
from two_step_stress.task import screens
from two_step_stress.task.instructions import (
    run_practice,
    show_block_banner,
    show_break,
    show_end,
    show_instructions,
    show_welcome,
)
from two_step_stress.task.nback import generate_letter_stream
from two_step_stress.task.rewards import initialise_reward_probs
from two_step_stress.task.transitions import build_transition_mapping
from two_step_stress.task.trial import run_trial

logger = logging.getLogger(__name__)

WINDOW_SIZE: tuple[int, int] = (1280, 720)
COMMON_REFRESH_RATES: tuple[float, ...] = (60.0, 120.0, 144.0)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Run the two-step task with a cognitive-load manipulation."
    )
    p.add_argument(
        "--participant", type=str, default=None,
        help="Participant ID (skips the GUI dialog; session defaults to 1).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed override for piloting (default: system entropy).",
    )
    p.add_argument(
        "--no-practice", action="store_true", help="Skip the practice block.",
    )
    p.add_argument(
        "--windowed", action="store_true",
        help="Run windowed (1280x720) instead of fullscreen.",
    )
    p.add_argument(
        "--trials", type=int, default=None, metavar="N",
        help=(
            "PILOTING ONLY: cap trials per main block at N (default: %d). "
            "Affects the main blocks and each load block's letter stream; "
            "practice is unaffected." % N_TRIALS_PER_BLOCK
        ),
    )
    return p.parse_args()


def _collect_subject_info(args: argparse.Namespace) -> tuple[str, int]:
    """Return ``(participant_id, session)`` from CLI (preferred) or a GUI dialog.

    Validates that ``participant_id`` is non-empty and filename-safe *before*
    the window opens; raises ``SystemExit`` otherwise (or if the dialog is
    cancelled).
    """
    if args.participant is not None:
        participant_id = args.participant.strip()
        session = 1
    else:
        info = {"participant_id": "", "session": 1}
        dlg = gui.DlgFromDict(info, title="Two-step task", order=["participant_id", "session"])
        if not dlg.OK:
            raise SystemExit("Cancelled at the subject-info dialog.")
        participant_id = str(info["participant_id"]).strip()
        try:
            session = int(info["session"])
        except (ValueError, TypeError):
            raise SystemExit("session must be an integer.")

    if not participant_id or not _SAFE_ID.match(participant_id):
        raise SystemExit(
            "participant_id must be non-empty and contain only letters, digits, "
            "'_' or '-'."
        )
    return participant_id, session


def _counterbalance_id(participant_id: str) -> int:
    """Resolve a participant ID to a stable non-negative int for counterbalancing.

    Numeric IDs are used directly; non-numeric IDs (e.g. names) are hashed with
    SHA-256 so the same ID always maps to the same counterbalance across runs
    (unlike the salted built-in ``hash``).
    """
    if participant_id.isdigit():
        return int(participant_id)
    digest = hashlib.sha256(participant_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100_000_000


def _measure_frame_rate(win: visual.Window) -> float:
    """Measure the refresh rate (plan §9 C3); fall back to 60 Hz if unavailable."""
    measured = win.getActualFrameRate(
        nIdentical=10, nMaxFrames=120, nWarmUpFrames=10, threshold=1
    )
    if measured is None:
        logger.warning("Could not measure refresh rate; falling back to 60 Hz.")
        return 60.0
    rate = float(measured)
    nearest = min(COMMON_REFRESH_RATES, key=lambda r: abs(r - rate))
    if abs(rate - nearest) / nearest > 0.05:
        logger.warning(
            "Measured refresh rate %.1f Hz deviates >5%% from 60/120/144 Hz; continuing.",
            rate,
        )
    logger.info("Measured refresh rate: %.2f Hz", rate)
    return rate


def _session_metadata(
    *,
    participant_id: str,
    session: int,
    timestamp: str,
    seed: int,
    cb_id: int,
    mapping: dict[int, int],
    block_order: str,
    frame_rate: float,
    win: visual.Window,
    windowed: bool,
    practice: bool,
    total_treasure: int,
    duration_s: float,
) -> dict:
    """Assemble the session-level metadata sidecar (plan §6)."""
    try:
        package_version = pkg_version("two-step-stress")
    except PackageNotFoundError:
        package_version = "unknown"
    return {
        "participant_id": participant_id,
        "session": session,
        "timestamp": timestamp,
        "rng_seed": seed,
        "counterbalance_id": cb_id,
        "transition_mapping": mapping,
        "block_order": block_order,
        "key_mapping": {
            "left": KEY_LEFT,
            "right": KEY_RIGHT,
            "nback_match": KEY_NBACK_MATCH,
            "nback_no_match": KEY_NBACK_NO_MATCH,
        },
        "frame_rate_hz": frame_rate,
        "window_size": [int(v) for v in win.size],
        "fullscreen": not windowed,
        "practice": practice,
        "psychopy_version": psychopy.__version__,
        "package_version": package_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "total_treasure": total_treasure,
        "duration_s": round(duration_s, 1),
    }


def main() -> None:
    """Run one full session of the two-step task.

    Parses CLI arguments, collects participant info (CLI or GUI dialog), seeds
    the RNG, opens and calibrates the window, then drives the session —
    welcome, instructions, optional practice, four main blocks with banners and
    breaks, and the end screen — logging one CSV row per trial.  On normal exit
    or an Escape ``KeyboardInterrupt`` it flushes the CSV, writes the session
    JSON sidecar, and closes the window cleanly.

    Returns
    -------
    None
    """
    start_time = time.time()
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting two-step task")

    # --- session setup (before the window opens) ---
    participant_id, session = _collect_subject_info(args)
    cb_id = _counterbalance_id(participant_id)
    mapping = build_transition_mapping(cb_id)
    block_order = BLOCK_ORDERS[cb_id % 2]  # decision Q8: derives from participant_id

    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info(
        "Participant %s | session %d | cb_id %d | mapping %s | order %s | seed %s",
        participant_id, session, cb_id, mapping, block_order, seed,
    )

    if args.trials is not None and args.trials < 1:
        raise SystemExit("--trials must be a positive integer.")
    trials_per_block = args.trials if args.trials is not None else N_TRIALS_PER_BLOCK
    if args.trials is not None:
        logger.warning(
            "PILOTING: capping trials per main block at %d (default %d).",
            trials_per_block, N_TRIALS_PER_BLOCK,
        )

    # --- window + calibration ---
    win_kwargs: dict = dict(
        units="height", color="black", colorSpace="rgb", fullscr=not args.windowed
    )
    if args.windowed:
        win_kwargs["size"] = WINDOW_SIZE
    win = visual.Window(**win_kwargs)
    win.mouseVisible = args.windowed
    frame_rate = _measure_frame_rate(win)

    # --- stimuli, logging, and the continuous reward walk ---
    stims = screens.build_session_stimuli(win)
    writer = TrialWriter(participant_id, timestamp)
    reward_probs = initialise_reward_probs(rng)  # RNG step 1; threaded everywhere below

    total_treasure = 0
    try:
        show_welcome(win, stims, frame_rate)
        show_instructions(win, stims, frame_rate)
        if not args.no_practice:
            run_practice(
                win, stims, rng, mapping, reward_probs, frame_rate, writer, participant_id
            )

        trial_global = 0
        for block_num, code in enumerate(block_order, start=1):
            block_type = "load" if code == "A" else "no_load"
            show_block_banner(win, stims, frame_rate, block_num, block_type)
            logger.info("Block %d start (%s)", block_num, block_type)

            letters = is_match = None
            if block_type == "load":
                letters, is_match = generate_letter_stream(
                    trials_per_block, NBACK_MATCH_RATE, rng
                )

            for t in range(trials_per_block):
                trial_global += 1
                letter = letters[t] if block_type == "load" else None
                match = is_match[t] if block_type == "load" else None
                row = run_trial(
                    win, stims, rng, mapping, reward_probs, frame_rate,
                    block=block_num,
                    block_type=block_type,
                    trial_in_block=t + 1,
                    trial_global=trial_global,
                    nback_letter=letter,
                    nback_is_match=match,
                )
                row["participant_id"] = participant_id
                writer.write_row(row)
                if row["reward"] == 1:
                    total_treasure += 1
                    screens.update_treasure_counter(stims.counter, total_treasure)

            logger.info("Block %d complete", block_num)
            if block_num < N_BLOCKS:
                show_break(win, stims, frame_rate, block_num)

        show_end(win, stims, frame_rate, total_treasure)
        logger.info("Session completed normally (treasure=%d)", total_treasure)
    except KeyboardInterrupt:
        logger.warning("Session aborted via Escape (treasure=%d)", total_treasure)
    finally:
        metadata = _session_metadata(
            participant_id=participant_id,
            session=session,
            timestamp=timestamp,
            seed=seed,
            cb_id=cb_id,
            mapping=mapping,
            block_order=block_order,
            frame_rate=frame_rate,
            win=win,
            windowed=args.windowed,
            practice=not args.no_practice,
            total_treasure=total_treasure,
            duration_s=time.time() - start_time,
        )
        writer.write_session_json(metadata)
        writer.close()
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
