"""Visual smoke test for the screens.py stimulus layer.

Opens a small windowed PsychoPy window and draws each major screen in sequence,
pausing 1.5 s between them, so you can eyeball that the emoji stimuli, colours,
and layout actually render on this machine. Press ESCAPE at any point to abort.

This is a throwaway dev script (NOT part of the package), so it is allowed to
use core.wait and print — neither belongs in the real experiment loop.

Run from the repo root:
    python scripts/smoke_test_screens.py
"""

from psychopy import core, event, visual

from two_step_stress.task import screens

WINDOW_SIZE = (1280, 720)   # windowed, per the plan's dev default
STEP_SECONDS = 1.5
DEMO_STATE = 1              # show planet B (red) on the transition / Stage-2 screens


def _check_escape() -> None:
    """Raise KeyboardInterrupt if ESCAPE was pressed (for a clean early exit)."""
    if "escape" in event.getKeys(keyList=["escape"]):
        raise KeyboardInterrupt


def main() -> None:
    win = visual.Window(
        size=WINDOW_SIZE,
        fullscr=False,
        units="height",
        color="black",
        colorSpace="rgb",
    )

    print("Building geometric stimuli...")
    s = screens.build_session_stimuli(win)

    def show(label: str, draw_fn) -> None:
        print(f"  -> {label}")
        draw_fn()
        core.wait(STEP_SECONDS)
        _check_escape()

    try:
        show("Fixation (ITI)",
             lambda: screens.draw_fixation(win, s.fixation, s.counter))

        show("Stage 1: two spaceships",
             lambda: screens.draw_stage1(win, s.stage1, s.counter))

        show(f"Transition reveal -> planet {DEMO_STATE}",
             lambda: screens.draw_transition(win, s.transition, DEMO_STATE, s.counter))

        show(f"Stage 2: aliens on planet {DEMO_STATE}",
             lambda: screens.draw_stage2(win, s.stage2, DEMO_STATE, s.counter))

        show("Reward feedback (rewarded)",
             lambda: screens.draw_reward(win, s.reward, True, s.counter))

        show("'Too slow!' timeout",
             lambda: screens.draw_too_slow(win, s.too_slow, s.counter))

        screens.set_nback_letter(s.nback_letter, "K")
        show("N-back letter over Stage 1",
             lambda: screens.draw_stage1(
                 win, s.stage1, s.counter, letter=s.nback_letter, show_letter=True))

        screens.update_treasure_counter(s.counter, 42)
        def _draw_counter() -> None:
            s.counter.draw()
            win.flip()
        show("Treasure counter (= 42)", _draw_counter)

        screens.set_message(
            s.message,
            "Section 2 of 4\n\nIn this section you'll also see a letter -\n"
            "keep tracking it.",
            "Press SPACE to begin",
        )
        show("Block-start banner", lambda: screens.show_message(win, s.message))

        screens.set_message(
            s.message,
            "Welcome!\n\nYou'll pilot spaceships to planets and meet aliens\n"
            "who sometimes share treasure. This takes about 20 minutes.",
            "Press SPACE to continue",
        )
        show("Welcome screen", lambda: screens.show_message(win, s.message))

        print("Smoke test complete.")
    except KeyboardInterrupt:
        print("Aborted by user (escape).")
    finally:
        win.close()

    core.quit()


if __name__ == "__main__":
    main()
