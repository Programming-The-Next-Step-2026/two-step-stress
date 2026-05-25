# PsychoPy display-layer design plan

Design for the live experiment that wraps the existing pure-logic engine
(`transitions`, `rewards`, `nback`) in a PsychoPy presentation loop. **No
experiment code is written yet** — this document is for review. All §8 open
questions have been resolved (see the decisions table); the four review concerns
(C1–C4) are resolved in §9.

The guiding principle: the engine already owns *all* randomness and task logic.
The display layer only (a) calls the engine, (b) draws what it returns, (c)
collects keypresses with accurate timing, and (d) logs. No task logic (transition
probabilities, reward draws, letter streams) is re-implemented here.

---

## 1. File structure

Following the layout already prescribed in `CLAUDE.md` (decision Q1 → `task/`):

```
src/two_step_stress/
    task/
        screens.py        # NEW: PsychoPy stimulus builders + draw helpers (emoji now, PNG-ready)
        trial.py          # NEW: run ONE two-step trial (load and no-load); returns a dict
        instructions.py   # NEW: welcome / instruction / break / debrief screens + practice
        run_experiment.py # NEW: entry point — session setup, block loop, teardown
    io/
        __init__.py       # NEW (empty)
        data_logging.py   # NEW: TrialWriter (CSV, one row/trial) + session JSON sidecar
```

Entry point (decision Q2): **`python -m two_step_stress.task.run_experiment`** only.
No `scripts/` shim and no `[project.scripts]` console command.

Responsibilities, kept small and single-purpose per `CLAUDE.md`:

| File | Owns | Does **not** own |
|------|------|------------------|
| `screens.py` | Building `visual.*` stimuli once; helper draw functions; cover-story emoji/colour/text constants | Timing, key collection, logging |
| `trial.py` | The single-trial timeline (flip-locked); calling engine fns; collecting keys + RT; returning a plain `dict` | Window creation, block/counterbalance logic, file I/O |
| `instructions.py` | Welcome, instruction pages, between-block break, end screen, and the **practice loop** | Main-block sequencing |
| `run_experiment.py` | Subject dialog/CLI, RNG seeding, refresh-rate calibration, block order, reward-walk lifecycle, block→trial loops, clean teardown on Escape | Drawing details, CSV column formatting |
| `io/data_logging.py` | CSV schema + safe (no-overwrite) file naming + session JSON | Anything PsychoPy |

`trial.py`'s main function should read end-to-end on roughly one screen, per
`CLAUDE.md`.

**Testing posture.** None of these files are unit-tested by screen interaction.
`trial.py` is structured so the engine calls it makes are the *same* functions the
sanity-check already exercises, so the integration test (MB shows the
reward×transition crossover, MF doesn't) stays the real guardrail. The PsychoPy
window is never run in an automated workflow (it needs a display).

---

## 2. Trial flow (one trial, step by step)

All onsets that matter are **frame-locked** (`win.flip()`-driven), never
`core.wait`, per `CLAUDE.md`. Fixed-duration screens (ITI, transition reveal,
feedback, n-back letter) are timed by **integer frame counts** derived from the
measured refresh rate (§9 C3). RTs come from a `core.Clock` / `Keyboard` reset on
the flip that reveals the response stimulus.

Cover story (same in load and no-load): **spaceships → planets → aliens with
treasure** (Decker et al., 2016). Stimuli are emoji placeholders (decision Q15):
🚀 spaceships (colour-coded left/right), 🪐 planets, 👽 aliens, 💰 reward, 🎁
no-reward. A small **running treasure total** is shown at the bottom of the screen
throughout (decision Q5).

```
┌── ITI (fixed 1000 ms) ──────────────────────────────────────────────┐
│  centred fixation cross on black; running treasure total at bottom   │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌── STAGE 1: two 🚀 spaceships (left & right, colour-coded) ──────────┐
│  In LOAD blocks the 1-back letter is overlaid for the first 500 ms   │
│  (see §3). Response window = STAGE1_WINDOW_MS (2000 ms) cap.          │
│  Self-paced (decision Q3): the choice may be made any time up to the │
│  2000 ms cap. Keys: f = left, j = right. Clock resets on the flip    │
│  that draws the spaceships.                                          │
│  → valid f/j: record stage1_choice (0/1) + stage1_rt.                │
│  → no f/j by 2000 ms: "Too slow!" feedback, log stage1_* = NA, and   │
│    SKIP straight to ITI (decision Q4): no transition, no Stage-2, no  │
│    reward draw, and the reward walk is FROZEN this trial (Q6/§9 C6).  │
│  (LOAD only) the transition reveal does not begin until the n-back   │
│  window also closes — see the §3 / §9 C2 timeline.                   │
└─────────────────────────────────────────────────────────────────────┘
         │ (valid choice; load: n-back window also closed)
         ▼
┌── TRANSITION REVEAL (fixed ~700 ms) ────────────────────────────────┐
│  apply_transition(stage1_choice, mapping, rng) → (stage2_state, kind)│
│  Show the chosen 🚀 arriving at the resulting 🪐 planet so the       │
│  participant perceives common vs rare landings. Common/rare is NOT   │
│  labelled in main blocks (only in practice — §5).                    │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌── STAGE 2: two 👽 aliens on the planet ─────────────────────────────┐
│  Planet-coloured background + its two aliens (left & right).         │
│  Window = STAGE2_WINDOW_MS (2000 ms) cap, self-paced. Keys: f / j.   │
│  Clock resets on the flip that draws the aliens.                     │
│  → valid f/j: record stage2_choice (0/1) + stage2_rt.                │
│  → no f/j by 2000 ms: "Too slow!", log stage2_* = NA, SKIP to ITI;   │
│    reward walk FROZEN this trial (Q6/§9 C6).                         │
└─────────────────────────────────────────────────────────────────────┘
         │ (valid choice)
         ▼
┌── REWARD FEEDBACK (fixed 1000 ms) ──────────────────────────────────┐
│  arm = stage2_state * 2 + stage2_choice   (flat index 0–3, §9 C1)    │
│  p   = reward_probs[arm]                                             │
│  reward = int(rng.random() < p)   (single Bernoulli)                 │
│  reward = 1 → 💰 + "+1" and running total increments;                │
│  reward = 0 → 🎁 (empty/no-treasure) and total unchanged.            │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
   step_reward_probs(reward_probs, rng)  ← ONLY on a completed trial
         │                                  (frozen on any skip; §9 C6)
         ▼
   write one CSV row, then loop to next trial's ITI.
```

Notes:
- The reward-probability vector is **owned by `run_experiment.py` for the whole
  session** and threaded trial→trial (initialised once at session start, stepped
  once **per completed trial only**). The walk is continuous within and across
  blocks (it does not reset per block). The 4 floats *at trial start* are logged
  before the step.
- Escape at any flip → flush CSV, write the session JSON, close the window cleanly
  (`CLAUDE.md`).

---

## 3. 1-back overlay for load blocks

The 1-back is a *concurrent* working-memory load during Stage-1 deliberation
(Otto et al., 2013). Present in **load** blocks only.

- **Letter source.** One
  `generate_letter_stream(N_TRIALS_PER_BLOCK, NBACK_MATCH_RATE, rng)` call per load
  block, made up-front in `run_experiment.py`. The `(letters, is_match)` lists are
  indexed by `trial_in_block`. The engine guarantees ~33% matches and
  `is_match[0] == False`.
- **Onset / offset.** At Stage-1 onset, the letter is drawn above the spaceships
  for `NBACK_LETTER_MS` (500 ms), then removed; the spaceships remain for the rest
  of the window. The letter competes with the *encoding* phase of the Stage-1
  decision, exactly when planning would otherwise happen.
- **Response.** Collected during the Stage-1 window in parallel with the f/j
  choice, on different keys: `z` = match, `m` = no-match. First `z`/`m` is the
  n-back response; first `f`/`j` is the choice. Order is free. RT measured from
  **letter onset**.
- **Scoring.**
  `score_nback_response(is_match, key, KEY_NBACK_MATCH, KEY_NBACK_NO_MATCH)`
  → `(nback_correct, response_type)`, logged.
- **Missed n-back.** If no `z`/`m` by the end of the Stage-1 window:
  `nback_response = NA`, `nback_correct = False`, `nback_rt = NA`. Silent — no
  on-screen nudge (decision Q7). A missed n-back does **not** abort the trial; only
  a missed *choice* does (§2).
- **No-load blocks.** No letter, no `z`/`m` collected; `nback_*` logged as NA. The
  Stage-1 screen is otherwise identical so visual demand is matched apart from the
  letter.

The precise interaction between the choice press and the n-back window (when the
transition reveal is allowed to start) is pinned down in **§9 C2**.

---

## 4. Block structure & counterbalancing

- **4 blocks × 50 trials** = 200 main trials (`N_BLOCKS`, `N_TRIALS_PER_BLOCK`).
- **Within-subject, blocked** load (not interleaved — `CLAUDE.md`). 2 load (A) +
  2 no-load (B).
- **Order** from `BLOCK_ORDERS = ("ABBA", "BAAB")`, selected by
  `BLOCK_ORDERS[participant_id % 2]`. The transition-mapping counterbalance uses
  the same id (`build_transition_mapping(participant_id)`) — both counterbalances
  derive from `participant_id` (decision Q8).
- **Breaks between blocks.** Self-paced break after blocks 1, 2, 3: "Block k of 4
  complete. Take a short break. Press SPACE to continue." Block 4 → end screen.
- **Block-start banner.** Before each block, one screen naming the upcoming
  condition in participant-friendly terms (load → "you'll also see a letter — keep
  tracking it"; no-load → "just choose spaceships"), SPACE to begin. Makes the
  manipulation explicit so participants engage the load.
- Reward walks are continuous across blocks; each load block draws its own letter
  stream.

---

## 5. Instructions & screens

Concise; text constants live in `instructions.py`. Sequence at session start:

1. **Welcome** — what they'll do, ~20 min, SPACE.
2. **Instruction pages** (a few short screens, SPACE advance / BACKSPACE back):
   - The 🚀 → 🪐 → 👽 → 💰 cover story.
   - Stage-1 and Stage-2 keys `f` / `j`; respond before the ship leaves.
   - Each spaceship *usually* flies to one planet but *sometimes* the other
     (common/rare in cover-story terms, no numbers).
   - Treasure odds drift slowly — keep tracking which aliens pay out.
   - **The n-back rule and keys `z` / `m`, introduced here for everyone**
     (decision Q9), before any load practice.
3. **Practice** — **12 trials total: 8 no-load then 4 load** (decision; reduced
   from 20), with **explicit transition feedback** (the one place "COMMON"/"RARE"
   is shown), per `CLAUDE.md`. Practice uses its own reward walks + letter stream
   from the participant RNG and **is logged** with `block_type="practice"`
   (decision Q10). After practice: "Ready? The real game begins now." Practice is
   on by default; `--no-practice` skips it for piloting (decision Q11).
4. **Main blocks 1–4** with banner + break screens (§4).
5. **End screen** — "All done, thank you!" with total treasure, SPACE/ESCAPE to
   exit. Then flush + close.

> **Parameter-change note (`CLAUDE.md` sync rule).** Reducing practice from 20 →
> 12 changes a task parameter. At implementation time this requires updating
> `config.py` (`N_TRIALS_PRACTICE: 20 → 12`, plus two new constants, e.g.
> `N_PRACTICE_NO_LOAD = 8`, `N_PRACTICE_LOAD = 4`) **and** the matching line in
> `CLAUDE.md` ("20 practice trials" → "12"), in the same change. These edits are
> deliberately **deferred to the coding phase** so `config.py` and `CLAUDE.md`
> stay in sync with each other (changing the doc now would desync it from
> `config.py`, which still reads 20). `docs/pre_registration.md` does not yet
> exist, so no pre-reg update is owed.

---

## 6. Data logging

**One CSV per participant, one row per trial**, flushed each trial so a crash
keeps completed trials. Columns use the **`CLAUDE.md` schema names verbatim**:

| Column | Type / values | Source |
|--------|---------------|--------|
| `participant_id` | str/int | dialog/CLI |
| `block` | int 1–4 (practice rows: 0) | block loop |
| `block_type` | `load` \| `no_load` \| `practice` | block order (Q10) |
| `trial_in_block` | int | trial loop |
| `trial_global` | int 1–200 (practice excluded from the main count) | counter |
| `stage1_choice` | 0/1 or `NA` | keypress |
| `stage1_rt` | float s or `NA` | clock |
| `transition` | `common` \| `rare` \| `NA` | `apply_transition` |
| `stage2_state` | 0/1 or `NA` | `apply_transition` |
| `stage2_choice` | 0/1 or `NA` | keypress |
| `stage2_rt` | float s or `NA` | clock |
| `reward` | 0/1 or `NA` | Bernoulli on `reward_probs[arm]` |
| `reward_probs` | JSON array of 4 floats **at trial start** | session walk state |
| `nback_letter` | str or `NA` (no-load) | letter stream |
| `nback_is_match` | bool or `NA` | letter stream |
| `nback_response` | `z`/`m` or `NA` | keypress |
| `nback_correct` | bool or `NA` | `score_nback_response` |
| `nback_rt` | float s or `NA` | clock |
| `timestamp` | ISO 8601 trial onset | clock |

**File location (decision Q12):** `data/raw/sub-<id>_<timestamp>.csv`.
`CLAUDE.md`'s data-schema line has been updated to this path. **Never overwrite**
— if the target exists, append a numeric suffix (`..._1.csv`), per `CLAUDE.md`.
`data/` is gitignored (confirmed); nothing under it is committed.

**Session JSON sidecar** (`..._session.json`, same stem), per `CLAUDE.md`:
PsychoPy version, package version (`0.1.0`), **RNG seed**, key mapping, block
order, stimulus/transition mapping, **measured screen refresh rate** (§9 C3),
total duration. Added (cheap + useful): Python version, platform, window size, and
the participant-level counterbalance ids.

**RNG (decision Q13).** One `numpy.random.Generator` per participant, seeded from
system entropy and the integer seed logged to JSON. A `--seed` CLI flag overrides
for piloting/repro. The single generator is consumed by transitions, rewards, the
reward Bernoulli, *and* letter streams, so the **call order is fixed and
documented** for exact reproducibility.

**Logging module.** `logging` at INFO for milestones (session/block start+end),
DEBUG per-trial. No `print` in task code (`CLAUDE.md`).

---

## 7. Subject-info collection (decision Q14)

**GUI dialog and CLI, CLI overrides.** `run_experiment.py` takes `argparse` flags
(`--participant`, `--seed`, `--no-practice`, `--windowed`). If `--participant` is
given it is used directly; otherwise a PsychoPy `gui.DlgFromDict` collects
`participant_id` (required) and `session` (default 1). `participant_id` must be
non-empty and filename-safe; rejected otherwise *before* the window opens.

---

## 8. Resolved decisions (was: open questions §8)

| # | Decision |
|---|----------|
| Q1 | Layout under `task/` (CLAUDE.md). |
| Q2 | Entry point `python -m two_step_stress.task.run_experiment` only. |
| Q3 | Stage-1/2 self-paced with a 2000 ms cap. |
| Q4 | No-response → "Too slow!", log NA, skip to ITI. |
| Q5 | **Show running treasure total** (small counter, bottom of screen). |
| Q6 | **Freeze the reward walk on timeout/skip trials**; step only on completed trials. |
| Q7 | Missed n-back: silent, just log NA. |
| Q8 | Both counterbalances (block order + transition mapping) derived from `participant_id`. |
| Q9 | Teach the n-back in the initial instructions for everyone. |
| Q10 | Log practice rows with `block_type="practice"`. |
| Q11 | Practice on by default; `--no-practice` flag to skip. |
| Q12 | Data path `data/raw/`; `CLAUDE.md` updated to match. |
| Q13 | Entropy seed by default; `--seed` override for piloting. |
| Q14 | GUI dialog **and** CLI; CLI overrides the dialog when given. |
| Q15 | Emoji placeholders (🚀 colour-coded / 🪐 / 👽 / 💰 reward / 🎁 no-reward); `screens.py` builders kept PNG-swappable. |
| Q16 | Windowed 1280×720 for dev (`--windowed`); fullscreen for real runs. Black background, white text. |
| Practice length | 12 trials total (8 no-load + 4 load), down from 20. |

---

## 9. Review concerns — resolutions

### C1 — Reward indexing (verified against the engine)

`src/two_step_stress/task/rewards.py`:
- `initialise_reward_probs(rng)` → `rng.uniform(REWARD_WALK_MIN, REWARD_WALK_MAX, size=N_ARMS)`
  with `N_ARMS = 4` ⇒ a **flat `numpy.ndarray` of shape `(4,)`**, values in
  `[0.25, 0.75]`.
- `step_reward_probs(probs, rng)` → same shape `(4,)`.

There is **no 2×2 array** anywhere — the four arms are a flat vector. The single
canonical indexing convention is therefore:

```
arm = stage2_state * 2 + stage2_choice      # ∈ {0, 1, 2, 3}
p   = reward_probs[arm]
```

This **matches `scripts/sanity_check.py:47`** exactly
(`reward_probs[s2 * 2 + a2]`). The display layer will use this identical pattern —
one convention across the whole codebase. (Arm layout: state 0 → arms {0,1}, state
1 → arms {2,3}; within a state, action 0 = arm `2·state`, action 1 = arm
`2·state+1`.)

### C2 — N-back response collection vs transition reveal (load blocks)

Two responses are collected on different keys during the Stage-1 window: the
**choice** (`f`/`j`) and the **n-back** (`z`/`m`). The choice ends Stage-1
deliberation, but the n-back response may still arrive afterward. The transition
reveal must **not** start while we are still listening for the letter response
(showing the planet would confuse the participant about what they're responding
to).

**Rule (load blocks):** after a valid `f`/`j` choice, keep the Stage-1 screen up
and keep listening for `z`/`m` until **whichever comes first**: (a) a `z`/`m`
press, or (b) the 2000 ms Stage-1 cap elapses. The transition reveal begins only
then. Equivalently, the reveal starts at
`min(t_nback, 2000 ms)` but never before the choice, and only if a valid choice
exists.

```
LOAD-block Stage-1 timeline (t measured from spaceship/letter onset)

t=0 ───────── spaceships + letter on; reset clocks; listen f/j AND z/m
              │
t=500 ──────── letter removed; spaceships remain; still listening f/j AND z/m
              │
   ┌──────────┴───────────────────────────────────────────────────────┐
   │ Case A: choice (f/j) at t_c, n-back (z/m) still pending            │
   │   → choice locked at t_c; SPACESHIPS STAY UP; keep listening z/m   │
   │   → z/m at t_n  →  reveal starts at t_n                            │
   │   → no z/m by 2000 → nback = NA; reveal starts at 2000             │
   ├────────────────────────────────────────────────────────────────────┤
   │ Case B: n-back (z/m) first at t_n, choice still pending            │
   │   → n-back locked at t_n; keep listening f/j                       │
   │   → f/j at t_c (≤2000) → reveal starts at t_c                      │
   │   → no f/j by 2000 → TIMEOUT (missing choice) → skip to ITI (§2)   │
   ├────────────────────────────────────────────────────────────────────┤
   │ Case C: both pressed → reveal starts at max(t_c, t_n)              │
   │ Case D: neither by 2000 → TIMEOUT → skip to ITI                    │
   └────────────────────────────────────────────────────────────────────┘
```

A missing n-back (Case A → timeout branch) is logged NA and is **not** a skip
(§3). A missing *choice* (Cases B/D) **is** a skip (§2), and on a skip the reward
walk is frozen (C6).

**No-load blocks:** no `z`/`m` listening, so the transition reveal begins
immediately on the `f`/`j` press (still self-paced, 2000 ms cap).

### C3 — Refresh-rate calibration

At session start, **after window creation and before instructions**:
`win.getActualFrameRate(nIdentical=…, nMaxFrames=≈120, …)`, take the modal value,
store as `frame_rate`, and **log it to the session JSON**. All millisecond
durations (ITI, transition reveal, feedback, n-back letter) are converted to
**integer frame counts** via this measured rate, e.g.
`n_frames = round(duration_ms / 1000 * frame_rate)`. If the measured rate deviates
from the nearest common value (60 / 120 / 144 Hz) by more than ~5%, **log a
warning and continue** (do not abort). If measurement returns `None` (rare),
fall back to an assumed 60 Hz, log a warning, and proceed.

### C4 — Stop-and-ask triggers (re-checked against `CLAUDE.md` line 126)

The stop-and-ask triggers are: changing the pre-registered analysis after data
collection has started; removing the cognitive-load manipulation; switching to
PsychoPy Builder; introducing a web/online deployment. **None of the decisions in
this plan hit any of them:**
- The 1-back cognitive-load manipulation is fully preserved (§3, blocked load,
  z/m responses, ~33% match).
- This is local PsychoPy **Coder**-mode code imported as a library — no Builder,
  no `.psyexp`, no web/online component.
- No analysis change; data collection has not started, and the headline
  stay-probability / regression / RL pipeline is untouched.
- The only task-parameter change (practice 20 → 12) is handled via the
  `config.py` + `CLAUDE.md` sync rule (§5 note), not a pre-registered-analysis
  change.

So the plan introduces no `CLAUDE.md` "stop and ask" condition.

---

## 10. Proposed build order (after sign-off)

1. `io/data_logging.py` — TrialWriter (no-overwrite naming, CSV schema) + session
   JSON. Lowest risk, no PsychoPy.
2. `screens.py` — emoji stimulus builders + draw helpers (PNG-swappable).
3. `trial.py` — single-trial timeline calling the engine; returns a row dict.
4. `instructions.py` — welcome / instructions / practice / breaks / end.
5. `run_experiment.py` — wire it together: dialog/CLI, seeding, refresh-rate
   calibration, block order, reward-walk lifecycle, teardown.
6. Config + doc sync (practice 20 → 12 in `config.py` and `CLAUDE.md`).

No PsychoPy window is run in any automated check; the simulated-agent integration
test remains the correctness guardrail.
