# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

A PsychoPy implementation of the Daw et al. (2011) two-step task with a within-subject cognitive-load manipulation (1-back letter task during Stage-1 deliberation), following Otto, Gershman, Markman & Daw (2013, *Psychological Science*, "The Curse of Planning").

**Research question.** Does concurrent working-memory load shift behavioural control on the two-step task from model-based toward model-free, replicating Otto et al. (2013) in a healthy student sample?

**Context.** Graduate psychology coursework (programming + experiment design). The task itself runs locally; analysis is in Python. This is *paradigm validation in healthy volunteers*, not a clinical study.

**Background reading.** Key papers for implementation: Daw et al. (2011, *Neuron*); Otto et al. (2013, *Psych Sci*); Decker et al. (2016, *Psych Sci*) for the spaceship cover story; Kool, Cushman & Gershman (2016, *PLOS Comp Biol*) and Feher da Silva & Hare (2018, *PLOS ONE*) for methodological caveats to acknowledge in the write-up.

## Stack

- **Python ≥ 3.10**, single conda or venv environment.
- **PsychoPy** (Coder mode, not Builder). Use `psychopy` as a library imported in plain `.py` files — do not generate `.psyexp` files.
- **NumPy, pandas, scipy, statsmodels** for analysis.
- **matplotlib** for plots. No seaborn unless explicitly needed.
- **pytest** for tests.
- Package installable with `pip install -e .` from the repo root.

Do not add web frameworks, GUI toolkits other than PsychoPy, or ML libraries.

## Repository layout

```
src/two_step_stress/
    __init__.py
    config.py             # all task parameters in one place
    task/
        __init__.py
        screens.py        # stimulus builders + draw helpers
        transitions.py    # Stage-1 → Stage-2 transition logic
        rewards.py        # drifting reward-probability walks
        nback.py          # 1-back letter stream + response collection
        trial.py          # one full two-step trial (load and no-load)
        instructions.py   # instruction screens, practice, debrief
        run_experiment.py # entry point: python -m two_step_stress.task.run_experiment
    analysis/
        __init__.py
        stay_probability.py   # 2x2 stay analysis + logistic regression
        plots.py
    io/
        __init__.py
        data_logging.py   # ExperimentHandler wrapper, CSV schema
tests/
    test_transitions.py
    test_rewards.py
    test_nback.py
docs/
    two_step_stress_tutorial.ipynb  # vignette / tutorial
data/                     # gitignored; per-participant CSVs land here
notebooks/                # exploratory only; not part of the package
pyproject.toml
README.md
LICENSE
CLAUDE.md
.gitignore
```

Keep modules small and single-purpose. The trial loop in `trial.py` should be readable end-to-end on one screen.

## Task parameters (defaults — change only with a comment justifying it)

All live in `config.py`:

- **Trials.** 200 main trials, split into 4 blocks of 50 with breaks. 12 practice trials (8 no-load + 4 load) with explicit transition feedback.
- **Transitions.** Common p = 0.7, rare p = 0.3. Each Stage-1 action has its own preferred Stage-2 state. Mapping fixed per participant, counterbalanced across participants.
- **Reward probabilities.** Four independent Gaussian random walks, σ = 0.025, reflecting boundaries [0.25, 0.75], independent seeds per participant logged to data.
- **Reward.** Binary (1/0).
- **Timing.** Stage-1 response window 2000 ms, Stage-2 response window 2000 ms, transition reveal 700 ms, ITI 1000 ms, feedback 1000 ms. Use frame-locked presentation (`win.flip()`-driven), not `core.wait`, for stimulus onsets that matter.
- **Cognitive load.** Within-subject, blocked (not interleaved trial-by-trial — too noisy). 2 load blocks and 2 no-load blocks, order ABBA / BAAB counterbalanced. In load blocks: a letter appears for 500 ms at trial start; participant must indicate (key press) whether it matches the letter from the *previous* trial in that block (1-back). The 1-back response is collected during the Stage-1 decision window. Letters drawn from a fixed consonant set; ~33% match rate.
- **Keys.** Stage-1/Stage-2 choice on `f` / `j`. 1-back match/no-match on `z` / `m`. Document in instructions and store in config.
- **Cover story.** Spaceships → planets → aliens with treasure (Decker et al., 2016). Same in load and no-load.

## Data schema

One CSV per participant in `data/raw/sub-<id>_<timestamp>.csv`. One row per trial. Columns:

`participant_id, block, block_type (load|no_load), trial_in_block, trial_global, stage1_choice, stage1_rt, transition (common|rare), stage2_state, stage2_choice, stage2_rt, reward, reward_probs (json of 4 floats at trial start), nback_letter, nback_is_match, nback_response, nback_correct, nback_rt, timestamp`

Plus a session-level JSON sidecar with: PsychoPy version, package version, RNG seed, key mapping, block order, stimulus mapping, screen refresh rate, total duration.

Never overwrite an existing participant file. Append a numeric suffix if the file exists.

## Coding conventions

- Type hints on public functions.
- Docstrings: one-line summary, then params/returns. NumPy style.
- Constants in `config.py`, not scattered.
- No magic numbers in the trial loop.
- Random number generation: one `numpy.random.Generator` per participant, seeded from system entropy and logged. Do not use `random.random()` or unseeded `np.random.*`.
- PsychoPy windows: always close cleanly on Escape; flush data before exit.
- Logging: use the `logging` module at INFO for milestones, DEBUG for trial-level events. Do not `print` in task code.

## Testing

PsychoPy code that touches the screen is hard to unit-test, so:

- **Pure-logic modules** (`transitions.py`, `rewards.py`, `nback.py` letter-stream generator, `stay_probability.py`) are fully unit-tested.
- **Trial-loop integration** is tested by running the simulated agent through the same `transitions` + `rewards` code the live task uses, then checking that a model-based agent produces the textbook reward × transition interaction and a model-free agent does not. This is the single most important test in the repo — if it fails, the task is broken.
- Run `pytest` before any commit that touches logic.

## Analysis pipeline

`analysis/stay_probability.py` implements:

1. **2 × 2 stay-probability plot** — P(repeat Stage-1 choice) by previous reward (yes/no) × previous transition (common/rare), separately for load and no-load blocks. This is the headline figure.
2. **Logistic regression** (statsmodels) of `stay ~ prev_reward * prev_transition * load`, with participant random effects if mixed-effects (`statsmodels.MixedLM` or `pymer4`); otherwise per-participant fits aggregated.
3. **Hybrid RL model fit** (stretch goal — not implemented): parameters β_MB, β_MF, α, λ, π. Negative log-likelihood minimised with `scipy.optimize.minimize` (L-BFGS-B, parameter bounds, ≥10 random starts). Compare against pure-MB and pure-MF nested models via AIC/BIC. The headline test is whether β_MB is lower under load.
4. **1-back accuracy** as a manipulation check — load blocks should show non-trivial dual-task cost; report mean accuracy and exclude participants below chance.

## Workflow rules for Claude Code

- **One concern per change.** Don't refactor while adding features.
- **Before adding a new task feature, write or update the test for the corresponding pure-logic module.**
- **Don't run the PsychoPy window in automated workflows** — it requires a display. Use the simulated agent for end-to-end checks.
- **Don't commit anything in `data/`.** It's gitignored; double-check.
- **Don't pull in new dependencies without updating `pyproject.toml` and explaining why in the commit.**
- **If a task parameter is changed, update `config.py` *and* the relevant section of this file.**
- **Stop and ask** if a request would: change the pre-registered analysis after data collection has started; remove the cognitive-load manipulation; switch to PsychoPy Builder; introduce a web/online deployment.

## Branch and PR conventions

- Working branch for this milestone: `week-1`.
- Commits: imperative mood, scoped (`task: add Stage-2 reward animation`, `analysis: fix stay-prob edge case at trial 1`).
- Keep PRs small and reviewable. The week-1 PR stays open for review and is *not* squashed-merged until the package installs cleanly and `pytest` passes.
