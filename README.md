# two-step-stress

A PsychoPy implementation of the two-step decision task (Daw et al., 2011)
with a within-subject cognitive load manipulation.

## Overview

This project investigates whether concurrent working-memory load shifts
behavioural control from model-based toward model-free reinforcement learning,
following Otto et al. (2013). Participants complete a two-step task while
performing a 1-back letter task during the deliberation period of load blocks.

## Installation

This project needs **Python 3.11** — PsychoPy does not build on newer Pythons,
so the package pins `requires-python = ">=3.11,<3.12"`.

### Recommended: conda (all platforms)

`conda` installs its *own* Python 3.11 into an isolated environment, so this
works regardless of which Python you already have — a system Python 3.12/3.13
is fine, conda doesn't touch it. From the repo root:

```bash
conda env create -f environment.yml   # `stress` env: Python 3.11, wxPython, and the package
conda activate stress
```

On **Ubuntu / Linux**, also install one system library — Qt6's startup dialog
needs it (wxPython is already handled by conda above):

```bash
sudo apt install libxcb-cursor0
```

Verify the install:

```bash
pytest -q          # expect: 55 passed
```

### Already have Python 3.11? (venv shortcut)

If a 3.11 interpreter already exists on your machine, a plain venv works.
(`venv` cannot *create* 3.11 for you — install it first via pyenv, Homebrew, or
python.org if you only have a newer Python.)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[task,dev]"   # or ".[dev]" for tests + analysis only (no PsychoPy)
```

On Linux, prefer the conda path above: pip has no wxPython wheel for Linux, so
this route would try to compile wxPython from source.

### What the extras mean

- base (`pip install -e .`) — engine + analysis only (NumPy, pandas, scipy,
  statsmodels, matplotlib)
- `task` — adds PsychoPy, needed only to run the live experiment
- `dev` — adds pytest, nbconvert, ipykernel (tests + vignette execution)

## Running

- Experiment: `python -m two_step_stress.task.run_experiment` (needs the `task` extra)
- Analysis:   `python scripts/run_analysis.py --csv data/raw/sub-<id>_<timestamp>.csv`
- Tests:      `pytest`

## Dependencies

- **Core:** NumPy, pandas, scipy, statsmodels, matplotlib
- **`task` extra:** PsychoPy (live experiment only)
- **`dev` extra:** pytest, nbconvert, ipykernel (tests + vignette execution)

## References

- Daw et al. (2011). Model-based influences on humans' choices and striatal
  prediction errors. *Neuron*.
- Otto et al. (2013). The curse of planning. *Psychological Science*.
