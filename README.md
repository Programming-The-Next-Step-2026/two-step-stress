# two-step-stress

A PsychoPy implementation of the two-step decision task (Daw et al., 2011)
with a within-subject cognitive load manipulation.

## Overview

This project investigates whether concurrent working-memory load shifts
behavioural control from model-based toward model-free reinforcement learning,
following Otto et al. (2013). Participants complete a two-step task while
performing a 1-back letter task during the deliberation period of load blocks.

## Installation

Use **Python 3.11**. PsychoPy does not build on Python 3.13+, so the package
pins `requires-python = ">=3.11,<3.12"`.

For development — running the tests and re-executing the tutorial notebook
(no PsychoPy needed):

```bash
pip install -e ".[dev]"
```

To also run the live experiment locally (adds PsychoPy):

```bash
pip install -e ".[task,dev]"
```

The base install (`pip install -e .`) provides the engine and analysis layer
only (NumPy, pandas, scipy, statsmodels, matplotlib). PsychoPy is an optional
`task` extra, needed only for stimulus presentation in the live experiment —
so the engine, analysis pipeline, tests, and vignette all run without it.

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
