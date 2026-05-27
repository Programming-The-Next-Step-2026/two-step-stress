"""Analyse one participant CSV: stay-probability figures + 1-back check.

Run from the repo root:
    python scripts/run_analysis.py --csv data/raw/sub-01_<timestamp>.csv

Saves three figures to figures/sub-<id>/ and prints the manipulation-check
summary.
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display

from two_step_stress.analysis.io import clean_for_stay_analysis, load_session
from two_step_stress.analysis.manipulation_check import nback_accuracy, summarise
from two_step_stress.analysis.plots import (
    plot_nback_accuracy,
    plot_stay_probability_2x2,
    plot_stay_probability_by_load,
)
from two_step_stress.analysis.stay_probability import compute_stay_probability


def main() -> None:
    """Run the full single-participant analysis from the command line.

    Loads the CSV named by ``--csv``, cleans it for the stay analysis, computes
    the overall and load-split stay probabilities and the 1-back manipulation
    check, prints the manipulation-check summary, and saves the three figures
    to ``figures/sub-<id>/``.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Analyse a two-step participant CSV.")
    parser.add_argument("--csv", required=True, help="Path to a participant CSV.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = load_session(args.csv)
    participant_id = str(df["participant_id"].iloc[0]) if len(df) else "unknown"
    logging.info("Loaded %d rows for participant %s", len(df), participant_id)

    clean = clean_for_stay_analysis(df)
    logging.info("%d completed main-task trials after cleaning", len(clean))

    stay = compute_stay_probability(clean, split_by_block_type=False)
    stay_split = compute_stay_probability(clean, split_by_block_type=True)
    accuracy = nback_accuracy(df)

    print(summarise(df))

    out_dir = Path("figures") / f"sub-{participant_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_stay_probability_2x2(stay, save_path=out_dir / "stay_probability_2x2.png")
    plot_stay_probability_by_load(stay_split, save_path=out_dir / "stay_probability_by_load.png")
    plot_nback_accuracy(accuracy, save_path=out_dir / "nback_accuracy.png")

    print(f"Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
