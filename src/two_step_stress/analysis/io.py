"""Loading and cleaning participant CSVs for analysis."""

from __future__ import annotations

import json

import pandas as pd


def _parse_probs(value: object) -> list[float] | None:
    """Parse one ``reward_probs`` cell (a JSON array string) to a list of floats."""
    if pd.isna(value):
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def load_session(csv_path: str) -> pd.DataFrame:
    """Read a participant CSV into a DataFrame.

    The ``"NA"`` sentinel written by the task is read as ``NaN`` (pandas
    default).  The ``reward_probs`` column is parsed from JSON into lists of
    floats.  All rows (including practice and timed-out trials) are returned.

    Parameters
    ----------
    csv_path : str
        Path to a ``sub-<id>_<timestamp>.csv`` file.

    Returns
    -------
    pandas.DataFrame
        The full session, one row per trial.
    """
    df = pd.read_csv(csv_path)
    if "reward_probs" in df.columns:
        df["reward_probs"] = df["reward_probs"].apply(_parse_probs)
    return df


def clean_for_stay_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to completed main-task trials for the stay-probability analysis.

    Drops practice rows (``block_type == "practice"``) and any trial that timed
    out (``stage1_choice`` or ``stage2_choice`` is ``NaN``).  Row order is
    preserved; the index is reset.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`load_session`.

    Returns
    -------
    pandas.DataFrame
        Completed, non-practice trials only.
    """
    out = df[df["block_type"] != "practice"]
    out = out.dropna(subset=["stage1_choice", "stage2_choice"])
    return out.reset_index(drop=True)
