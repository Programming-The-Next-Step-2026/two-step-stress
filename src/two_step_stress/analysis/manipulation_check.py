"""1-back manipulation check: dual-task accuracy in the load blocks."""

from __future__ import annotations

import pandas as pd


def _to_bool(value: object) -> bool:
    """Coerce a CSV-read ``nback_correct`` value ("True"/"False"/bool) to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def nback_accuracy(df: pd.DataFrame) -> dict:
    """Summarise 1-back accuracy in the load blocks.

    Accuracy is computed over trials where a response was made; missed trials
    (no ``z``/``m`` press) are reported separately rather than scored as errors.

    Parameters
    ----------
    df : pandas.DataFrame
        Full session (output of :func:`io.load_session`).  Practice rows have
        ``block_type == "practice"`` and are excluded automatically.

    Returns
    -------
    dict
        ``overall_accuracy`` (float; NaN if nothing responded),
        ``per_block_accuracy`` (dict ``{block: accuracy}``),
        ``n_responded`` (int), ``n_missed`` (int).
    """
    load = df[df["block_type"] == "load"]
    responded = load[load["nback_response"].notna()]
    n_responded = int(len(responded))
    n_missed = int(len(load) - n_responded)

    if n_responded == 0:
        return {
            "overall_accuracy": float("nan"),
            "per_block_accuracy": {},
            "n_responded": 0,
            "n_missed": n_missed,
        }

    overall = float(responded["nback_correct"].map(_to_bool).mean())
    per_block = {
        int(blk): float(g["nback_correct"].map(_to_bool).mean())
        for blk, g in responded.groupby("block")
    }
    return {
        "overall_accuracy": overall,
        "per_block_accuracy": per_block,
        "n_responded": n_responded,
        "n_missed": n_missed,
    }


def summarise(df: pd.DataFrame) -> str:
    """Return a human-readable 1-back manipulation-check summary string.

    Parameters
    ----------
    df : pandas.DataFrame
        Full session (output of :func:`io.load_session`).

    Returns
    -------
    str
        A multi-line summary of overall and per-block 1-back accuracy, the
        responded/missed counts, and a below-chance exclusion warning where
        applicable.
    """
    acc = nback_accuracy(df)
    lines = ["1-back manipulation check (load blocks):"]

    if acc["n_responded"] == 0:
        lines.append(
            f"  No n-back responses recorded ({acc['n_missed']} missed / no load blocks)."
        )
        return "\n".join(lines)

    lines.append(
        f"  Overall accuracy: {acc['overall_accuracy']:.1%} "
        f"({acc['n_responded']} responded, {acc['n_missed']} missed)"
    )
    for blk in sorted(acc["per_block_accuracy"]):
        lines.append(f"  Block {blk}: {acc['per_block_accuracy'][blk]:.1%}")
    if acc["overall_accuracy"] < 0.5:
        lines.append("  WARNING: below chance (0.5) — consider excluding this participant.")
    return "\n".join(lines)
