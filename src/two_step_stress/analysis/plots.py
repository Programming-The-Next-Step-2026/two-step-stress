"""Matplotlib figures for the two-step analysis.

Plotting only — these functions take the tidy frames / dicts produced by the
other analysis modules and never touch raw CSVs.  Set a non-interactive
backend (e.g. ``matplotlib.use("Agg")``) before importing this module if you
only need to save figures (see ``scripts/run_analysis.py``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Bars on the stay plot are coloured by previous reward.
REWARDED_COLOR: str = "#4472C4"    # project blue
UNREWARDED_COLOR: str = "#9AA0A6"  # grey
CHANCE_COLOR: str = "#C0504D"      # project red (chance line / accents)

_TRANSITIONS: tuple[str, ...] = ("common", "rare")


def _cell_values(df: pd.DataFrame, reward: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (heights, asymmetric_yerr[2xN]) for one reward level across transitions."""
    heights, lows, highs = [], [], []
    for transition in _TRANSITIONS:
        cell = df[(df["transition"] == transition) & (df["reward"] == reward)]
        if len(cell):
            heights.append(float(cell["stay_prob"].iloc[0]))
            lows.append(float(cell["ci_lower"].iloc[0]))
            highs.append(float(cell["ci_upper"].iloc[0]))
        else:
            heights.append(np.nan)
            lows.append(np.nan)
            highs.append(np.nan)
    heights = np.array(heights)
    yerr = np.vstack([heights - np.array(lows), np.array(highs) - heights])
    yerr = np.nan_to_num(np.clip(yerr, 0.0, None), nan=0.0)
    return heights, yerr


def _draw_panel(
    ax: plt.Axes, df: pd.DataFrame, title: str = "", show_legend: bool = True
) -> None:
    """Draw one grouped 2 x 2 panel: x = transition, bars = previous reward.

    Uses explicit numeric x positions (not the string transition labels) so
    matplotlib never falls back to categorical units.  The legend, when shown,
    is placed outside the axes so it can't overlap the bars.
    """
    x = np.arange(len(_TRANSITIONS))
    width = 0.38

    rew_h, rew_err = _cell_values(df, 1)
    unrew_h, unrew_err = _cell_values(df, 0)

    ax.bar(x - width / 2, rew_h, width, yerr=rew_err, capsize=4,
           color=REWARDED_COLOR, label="Rewarded")
    ax.bar(x + width / 2, unrew_h, width, yerr=unrew_err, capsize=4,
           color=UNREWARDED_COLOR, label="Unrewarded")

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in _TRANSITIONS])
    ax.set_xlabel("Transition")
    ax.set_ylabel("Stay probability")
    ax.set_ylim(0.0, 1.0)
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)


def plot_stay_probability_2x2(stay_df: pd.DataFrame, save_path=None) -> plt.Figure:
    """Daw-2011-style 2 x 2 stay-probability bar chart (single panel)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    _draw_panel(ax, stay_df, title="Stay probability")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_stay_probability_by_load(stay_df_split: pd.DataFrame, save_path=None) -> plt.Figure:
    """Two-panel stay plot: load (left) vs no_load (right), shared y-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, block_type, title in (
        (axes[0], "load", "Load"),
        (axes[1], "no_load", "No load"),
    ):
        _draw_panel(
            ax, stay_df_split[stay_df_split["block_type"] == block_type], title,
            show_legend=False,
        )
    # Single legend outside the right panel (avoids one-per-panel overlap).
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Stay probability by reward x transition")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_nback_accuracy(accuracy_dict: dict, save_path=None) -> plt.Figure:
    """Bar chart of per-block 1-back accuracy with a chance line at 0.5."""
    per_block = accuracy_dict["per_block_accuracy"]
    blocks = sorted(per_block)
    values = [per_block[b] for b in blocks]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(blocks))
    ax.bar(x, values, color=REWARDED_COLOR, width=0.5)
    ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("Block")
    ax.set_ylabel("1-back accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("1-back manipulation check")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
