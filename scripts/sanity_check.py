"""Sanity check: MF vs MB agent stay-probability patterns (Daw et al., 2011 Fig 2).

Run from the repo root:
    python scripts/sanity_check.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from two_step_stress.config import P_COMMON, P_RARE
from two_step_stress.task.transitions import apply_transition, build_transition_mapping
from two_step_stress.task.rewards import initialise_reward_probs, step_reward_probs

N_TRIALS = 5_000
ALPHA    = 0.3
BETA     = 5.0
SEED     = 42


def softmax_choice(q: np.ndarray, rng: np.random.Generator) -> int:
    """Sample an action from a softmax over action values.

    Parameters
    ----------
    q : numpy.ndarray
        Action values (one per available action).
    rng : numpy.random.Generator
        Random generator used to draw the choice.

    Returns
    -------
    int
        Index of the chosen action, drawn with probability proportional to
        ``exp(BETA * q)``.
    """
    e = np.exp(BETA * (q - q.max()))
    return int(rng.choice(len(q), p=e / e.sum()))


def simulate(mode: str, rng: np.random.Generator) -> list[tuple]:
    """Run one agent through ``N_TRIALS`` of the real two-step engine.

    Both agents use SARSA(0) learning at Stage 2 and differ only at Stage 1:
    the model-free agent reinforces the chosen Stage-1 action with the same
    reward-prediction error, whereas the model-based agent re-derives Stage-1
    values each trial from the Stage-2 values, weighted by the known 70/30
    transition structure.

    Parameters
    ----------
    mode : str
        ``"mf"`` (model-free) or ``"mb"`` (model-based).
    rng : numpy.random.Generator
        Random generator driving choices, transitions, rewards, and the walk.

    Returns
    -------
    list of tuple
        One ``(stage1_choice, reward, transition_type)`` tuple per trial.
    """
    mapping       = build_transition_mapping(0)
    reward_probs  = initialise_reward_probs(rng)
    q2 = np.full((2, 2), 0.5)   # Stage-2 Q-values [state, action]
    q1 = np.full(2, 0.5)         # Stage-1 Q-values (MF only; MB recomputes each trial)

    records = []
    for _ in range(N_TRIALS):
        # Stage-1: MB recomputes q1 from q2 + known transition model each trial
        if mode == "mb":
            q1 = np.array([
                P_COMMON * q2[mapping[a]].max() + P_RARE * q2[1 - mapping[a]].max()
                for a in range(2)
            ])
        a1 = softmax_choice(q1, rng)

        s2, transition_type = apply_transition(a1, mapping, rng)

        a2 = softmax_choice(q2[s2], rng)

        r = int(rng.random() < reward_probs[s2 * 2 + a2])

        # Q updates
        delta = r - q2[s2, a2]
        q2[s2, a2] += ALPHA * delta
        if mode == "mf":
            q1[a1] += ALPHA * delta   # direct reward signal, ignores transition structure

        reward_probs = step_reward_probs(reward_probs, rng)
        records.append((a1, r, transition_type))

    return records


def stay_matrix(records: list[tuple]) -> dict:
    """Return {reward: {transition: P(stay)}} from consecutive-trial pairs."""
    counts = {r: {t: [0, 0] for t in ("common", "rare")} for r in (0, 1)}
    for i in range(1, len(records)):
        prev_a1, prev_r, prev_t = records[i - 1]
        curr_a1, _, _           = records[i]
        counts[prev_r][prev_t][0] += int(curr_a1 == prev_a1)
        counts[prev_r][prev_t][1] += 1
    return {
        r: {t: counts[r][t][0] / counts[r][t][1] for t in ("common", "rare")}
        for r in (0, 1)
    }


def plot_panel(ax: plt.Axes, mat: dict, title: str) -> None:
    x, w = np.array([0.0, 1.0]), 0.3
    ax.bar(x - w/2, [mat[1]["common"], mat[0]["common"]], w, label="Common", color="#4472C4")
    ax.bar(x + w/2, [mat[1]["rare"],   mat[0]["rare"]],   w, label="Rare",   color="#C0504D")
    ax.set_xticks(x)
    ax.set_xticklabels(["Rewarded", "Unrewarded"])
    ax.set_ylabel("P(stay)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


# --- run ---
rng_mf = np.random.default_rng(SEED)
rng_mb = np.random.default_rng(SEED + 1)

print("Simulating MF agent (5 000 trials)…")
mf_mat = stay_matrix(simulate("mf", rng_mf))

print("Simulating MB agent (5 000 trials)…")
mb_mat = stay_matrix(simulate("mb", rng_mb))

for name, mat in [("MF", mf_mat), ("MB", mb_mat)]:
    print(f"\n{name} stay probabilities:")
    print(f"  Rewarded   — common: {mat[1]['common']:.3f}   rare: {mat[1]['rare']:.3f}")
    print(f"  Unrewarded — common: {mat[0]['common']:.3f}   rare: {mat[0]['rare']:.3f}")

# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
plot_panel(axes[0], mf_mat, "Model-Free")
plot_panel(axes[1], mb_mat, "Model-Based")
fig.suptitle("Stay probability by reward × transition (Daw et al., 2011)", fontsize=11)
fig.tight_layout()

out = Path(__file__).parent / "sanity_check.png"
fig.savefig(out, dpi=150)
print(f"\nFigure saved → {out}")
