# Transition probabilities (Daw et al., 2011)
P_COMMON: float = 0.7  # probability of the "common" Stage-1 → Stage-2 transition
P_RARE: float = 0.3    # probability of the "rare" transition

# Reward random-walk parameters (Daw et al., 2011)
REWARD_WALK_SIGMA: float = 0.025   # Gaussian step SD
REWARD_WALK_MIN: float = 0.25      # reflecting lower boundary
REWARD_WALK_MAX: float = 0.75      # reflecting upper boundary

# Trial counts
N_TRIALS_MAIN: int = 200       # total main-task trials
N_BLOCKS: int = 4              # number of blocks
N_TRIALS_PER_BLOCK: int = 50   # N_TRIALS_MAIN / N_BLOCKS
N_TRIALS_PRACTICE: int = 12       # practice trials (8 no-load + 4 load); reduced from 20 — see docs/psychopy_plan.md §5
N_PRACTICE_NO_LOAD: int = 8       # no-load practice trials (shown first)
N_PRACTICE_LOAD: int = 4          # load practice trials (shown after)

# Response keys
KEY_LEFT: str = "f"   # Stage-1 and Stage-2 left choice
KEY_RIGHT: str = "j"  # Stage-1 and Stage-2 right choice
KEY_NBACK_MATCH: str = "z"     # 1-back: letter matches previous
KEY_NBACK_NO_MATCH: str = "m"  # 1-back: letter does not match

# Response windows (milliseconds)
STAGE1_WINDOW_MS: int = 2000
STAGE2_WINDOW_MS: int = 2000
ITI_MS: int = 1000
FEEDBACK_MS: int = 1000
NBACK_LETTER_MS: int = 500       # duration the 1-back letter is displayed
TRANSITION_REVEAL_MS: int = 700  # Stage-1 → Stage-2 transition reveal (planet arrival)

# Cognitive-load block structure
N_LOAD_BLOCKS: int = 2
N_NO_LOAD_BLOCKS: int = 2
# Block orders counterbalanced across participants: load=A, no-load=B
BLOCK_ORDERS: tuple[str, ...] = ("ABBA", "BAAB")

# 1-back consonant pool (~33% match rate achieved by trial-level RNG)
NBACK_CONSONANTS: tuple[str, ...] = (
    "B", "D", "F", "G", "H", "J", "K", "L", "M",
    "N", "P", "R", "S", "T", "V", "W", "X", "Z",
)
NBACK_MATCH_RATE: float = 0.33  # target proportion of match trials
