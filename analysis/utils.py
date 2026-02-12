"""
Utility functions for model evaluation and visualization.

These functions are shared across model_analyses.qmd and supplement.qmd
to ensure consistent computation of metrics and avoid code duplication.
"""

import numpy as np
import pandas as pd

# Domain structure (consistent with models/model.py)
DOMAIN_RANGES = {
    "arbitrary": (0, 5),
    "background": (5, 10),
    "identity": (10, 15),
    "morality": (15, 20),
    "politics": (20, 25),
    "preferences": (25, 30),
    "religion": (30, 35),
}

# Standard color palette (consistent across figures)
COLORS = {
    "opposing": "#648FFF",  # Blue (opposing stance)
    "shared": "#DC267F",  # Pink (shared stance)
    "human": "#2c3e50",  # Dark gray
    "bayesian": "#8576FF",  # Purple
    "egocentric": "#7BC9FF",  # Light blue
    "scrambled": "#A3FFD6",  # Mint green
    "focal": "#2E86AB",  # Dark blue for focal/discussed
    "same": "#9F8AC2",  # Wisteria lavender for same domain
    "diff": "#C8C8C8",  # Light gray for different domain
}


# =============================================================================
# GRADIENT AND RATE COMPUTATION
# =============================================================================


def get_rates(df: pd.DataFrame, col: str) -> dict:
    """
    Compute mean rates by question_type × stance.

    Args:
        df: DataFrame with 'question_type', 'stance', and value columns
        col: Column name to aggregate (e.g., 'pred_prob', 'actual')

    Returns:
        Dict mapping (question_type, stance) -> mean rate
    """
    rates = {}
    for qt in ["observed", "same_domain", "different_domain"]:
        for stance in ["shared", "opposing"]:
            cell = df[(df["question_type"] == qt) & (df["stance"] == stance)]
            rates[(qt, stance)] = cell[col].mean() if len(cell) > 0 else np.nan
    return rates


def compute_gradient(df: pd.DataFrame, col: str = "pred_prob") -> float:
    """
    Compute the generalization gradient: (same_shared - same_opposing) - (diff_shared - diff_opposing).

    This measures domain-specific transfer: how much more does agreement/disagreement
    affect same-domain questions vs. different-domain questions.

    Args:
        df: DataFrame with 'question_type', 'stance', and value columns
        col: Column name to aggregate

    Returns:
        Gradient value (positive = stronger same-domain effects)
    """
    rates = {}
    for qt in ["same_domain", "different_domain"]:
        for stance in ["shared", "opposing"]:
            cell = df[(df["question_type"] == qt) & (df["stance"] == stance)]
            rates[(qt, stance)] = cell[col].mean() if len(cell) > 0 else np.nan

    if any(pd.isna(v) for v in rates.values()):
        return np.nan

    return (rates[("same_domain", "shared")] - rates[("same_domain", "opposing")]) - (
        rates[("different_domain", "shared")] - rates[("different_domain", "opposing")]
    )


# =============================================================================
# BOOTSTRAP FUNCTIONS
# =============================================================================


def bootstrap_gradient(
    df: pd.DataFrame, col: str, n_boot: int = 1000, seed: int = 42
) -> np.ndarray:
    """
    Bootstrap confidence interval for gradient.

    Resamples participants (not individual observations) to preserve
    within-participant correlation structure.

    Args:
        df: DataFrame with 'pid', 'question_type', 'stance', and value columns
        col: Column name to aggregate
        n_boot: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Array of bootstrapped gradient values
    """
    np.random.seed(seed)
    pids = df["pid"].unique()
    boot_grads = []

    for _ in range(n_boot):
        boot_pids = np.random.choice(pids, size=len(pids), replace=True)
        boot_df = df[df["pid"].isin(boot_pids)]
        g = compute_gradient(boot_df, col)
        if not np.isnan(g):
            boot_grads.append(g)

    return np.array(boot_grads)


def bootstrap_rates(
    df: pd.DataFrame, col: str, n_boot: int = 1000, seed: int = 42
) -> dict:
    """
    Bootstrap confidence intervals for cell rates.

    Args:
        df: DataFrame with 'pid', 'question_type', 'stance', and value columns
        col: Column name to aggregate
        n_boot: Number of bootstrap iterations
        seed: Random seed

    Returns:
        Dict mapping (question_type, stance) -> (lower, upper) CI bounds
    """
    np.random.seed(seed)
    pids = df["pid"].unique()

    boot_rates = {
        (qt, stance): []
        for qt in ["observed", "same_domain", "different_domain"]
        for stance in ["shared", "opposing"]
    }

    for _ in range(n_boot):
        boot_pids = np.random.choice(pids, size=len(pids), replace=True)
        boot_df = df[df["pid"].isin(boot_pids)]
        for qt in ["observed", "same_domain", "different_domain"]:
            for stance in ["shared", "opposing"]:
                cell = boot_df[
                    (boot_df["question_type"] == qt) & (boot_df["stance"] == stance)
                ]
                boot_rates[(qt, stance)].append(
                    cell[col].mean() if len(cell) else np.nan
                )

    return {
        k: (np.percentile(v, 2.5), np.percentile(v, 97.5))
        for k, v in boot_rates.items()
    }


# =============================================================================
# SELF-STRUCTURE ANALYSIS
# =============================================================================


def compute_self_structure_gradient(responses: np.ndarray) -> float:
    """
    Compute how much a participant's responses cluster by domain.

    Measures: mean(within-domain similarity) - mean(cross-domain similarity)

    Positive values indicate responses cluster by domain (typical).
    Negative/zero values indicate responses don't cluster by domain (anomalous).

    Args:
        responses: Array of 35 responses (one per question)

    Returns:
        Self-structure gradient (positive = domain-structured responses)
    """
    within_sims, cross_sims = [], []

    for d, (start, end) in DOMAIN_RANGES.items():
        # Within-domain similarities
        for i in range(start, end):
            for j in range(i + 1, end):
                sim = 1 - abs(responses[i] - responses[j]) / 4.0
                within_sims.append(sim)

        # Cross-domain similarities
        for d2, (s2, e2) in DOMAIN_RANGES.items():
            if d != d2:
                for i in range(start, end):
                    for j in range(s2, e2):
                        sim = 1 - abs(responses[i] - responses[j]) / 4.0
                        cross_sims.append(sim)

    return np.mean(within_sims) - np.mean(cross_sims)


def compute_all_self_structure_gradients(response_matrix: pd.DataFrame) -> dict:
    """
    Compute self-structure gradient for all participants.

    Args:
        response_matrix: DataFrame with participants as rows, questions as columns

    Returns:
        Dict mapping pid -> self-structure gradient
    """
    gradients = {}
    for pid in response_matrix.index:
        responses = response_matrix.loc[pid].values
        gradients[pid] = compute_self_structure_gradient(responses)
    return gradients


# =============================================================================
# PLOTTING HELPERS
# =============================================================================


def set_plot_style():
    """Set consistent plot style for all figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Helvetica Neue",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
        }
    )


def add_significance_bracket(ax, x1, x2, y, height=0.02, text="***"):
    """Add a significance bracket between two x positions."""
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        "k-",
        lw=1,
        solid_capstyle="butt",
    )
    ax.text(
        (x1 + x2) / 2,
        y + height + 0.003,
        text,
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
