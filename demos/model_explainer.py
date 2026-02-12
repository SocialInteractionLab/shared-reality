"""
Bayesian Factor Model Explainer.

An interactive lesson that walks through how the commonality inference model works,
from factor analysis to Bayesian updating to predictions.

Run with: uv run marimo run demos/model_explainer.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


# =============================================================================
# SETUP
# =============================================================================


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    import sys
    from pathlib import Path

    _project_root = Path(__file__).parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    from scipy.stats import norm

    from models.model import (
        load_factor_loadings,
        load_question_means,
        load_responses,
        load_correlation_matrix,
        DOMAIN_RANGES,
    )

    return (
        np,
        pd,
        plt,
        Ellipse,
        transforms,
        Path,
        norm,
        load_factor_loadings,
        load_question_means,
        load_responses,
        load_correlation_matrix,
        DOMAIN_RANGES,
    )


@app.cell
def __(
    load_factor_loadings,
    load_question_means,
    load_responses,
    load_correlation_matrix,
    DOMAIN_RANGES,
    Path,
    pd,
    np,
):
    # Load all data once
    all_loadings = load_factor_loadings()  # Full 35x35
    means = load_question_means()
    responses_df = load_responses()

    # Correlation matrix
    corr_matrix = load_correlation_matrix()

    # Eigenvalues for variance explained
    eigvals = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]
    variance_explained = eigvals / eigvals.sum() * 100

    # Question text
    _data_dir = Path(__file__).parent.parent / "data"
    _df = pd.read_csv(_data_dir / "responses.csv", low_memory=False)
    _q_map = _df.groupby("question")["preChatQuestion"].first().to_dict()
    questions = [_q_map.get(i + 1, f"Question {i + 1}") for i in range(35)]

    # Domain labels and colors
    domain_labels = []
    for _i in range(35):
        for _domain, (_start, _end) in DOMAIN_RANGES.items():
            if _start <= _i < _end:
                domain_labels.append(_domain)
                break

    DOMAIN_COLORS = {
        "arbitrary": "#00BFFF",
        "background": "#FF6600",
        "identity": "#39FF14",
        "morality": "#FF1493",
        "politics": "#9D00FF",
        "preferences": "#FFD700",
        "religion": "#00FFFF",
    }

    # Cyberpunk palette
    CYBER_BG = "#0D0221"
    CYBER_GRID = "#1a1a3a"
    CYBER_CYAN = "#00FFFF"
    CYBER_MAGENTA = "#FF00FF"
    CYBER_YELLOW = "#FFFF00"
    CYBER_ORANGE = "#FF6600"
    CYBER_GREEN = "#39FF14"

    def style_ax(ax):
        """Apply cyberpunk styling to an axis."""
        ax.set_facecolor(CYBER_BG)
        ax.tick_params(colors="white", labelsize=10)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_color(CYBER_GRID)

    def confidence_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
        """Draw a confidence ellipse for a 2D Gaussian."""
        if cov.shape[0] < 2:
            return None
        _pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-10)
        _ell_radius_x = np.sqrt(1 + _pearson)
        _ell_radius_y = np.sqrt(1 - _pearson)
        _ellipse = Ellipse(
            (0, 0), width=_ell_radius_x * 2, height=_ell_radius_y * 2, **kwargs
        )
        _scale_x = np.sqrt(cov[0, 0]) * n_std
        _scale_y = np.sqrt(cov[1, 1]) * n_std
        _transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(_scale_x, _scale_y)
            .translate(mean[0], mean[1])
        )
        _ellipse.set_transform(_transf + ax.transData)
        return ax.add_patch(_ellipse)

    return (
        all_loadings,
        means,
        responses_df,
        corr_matrix,
        eigvals,
        variance_explained,
        questions,
        domain_labels,
        DOMAIN_COLORS,
        CYBER_BG,
        CYBER_GRID,
        CYBER_CYAN,
        CYBER_MAGENTA,
        CYBER_YELLOW,
        CYBER_ORANGE,
        CYBER_GREEN,
        style_ax,
        confidence_ellipse,
    )


# =============================================================================
# SECTION 1: OVERVIEW
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
# The Bayesian Factor Model: An Interactive Explainer

## The Setup

Imagine you're in an experiment. You and a stranger both answered **35 opinion questions** (covering politics, religion, morality, identity, preferences, background, and arbitrary topics). You can see your own answers, but not your partner's.

Then you learn **one thing**: your partner's response to a single question.

**The question**: How should you update your beliefs about their responses to the other 34 questions?

This notebook walks through the **Bayesian factor model** — a computational model of how people make these inferences. The key idea:

1. **Factor analysis** extracts the hidden structure of how opinions covary in the population
2. **Bayesian inference** uses that structure to update beliefs about a partner after observing one response
3. The model predicts **domain-specific generalization**: observing a politics response updates predictions about other politics questions more than about religion questions

---

**Roadmap**:
- Sections 2-3: What factor analysis is and what θ means
- Sections 4-6: The Bayesian update (prior → observation → posterior)
- Sections 7-8: How predictions and P(match) are computed
- Section 9: Why the generalization gradient is the right evaluation metric
    """)
    return


# =============================================================================
# SECTION 2: FACTOR ANALYSIS PRIMER
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 2: What Is a Factor? — Factor Analysis Primer

## The Problem

People's responses to 35 questions aren't independent. Someone who is politically liberal will probably answer several political questions similarly, and maybe some morality questions too. There's **hidden structure** — a smaller number of underlying dimensions that drive the correlations.

## The Correlation Matrix

First, we compute a **35 × 35 correlation matrix** across all participants. Each entry $r_{ij}$ measures how correlated question $i$ and question $j$ are across the population. If $r_{ij}$ is high, people who answer question $i$ high tend to answer question $j$ high too.

## What Factor Analysis Does

Factor analysis decomposes this 35 × 35 matrix into a smaller set of $k$ **latent dimensions** (factors) that explain most of the correlations. Think of it as dimensionality reduction: 35 correlated questions → $k$ uncorrelated factors.

## How It Works in Our Code

1. Compute the correlation matrix $R$ (35 × 35)
2. Eigendecompose: $R = V \\cdot \\text{diag}(\\lambda) \\cdot V^\\top$
3. Factor loadings: $\\Lambda = V \\cdot \\sqrt{\\lambda}$
4. Keep the top $k$ factors (largest eigenvalues)

## What Is a Factor?

A factor is an **abstract latent dimension** that multiple questions share. We don't label factors in advance — we infer their meaning from which questions load on them:

- If a factor has high loadings for all politics questions → it probably captures political orientation
- If a factor has high loadings for religion questions → it probably captures religiosity

## The Key Matrices

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $\\Lambda$ | 35 × $k$ | Factor loadings. Row $q$ = how question $q$ relates to each factor |
| $\\mu$ | 35 | Population mean response per question |
| $\\theta$ | $k$ | A person's latent factor scores (unobserved) |

The **generative model**: $r = \\Lambda\\theta + \\mu$ — factor scores × loadings + means = predicted responses.

**Important**: This is standard factor analysis — there's nothing Bayesian about it yet. It just gives us the "map" of how questions relate in the population.
    """)
    return


# Widget cell for Section 2
@app.cell
def __(mo):
    fa_k = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=5,
        label="Number of factors (k)",
        show_value=True,
    )
    return (fa_k,)


# Visualization cell for Section 2
@app.cell
def __(
    mo,
    fa_k,
    all_loadings,
    corr_matrix,
    variance_explained,
    domain_labels,
    DOMAIN_COLORS,
    DOMAIN_RANGES,
    CYBER_BG,
    CYBER_GRID,
    CYBER_CYAN,
    CYBER_YELLOW,
    style_ax,
    np,
    plt,
):
    _k = int(fa_k.value)
    _loadings_k = all_loadings[:, :_k]

    _fig, _axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
        facecolor=CYBER_BG,
        gridspec_kw={"width_ratios": [1.2, 1, 0.8]},
    )

    # Panel 1: Correlation matrix
    _ax1 = _axes[0]
    _ax1.set_facecolor(CYBER_BG)
    _im = _ax1.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        _ax1.axhline(y=_start - 0.5, color="white", linewidth=0.5, alpha=0.5)
        _ax1.axvline(x=_start - 0.5, color="white", linewidth=0.5, alpha=0.5)
        _ax1.text(
            (_start + _end) / 2 - 0.5,
            -1.5,
            _domain[:3].upper(),
            ha="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
    _ax1.set_title(
        "Population Correlation Matrix (35 × 35)",
        fontsize=13,
        fontweight="bold",
        color=CYBER_CYAN,
    )
    _ax1.tick_params(colors="white", labelsize=7)
    _cbar = plt.colorbar(_im, ax=_ax1, fraction=0.046, pad=0.04)
    _cbar.ax.tick_params(colors="white", labelsize=8)
    for _spine in _ax1.spines.values():
        _spine.set_color(CYBER_GRID)

    # Panel 2: Factor loadings heatmap
    _ax2 = _axes[1]
    _ax2.set_facecolor(CYBER_BG)
    _max_abs = np.abs(_loadings_k).max()
    _im2 = _ax2.imshow(
        _loadings_k, cmap="RdBu_r", vmin=-_max_abs, vmax=_max_abs, aspect="auto"
    )
    _ax2.set_xlabel("Factor", fontsize=11, color="white")
    _ax2.set_xticks(range(_k))
    _ax2.set_xticklabels([f"F{i+1}" for i in range(_k)])
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        _ax2.axhline(y=_start - 0.5, color="white", linewidth=0.5, alpha=0.5)
        _mid = (_start + _end) / 2 - 0.5
        _ax2.text(
            -0.8,
            _mid,
            _domain[:4].upper(),
            ha="right",
            va="center",
            fontsize=7,
            color=DOMAIN_COLORS[_domain],
            fontweight="bold",
        )
    _ax2.set_title(
        f"Factor Loadings Λ (35 × {_k})",
        fontsize=13,
        fontweight="bold",
        color=CYBER_YELLOW,
    )
    _ax2.tick_params(colors="white", labelsize=7)
    _cbar2 = plt.colorbar(_im2, ax=_ax2, fraction=0.046, pad=0.04)
    _cbar2.ax.tick_params(colors="white", labelsize=8)
    for _spine in _ax2.spines.values():
        _spine.set_color(CYBER_GRID)

    # Panel 3: Variance explained
    _ax3 = _axes[2]
    style_ax(_ax3)
    _x_eig = np.arange(1, min(15, len(variance_explained)) + 1)
    _ax3.bar(
        _x_eig,
        variance_explained[: len(_x_eig)],
        color=CYBER_GRID,
        edgecolor="white",
        linewidth=0.5,
    )
    _ax3.bar(
        _x_eig[:_k],
        variance_explained[:_k],
        color=CYBER_CYAN,
        edgecolor="white",
        linewidth=0.5,
    )
    _cum_var = variance_explained[:_k].sum()
    _ax3.set_xlabel("Factor", fontsize=11)
    _ax3.set_ylabel("% Variance Explained", fontsize=11)
    _ax3.set_title(
        f"k={_k} → {_cum_var:.1f}% total",
        fontsize=13,
        fontweight="bold",
        color=CYBER_CYAN,
    )
    _ax3.grid(True, color=CYBER_GRID, alpha=0.3, axis="y")

    plt.tight_layout()

    mo.vstack(
        [
            mo.hstack([fa_k], justify="center"),
            _fig,
            mo.md(f"""
**Left**: The 35 × 35 correlation matrix shows block structure — questions within the same domain tend to be more correlated. **Center**: The factor loadings Λ show which questions load on which factors. With k={_k}, the model compresses 35 questions into {_k} latent dimensions explaining **{_cum_var:.1f}%** of variance. **Right**: Variance explained per factor (cyan = included).
        """),
        ]
    )
    return


# =============================================================================
# SECTION 3: θ — THE PARTNER'S POSITION IN FACTOR SPACE
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 3: θ — The Partner's Position in Factor Space

$\\theta \\in \\mathbb{R}^k$ is where a person sits on the latent factor dimensions. It's **not** their predicted responses — it's the **compressed, lower-dimensional representation** that generates those responses.

With $k=5$, $\\theta = [1.2, -0.8, 0.3, 0.1, -1.5]$ would mean this person is high on factor 1, low on factor 2, and so on.

The mapping from $\\theta$ to responses is:

$$r = \\Lambda\\theta + \\mu$$

- $\\theta \\in \\mathbb{R}^k$ — where they are on $k$ latent dimensions
- $\\Lambda\\theta + \\mu \\in \\mathbb{R}^{35}$ — what that implies for their 35 responses

We **never observe** $\\theta$ directly. The whole model is about inferring it from one observed response. Set θ manually below to see what response profile it implies.
    """)
    return


# Widget cell for Section 3
@app.cell
def __(mo):
    theta_f1 = mo.ui.slider(
        start=-3.0,
        stop=3.0,
        step=0.5,
        value=0.0,
        label="θ₁ (Factor 1)",
        show_value=True,
    )
    theta_f2 = mo.ui.slider(
        start=-3.0,
        stop=3.0,
        step=0.5,
        value=0.0,
        label="θ₂ (Factor 2)",
        show_value=True,
    )
    return (
        theta_f1,
        theta_f2,
    )


# Visualization cell for Section 3
@app.cell
def __(
    mo,
    theta_f1,
    theta_f2,
    all_loadings,
    means,
    domain_labels,
    DOMAIN_COLORS,
    DOMAIN_RANGES,
    CYBER_BG,
    CYBER_GRID,
    CYBER_CYAN,
    CYBER_YELLOW,
    style_ax,
    np,
    plt,
):
    # Use k=2 for this demo to match the two sliders
    _loadings_2 = all_loadings[:, :2]
    _theta = np.array([float(theta_f1.value), float(theta_f2.value)])
    _pred = _loadings_2 @ _theta + means

    _fig, _axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=CYBER_BG)

    # Panel 1: θ in factor space
    _ax1 = _axes[0]
    style_ax(_ax1)
    _ax1.scatter(
        [_theta[0]],
        [_theta[1]],
        color=CYBER_YELLOW,
        s=200,
        marker="X",
        linewidths=2,
        zorder=5,
    )
    _ax1.scatter(
        [0],
        [0],
        color=CYBER_CYAN,
        s=100,
        marker="+",
        linewidths=2,
        zorder=4,
        label="Population mean (θ=0)",
    )
    _ax1.axhline(y=0, color=CYBER_GRID, linewidth=0.5)
    _ax1.axvline(x=0, color=CYBER_GRID, linewidth=0.5)
    _ax1.set_xlim(-4, 4)
    _ax1.set_ylim(-4, 4)
    _ax1.set_aspect("equal")
    _ax1.set_xlabel("θ₁ (Factor 1)", fontsize=12)
    _ax1.set_ylabel("θ₂ (Factor 2)", fontsize=12)
    _ax1.set_title(
        "Partner's Position in Factor Space",
        fontsize=14,
        fontweight="bold",
        color=CYBER_YELLOW,
    )
    _ax1.legend(
        fontsize=9, facecolor=CYBER_BG, edgecolor=CYBER_GRID, labelcolor="white"
    )
    _ax1.grid(True, color=CYBER_GRID, alpha=0.3)

    # Panel 2: Implied response profile
    _ax2 = _axes[1]
    style_ax(_ax2)
    _x = np.arange(35)
    _colors = [DOMAIN_COLORS[d] for d in domain_labels]
    _ax2.bar(
        _x,
        means,
        color=CYBER_GRID,
        alpha=0.5,
        edgecolor="white",
        linewidth=0.3,
        label="Population mean (μ)",
    )
    _ax2.bar(
        _x,
        _pred,
        color=_colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        label="Λθ + μ",
    )
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax2.axvline(x=_start - 0.5, color=CYBER_GRID, alpha=0.8, linewidth=1)
        _ax2.text(
            (_start + _end) / 2 - 0.5,
            0.3,
            _domain[:3].upper(),
            ha="center",
            fontsize=7,
            color="white",
            alpha=0.7,
        )
    _ax2.set_xlim(-0.5, 34.5)
    _ax2.set_ylim(0, 7)
    _ax2.set_xlabel("Question", fontsize=12)
    _ax2.set_ylabel("Response", fontsize=12)
    _ax2.set_title(
        "Implied Response Profile: r = Λθ + μ",
        fontsize=14,
        fontweight="bold",
        color=CYBER_CYAN,
    )
    _ax2.legend(
        fontsize=9, facecolor=CYBER_BG, edgecolor=CYBER_GRID, labelcolor="white"
    )

    plt.tight_layout()

    _is_zero = abs(_theta[0]) < 0.01 and abs(_theta[1]) < 0.01
    _note = (
        "θ = 0 → predicted responses are just the population means μ."
        if _is_zero
        else f"θ = [{_theta[0]:.1f}, {_theta[1]:.1f}] shifts predictions away from the population mean, more for questions that load heavily on these factors."
    )

    mo.vstack(
        [
            mo.hstack([theta_f1, theta_f2], justify="center"),
            _fig,
            mo.md(
                f"**Note**: {_note} (Using k=2 here for visualization; the full model uses k=5.)"
            ),
        ]
    )
    return


# =============================================================================
# SECTION 4: THE PRIOR
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 4: The Prior — Before Observing Anything

Before seeing any of the partner's responses, we assume they're **average**:

$$\\theta \\sim \\mathcal{N}(0, \\sigma^2_{\\text{prior}} \\cdot I_k)$$

This means:
- **Prior mean** = $[0, 0, \\ldots, 0]$ — centered at the population average in factor space
- **Prior covariance** = $\\sigma^2_{\\text{prior}} \\cdot I$ — equal uncertainty in all directions

Since the generative model is $r = \\Lambda\\theta + \\mu$, plugging in $\\theta = 0$ gives predicted responses = $\\mu$ (the population means). The prior on $\\theta$ *implies* a prior on responses, but the prior lives in the compressed latent space — not in the 35-dimensional response space.

$\\sigma_{\\text{prior}}$ controls how uncertain we are: larger $\\sigma_{\\text{prior}}$ means we think the partner could be anywhere in factor space, so the observation will have more influence on the posterior.
    """)
    return


# Widget cell for Section 4
@app.cell
def __(mo):
    prior_sigma = mo.ui.slider(
        start=0.5, stop=3.0, step=0.25, value=1.5, label="σ_prior", show_value=True
    )
    return (prior_sigma,)


# Visualization cell for Section 4
@app.cell
def __(
    mo,
    prior_sigma,
    means,
    DOMAIN_RANGES,
    domain_labels,
    DOMAIN_COLORS,
    CYBER_BG,
    CYBER_GRID,
    CYBER_CYAN,
    style_ax,
    np,
    plt,
    Ellipse,
):
    _sigma = float(prior_sigma.value)

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=CYBER_BG)

    # Panel 1: Prior in factor space (2D)
    _ax1 = _axes[0]
    style_ax(_ax1)
    for _n_std, _alpha, _ls in [(1, 1.0, "-"), (2, 0.6, "--")]:
        _ell = Ellipse(
            (0, 0),
            width=2 * _sigma * _n_std,
            height=2 * _sigma * _n_std,
            facecolor="none",
            edgecolor=CYBER_CYAN,
            linewidth=2.5 if _n_std == 1 else 1.5,
            linestyle=_ls,
            alpha=_alpha,
        )
        _ax1.add_patch(_ell)
    _ax1.scatter([0], [0], color=CYBER_CYAN, s=150, marker="+", linewidths=3, zorder=5)
    _ax1.set_xlim(-4, 4)
    _ax1.set_ylim(-4, 4)
    _ax1.set_aspect("equal")
    _ax1.set_xlabel("θ₁", fontsize=14)
    _ax1.set_ylabel("θ₂", fontsize=14)
    _ax1.set_title(
        "Prior: θ ~ N(0, σ²·I)", fontsize=14, fontweight="bold", color=CYBER_CYAN
    )
    _ax1.axhline(y=0, color=CYBER_GRID, linewidth=0.5)
    _ax1.axvline(x=0, color=CYBER_GRID, linewidth=0.5)
    _ax1.grid(True, color=CYBER_GRID, alpha=0.3)
    _ax1.text(
        0.05,
        0.95,
        f"σ_prior = {_sigma:.1f}\n1σ and 2σ ellipses",
        transform=_ax1.transAxes,
        fontsize=11,
        va="top",
        family="monospace",
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round", fc=CYBER_BG, ec=CYBER_CYAN, alpha=0.9, pad=0.5),
    )

    # Panel 2: Implied prior on responses (= population means)
    _ax2 = _axes[1]
    style_ax(_ax2)
    _x = np.arange(35)
    _colors = [DOMAIN_COLORS[d] for d in domain_labels]
    _ax2.bar(_x, means, color=_colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax2.axvline(x=_start - 0.5, color=CYBER_GRID, alpha=0.8, linewidth=1)
        _ax2.text(
            (_start + _end) / 2 - 0.5,
            0.3,
            _domain[:3].upper(),
            ha="center",
            fontsize=7,
            color="white",
            alpha=0.7,
        )
    _ax2.set_xlim(-0.5, 34.5)
    _ax2.set_ylim(0, 6)
    _ax2.set_xlabel("Question", fontsize=12)
    _ax2.set_ylabel("Predicted Response", fontsize=12)
    _ax2.set_title(
        "Implied Prior: E[r] = Λ·0 + μ = μ (population means)",
        fontsize=14,
        fontweight="bold",
        color=CYBER_CYAN,
    )

    plt.tight_layout()

    mo.vstack(
        [
            mo.hstack([prior_sigma], justify="center"),
            _fig,
            mo.md(f"""
With θ = 0, the predicted response for every question is just the population mean. The prior doesn't favor any particular response profile — it says "this partner is probably average, but I'm not sure." The uncertainty (σ_prior = {_sigma:.1f}) determines how much a single observation can shift us away from this default.
        """),
        ]
    )
    return


# =============================================================================
# SECTION 5: THE OBSERVATION
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 5: The Observation — One Response Constrains θ

Now we observe the partner's response to **one question**. If we observe $r_{\\text{obs}}$ on question $q$, the generative model says:

$$r_{\\text{obs}} = \\Lambda_q^\\top \\theta + \\mu_q$$

This defines a **hyperplane** (a line in 2D) in factor space — all values of $\\theta$ that are consistent with what we observed. The **loading vector** $\\Lambda_q$ determines the orientation of this constraint.

Key insight: questions with **different loadings** constrain $\\theta$ in different directions. A politics question constrains along the "political" axis, a religion question constrains along the "religious" axis.
    """)
    return


# Widget cell for Section 5
@app.cell
def __(mo, questions, domain_labels):
    _question_options = {
        f"Q{i}: [{domain_labels[i][:4].upper()}] {q[:45]}{'...' if len(q) > 45 else ''}": i
        for i, q in enumerate(questions)
    }
    obs_question = mo.ui.dropdown(
        options=_question_options,
        value=list(_question_options.keys())[22],
        label="Observed Question",
    )
    obs_response = mo.ui.slider(
        start=1, stop=5, step=1, value=5, label="Partner's Response", show_value=True
    )
    return (
        obs_question,
        obs_response,
    )


# Visualization cell for Section 5
@app.cell
def __(
    mo,
    prior_sigma,
    obs_question,
    obs_response,
    all_loadings,
    means,
    domain_labels,
    CYBER_BG,
    CYBER_GRID,
    CYBER_CYAN,
    CYBER_YELLOW,
    CYBER_ORANGE,
    style_ax,
    np,
    plt,
    Ellipse,
):
    _obs_q = obs_question.value
    _r_obs = float(obs_response.value)
    _sigma = float(prior_sigma.value)
    _loadings_2 = all_loadings[:, :2]
    _L_obs = _loadings_2[_obs_q]
    _mu_obs = means[_obs_q]
    _constraint_val = _r_obs - _mu_obs

    _fig, _ax = plt.subplots(figsize=(8, 8), facecolor=CYBER_BG)
    style_ax(_ax)

    # Prior ellipses (faint)
    for _n_std, _alpha, _ls in [(1, 0.3, ":"), (2, 0.2, ":")]:
        _ell = Ellipse(
            (0, 0),
            width=2 * _sigma * _n_std,
            height=2 * _sigma * _n_std,
            facecolor="none",
            edgecolor=CYBER_CYAN,
            linewidth=1.5,
            linestyle=_ls,
            alpha=_alpha,
        )
        _ax.add_patch(_ell)
    _ax.scatter(
        [0],
        [0],
        color=CYBER_CYAN,
        s=80,
        marker="+",
        linewidths=2,
        alpha=0.5,
        label="Prior mean",
    )

    # Likelihood constraint line
    _plot_range = 4.0
    if abs(_L_obs[1]) > 1e-6:
        _theta1 = np.linspace(-_plot_range, _plot_range, 200)
        _theta2 = (_constraint_val - _L_obs[0] * _theta1) / _L_obs[1]
        _valid = np.abs(_theta2) < _plot_range * 1.5
        _ax.plot(
            _theta1[_valid],
            _theta2[_valid],
            color=CYBER_YELLOW,
            linewidth=3.5,
            label=f"Constraint: Λ'θ = {_constraint_val:.2f}",
        )
    elif abs(_L_obs[0]) > 1e-6:
        _ax.axvline(
            x=_constraint_val / _L_obs[0],
            color=CYBER_YELLOW,
            linewidth=3.5,
            label=f"Constraint: Λ'θ = {_constraint_val:.2f}",
        )

    # Loading vector arrow
    _arrow_scale = 2.5
    _ax.arrow(
        0,
        0,
        _L_obs[0] * _arrow_scale,
        _L_obs[1] * _arrow_scale,
        head_width=0.15,
        head_length=0.1,
        fc=CYBER_ORANGE,
        ec=CYBER_ORANGE,
        lw=2.5,
        zorder=5,
    )
    _ax.text(
        _L_obs[0] * _arrow_scale * 1.15,
        _L_obs[1] * _arrow_scale * 1.15,
        f"Λ_{_obs_q}",
        fontsize=14,
        color=CYBER_ORANGE,
        fontweight="bold",
        ha="center",
    )

    _ax.set_xlim(-_plot_range, _plot_range)
    _ax.set_ylim(-_plot_range, _plot_range)
    _ax.set_aspect("equal")
    _ax.set_xlabel("θ₁", fontsize=14)
    _ax.set_ylabel("θ₂", fontsize=14)
    _ax.set_title(
        f"Observation: Q{_obs_q} [{domain_labels[_obs_q]}] = {_r_obs:.0f}",
        fontsize=14,
        fontweight="bold",
        color=CYBER_YELLOW,
    )
    _ax.axhline(y=0, color=CYBER_GRID, linewidth=0.5)
    _ax.axvline(x=0, color=CYBER_GRID, linewidth=0.5)
    _ax.grid(True, color=CYBER_GRID, alpha=0.3)
    _ax.legend(
        fontsize=10, facecolor=CYBER_BG, edgecolor=CYBER_GRID, labelcolor="white"
    )
    plt.tight_layout()

    mo.vstack(
        [
            mo.hstack([obs_question, obs_response], justify="center"),
            _fig,
            mo.md(f"""
The **yellow line** shows all values of θ consistent with observing r = {_r_obs:.0f} on Q{_obs_q} ({domain_labels[_obs_q]}). The **orange arrow** is the loading vector Λ_{_obs_q} = [{_L_obs[0]:.3f}, {_L_obs[1]:.3f}], which points perpendicular to the constraint. Try switching the observed question to a different domain — the constraint line rotates because each question has different factor loadings.
        """),
        ]
    )
    return


# =============================================================================
# SECTION 6: THE BAYESIAN UPDATE
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 6: The Bayesian Update — What Makes This Bayesian

Now we combine the **prior** (what we believed before) with the **likelihood** (what the observation tells us) to get the **posterior** (our updated belief):

$$P(\\theta \\mid r_{\\text{obs}}) \\propto P(r_{\\text{obs}} \\mid \\theta) \\cdot P(\\theta)$$

Since everything is Gaussian, the posterior is also Gaussian: $\\theta \\mid r_{\\text{obs}} \\sim \\mathcal{N}(\\mu_{\\text{post}}, \\Sigma_{\\text{post}})$

**This is what's Bayesian**: we don't just get a point estimate $\\hat{\\theta}$ — we maintain a full **distribution** over $\\theta$, with:
- $\\mu_{\\text{post}}$ = our best guess (the posterior mean)
- $\\Sigma_{\\text{post}}$ = how uncertain we still are (the posterior covariance)

The update uses **Kalman filtering**:

$$K = \\frac{\\Sigma_{\\text{prior}} \\Lambda_q}{\\Lambda_q^\\top \\Sigma_{\\text{prior}} \\Lambda_q}
\\qquad
\\mu_{\\text{post}} = K \\cdot (r_{\\text{obs}} - \\mu_q)
\\qquad
\\Sigma_{\\text{post}} = \\Sigma_{\\text{prior}} - K \\Lambda_q^\\top \\Sigma_{\\text{prior}}$$

where $K$ is the **Kalman gain** — how much to shift $\\theta$ per unit of surprise.

The posterior is **not** $\\theta$ itself. $\\theta$ is the partner's true position, which we can never observe. The posterior is our **belief** about $\\theta$ — the best we can do given one data point.
    """)
    return


# Computation + visualization cell for Section 6
@app.cell
def __(
    mo,
    obs_question,
    obs_response,
    prior_sigma,
    all_loadings,
    means,
    domain_labels,
    CYBER_BG,
    CYBER_GRID,
    CYBER_CYAN,
    CYBER_YELLOW,
    CYBER_MAGENTA,
    CYBER_ORANGE,
    style_ax,
    confidence_ellipse,
    np,
    plt,
):
    _obs_q = obs_question.value
    _r_obs = float(obs_response.value)
    _sigma = float(prior_sigma.value)

    _loadings_2 = all_loadings[:, :2]
    _L_obs = _loadings_2[_obs_q]
    _mu_obs = means[_obs_q]

    # Prior
    _prior_mean = np.zeros(2)
    _prior_cov = _sigma**2 * np.eye(2)

    # Kalman update
    _L_cov_L = _L_obs @ _prior_cov @ _L_obs
    _K = _prior_cov @ _L_obs / (_L_cov_L + 1e-10)
    _innovation = (_r_obs - _mu_obs) - _L_obs @ _prior_mean
    bay_post_mean = _prior_mean + _K * _innovation
    bay_post_cov = _prior_cov - np.outer(_K, _L_obs @ _prior_cov)

    _plot_range = 4.0
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=CYBER_BG)

    # Panel 1: PRIOR
    _ax1 = _axes[0]
    style_ax(_ax1)
    confidence_ellipse(
        _prior_mean,
        _prior_cov,
        _ax1,
        n_std=1,
        facecolor="none",
        edgecolor=CYBER_CYAN,
        linewidth=2.5,
    )
    confidence_ellipse(
        _prior_mean,
        _prior_cov,
        _ax1,
        n_std=2,
        facecolor="none",
        edgecolor=CYBER_CYAN,
        linewidth=1.5,
        linestyle="--",
        alpha=0.6,
    )
    _ax1.scatter([0], [0], color=CYBER_CYAN, s=150, marker="+", linewidths=3, zorder=5)
    _ax1.set_xlim(-_plot_range, _plot_range)
    _ax1.set_ylim(-_plot_range, _plot_range)
    _ax1.set_aspect("equal")
    _ax1.set_xlabel("θ₁", fontsize=14)
    _ax1.set_ylabel("θ₂", fontsize=14)
    _ax1.set_title("PRIOR", fontsize=16, fontweight="bold", color=CYBER_CYAN)
    _ax1.axhline(y=0, color=CYBER_GRID, linewidth=0.5)
    _ax1.axvline(x=0, color=CYBER_GRID, linewidth=0.5)
    _ax1.grid(True, color=CYBER_GRID, alpha=0.3)
    _ax1.text(
        0.05,
        0.95,
        f"θ ~ N(0, {_sigma:.1f}²·I)",
        transform=_ax1.transAxes,
        fontsize=12,
        va="top",
        family="monospace",
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round", fc=CYBER_BG, ec=CYBER_CYAN, alpha=0.9, pad=0.5),
    )

    # Panel 2: LIKELIHOOD
    _ax2 = _axes[1]
    style_ax(_ax2)
    _constraint_val = _r_obs - _mu_obs
    if abs(_L_obs[1]) > 1e-6:
        _theta1 = np.linspace(-_plot_range, _plot_range, 200)
        _theta2 = (_constraint_val - _L_obs[0] * _theta1) / _L_obs[1]
        _valid = np.abs(_theta2) < _plot_range * 1.5
        _ax2.plot(_theta1[_valid], _theta2[_valid], color=CYBER_YELLOW, linewidth=3.5)
    elif abs(_L_obs[0]) > 1e-6:
        _ax2.axvline(x=_constraint_val / _L_obs[0], color=CYBER_YELLOW, linewidth=3.5)
    _arrow_scale = 2.0
    _ax2.arrow(
        0,
        0,
        _L_obs[0] * _arrow_scale,
        _L_obs[1] * _arrow_scale,
        head_width=0.15,
        head_length=0.1,
        fc=CYBER_ORANGE,
        ec=CYBER_ORANGE,
        lw=2.5,
        zorder=5,
    )
    _ax2.text(
        _L_obs[0] * _arrow_scale * 1.15 + 0.1,
        _L_obs[1] * _arrow_scale * 1.15,
        "Λ",
        fontsize=16,
        color=CYBER_ORANGE,
        fontweight="bold",
    )
    _ax2.set_xlim(-_plot_range, _plot_range)
    _ax2.set_ylim(-_plot_range, _plot_range)
    _ax2.set_aspect("equal")
    _ax2.set_xlabel("θ₁", fontsize=14)
    _ax2.set_ylabel("θ₂", fontsize=14)
    _ax2.set_title("LIKELIHOOD", fontsize=16, fontweight="bold", color=CYBER_YELLOW)
    _ax2.axhline(y=0, color=CYBER_GRID, linewidth=0.5)
    _ax2.axvline(x=0, color=CYBER_GRID, linewidth=0.5)
    _ax2.grid(True, color=CYBER_GRID, alpha=0.3)
    _ax2.text(
        0.05,
        0.95,
        f"r = {_r_obs:.0f}, μ = {_mu_obs:.2f}\nΛ'θ + μ = r",
        transform=_ax2.transAxes,
        fontsize=12,
        va="top",
        family="monospace",
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round", fc=CYBER_BG, ec=CYBER_YELLOW, alpha=0.9, pad=0.5),
    )

    # Panel 3: POSTERIOR
    _ax3 = _axes[2]
    style_ax(_ax3)
    confidence_ellipse(
        bay_post_mean,
        bay_post_cov,
        _ax3,
        n_std=1,
        facecolor="none",
        edgecolor=CYBER_MAGENTA,
        linewidth=2.5,
    )
    confidence_ellipse(
        bay_post_mean,
        bay_post_cov,
        _ax3,
        n_std=2,
        facecolor="none",
        edgecolor=CYBER_MAGENTA,
        linewidth=1.5,
        linestyle="--",
        alpha=0.6,
    )
    confidence_ellipse(
        _prior_mean,
        _prior_cov,
        _ax3,
        n_std=1,
        facecolor="none",
        edgecolor=CYBER_CYAN,
        linewidth=1,
        linestyle=":",
        alpha=0.4,
    )
    _ax3.scatter([0], [0], color=CYBER_CYAN, s=80, marker="+", linewidths=2, alpha=0.5)
    _ax3.scatter(
        [bay_post_mean[0]],
        [bay_post_mean[1]],
        color=CYBER_MAGENTA,
        s=120,
        marker="X",
        linewidths=2,
        zorder=5,
    )
    _ax3.annotate(
        "",
        xy=(bay_post_mean[0], bay_post_mean[1]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#FF1493", lw=2.5),
    )
    _ax3.set_xlim(-_plot_range, _plot_range)
    _ax3.set_ylim(-_plot_range, _plot_range)
    _ax3.set_aspect("equal")
    _ax3.set_xlabel("θ₁", fontsize=14)
    _ax3.set_ylabel("θ₂", fontsize=14)
    _ax3.set_title("POSTERIOR", fontsize=16, fontweight="bold", color=CYBER_MAGENTA)
    _ax3.axhline(y=0, color=CYBER_GRID, linewidth=0.5)
    _ax3.axvline(x=0, color=CYBER_GRID, linewidth=0.5)
    _ax3.grid(True, color=CYBER_GRID, alpha=0.3)
    _ax3.text(
        0.05,
        0.95,
        "Prior × Likelihood\n= Posterior",
        transform=_ax3.transAxes,
        fontsize=12,
        va="top",
        family="monospace",
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round", fc=CYBER_BG, ec=CYBER_MAGENTA, alpha=0.9, pad=0.5),
    )
    plt.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.md(f"""
### Numerical Walkthrough

| | Value |
|---|---|
| **Observed** | Q{_obs_q} [{domain_labels[_obs_q]}] = {_r_obs:.0f} |
| **Population mean** (μ_{_obs_q}) | {_mu_obs:.3f} |
| **Centered observation** (r - μ) | {_r_obs - _mu_obs:.3f} |
| **Loading vector** (Λ_{_obs_q}) | [{_L_obs[0]:.3f}, {_L_obs[1]:.3f}] |
| **Kalman gain** (K) | [{_K[0]:.3f}, {_K[1]:.3f}] |
| **Posterior mean** (μ_post) | [{bay_post_mean[0]:.3f}, {bay_post_mean[1]:.3f}] |
| **Posterior std** (diagonal) | [{np.sqrt(bay_post_cov[0,0]):.3f}, {np.sqrt(bay_post_cov[1,1]):.3f}] |

The posterior (magenta) is **compressed along the loading direction** — we're now certain about the partner's position along that axis, but still uncertain in the perpendicular direction. This is why one observation doesn't fully determine θ.
        """),
        ]
    )
    return (
        bay_post_mean,
        bay_post_cov,
    )


# =============================================================================
# SECTION 7: FROM θ TO PREDICTIONS
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 7: From θ to Predictions

Now we have a posterior distribution over θ. How does this translate to predictions about the partner's responses on all 35 questions?

For each question $q$, the **predictive distribution** is:

$$\\hat{r}_q \\sim \\mathcal{N}\\left(\\Lambda_q^\\top \\mu_{\\text{post}} + \\mu_q, \\;\\; \\Lambda_q^\\top \\Sigma_{\\text{post}} \\Lambda_q\\right)$$

- **Predicted mean**: $\\Lambda_q^\\top \\mu_{\\text{post}} + \\mu_q$ — our best guess of what they'd say
- **Predicted variance**: $\\Lambda_q^\\top \\Sigma_{\\text{post}} \\Lambda_q$ — how uncertain we are

Questions that **load on similar factors** as the observed question get more transfer (lower uncertainty). This is where domain-specific gradients come from — politics questions load on the same factors, so observing one politics response updates predictions for other politics questions more than for religion questions.
    """)
    return


# Visualization cell for Section 7
@app.cell
def __(
    mo,
    obs_question,
    bay_post_mean,
    bay_post_cov,
    all_loadings,
    means,
    domain_labels,
    DOMAIN_COLORS,
    DOMAIN_RANGES,
    CYBER_BG,
    CYBER_GRID,
    CYBER_CYAN,
    CYBER_MAGENTA,
    style_ax,
    np,
    plt,
):
    _obs_q = obs_question.value
    _loadings_2 = all_loadings[:, :2]

    # Predicted means and stds for all 35 questions
    _pred_means = _loadings_2 @ bay_post_mean + means
    _pred_vars = np.array(
        [_loadings_2[i] @ bay_post_cov @ _loadings_2[i] for i in range(35)]
    )
    _pred_vars = np.maximum(_pred_vars, 1e-10)
    _pred_stds = np.sqrt(_pred_vars)

    _fig, _axes = plt.subplots(2, 1, figsize=(16, 8), facecolor=CYBER_BG)
    _x = np.arange(35)
    _colors = [DOMAIN_COLORS[d] for d in domain_labels]

    # Panel 1: Predicted response means ± std
    _ax1 = _axes[0]
    style_ax(_ax1)
    _ax1.bar(
        _x,
        _pred_means,
        yerr=_pred_stds,
        color=_colors,
        alpha=0.8,
        capsize=2,
        error_kw={"linewidth": 1.5, "alpha": 0.8, "color": "white"},
        edgecolor="white",
        linewidth=0.5,
    )
    _ax1.bar(
        [_obs_q],
        [_pred_means[_obs_q]],
        color=CYBER_MAGENTA,
        edgecolor="white",
        linewidth=2,
    )
    _ax1.scatter(
        _x,
        means,
        color="white",
        s=15,
        marker="_",
        linewidths=2,
        label="Population mean (μ)",
        zorder=5,
    )
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax1.axvline(x=_start - 0.5, color=CYBER_GRID, alpha=0.8, linewidth=1)
        _ax1.text(
            (_start + _end) / 2 - 0.5,
            0.3,
            _domain[:3].upper(),
            ha="center",
            fontsize=7,
            color="white",
            alpha=0.7,
        )
    _ax1.set_xlim(-0.5, 34.5)
    _ax1.set_ylim(0, 7)
    _ax1.set_ylabel("Predicted Response", fontsize=12)
    _ax1.set_title(
        "Predicted Partner Responses: E[r_q] = Λ_q'·θ_post + μ_q",
        fontsize=14,
        fontweight="bold",
        color=CYBER_MAGENTA,
    )
    _ax1.legend(
        fontsize=9, facecolor=CYBER_BG, edgecolor=CYBER_GRID, labelcolor="white"
    )

    # Panel 2: Uncertainty
    _ax2 = _axes[1]
    style_ax(_ax2)
    _ax2.bar(_x, _pred_stds, color=_colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    _ax2.bar(
        [_obs_q],
        [_pred_stds[_obs_q]],
        color=CYBER_MAGENTA,
        edgecolor="white",
        linewidth=2,
    )
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax2.axvline(x=_start - 0.5, color=CYBER_GRID, alpha=0.8, linewidth=1)
        _ax2.text(
            (_start + _end) / 2 - 0.5,
            0.02,
            _domain[:3].upper(),
            ha="center",
            fontsize=7,
            color="white",
            alpha=0.7,
        )
    _ax2.set_xlim(-0.5, 34.5)
    _ax2.set_xlabel("Question", fontsize=12)
    _ax2.set_ylabel("Prediction Uncertainty (std)", fontsize=12)
    _ax2.set_title(
        "Predictive Uncertainty: √(Λ_q'·Σ_post·Λ_q)",
        fontsize=14,
        fontweight="bold",
        color=CYBER_CYAN,
    )
    plt.tight_layout()

    # Domain-level summary table
    _domain_rows = []
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        _dmean = _pred_means[_start:_end].mean()
        _dstd = _pred_stds[_start:_end].mean()
        _is_obs = "observed" if _start <= _obs_q < _end else ""
        _domain_rows.append(f"| {_domain} | {_dmean:.2f} | {_dstd:.3f} | {_is_obs} |")

    mo.vstack(
        [
            _fig,
            mo.md(f"""
### Domain Summary

| Domain | Mean E[r] | Mean Uncertainty | |
|--------|-----------|------------------|-|
{chr(10).join(_domain_rows)}

**Top**: Predicted responses (bars) with uncertainty (error bars). The observed question (magenta) is pinned exactly. **Bottom**: Prediction uncertainty — questions loading on similar factors as the observed question have lower uncertainty.

*(Using k=2 for visualization. The full model with k=5 produces richer domain-specific gradients.)*
        """),
        ]
    )
    return


# =============================================================================
# SECTION 8: P(MATCH) — THE FINAL OUTPUT
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 8: P(match) — The Final Output

The model's final output is a **probability of agreement** for each question:

$$P(\\text{match}_q) = P(|r_{\\text{partner}} - r_{\\text{self}}| \\leq \\tau)$$

This uses the **full predictive distribution** — both the mean AND the variance matter. A question might have a predicted mean close to your response, but if the uncertainty is high, P(match) will be lower.

Using the Gaussian predictive distribution:

$$P(\\text{match}_q) = \\Phi\\left(\\frac{r_{\\text{self}} + \\tau - \\hat{\\mu}_q}{\\hat{\\sigma}_q}\\right) - \\Phi\\left(\\frac{r_{\\text{self}} - \\tau - \\hat{\\mu}_q}{\\hat{\\sigma}_q}\\right)$$

Finally, a **lapse rate** $\\varepsilon$ accounts for random guessing:

$$P_{\\text{final}} = (1 - \\varepsilon) \\cdot P(\\text{match}) + \\varepsilon \\cdot 0.5$$

This shrinks all predictions toward 0.5 (chance). The lapse rate captures attentional lapses or noisy responding.
    """)
    return


# Widget cell for Section 8
@app.cell
def __(mo):
    match_threshold = mo.ui.slider(
        start=0.5,
        stop=4.0,
        step=0.25,
        value=2.0,
        label="τ (match threshold)",
        show_value=True,
    )
    epsilon = mo.ui.slider(
        start=0.0,
        stop=0.5,
        step=0.05,
        value=0.1,
        label="ε (lapse rate)",
        show_value=True,
    )
    return (
        match_threshold,
        epsilon,
    )


# Visualization cell for Section 8
@app.cell
def __(
    mo,
    obs_question,
    match_threshold,
    epsilon,
    bay_post_mean,
    bay_post_cov,
    all_loadings,
    means,
    domain_labels,
    DOMAIN_COLORS,
    DOMAIN_RANGES,
    CYBER_BG,
    CYBER_GRID,
    CYBER_GREEN,
    style_ax,
    np,
    plt,
    norm,
):
    _obs_q = obs_question.value
    _tau = float(match_threshold.value)
    _eps = float(epsilon.value)
    _loadings_2 = all_loadings[:, :2]

    # Use population mean as r_self
    _r_self = means.copy()

    # Predicted means and stds
    _pred_means = _loadings_2 @ bay_post_mean + means
    _pred_vars = np.array(
        [_loadings_2[i] @ bay_post_cov @ _loadings_2[i] for i in range(35)]
    )
    _pred_vars = np.maximum(_pred_vars, 1e-10)
    _pred_stds = np.sqrt(_pred_vars)

    # P(match)
    _upper = (_r_self + _tau - _pred_means) / _pred_stds
    _lower = (_r_self - _tau - _pred_means) / _pred_stds
    _p_match_raw = np.clip(norm.cdf(_upper) - norm.cdf(_lower), 0.0, 1.0)
    _p_match = (1 - _eps) * _p_match_raw + _eps * 0.5

    _fig, _ax = plt.subplots(figsize=(16, 5), facecolor=CYBER_BG)
    style_ax(_ax)
    _x = np.arange(35)
    _colors = [DOMAIN_COLORS[d] for d in domain_labels]
    _ax.bar(_x, _p_match, color=_colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    _ax.bar(
        [_obs_q], [_p_match[_obs_q]], color=CYBER_GREEN, edgecolor="white", linewidth=2
    )
    _ax.axhline(
        y=0.5,
        color="white",
        linewidth=1,
        linestyle="--",
        alpha=0.3,
        label="Chance (0.5)",
    )
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax.axvline(x=_start - 0.5, color=CYBER_GRID, alpha=0.8, linewidth=1)
        _ax.text(
            (_start + _end) / 2 - 0.5,
            0.03,
            _domain[:3].upper(),
            ha="center",
            fontsize=7,
            color="white",
            alpha=0.7,
        )
    _ax.set_xlim(-0.5, 34.5)
    _ax.set_ylim(0, 1.05)
    _ax.set_xlabel("Question", fontsize=12)
    _ax.set_ylabel("P(match)", fontsize=12)
    _ax.set_title(
        f"P(match) = P(|r_partner - r_self| ≤ τ={_tau:.1f}), ε={_eps:.2f}",
        fontsize=14,
        fontweight="bold",
        color=CYBER_GREEN,
    )
    _ax.legend(fontsize=9, facecolor=CYBER_BG, edgecolor=CYBER_GRID, labelcolor="white")
    plt.tight_layout()

    # Domain averages
    _domain_rows = []
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        _dmean = _p_match[_start:_end].mean()
        _is_obs = "observed" if _start <= _obs_q < _end else ""
        _domain_rows.append(f"| {_domain} | {_dmean:.3f} | {_is_obs} |")

    mo.vstack(
        [
            mo.hstack([match_threshold, epsilon], justify="center"),
            _fig,
            mo.md(f"""
### Domain Averages

| Domain | P(match) | |
|--------|----------|-|
{chr(10).join(_domain_rows)}

Try adjusting τ and ε — the *magnitude* changes but the *relative pattern across domains* stays the same. Neither parameter can create domain-specific gradients.

*(Using k=2 and r_self = population mean.)*
        """),
        ]
    )
    return


# =============================================================================
# SECTION 9: WHY THE GRADIENT?
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 9: Why the Gradient? — What the Parameters Can and Can't Do

## The Evaluation Metric

The model is evaluated by the **generalization gradient** — a difference-in-differences:

$$\\text{Gradient} = \\underbrace{(P_{\\text{same}}^{\\text{shared}} - P_{\\text{same}}^{\\text{opposing}})}_{\\text{same-domain effect}} - \\underbrace{(P_{\\text{diff}}^{\\text{shared}} - P_{\\text{diff}}^{\\text{opposing}})}_{\\text{different-domain effect}}$$

A positive gradient means: the shared/opposing difference is **larger for same-domain questions** than different-domain questions — the signature of structured generalization.

## Why Not Just Accuracy?

The model's accuracy (~55.7%) is modest — predicting 34 questions from 1 observation is inherently hard. The model's claim isn't "I can predict individual responses." It's "the **pattern** of how predictions transfer across domains matches how humans do it."

## What Creates the Gradient?

σ_prior, τ, and ε are **uniform transformations** — they apply to every question equally. The gradient (difference-in-differences) cancels out uniform shifts. These parameters control the *magnitude* of the gradient, but cannot create *domain-specificity* from nothing.

**Domain-specificity comes entirely from the factor loadings Λ.** The visualization below demonstrates this: when you scramble the loadings, the gradient disappears — even with the same parameters.
    """)
    return


# Widget cell for Section 9
@app.cell
def __(mo):
    scramble_toggle = mo.ui.dropdown(
        options={
            "Real factor loadings (structured)": False,
            "Scrambled loadings (random permutation)": True,
        },
        value="Real factor loadings (structured)",
        label="Factor loadings",
    )
    return (scramble_toggle,)


# Visualization cell for Section 9
@app.cell
def __(
    mo,
    scramble_toggle,
    obs_question,
    obs_response,
    prior_sigma,
    all_loadings,
    means,
    domain_labels,
    DOMAIN_COLORS,
    DOMAIN_RANGES,
    CYBER_BG,
    CYBER_GRID,
    CYBER_CYAN,
    CYBER_YELLOW,
    CYBER_GREEN,
    style_ax,
    np,
    plt,
    norm,
):
    _obs_q = obs_question.value
    _r_obs = float(obs_response.value)
    _sigma = float(prior_sigma.value)
    _tau = 2.0
    _eps = 0.1

    _loadings_2 = all_loadings[:, :2]

    # Optionally scramble
    if scramble_toggle.value:
        _rng = np.random.default_rng(42)
        _perm = _rng.permutation(35)
        _loadings_2 = _loadings_2[_perm, :]

    _L_obs = _loadings_2[_obs_q]
    _mu_obs = means[_obs_q]

    # Bayesian update
    _prior_cov = _sigma**2 * np.eye(2)
    _L_cov_L = _L_obs @ _prior_cov @ _L_obs
    _K = _prior_cov @ _L_obs / (_L_cov_L + 1e-10)
    _post_mean = _K * (_r_obs - _mu_obs)
    _post_cov = _prior_cov - np.outer(_K, _L_obs @ _prior_cov)

    # Predictions
    _pred_means = _loadings_2 @ _post_mean + means
    _pred_vars = np.array(
        [_loadings_2[i] @ _post_cov @ _loadings_2[i] for i in range(35)]
    )
    _pred_vars = np.maximum(_pred_vars, 1e-10)
    _pred_stds = np.sqrt(_pred_vars)

    _r_self = means.copy()
    _upper = (_r_self + _tau - _pred_means) / _pred_stds
    _lower = (_r_self - _tau - _pred_means) / _pred_stds
    _p_match = np.clip(norm.cdf(_upper) - norm.cdf(_lower), 0.0, 1.0)
    _p_match = (1 - _eps) * _p_match + _eps * 0.5

    # Compute domain averages for gradient
    _domain_avgs = {}
    for _domain, (_start, _end) in DOMAIN_RANGES.items():
        _domain_avgs[_domain] = _p_match[_start:_end].mean()
    _obs_domain = domain_labels[_obs_q]
    _same = np.mean([v for d, v in _domain_avgs.items() if d == _obs_domain])
    _diff = np.mean([v for d, v in _domain_avgs.items() if d != _obs_domain])
    _gradient_proxy = _same - _diff

    _fig, _axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=CYBER_BG)

    # Panel 1: P(match) bar chart
    _ax1 = _axes[0]
    style_ax(_ax1)
    _x = np.arange(35)
    _colors = [DOMAIN_COLORS[d] for d in domain_labels]
    _ax1.bar(_x, _p_match, color=_colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    _ax1.bar(
        [_obs_q], [_p_match[_obs_q]], color=CYBER_GREEN, edgecolor="white", linewidth=2
    )
    _ax1.axhline(y=0.5, color="white", linewidth=1, linestyle="--", alpha=0.3)
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax1.axvline(x=_start - 0.5, color=CYBER_GRID, alpha=0.8, linewidth=1)
        _ax1.text(
            (_start + _end) / 2 - 0.5,
            0.03,
            _domain[:3].upper(),
            ha="center",
            fontsize=7,
            color="white",
            alpha=0.7,
        )
    _ax1.set_xlim(-0.5, 34.5)
    _ax1.set_ylim(0, 1.05)
    _ax1.set_xlabel("Question", fontsize=12)
    _ax1.set_ylabel("P(match)", fontsize=12)
    _title_label = "SCRAMBLED Loadings" if scramble_toggle.value else "Real Loadings"
    _title_color = CYBER_YELLOW if scramble_toggle.value else CYBER_GREEN
    _ax1.set_title(
        f"P(match) with {_title_label}",
        fontsize=14,
        fontweight="bold",
        color=_title_color,
    )

    # Panel 2: Domain averages
    _ax2 = _axes[1]
    style_ax(_ax2)
    _sorted_domains = sorted(DOMAIN_RANGES.keys(), key=lambda d: DOMAIN_RANGES[d][0])
    _domain_vals = [_domain_avgs[d] for d in _sorted_domains]
    _domain_cols = [DOMAIN_COLORS[d] for d in _sorted_domains]
    _bars = _ax2.bar(
        range(len(_sorted_domains)),
        _domain_vals,
        color=_domain_cols,
        alpha=0.8,
        edgecolor="white",
        linewidth=1.5,
    )
    for _i, _d in enumerate(_sorted_domains):
        if _d == _obs_domain:
            _ax2.bar(
                [_i],
                [_domain_avgs[_d]],
                color=CYBER_GREEN,
                edgecolor="white",
                linewidth=2,
            )
    _ax2.set_xticks(range(len(_sorted_domains)))
    _ax2.set_xticklabels([d[:4].upper() for d in _sorted_domains], fontsize=9)
    _ax2.set_ylabel("Mean P(match)", fontsize=12)
    _ax2.set_ylim(0, 1.05)
    _gradient_text = f"Same-domain - Diff-domain = {_gradient_proxy:+.3f}"
    _ax2.set_title(
        f"Domain Averages ({_gradient_text})",
        fontsize=14,
        fontweight="bold",
        color=_title_color,
    )
    _ax2.axhline(
        y=np.mean(_domain_vals),
        color="white",
        linewidth=1,
        linestyle="--",
        alpha=0.3,
        label="Overall mean",
    )
    _ax2.legend(
        fontsize=9, facecolor=CYBER_BG, edgecolor=CYBER_GRID, labelcolor="white"
    )
    plt.tight_layout()

    _scramble_note = (
        (
            "**Scrambled**: The loadings have been randomly permuted, destroying the domain structure. "
            "Notice how the domain averages are nearly **flat** — no domain-specific gradient, "
            "even with the same σ_prior, τ, and ε. **The parameters can't create structure that isn't in Λ.**"
        )
        if scramble_toggle.value
        else (
            "**Real loadings**: The domain structure in Λ creates a gradient — the observed domain "
            f"({_obs_domain}) has {'higher' if _gradient_proxy > 0 else 'different'} P(match) than other domains. "
            "Switch to scrambled loadings to see this structure disappear."
        )
    )

    mo.vstack(
        [
            mo.hstack([scramble_toggle], justify="center"),
            _fig,
            mo.md(f"""
{_scramble_note}

### Summary

| Component | Role | Creates gradients? |
|-----------|------|--------------------|
| Factor loadings (Λ) | Population map of belief structure | **Yes** — the sole source |
| Prior (θ ~ N(0, σ²·I)) | Default assumption | No — uniform |
| Bayesian update | Inference engine | No — but propagates Λ's structure |
| Match threshold (τ) | Define agreement | No — uniform |
| Lapse rate (ε) | Random guessing rate | No — uniform |

The factor analysis gives the map. Bayesian inference navigates the map. The parameters tune the scale. The gradient tests the map's structure.
        """),
        ]
    )
    return


# =============================================================================
# SECTION 10: QUICK REFERENCE
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Quick Reference

## The Full Pipeline

```
Population responses (N × 35)
    → Correlation matrix (35 × 35)
    → Eigendecomposition → Factor loadings Λ (35 × k) + means μ (35)
    → Prior: θ ~ N(0, σ²·I)
    → Observe one response → Bayesian update → Posterior: θ|r ~ N(μ_post, Σ_post)
    → Predictive distribution: r_q ~ N(Λ_q'μ_post + μ_q, Λ_q'Σ_post Λ_q)
    → P(match) = P(|r_partner - r_self| ≤ τ)
    → Apply lapse rate: (1-ε)·P(match) + ε·0.5
```

## Key Equations

| Step | Equation |
|------|----------|
| Generative model | $r = \\Lambda\\theta + \\mu$ |
| Prior | $\\theta \\sim \\mathcal{N}(0, \\sigma^2 I)$ |
| Kalman gain | $K = \\Sigma \\Lambda_q / (\\Lambda_q^\\top \\Sigma \\Lambda_q)$ |
| Posterior mean | $\\mu_{\\text{post}} = K \\cdot (r_{\\text{obs}} - \\mu_q)$ |
| Posterior covariance | $\\Sigma_{\\text{post}} = \\Sigma - K \\Lambda_q^\\top \\Sigma$ |
| Predicted mean | $\\hat{\\mu}_q = \\Lambda_q^\\top \\mu_{\\text{post}} + \\mu_q$ |
| Predicted variance | $\\hat{\\sigma}^2_q = \\Lambda_q^\\top \\Sigma_{\\text{post}} \\Lambda_q$ |
| P(match) | $\\Phi\\left(\\frac{r_{\\text{self}} + \\tau - \\hat{\\mu}_q}{\\hat{\\sigma}_q}\\right) - \\Phi\\left(\\frac{r_{\\text{self}} - \\tau - \\hat{\\mu}_q}{\\hat{\\sigma}_q}\\right)$ |
| With lapse | $(1-\\varepsilon) \\cdot P(\\text{match}) + \\varepsilon \\cdot 0.5$ |

## Parameters

| Parameter | Symbol | Fitted Value | Role |
|-----------|--------|-------------|------|
| Factors | $k$ | 5 | Dimensionality of latent space |
| Prior std | $\\sigma_{\\text{prior}}$ | 1.5 | Uncertainty before observing |
| Match threshold | $\\tau$ | 2.0 | Definition of "agreement" |
| Lapse rate | $\\varepsilon$ | 0.1 | Random guessing rate |
| Obs. noise | $\\sigma_{\\text{obs}}$ | 0.0 | 0 = exact observation (no-chat) |
| Mixture weight | $\\lambda$ | 0.0 | 0 = pure Bayesian (best fit) |
    """)
    return


if __name__ == "__main__":
    app.run()
