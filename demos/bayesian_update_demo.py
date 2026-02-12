"""
Interactive demo comparing Bayesian Factor Model vs Egocentric Projection.

Shows how each model updates beliefs about a partner after observing a single response,
then compares their predictions side-by-side.

Run with: uv run marimo run demos/bayesian_update_demo.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md("""
# Commonality Inference Models: Interactive Demo

This notebook explores two computational models of how people infer commonality with a conversation partner after observing a single response.

| Model | Key Idea | Source of Structure |
|-------|----------|---------------------|
| **Bayesian Factor** (Section 1) | Partner's beliefs live in a low-dimensional factor space | Population covariance |
| **Egocentric Projection** (Section 2) | Partner is "like me" → project my beliefs | Your own responses |

To formally compare these accounts, we nest both within a **mixture framework** (Sections 3-4) controlled by weight $\\lambda \\in [0,1]$:

$$P(\\text{commonality}) = (1 - \\lambda) \\cdot P_{\\text{Bayesian}} + \\lambda \\cdot P_{\\text{Egocentric}}$$

where $\\lambda = 0$ yields pure Bayesian inference and $\\lambda = 1$ yields pure egocentric projection.

**Empirical result**: The Bayesian model ($\\lambda=0$) captures **95%** of the human generalization gradient; the egocentric model ($\\lambda=1$) captures only **12%**.

---

**How to use this notebook**: Explore each model independently in Sections 1-2, compare predictions side-by-side in Section 3, and see how mixture weight affects human fit in Section 4.
    """)
    return


@app.cell
def __():
    import sys
    from pathlib import Path

    # Add project root to path
    _project_root = Path(__file__).parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    from scipy.special import erf

    from models.model import (
        load_factor_loadings,
        load_question_means,
        load_responses,
        DOMAIN_RANGES,
    )

    return (
        np,
        plt,
        Ellipse,
        transforms,
        Path,
        erf,
        load_factor_loadings,
        load_question_means,
        load_responses,
        DOMAIN_RANGES,
    )


@app.cell
def __(load_factor_loadings, load_question_means, load_responses, DOMAIN_RANGES, Path):
    import pandas as pd

    # Load all factor loadings (up to k=10 for demo)
    all_loadings = load_factor_loadings(k=10)
    means = load_question_means()

    # Load question text from raw data
    _data_dir = Path(__file__).parent.parent / "data"
    _df = pd.read_csv(_data_dir / "responses.csv", low_memory=False)

    # Get unique question text for each question ID (1-35)
    _q_map = _df.groupby("question")["preChatQuestion"].first().to_dict()
    questions = [_q_map.get(i + 1, f"Question {i+1}") for i in range(35)]

    # Create domain labels
    domain_labels = []
    for _i in range(35):
        for _domain, (_start, _end) in DOMAIN_RANGES.items():
            if _start <= _i < _end:
                domain_labels.append(_domain)
                break

    # Load sample participant responses for egocentric model demo
    _responses_df = load_responses()
    sample_participants = {
        "Population mean": means.copy(),
        "Participant A": _responses_df.iloc[0].values.copy(),
        "Participant B": _responses_df.iloc[50].values.copy(),
        "Participant C": _responses_df.iloc[100].values.copy(),
    }

    return all_loadings, means, questions, domain_labels, sample_participants


# =============================================================================
# SECTION 1: BAYESIAN FACTOR MODEL
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 1: Bayesian Factor Model

The Bayesian model represents a partner's beliefs as a latent position $\\theta \\in \\mathbb{R}^k$ in **factor space**—a low-dimensional space capturing how beliefs covary *in the population*.

### Key Equations

**Prior**: $\\theta \\sim \\mathcal{N}(0, \\sigma^2_{\\text{prior}} \\cdot I_k)$

**Generative model**: $r_q \\mid \\theta \\sim \\mathcal{N}(\\Lambda_q^\\top \\theta + \\mu_q, \\sigma^2_{\\text{obs}})$

**Posterior**: $P(\\theta \\mid r_{\\text{obs}}) \\propto P(r_{\\text{obs}} \\mid \\theta) \\cdot P(\\theta)$ (closed-form Gaussian)

**Predictive**: $\\hat{r}_q \\sim \\mathcal{N}(\\Lambda_q^\\top \\mu_{\\text{post}} + \\mu_q, \\; \\Lambda_q^\\top \\Sigma_{\\text{post}} \\Lambda_q)$

**Number of factors**: Parallel analysis suggests **k=5** factors (explaining 36% of variance). These correspond roughly to political, religious, moral, identity, and preference dimensions.

| Parameter | Symbol | Psychological Meaning |
|-----------|--------|----------------------|
| Latent position | $\\theta$ | Partner's position in belief space (unobserved) |
| Factor loadings | $\\Lambda_q$ | How question $q$ relates to latent factors |
| Population mean | $\\mu_q$ | Average response to question $q$ |
| Prior variance | $\\sigma^2_{\\text{prior}}$ | Uncertainty about partner before observing |
| Observation noise | $\\sigma^2_{\\text{obs}}$ | Uncertainty in the observed response |

**Key insight**: Gradients emerge from *population covariance structure*. Political questions load on the same factors, so observing one updates predictions for others.
    """)
    return


@app.cell
def __(mo, questions, domain_labels):
    # Bayesian model controls
    _question_options = {
        f"Q{i}: [{domain_labels[i][:4].upper()}] {q[:45]}{'...' if len(q) > 45 else ''}": i
        for i, q in enumerate(questions)
    }

    bayes_question = mo.ui.dropdown(
        options=_question_options,
        value=list(_question_options.keys())[22],  # Start with a politics question
        label="Observed Question",
    )

    bayes_response = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=5,
        label="Partner's Response",
        show_value=True,
    )

    bayes_k = mo.ui.slider(
        start=1,
        stop=6,
        step=1,
        value=5,
        label="Factors (k)",
        show_value=True,
    )

    bayes_sigma = mo.ui.slider(
        start=0.5,
        stop=3.0,
        step=0.25,
        value=1.5,
        label="Prior σ",
        show_value=True,
    )

    return bayes_question, bayes_response, bayes_k, bayes_sigma


@app.cell
def __(
    bayes_question,
    bayes_response,
    bayes_k,
    bayes_sigma,
    all_loadings,
    means,
    np,
):
    # Compute Bayesian posterior
    b_obs_q = bayes_question.value
    b_r_obs = float(bayes_response.value)
    b_k = int(bayes_k.value)
    b_sigma_prior = float(bayes_sigma.value)

    # Get factor loadings for this k
    b_loadings_k = all_loadings[:, :b_k]
    b_L_obs = b_loadings_k[b_obs_q]
    b_mu_obs = means[b_obs_q]

    # Prior parameters
    b_prior_mean = np.zeros(b_k)
    b_prior_cov = b_sigma_prior**2 * np.eye(b_k)

    # Posterior update (delta observation)
    _r_centered = b_r_obs - b_mu_obs
    _L_cov_L = b_L_obs @ b_prior_cov @ b_L_obs
    _K = b_prior_cov @ b_L_obs / (_L_cov_L + 1e-10)
    _innovation = _r_centered - b_L_obs @ b_prior_mean

    b_posterior_mean = b_prior_mean + _K * _innovation
    b_posterior_cov = b_prior_cov - np.outer(_K, b_L_obs @ b_prior_cov)

    return (
        b_obs_q,
        b_r_obs,
        b_k,
        b_sigma_prior,
        b_loadings_k,
        b_L_obs,
        b_mu_obs,
        b_prior_mean,
        b_prior_cov,
        b_posterior_mean,
        b_posterior_cov,
    )


@app.cell
def __(
    mo,
    bayes_question,
    bayes_response,
    bayes_k,
    bayes_sigma,
    b_obs_q,
    b_r_obs,
    b_k,
    b_sigma_prior,
    b_L_obs,
    b_mu_obs,
    b_prior_mean,
    b_prior_cov,
    b_posterior_mean,
    b_posterior_cov,
    questions,
    domain_labels,
    np,
    plt,
    Ellipse,
    transforms,
):
    # Cyberpunk color palette
    _CYBER_CYAN = "#00FFFF"
    _CYBER_MAGENTA = "#FF00FF"
    _CYBER_YELLOW = "#FFFF00"
    _CYBER_ORANGE = "#FF6600"
    _CYBER_BG = "#0D0221"
    _CYBER_GRID = "#1a1a3a"

    def _confidence_ellipse(mean, cov, ax, n_std=2.0, facecolor="none", **kwargs):
        if len(mean) < 2:
            return None
        _mean_2d = mean[:2]
        _cov_2d = cov[:2, :2]
        _pearson = _cov_2d[0, 1] / np.sqrt(_cov_2d[0, 0] * _cov_2d[1, 1] + 1e-10)
        _ell_radius_x = np.sqrt(1 + _pearson)
        _ell_radius_y = np.sqrt(1 - _pearson)
        _ellipse = Ellipse(
            (0, 0),
            width=_ell_radius_x * 2,
            height=_ell_radius_y * 2,
            facecolor=facecolor,
            **kwargs,
        )
        _scale_x = np.sqrt(_cov_2d[0, 0]) * n_std
        _scale_y = np.sqrt(_cov_2d[1, 1]) * n_std
        _transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(_scale_x, _scale_y)
            .translate(_mean_2d[0], _mean_2d[1])
        )
        _ellipse.set_transform(_transf + ax.transData)
        return ax.add_patch(_ellipse)

    # Create 3-panel figure
    bayes_fig, _axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=_CYBER_BG)
    _plot_range = 4.0

    for _ax in _axes:
        _ax.set_facecolor(_CYBER_BG)
        _ax.tick_params(colors="white", labelsize=12)
        _ax.xaxis.label.set_color("white")
        _ax.yaxis.label.set_color("white")
        for _spine in _ax.spines.values():
            _spine.set_color(_CYBER_GRID)

    # Panel 1: PRIOR
    _ax1 = _axes[0]
    _ax1.set_title(
        "PRIOR\n(belief before observation)",
        fontsize=16,
        fontweight="bold",
        color=_CYBER_CYAN,
    )
    if b_k >= 2:
        _confidence_ellipse(
            b_prior_mean,
            b_prior_cov,
            _ax1,
            n_std=1,
            edgecolor=_CYBER_CYAN,
            linewidth=2.5,
        )
        _confidence_ellipse(
            b_prior_mean,
            b_prior_cov,
            _ax1,
            n_std=2,
            edgecolor=_CYBER_CYAN,
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
        )
        _ax1.scatter(
            [0], [0], color=_CYBER_CYAN, s=150, marker="+", linewidths=3, zorder=5
        )
        _ax1.set_xlabel("θ₁", fontsize=14)
        _ax1.set_ylabel("θ₂", fontsize=14)
    else:
        _x = np.linspace(-_plot_range, _plot_range, 200)
        _y = np.exp(-0.5 * (_x - b_prior_mean[0]) ** 2 / b_prior_cov[0, 0])
        _ax1.fill_between(_x, _y, alpha=0.4, color=_CYBER_CYAN)
        _ax1.plot(_x, _y, color=_CYBER_CYAN, linewidth=2.5)
        _ax1.set_xlabel("θ₁", fontsize=14)
        _ax1.set_ylabel("Density", fontsize=14)
    _ax1.set_xlim(-_plot_range, _plot_range)
    if b_k >= 2:
        _ax1.set_ylim(-_plot_range, _plot_range)
    _ax1.set_aspect("equal" if b_k >= 2 else "auto")
    _ax1.axhline(y=0, color=_CYBER_GRID, linestyle="-", alpha=0.5, linewidth=0.5)
    _ax1.axvline(x=0, color=_CYBER_GRID, linestyle="-", alpha=0.5, linewidth=0.5)
    _ax1.grid(True, color=_CYBER_GRID, alpha=0.3, linewidth=0.5)
    _ax1.text(
        0.05,
        0.95,
        f"θ ~ N(0, {b_sigma_prior:.1f}²·I)",
        transform=_ax1.transAxes,
        fontsize=13,
        va="top",
        family="monospace",
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round", fc=_CYBER_BG, ec=_CYBER_CYAN, alpha=0.9, pad=0.5),
    )

    # Panel 2: LIKELIHOOD
    _ax2 = _axes[1]
    _ax2.set_title(
        "LIKELIHOOD\n(constraint from observation)",
        fontsize=16,
        fontweight="bold",
        color=_CYBER_YELLOW,
    )
    _constraint_val = b_r_obs - b_mu_obs
    if b_k >= 2:
        if abs(b_L_obs[1]) > 1e-6:
            _theta1 = np.linspace(-_plot_range, _plot_range, 100)
            _theta2 = (_constraint_val - b_L_obs[0] * _theta1) / b_L_obs[1]
            _valid = np.abs(_theta2) < _plot_range * 1.5
            _ax2.plot(
                _theta1[_valid], _theta2[_valid], color=_CYBER_YELLOW, linewidth=3.5
            )
        elif abs(b_L_obs[0]) > 1e-6:
            _ax2.axvline(
                x=_constraint_val / b_L_obs[0], color=_CYBER_YELLOW, linewidth=3.5
            )
        _arrow_scale = 1.8
        _ax2.arrow(
            0,
            0,
            b_L_obs[0] * _arrow_scale,
            b_L_obs[1] * _arrow_scale,
            head_width=0.2,
            head_length=0.12,
            fc=_CYBER_ORANGE,
            ec=_CYBER_ORANGE,
            lw=2.5,
            zorder=5,
        )
        _ax2.text(
            b_L_obs[0] * _arrow_scale * 1.15,
            b_L_obs[1] * _arrow_scale * 1.15,
            "Λ",
            fontsize=16,
            color=_CYBER_ORANGE,
            fontweight="bold",
            ha="center",
        )
        _ax2.set_xlabel("θ₁", fontsize=14)
        _ax2.set_ylabel("θ₂", fontsize=14)
    else:
        _theta_constrained = _constraint_val / (b_L_obs[0] + 1e-10)
        _ax2.axvline(x=_theta_constrained, color=_CYBER_YELLOW, linewidth=3.5)
        _ax2.set_xlabel("θ₁", fontsize=14)
        _ax2.set_ylim(0, 1)
    _ax2.set_xlim(-_plot_range, _plot_range)
    if b_k >= 2:
        _ax2.set_ylim(-_plot_range, _plot_range)
    _ax2.set_aspect("equal" if b_k >= 2 else "auto")
    _ax2.axhline(y=0, color=_CYBER_GRID, linestyle="-", alpha=0.5, linewidth=0.5)
    _ax2.axvline(x=0, color=_CYBER_GRID, linestyle="-", alpha=0.5, linewidth=0.5)
    _ax2.grid(True, color=_CYBER_GRID, alpha=0.3, linewidth=0.5)
    _ax2.text(
        0.05,
        0.95,
        f"r = {b_r_obs:.0f}\nΛ'θ + μ = r",
        transform=_ax2.transAxes,
        fontsize=13,
        va="top",
        family="monospace",
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round", fc=_CYBER_BG, ec=_CYBER_YELLOW, alpha=0.9, pad=0.5),
    )

    # Panel 3: POSTERIOR
    _ax3 = _axes[2]
    _ax3.set_title(
        "POSTERIOR\n(updated belief)",
        fontsize=16,
        fontweight="bold",
        color=_CYBER_MAGENTA,
    )
    if b_k >= 2:
        _confidence_ellipse(
            b_posterior_mean,
            b_posterior_cov,
            _ax3,
            n_std=1,
            edgecolor=_CYBER_MAGENTA,
            linewidth=2.5,
        )
        _confidence_ellipse(
            b_posterior_mean,
            b_posterior_cov,
            _ax3,
            n_std=2,
            edgecolor=_CYBER_MAGENTA,
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
        )
        _confidence_ellipse(
            b_prior_mean,
            b_prior_cov,
            _ax3,
            n_std=1,
            edgecolor=_CYBER_CYAN,
            linewidth=1,
            linestyle=":",
            alpha=0.4,
        )
        _ax3.scatter(
            [0], [0], color=_CYBER_CYAN, s=80, marker="+", linewidths=2, alpha=0.5
        )
        _ax3.scatter(
            [b_posterior_mean[0]],
            [b_posterior_mean[1]],
            color=_CYBER_MAGENTA,
            s=120,
            marker="X",
            linewidths=2,
            zorder=5,
        )
        _ax3.annotate(
            "",
            xy=(b_posterior_mean[0], b_posterior_mean[1]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#FF1493", lw=2.5),
        )
        _ax3.set_xlabel("θ₁", fontsize=14)
        _ax3.set_ylabel("θ₂", fontsize=14)
    else:
        _x = np.linspace(-_plot_range, _plot_range, 200)
        _y_prior = np.exp(-0.5 * (_x - b_prior_mean[0]) ** 2 / b_prior_cov[0, 0])
        _y_post = np.exp(
            -0.5 * (_x - b_posterior_mean[0]) ** 2 / (b_posterior_cov[0, 0] + 1e-10)
        )
        _ax3.fill_between(_x, _y_prior, alpha=0.3, color=_CYBER_CYAN)
        _ax3.fill_between(_x, _y_post, alpha=0.5, color=_CYBER_MAGENTA)
        _ax3.plot(_x, _y_post, color=_CYBER_MAGENTA, linewidth=2.5)
        _ax3.set_xlabel("θ₁", fontsize=14)
        _ax3.set_ylabel("Density", fontsize=14)
    _ax3.set_xlim(-_plot_range, _plot_range)
    if b_k >= 2:
        _ax3.set_ylim(-_plot_range, _plot_range)
    _ax3.set_aspect("equal" if b_k >= 2 else "auto")
    _ax3.axhline(y=0, color=_CYBER_GRID, linestyle="-", alpha=0.5, linewidth=0.5)
    _ax3.axvline(x=0, color=_CYBER_GRID, linestyle="-", alpha=0.5, linewidth=0.5)
    _ax3.grid(True, color=_CYBER_GRID, alpha=0.3, linewidth=0.5)
    _ax3.text(
        0.05,
        0.95,
        "Prior × Likelihood\n= Posterior",
        transform=_ax3.transAxes,
        fontsize=13,
        va="top",
        family="monospace",
        color="white",
        fontweight="bold",
        bbox=dict(
            boxstyle="round", fc=_CYBER_BG, ec=_CYBER_MAGENTA, alpha=0.9, pad=0.5
        ),
    )

    plt.tight_layout()

    # Layout
    _controls = mo.vstack(
        [
            mo.md("### Bayesian Controls"),
            bayes_question,
            bayes_response,
            mo.hstack([bayes_k, bayes_sigma], justify="start"),
        ],
        gap=0.5,
    )

    mo.hstack([_controls, bayes_fig], widths=[1, 4], gap=1)
    return (bayes_fig,)


@app.cell
def __(mo, questions, domain_labels, b_obs_q):
    # Bayesian target question selector
    _target_options = {
        f"Q{i}: [{domain_labels[i][:4].upper()}] {q[:40]}...": i
        for i, q in enumerate(questions)
    }
    _default_idx = (b_obs_q + 15) % 35

    bayes_target = mo.ui.dropdown(
        options=_target_options,
        value=list(_target_options.keys())[_default_idx],
        label="Target Question",
    )
    return (bayes_target,)


@app.cell
def __(
    mo,
    bayes_target,
    b_obs_q,
    b_r_obs,
    b_loadings_k,
    means,
    b_posterior_mean,
    b_posterior_cov,
    questions,
    domain_labels,
    np,
    plt,
    erf,
):
    _CYBER_BG = "#0D0221"
    _CYBER_GRID = "#1a1a3a"
    _CYBER_CYAN = "#00FFFF"
    _CYBER_GREEN = "#39FF14"

    b_target_q = bayes_target.value
    _L_target = b_loadings_k[b_target_q]
    _mu_target = means[b_target_q]
    _pred_mean = _L_target @ b_posterior_mean + _mu_target
    _pred_var = _L_target @ b_posterior_cov @ _L_target
    _pred_std = np.sqrt(_pred_var + 1e-10)

    _responses = np.array([1, 2, 3, 4, 5])
    _probs = []
    for _r in _responses:
        _p_upper = 0.5 * (1 + erf((_r + 0.5 - _pred_mean) / (_pred_std * np.sqrt(2))))
        _p_lower = 0.5 * (1 + erf((_r - 0.5 - _pred_mean) / (_pred_std * np.sqrt(2))))
        _probs.append(_p_upper - _p_lower)
    _probs = np.array(_probs)
    _probs = _probs / (_probs.sum() + 1e-10)

    bayes_pairwise_fig, _ax = plt.subplots(figsize=(8, 4), facecolor=_CYBER_BG)
    _ax.set_facecolor(_CYBER_BG)
    _bars = _ax.bar(
        _responses, _probs, color=_CYBER_CYAN, alpha=0.8, edgecolor="white", linewidth=2
    )
    _ax.axvline(
        x=_pred_mean,
        color=_CYBER_GREEN,
        linewidth=3,
        linestyle="--",
        label=f"E[r]={_pred_mean:.2f}",
    )

    for _bar, _p in zip(_bars, _probs):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _bar.get_height() + 0.01,
            f"{_p:.0%}",
            ha="center",
            va="bottom",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

    _ax.set_xlabel("Partner's Response", fontsize=12, color="white")
    _ax.set_ylabel("Probability", fontsize=12, color="white")
    _ax.set_title(
        f"Bayesian: P(response on Q{b_target_q} | observed Q{b_obs_q}={b_r_obs:.0f})",
        fontsize=14,
        fontweight="bold",
        color=_CYBER_CYAN,
    )
    _ax.set_xticks([1, 2, 3, 4, 5])
    _ax.set_ylim(0, max(_probs) * 1.4)
    _ax.tick_params(colors="white", labelsize=11)
    _ax.legend(
        loc="upper right",
        fontsize=10,
        facecolor=_CYBER_BG,
        edgecolor=_CYBER_GRID,
        labelcolor="white",
    )
    _ax.grid(True, color=_CYBER_GRID, alpha=0.3, axis="y")
    for _spine in _ax.spines.values():
        _spine.set_color(_CYBER_GRID)
    plt.tight_layout()

    _same_domain = domain_labels[b_obs_q] == domain_labels[b_target_q]
    _domain_note = (
        "**Same domain** → stronger transfer"
        if _same_domain
        else "**Different domain** → weaker transfer"
    )

    mo.vstack(
        [
            mo.md(f"""
### Bayesian Pairwise Prediction

**Observed**: Q{b_obs_q} ({domain_labels[b_obs_q]}) → {b_r_obs:.0f} | **Predicting**: Q{b_target_q} ({domain_labels[b_target_q]})

{_domain_note} | Predicted mean: **{_pred_mean:.2f}** ± {_pred_std:.2f}
"""),
            bayes_target,
            bayes_pairwise_fig,
        ]
    )
    return (bayes_pairwise_fig, b_target_q)


# =============================================================================
# SECTION 2: EGOCENTRIC PROJECTION MODEL
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 2: Egocentric Projection Model

The egocentric model uses **self as a model for others**. It combines two self-referential components:

1. **Perceived similarity** (global): Did the partner agree with me on the observed question?
2. **Self-response similarity** (local): Did I answer this new question similarly to the observed question?

### Key Equation

$$P(\\text{commonality}_q) = \\gamma_0 + \\gamma_1 \\cdot \\underbrace{\\exp\\left(-\\frac{|r_{\\text{obs}} - s_{q^*}|}{\\tau}\\right)}_{\\text{perceived similarity}} \\cdot \\underbrace{\\exp\\left(-\\frac{|s_q - s_{q^*}|}{\\tau}\\right)}_{\\text{self-response similarity}}$$

where $s_q$ is YOUR response to question $q$, and $s_{q^*}$ is your response to the observed question.

| Parameter | Symbol | Psychological Meaning |
|-----------|--------|----------------------|
| Your response | $s_q$ | Your own answer to question $q$ |
| Partner's response | $r_{\\text{obs}}$ | Partner's response to focal question |
| Perceived similarity | $\\exp(-\|r - s\|/\\tau)$ | "They agreed with me" → they're like me |
| Self-similarity | $\\exp(-\|s_q - s_{q^*}\|/\\tau)$ | How similarly YOU answered $q$ vs observed |
| Scale parameter | $\\tau$ | Tolerance for disagreement (larger τ = more forgiving) |
| Base rate | $\\gamma_0$ | Default P(commonality) with no information |
| Projection weight | $\\gamma_1$ | How much agreement boosts P(commonality) |

**Key insight**: Gradients emerge from *the structure of YOUR beliefs*. If YOUR political responses are correlated, projection transfers within politics. **No population knowledge required.**
    """)
    return


@app.cell
def __(mo, questions, domain_labels, sample_participants):
    # Egocentric model controls
    _question_options = {
        f"Q{i}: [{domain_labels[i][:4].upper()}] {q[:45]}{'...' if len(q) > 45 else ''}": i
        for i, q in enumerate(questions)
    }

    ego_question = mo.ui.dropdown(
        options=_question_options,
        value=list(_question_options.keys())[22],  # Same default as Bayesian
        label="Observed Question",
    )

    ego_response = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=5,
        label="Partner's Response",
        show_value=True,
    )

    ego_profile = mo.ui.dropdown(
        options=list(sample_participants.keys()),
        value="Participant A",
        label="Your Belief Profile",
    )

    ego_tau = mo.ui.slider(
        start=0.5,
        stop=4.0,
        step=0.5,
        value=2.0,
        label="τ (similarity scale)",
        show_value=True,
    )

    return ego_question, ego_response, ego_profile, ego_tau


@app.cell
def __(
    ego_question,
    ego_response,
    ego_profile,
    ego_tau,
    sample_participants,
    np,
):
    # Compute egocentric model predictions
    e_obs_q = ego_question.value
    e_r_obs = float(ego_response.value)
    e_r_self = sample_participants[ego_profile.value].copy()
    e_tau = float(ego_tau.value)

    # Model parameters
    _base_rate = 0.3
    _projection_weight = 0.4

    # Perceived similarity: did partner agree with me on observed question?
    e_r_self_obs = e_r_self[e_obs_q]
    e_perceived_sim = np.exp(-np.abs(e_r_obs - e_r_self_obs) / e_tau)

    # Self-response similarity: how similar is my response on each question to observed?
    e_self_sim = np.exp(-np.abs(e_r_self - e_r_self_obs) / e_tau)

    # Egocentric predictions
    e_predictions = np.clip(
        _base_rate + e_perceived_sim * e_self_sim * _projection_weight, 0.01, 0.99
    )

    return (
        e_obs_q,
        e_r_obs,
        e_r_self,
        e_tau,
        e_r_self_obs,
        e_perceived_sim,
        e_self_sim,
        e_predictions,
    )


@app.cell
def __(
    mo,
    ego_question,
    ego_response,
    ego_profile,
    ego_tau,
    e_obs_q,
    e_r_obs,
    e_r_self,
    e_tau,
    e_r_self_obs,
    e_perceived_sim,
    e_self_sim,
    e_predictions,
    questions,
    domain_labels,
    DOMAIN_RANGES,
    np,
    plt,
):
    _CYBER_BG = "#0D0221"
    _CYBER_GRID = "#1a1a3a"
    _CYBER_YELLOW = "#FFFF00"
    _CYBER_ORANGE = "#FF6600"
    _CYBER_GREEN = "#39FF14"

    ego_fig, _axes = plt.subplots(3, 1, figsize=(14, 9), facecolor=_CYBER_BG)

    _x = np.arange(35)
    _domain_color_map = {
        "arbitrary": "#00BFFF",
        "background": "#FF6600",
        "identity": "#39FF14",
        "morality": "#FF1493",
        "politics": "#9D00FF",
        "preferences": "#FFD700",
        "religion": "#00FFFF",
    }
    _colors = [_domain_color_map[d] for d in domain_labels]

    # Panel 1: Your responses
    _ax1 = _axes[0]
    _ax1.set_facecolor(_CYBER_BG)
    _ax1.bar(_x, e_r_self, color=_colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    _ax1.bar(
        [e_obs_q],
        [e_r_self[e_obs_q]],
        color=_CYBER_YELLOW,
        edgecolor="white",
        linewidth=2,
    )
    _ax1.axhline(
        y=e_r_self_obs,
        color=_CYBER_YELLOW,
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Your obs response: {e_r_self_obs:.1f}",
    )
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax1.axvline(x=_start - 0.5, color=_CYBER_GRID, alpha=0.8, linewidth=1)
        _ax1.text(
            (_start + _end) / 2 - 0.5,
            0.3,
            _domain[:3].upper(),
            ha="center",
            fontsize=8,
            color="white",
            alpha=0.7,
        )
    _ax1.set_ylabel("Your Response", color="white", fontsize=12)
    _ax1.set_title(
        f"YOUR RESPONSES ({ego_profile.value}) — Yellow = observed question",
        fontsize=14,
        fontweight="bold",
        color=_CYBER_YELLOW,
    )
    _ax1.set_ylim(0, 6)
    _ax1.set_xlim(-0.5, 34.5)
    _ax1.tick_params(colors="white")
    _ax1.set_xticks([])
    _ax1.legend(
        loc="upper right",
        fontsize=9,
        facecolor=_CYBER_BG,
        edgecolor=_CYBER_GRID,
        labelcolor="white",
    )
    for _spine in _ax1.spines.values():
        _spine.set_color(_CYBER_GRID)

    # Panel 2: Self-similarity to observed question
    _ax2 = _axes[1]
    _ax2.set_facecolor(_CYBER_BG)
    _ax2.bar(_x, e_self_sim, color=_colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    _ax2.bar(
        [e_obs_q],
        [e_self_sim[e_obs_q]],
        color=_CYBER_ORANGE,
        edgecolor="white",
        linewidth=2,
    )
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax2.axvline(x=_start - 0.5, color=_CYBER_GRID, alpha=0.8, linewidth=1)
    _ax2.set_ylabel("Self-Similarity", color="white", fontsize=12)
    _ax2.set_title(
        f"SELF-RESPONSE SIMILARITY: exp(-|s_q - s_obs| / τ) where τ={e_tau:.1f}",
        fontsize=14,
        fontweight="bold",
        color=_CYBER_ORANGE,
    )
    _ax2.set_ylim(0, 1.1)
    _ax2.set_xlim(-0.5, 34.5)
    _ax2.tick_params(colors="white")
    _ax2.set_xticks([])
    for _spine in _ax2.spines.values():
        _spine.set_color(_CYBER_GRID)

    # Panel 3: P(commonality) predictions
    _ax3 = _axes[2]
    _ax3.set_facecolor(_CYBER_BG)
    _ax3.bar(
        _x, e_predictions, color=_colors, alpha=0.8, edgecolor="white", linewidth=0.5
    )
    _ax3.bar(
        [e_obs_q],
        [e_predictions[e_obs_q]],
        color=_CYBER_GREEN,
        edgecolor="white",
        linewidth=2,
    )
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax3.axvline(x=_start - 0.5, color=_CYBER_GRID, alpha=0.8, linewidth=1)
    _ax3.set_xlabel("Question", color="white", fontsize=12)
    _ax3.set_ylabel("P(commonality)", color="white", fontsize=12)
    _ax3.set_title(
        f"EGOCENTRIC PREDICTION: P(commonality) = base + perceived_sim × self_sim × weight",
        fontsize=14,
        fontweight="bold",
        color=_CYBER_GREEN,
    )
    _ax3.set_ylim(0, 1.0)
    _ax3.set_xlim(-0.5, 34.5)
    _ax3.tick_params(colors="white")
    for _spine in _ax3.spines.values():
        _spine.set_color(_CYBER_GRID)

    plt.tight_layout()

    _agreement = "✓ AGREE" if abs(e_r_obs - e_r_self_obs) <= 1 else "✗ DISAGREE"
    _agreement_color = "green" if abs(e_r_obs - e_r_self_obs) <= 1 else "red"

    _controls = mo.vstack(
        [
            mo.md("### Egocentric Controls"),
            ego_question,
            ego_response,
            ego_profile,
            ego_tau,
            mo.md(f"""
---
Partner: **{e_r_obs:.0f}** | You: **{e_r_self_obs:.1f}** → <span style="color:{_agreement_color}">**{_agreement}**</span>

Perceived similarity: **{e_perceived_sim:.2f}**
"""),
        ],
        gap=0.5,
    )

    mo.hstack([_controls, ego_fig], widths=[1, 4], gap=1)
    return (ego_fig,)


@app.cell
def __(mo, questions, domain_labels, e_obs_q):
    # Egocentric target question selector
    _target_options = {
        f"Q{i}: [{domain_labels[i][:4].upper()}] {q[:40]}...": i
        for i, q in enumerate(questions)
    }
    _default_idx = (e_obs_q + 15) % 35

    ego_target = mo.ui.dropdown(
        options=_target_options,
        value=list(_target_options.keys())[_default_idx],
        label="Target Question",
    )
    return (ego_target,)


@app.cell
def __(
    mo,
    ego_target,
    e_obs_q,
    e_r_obs,
    e_r_self,
    e_predictions,
    questions,
    domain_labels,
    np,
    plt,
    erf,
):
    _CYBER_BG = "#0D0221"
    _CYBER_GRID = "#1a1a3a"
    _CYBER_YELLOW = "#FFFF00"
    _CYBER_GREEN = "#39FF14"

    e_target_q = ego_target.value
    _self_response = e_r_self[e_target_q]
    _ego_p_common = e_predictions[e_target_q]

    # Create distribution centered on self, width based on P(commonality)
    _ego_std = 1.5 * (1 - _ego_p_common) + 0.3
    _responses = np.array([1, 2, 3, 4, 5])
    _probs = []
    for _r in _responses:
        _p_upper = 0.5 * (
            1 + erf((_r + 0.5 - _self_response) / (_ego_std * np.sqrt(2)))
        )
        _p_lower = 0.5 * (
            1 + erf((_r - 0.5 - _self_response) / (_ego_std * np.sqrt(2)))
        )
        _probs.append(_p_upper - _p_lower)
    _probs = np.array(_probs)
    _probs = _probs / (_probs.sum() + 1e-10)

    ego_pairwise_fig, _ax = plt.subplots(figsize=(8, 4), facecolor=_CYBER_BG)
    _ax.set_facecolor(_CYBER_BG)
    _bars = _ax.bar(
        _responses,
        _probs,
        color=_CYBER_YELLOW,
        alpha=0.8,
        edgecolor="white",
        linewidth=2,
    )
    _ax.axvline(
        x=_self_response,
        color=_CYBER_GREEN,
        linewidth=3,
        linestyle="--",
        label=f"Your r={_self_response:.1f}",
    )

    for _bar, _p in zip(_bars, _probs):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _bar.get_height() + 0.01,
            f"{_p:.0%}",
            ha="center",
            va="bottom",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

    _ax.set_xlabel("Partner's Response", fontsize=12, color="white")
    _ax.set_ylabel("Probability", fontsize=12, color="white")
    _ax.set_title(
        f"Egocentric: P(response on Q{e_target_q} | observed Q{e_obs_q}={e_r_obs:.0f})",
        fontsize=14,
        fontweight="bold",
        color=_CYBER_YELLOW,
    )
    _ax.set_xticks([1, 2, 3, 4, 5])
    _ax.set_ylim(0, max(_probs) * 1.4)
    _ax.tick_params(colors="white", labelsize=11)
    _ax.legend(
        loc="upper right",
        fontsize=10,
        facecolor=_CYBER_BG,
        edgecolor=_CYBER_GRID,
        labelcolor="white",
    )
    _ax.grid(True, color=_CYBER_GRID, alpha=0.3, axis="y")
    for _spine in _ax.spines.values():
        _spine.set_color(_CYBER_GRID)
    plt.tight_layout()

    mo.vstack(
        [
            mo.md(f"""
### Egocentric Pairwise Prediction

**Observed**: Q{e_obs_q} ({domain_labels[e_obs_q]}) → {e_r_obs:.0f} | **Predicting**: Q{e_target_q} ({domain_labels[e_target_q]})

Your response on target: **{_self_response:.1f}** | P(commonality): **{_ego_p_common:.1%}**

*Egocentric predicts partner will respond like YOU on questions where YOU answered similarly to observed.*
"""),
            ego_target,
            ego_pairwise_fig,
        ]
    )
    return (ego_pairwise_fig, e_target_q)


# =============================================================================
# SECTION 3: MODEL COMPARISON
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 3: Model Comparison

Now let's compare the models side-by-side with the **mixture framework**:

$$P(\\text{commonality}) = (1 - \\lambda) \\cdot P_{\\text{Bayesian}} + \\lambda \\cdot P_{\\text{Projection}}$$

- $\\lambda = 0$: Pure Bayesian (population structure)
- $\\lambda = 1$: Pure egocentric (self-structure)

Use the controls below to compare predictions from both models for the same observation.
    """)
    return


@app.cell
def __(mo, questions, domain_labels, sample_participants):
    # Comparison controls
    _question_options = {
        f"Q{i}: [{domain_labels[i][:4].upper()}] {q[:40]}...": i
        for i, q in enumerate(questions)
    }

    cmp_question = mo.ui.dropdown(
        options=_question_options,
        value=list(_question_options.keys())[22],
        label="Observed Question",
    )

    cmp_response = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=5,
        label="Partner's Response",
        show_value=True,
    )

    cmp_lambda = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.1,
        value=0.0,
        label="λ (0=Bayesian, 1=Egocentric)",
        show_value=True,
    )

    cmp_profile = mo.ui.dropdown(
        options=list(sample_participants.keys()),
        value="Participant A",
        label="Your Profile (for Egocentric)",
    )

    cmp_k = mo.ui.slider(
        start=1,
        stop=6,
        step=1,
        value=5,
        label="k factors (for Bayesian)",
        show_value=True,
    )

    return cmp_question, cmp_response, cmp_lambda, cmp_profile, cmp_k


@app.cell
def __(
    cmp_question,
    cmp_response,
    cmp_lambda,
    cmp_profile,
    cmp_k,
    sample_participants,
    all_loadings,
    means,
    np,
):
    # Compute both models
    c_obs_q = cmp_question.value
    c_r_obs = float(cmp_response.value)
    c_lambda = float(cmp_lambda.value)
    c_r_self = sample_participants[cmp_profile.value].copy()
    c_k = int(cmp_k.value)

    # Bayesian
    c_loadings = all_loadings[:, :c_k]
    c_L_obs = c_loadings[c_obs_q]
    c_prior_cov = 1.5**2 * np.eye(c_k)
    _r_centered = c_r_obs - means[c_obs_q]
    _L_cov_L = c_L_obs @ c_prior_cov @ c_L_obs
    _K = c_prior_cov @ c_L_obs / (_L_cov_L + 1e-10)
    c_post_mean = _K * _r_centered
    c_post_cov = c_prior_cov - np.outer(_K, c_L_obs @ c_prior_cov)

    c_bayes_means = c_loadings @ c_post_mean + means
    c_bayes_vars = np.array(
        [c_loadings[i] @ c_post_cov @ c_loadings[i] for i in range(35)]
    )

    # Egocentric
    _tau = 2.0
    c_r_self_obs = c_r_self[c_obs_q]
    c_perceived_sim = np.exp(-np.abs(c_r_obs - c_r_self_obs) / _tau)
    c_self_sim = np.exp(-np.abs(c_r_self - c_r_self_obs) / _tau)
    c_ego_preds = np.clip(0.3 + c_perceived_sim * c_self_sim * 0.4, 0.01, 0.99)

    return (
        c_obs_q,
        c_r_obs,
        c_lambda,
        c_r_self,
        c_k,
        c_loadings,
        c_post_mean,
        c_post_cov,
        c_bayes_means,
        c_bayes_vars,
        c_r_self_obs,
        c_perceived_sim,
        c_ego_preds,
    )


@app.cell
def __(
    mo,
    cmp_question,
    cmp_response,
    cmp_lambda,
    cmp_profile,
    cmp_k,
    c_obs_q,
    c_r_obs,
    c_lambda,
    c_r_self,
    c_k,
    c_bayes_means,
    c_bayes_vars,
    c_ego_preds,
    c_r_self_obs,
    c_perceived_sim,
    means,
    domain_labels,
    DOMAIN_RANGES,
    np,
    plt,
):
    _CYBER_BG = "#0D0221"
    _CYBER_GRID = "#1a1a3a"
    _CYBER_CYAN = "#00FFFF"
    _CYBER_YELLOW = "#FFFF00"
    _CYBER_MAGENTA = "#FF00FF"

    cmp_fig, _axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=_CYBER_BG)

    _x = np.arange(35)
    _domain_color_map = {
        "arbitrary": "#00BFFF",
        "background": "#FF6600",
        "identity": "#39FF14",
        "morality": "#FF1493",
        "politics": "#9D00FF",
        "preferences": "#FFD700",
        "religion": "#00FFFF",
    }
    _colors = [_domain_color_map[d] for d in domain_labels]

    # Top: Bayesian
    _ax1 = _axes[0]
    _ax1.set_facecolor(_CYBER_BG)
    _bayes_stds = np.sqrt(c_bayes_vars + 1e-10)
    _ax1.bar(
        _x,
        c_bayes_means,
        yerr=_bayes_stds,
        color=_colors,
        alpha=0.8,
        capsize=2,
        error_kw={"linewidth": 1, "alpha": 0.6, "color": "white"},
        edgecolor="white",
        linewidth=0.5,
    )
    _ax1.bar(
        [c_obs_q],
        [c_bayes_means[c_obs_q]],
        color=_CYBER_CYAN,
        edgecolor="white",
        linewidth=2,
    )
    _ax1.scatter(
        _x,
        means,
        color="white",
        s=20,
        marker="_",
        linewidths=2,
        label="Pop. mean",
        zorder=5,
    )
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax1.axvline(x=_start - 0.5, color=_CYBER_GRID, alpha=0.8, linewidth=1)
        _ax1.text(
            (_start + _end) / 2 - 0.5,
            0.4,
            _domain[:3].upper(),
            ha="center",
            fontsize=8,
            color="white",
            alpha=0.7,
        )
    _ax1.set_ylabel("Predicted Response", color="white", fontsize=12)
    _ax1.set_title(
        f"BAYESIAN (k={c_k}) — Population covariance structure",
        fontsize=14,
        fontweight="bold",
        color=_CYBER_CYAN,
    )
    _ax1.set_ylim(0, 6)
    _ax1.set_xlim(-0.5, 34.5)
    _ax1.tick_params(colors="white")
    _ax1.set_xticks([])
    _ax1.legend(
        loc="upper right",
        fontsize=9,
        facecolor=_CYBER_BG,
        edgecolor=_CYBER_GRID,
        labelcolor="white",
    )
    for _spine in _ax1.spines.values():
        _spine.set_color(_CYBER_GRID)

    # Bottom: Egocentric
    _ax2 = _axes[1]
    _ax2.set_facecolor(_CYBER_BG)
    _ego_scaled = c_ego_preds * 4 + 1
    _ax2.bar(
        _x, _ego_scaled, color=_colors, alpha=0.8, edgecolor="white", linewidth=0.5
    )
    _ax2.bar(
        [c_obs_q],
        [_ego_scaled[c_obs_q]],
        color=_CYBER_YELLOW,
        edgecolor="white",
        linewidth=2,
    )
    _ax2.scatter(
        _x,
        c_r_self,
        color="white",
        s=30,
        marker="o",
        alpha=0.6,
        label="Your responses",
        zorder=3,
    )
    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax2.axvline(x=_start - 0.5, color=_CYBER_GRID, alpha=0.8, linewidth=1)
    _ax2.set_xlabel("Question", color="white", fontsize=12)
    _ax2.set_ylabel("P(commonality) scaled", color="white", fontsize=12)
    _ax2.set_title(
        f"EGOCENTRIC ({cmp_profile.value}) — Your belief structure",
        fontsize=14,
        fontweight="bold",
        color=_CYBER_YELLOW,
    )
    _ax2.set_ylim(0, 6)
    _ax2.set_xlim(-0.5, 34.5)
    _ax2.tick_params(colors="white")
    _ax2.legend(
        loc="upper right",
        fontsize=9,
        facecolor=_CYBER_BG,
        edgecolor=_CYBER_GRID,
        labelcolor="white",
    )
    for _spine in _ax2.spines.values():
        _spine.set_color(_CYBER_GRID)

    plt.tight_layout()

    _agreement = "AGREE" if abs(c_r_obs - c_r_self_obs) <= 1 else "DISAGREE"

    _controls = mo.vstack(
        [
            mo.md("### Comparison Controls"),
            cmp_question,
            cmp_response,
            cmp_lambda,
            cmp_profile,
            cmp_k,
            mo.md(f"""
---
Partner: **{c_r_obs:.0f}** | You: **{c_r_self_obs:.1f}** → **{_agreement}**

λ = {c_lambda:.1f} → **{(1-c_lambda)*100:.0f}% Bayesian + {c_lambda*100:.0f}% Egocentric**
"""),
        ],
        gap=0.5,
    )

    mo.hstack([_controls, cmp_fig], widths=[1, 4], gap=1)
    return (cmp_fig,)


# =============================================================================
# SECTION 4: MIXTURE MODEL PERFORMANCE
# =============================================================================


@app.cell
def __(mo):
    mo.md("""
---
# Section 4: Mixture Model Performance

The mixture model interpolates between pure Bayesian ($\\lambda=0$) and pure Egocentric ($\\lambda=1$):

$$P(\\text{commonality}) = (1 - \\lambda) \\cdot P_{\\text{Bayesian}} + \\lambda \\cdot P_{\\text{Egocentric}}$$

**Key question**: What value of $\\lambda$ best captures human generalization?

The **generalization gradient** measures structured transfer—the difference-in-differences:

$$\\text{Gradient} = \\underbrace{(P_{\\text{same}}^{\\text{shared}} - P_{\\text{same}}^{\\text{opposing}})}_{\\text{same-domain effect}} - \\underbrace{(P_{\\text{diff}}^{\\text{shared}} - P_{\\text{diff}}^{\\text{opposing}})}_{\\text{different-domain effect}}$$

A positive gradient means the shared/opposing difference is larger for same-domain questions than different-domain questions—the signature of structured generalization.
    """)
    return


@app.cell
def __(mo):
    mix_lambda = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.0,
        label="λ (mixture weight)",
        show_value=True,
    )
    return (mix_lambda,)


@app.cell
def __(mix_lambda, np, plt, mo):
    _CYBER_BG = "#0D0221"
    _CYBER_GRID = "#1a1a3a"
    _CYBER_CYAN = "#00FFFF"
    _CYBER_YELLOW = "#FFFF00"
    _CYBER_MAGENTA = "#FF00FF"
    _CYBER_GREEN = "#39FF14"

    # Human gradient from fitted_params.json
    _human_gradient = (
        0.124  # (same_shared - same_opposing) - (diff_shared - diff_opposing)
    )

    # Model gradients at different λ (from supplement analysis)
    # These are approximate values showing the trend
    _lambda_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Bayesian gradient ≈ 0.118 (95% of human), Egocentric ≈ 0.015 (12% of human)
    _bayes_grad = 0.118
    _ego_grad = 0.015
    # Linear interpolation (approximate)
    _model_grads = (1 - _lambda_vals) * _bayes_grad + _lambda_vals * _ego_grad
    _pct_captured = _model_grads / _human_gradient * 100

    # Current λ value
    _curr_lambda = float(mix_lambda.value)
    _curr_grad = (1 - _curr_lambda) * _bayes_grad + _curr_lambda * _ego_grad
    _curr_pct = _curr_grad / _human_gradient * 100

    # Create figure
    mix_fig, _axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_CYBER_BG)

    # Left panel: Gradient vs λ
    _ax1 = _axes[0]
    _ax1.set_facecolor(_CYBER_BG)
    _ax1.plot(
        _lambda_vals,
        _model_grads,
        color=_CYBER_MAGENTA,
        linewidth=3,
        label="Model gradient",
    )
    _ax1.axhline(
        y=_human_gradient,
        color="white",
        linestyle="--",
        linewidth=2,
        label=f"Human gradient ({_human_gradient:.3f})",
    )
    _ax1.axhline(y=0, color=_CYBER_GRID, linestyle="-", linewidth=1)

    # Mark current λ
    _ax1.scatter(
        [_curr_lambda],
        [_curr_grad],
        color=_CYBER_GREEN,
        s=200,
        zorder=5,
        edgecolor="white",
        linewidth=2,
    )
    _ax1.axvline(
        x=_curr_lambda, color=_CYBER_GREEN, linestyle=":", linewidth=2, alpha=0.7
    )

    # Mark endpoints
    _ax1.scatter(
        [0],
        [_bayes_grad],
        color=_CYBER_CYAN,
        s=150,
        zorder=4,
        marker="s",
        label="Bayesian (λ=0)",
    )
    _ax1.scatter(
        [1],
        [_ego_grad],
        color=_CYBER_YELLOW,
        s=150,
        zorder=4,
        marker="s",
        label="Egocentric (λ=1)",
    )

    _ax1.set_xlabel("λ (mixture weight)", fontsize=14, color="white")
    _ax1.set_ylabel("Generalization Gradient", fontsize=14, color="white")
    _ax1.set_title(
        "Model Gradient vs Mixture Weight",
        fontsize=16,
        fontweight="bold",
        color="white",
    )
    _ax1.set_xlim(-0.05, 1.05)
    _ax1.set_ylim(-0.01, 0.15)
    _ax1.tick_params(colors="white", labelsize=11)
    _ax1.legend(
        loc="upper right",
        fontsize=10,
        facecolor=_CYBER_BG,
        edgecolor=_CYBER_GRID,
        labelcolor="white",
    )
    _ax1.grid(True, color=_CYBER_GRID, alpha=0.3)
    for _spine in _ax1.spines.values():
        _spine.set_color(_CYBER_GRID)

    # Right panel: % of human gradient captured
    _ax2 = _axes[1]
    _ax2.set_facecolor(_CYBER_BG)

    # Bar showing current % captured
    _bar_colors = [
        _CYBER_CYAN if i == 0 else _CYBER_YELLOW if i == 2 else _CYBER_MAGENTA
        for i in range(3)
    ]
    _bar_vals = [95, _curr_pct, 12]
    _bar_labels = [
        "Bayesian\n(λ=0)",
        f"Current\n(λ={_curr_lambda:.2f})",
        "Egocentric\n(λ=1)",
    ]

    _bars = _ax2.bar(
        _bar_labels,
        _bar_vals,
        color=_bar_colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=2,
    )

    # Add percentage labels
    for _bar, _val in zip(_bars, _bar_vals):
        _ax2.text(
            _bar.get_x() + _bar.get_width() / 2,
            _bar.get_height() + 2,
            f"{_val:.0f}%",
            ha="center",
            va="bottom",
            color="white",
            fontsize=14,
            fontweight="bold",
        )

    _ax2.axhline(
        y=100,
        color="white",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Human (100%)",
    )
    _ax2.set_ylabel("% of Human Gradient Captured", fontsize=14, color="white")
    _ax2.set_title(
        "Model Fit to Human Data", fontsize=16, fontweight="bold", color="white"
    )
    _ax2.set_ylim(0, 120)
    _ax2.tick_params(colors="white", labelsize=12)
    _ax2.grid(True, color=_CYBER_GRID, alpha=0.3, axis="y")
    for _spine in _ax2.spines.values():
        _spine.set_color(_CYBER_GRID)

    plt.tight_layout()

    mo.vstack(
        [
            mo.hstack([mix_lambda], justify="center"),
            mix_fig,
            mo.md(f"""
### Current Selection: λ = {_curr_lambda:.2f}

- **Model gradient**: {_curr_grad:.3f}
- **Human gradient**: {_human_gradient:.3f}
- **% captured**: {_curr_pct:.1f}%

{
"**Best fit!** Pure Bayesian (λ=0) captures 95% of human generalization gradient." if _curr_lambda == 0 else
"**Worst fit!** Pure Egocentric (λ=1) captures only 12% of human gradient." if _curr_lambda == 1 else
f"Mixture model captures {_curr_pct:.0f}% of human gradient."
}
"""),
        ]
    )
    return (mix_fig,)


@app.cell
def __(mo):
    mo.md("""
---
## Key Takeaways

| Model | λ | Source of Gradients | Human Fit |
|-------|---|---------------------|-----------|
| **Bayesian Factor** | 0 | Population covariance structure | **95%** of gradient |
| **Egocentric Projection** | 1 | Individual's own belief structure | **12%** of gradient |

**Why the difference?**

- The Bayesian model captures *population-level regularities* (political beliefs covary in the population)
- The egocentric model only captures *individual-level structure* (how YOUR beliefs happen to cluster)
- Human generalization follows population structure, even when individual structure differs

**Implication**: People have implicit knowledge about how beliefs covary in populations, not just in themselves.

**Best-fitting model**: Pure Bayesian (λ=0, k=5) with fitted parameters σ_prior=1.5, τ=2.0, ε=0.1
    """)
    return


if __name__ == "__main__":
    app.run()
