"""
Interactive demo of the Bayesian posterior update in factor space.

Shows how observing a single response updates beliefs about a partner's
position in latent factor space.

Run with: uv run marimo run notebooks/bayesian_update_demo.py
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

This notebook compares two computational models of how people infer commonality with a conversation partner after observing a single response.

---

## Model 1: Bayesian Factor Model (Population Structure)

The Bayesian model represents a partner's beliefs as a latent position $\\theta \\in \\mathbb{R}^k$ in **factor space**—a low-dimensional space where dimensions capture how beliefs covary *in the population*.

### Key Equations

**Prior** — Before observing anything, we assume the partner is drawn from the population:

$$\\theta \\sim \\mathcal{N}(0, \\sigma^2_{\\text{prior}} \\cdot I_k)$$

**Generative model** — Each survey response is a noisy projection from factor space:

$$r_q \\mid \\theta \\sim \\mathcal{N}(\\Lambda_q^\\top \\theta + \\mu_q, \\sigma^2_{\\text{obs}})$$

where $\\Lambda_q$ is the factor loading vector for question $q$ and $\\mu_q$ is the population mean.

**Posterior** — After observing $r_{\\text{obs}}$, we update via Bayes' rule (closed-form Gaussian):

$$\\mu_{\\text{post}} = \\Sigma_{\\text{post}} \\left( \\frac{\\Lambda_{\\text{obs}} (r_{\\text{obs}} - \\mu_{\\text{obs}})}{\\sigma^2_{\\text{obs}}} \\right), \\quad \\Sigma_{\\text{post}}^{-1} = \\Sigma_0^{-1} + \\frac{\\Lambda_{\\text{obs}} \\Lambda_{\\text{obs}}^\\top}{\\sigma^2_{\\text{obs}}}$$

**Predictive** — For target question $q$: $\\;\\hat{r}_q \\sim \\mathcal{N}(\\Lambda_q^\\top \\mu_{\\text{post}} + \\mu_q, \\; \\Lambda_q^\\top \\Sigma_{\\text{post}} \\Lambda_q)$

**Key insight**: Gradients emerge from *population covariance structure*. Political questions load on the same factors, so observing one updates predictions for others.

---

## Model 2: Similarity Projection (Egocentric Heuristic)

The similarity projection model uses **self as a model for others**. It combines two self-referential components:

1. **Perceived similarity** (global): Did the partner agree with me on the observed question?
2. **Self-response similarity** (local): Did I answer this new question similarly to the observed question?

### Key Equation

$$P(\\text{commonality}_q) = \\gamma_0 + \\gamma_1 \\cdot \\underbrace{\\exp\\left(-\\frac{|r_{\\text{obs}} - s_{q^*}|}{\\tau}\\right)}_{\\text{perceived similarity}} \\cdot \\underbrace{\\exp\\left(-\\frac{|s_q - s_{q^*}|}{\\tau}\\right)}_{\\text{self-response similarity}}$$

where $s_q$ is the participant's own response to question $q$, and $s_{q^*}$ is their response to the observed question.

**Key insight**: Gradients emerge from *the structure of your own beliefs*. If your political responses are correlated, projection transfers within politics. **No population knowledge required.**

---

## Model Comparison: Mixture Framework

The models are nested via mixture weight $\\lambda \\in [0,1]$:

$$P(\\text{commonality}) = (1 - \\lambda) \\cdot P_{\\text{Bayesian}} + \\lambda \\cdot P_{\\text{Projection}}$$

- $\\lambda = 0$: Pure Bayesian (population structure)
- $\\lambda = 1$: Pure egocentric projection (self-structure)

**Empirical result**: The Bayesian model ($\\lambda=0$) captures **95%** of the human generalization gradient; the egocentric model ($\\lambda=1$) captures only **12%**.

---
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
    _q_map = _df.groupby('question')['preChatQuestion'].first().to_dict()
    questions = [_q_map.get(i+1, f"Question {i+1}") for i in range(35)]

    # Create domain labels
    domain_labels = []
    for _i in range(35):
        for _domain, (_start, _end) in DOMAIN_RANGES.items():
            if _start <= _i < _end:
                domain_labels.append(_domain)
                break

    # Load sample participant responses for egocentric model demo
    # Get a few diverse participants with different belief structures
    _responses_df = load_responses()
    sample_participants = {
        "Population mean": means.copy(),
        "Participant A": _responses_df.iloc[0].values.copy(),
        "Participant B": _responses_df.iloc[50].values.copy(),
        "Participant C": _responses_df.iloc[100].values.copy(),
    }

    return all_loadings, means, questions, domain_labels, sample_participants


@app.cell
def __(mo):
    mo.md("""
## Interactive Visualization

Use the controls below to explore how a single observed response updates the posterior distribution in factor space.

- **Question**: Select which question the partner responded to
- **Partner's Response**: The observed Likert response (1-5)
- **Factors (k)**: Number of latent dimensions in the factor model
- **Prior σ**: Standard deviation of the prior (how uncertain you are before observing)
    """)
    return


@app.cell
def __(mo, questions, domain_labels, sample_participants):
    # Question dropdown with domain labels
    _question_options = {
        f"Q{i}: [{domain_labels[i][:4].upper()}] {q[:45]}{'...' if len(q) > 45 else ''}": i
        for i, q in enumerate(questions)
    }

    question_selector = mo.ui.dropdown(
        options=_question_options,
        value=list(_question_options.keys())[30],  # Start with a religion question
        label="Observed Question",
    )

    response_slider = mo.ui.slider(
        start=1, stop=5, step=1, value=5,
        label="Partner's Response",
        show_value=True,
    )

    k_slider = mo.ui.slider(
        start=1, stop=6, step=1, value=2,
        label="Factors (k)",
        show_value=True,
    )

    sigma_prior_slider = mo.ui.slider(
        start=0.5, stop=3.0, step=0.25, value=1.5,
        label="Prior σ",
        show_value=True,
    )

    # New controls for model comparison
    lambda_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.1, value=0.0,
        label="λ (0=Bayesian, 1=Egocentric)",
        show_value=True,
    )

    self_profile_selector = mo.ui.dropdown(
        options=list(sample_participants.keys()),
        value="Population mean",
        label="Your Belief Profile",
    )

    return question_selector, response_slider, k_slider, sigma_prior_slider, lambda_slider, self_profile_selector


@app.cell
def __(
    question_selector,
    response_slider,
    k_slider,
    sigma_prior_slider,
    lambda_slider,
    self_profile_selector,
    sample_participants,
    all_loadings,
    means,
    np,
):
    # Extract current values (dropdown returns the value directly, which is the index)
    obs_q = question_selector.value
    r_obs = float(response_slider.value)
    k = int(k_slider.value)
    sigma_prior = float(sigma_prior_slider.value)
    lambda_mix = float(lambda_slider.value)

    # Get self responses for egocentric model
    r_self = sample_participants[self_profile_selector.value].copy()

    # Get factor loadings for this k
    loadings_k = all_loadings[:, :k]
    L_obs = loadings_k[obs_q]  # (k,) loading vector for observed question
    mu_obs = means[obs_q]  # Population mean for observed question

    # Prior parameters
    prior_mean = np.zeros(k)
    prior_cov = sigma_prior**2 * np.eye(k)

    # Posterior update (delta observation, sigma_obs = 0)
    _r_centered = r_obs - mu_obs
    _L_cov_L = L_obs @ prior_cov @ L_obs
    _K = prior_cov @ L_obs / (_L_cov_L + 1e-10)  # Kalman gain
    _innovation = _r_centered - L_obs @ prior_mean

    posterior_mean = prior_mean + _K * _innovation
    posterior_cov = prior_cov - np.outer(_K, L_obs @ prior_cov)

    # === Egocentric model predictions ===
    # Parameters (from paper's best fit)
    _base_rate = 0.3
    _projection_weight = 0.4
    _tau = 2.0  # Scale for similarity decay

    # Perceived similarity: did partner agree with me on observed question?
    _r_self_obs = r_self[obs_q]
    _perceived_sim = np.exp(-np.abs(r_obs - _r_self_obs) / _tau)

    # Self-response similarity: how similarly did I answer each question vs observed?
    _self_sim = np.exp(-np.abs(r_self - _r_self_obs) / _tau)

    # Egocentric predictions for all 35 questions
    ego_predictions = np.clip(_base_rate + _perceived_sim * _self_sim * _projection_weight, 0.01, 0.99)

    return (
        obs_q, r_obs, k, sigma_prior, lambda_mix,
        r_self,
        loadings_k, L_obs, mu_obs,
        prior_mean, prior_cov,
        posterior_mean, posterior_cov,
        ego_predictions,
    )


@app.cell
def __(
    mo,
    question_selector,
    response_slider,
    k_slider,
    sigma_prior_slider,
    lambda_slider,
    self_profile_selector,
    obs_q, r_obs, k, sigma_prior, lambda_mix,
    r_self,
    L_obs, mu_obs,
    prior_mean, prior_cov,
    posterior_mean, posterior_cov,
    questions, domain_labels,
    np, plt, Ellipse, transforms,
):
    # Cyberpunk color palette
    CYBER_CYAN = '#00FFFF'
    CYBER_MAGENTA = '#FF00FF'
    CYBER_YELLOW = '#FFFF00'
    CYBER_PINK = '#FF1493'
    CYBER_PURPLE = '#9D00FF'
    CYBER_ORANGE = '#FF6600'
    CYBER_BG = '#0D0221'
    CYBER_GRID = '#1a1a3a'

    def _confidence_ellipse(mean, cov, ax, n_std=2.0, facecolor='none', **kwargs):
        """Draw a covariance ellipse (2D projection)."""
        if len(mean) < 2:
            return None
        _mean_2d = mean[:2]
        _cov_2d = cov[:2, :2]
        _pearson = _cov_2d[0, 1] / np.sqrt(_cov_2d[0, 0] * _cov_2d[1, 1] + 1e-10)
        _ell_radius_x = np.sqrt(1 + _pearson)
        _ell_radius_y = np.sqrt(1 - _pearson)
        _ellipse = Ellipse((0, 0), width=_ell_radius_x * 2, height=_ell_radius_y * 2,
                          facecolor=facecolor, **kwargs)
        _scale_x = np.sqrt(_cov_2d[0, 0]) * n_std
        _scale_y = np.sqrt(_cov_2d[1, 1]) * n_std
        _transf = transforms.Affine2D().rotate_deg(45).scale(_scale_x, _scale_y).translate(_mean_2d[0], _mean_2d[1])
        _ellipse.set_transform(_transf + ax.transData)
        return ax.add_patch(_ellipse)

    # Create the 3-panel figure with dark background
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=CYBER_BG)
    _plot_range = 4.0

    for _ax in axes:
        _ax.set_facecolor(CYBER_BG)
        _ax.tick_params(colors='white', labelsize=13)
        _ax.xaxis.label.set_color('white')
        _ax.yaxis.label.set_color('white')
        for _spine in _ax.spines.values():
            _spine.set_color(CYBER_GRID)

    # === Panel 1: PRIOR (in θ-space) ===
    _ax1 = axes[0]
    _ax1.set_title("PRIOR\n(belief before observation)", fontsize=18, fontweight='bold', color=CYBER_CYAN)
    if k >= 2:
        _confidence_ellipse(prior_mean, prior_cov, _ax1, n_std=1, edgecolor=CYBER_CYAN, linewidth=2.5, linestyle='-')
        _confidence_ellipse(prior_mean, prior_cov, _ax1, n_std=2, edgecolor=CYBER_CYAN, linewidth=1.5, linestyle='--', alpha=0.6)
        _ax1.scatter([0], [0], color=CYBER_CYAN, s=150, marker='+', linewidths=3, zorder=5)
        _ax1.set_xlabel("θ₁ (Factor 1)", fontsize=14); _ax1.set_ylabel("θ₂ (Factor 2)", fontsize=14)
    else:
        _x = np.linspace(-_plot_range, _plot_range, 200)
        _y = np.exp(-0.5 * (_x - prior_mean[0])**2 / prior_cov[0, 0])
        _ax1.fill_between(_x, _y, alpha=0.4, color=CYBER_CYAN)
        _ax1.plot(_x, _y, color=CYBER_CYAN, linewidth=2.5)
        _ax1.set_xlabel("θ₁ (Factor 1)", fontsize=14); _ax1.set_ylabel("Density", fontsize=14)
    _ax1.set_xlim(-_plot_range, _plot_range)
    _ax1.set_ylim(-_plot_range, _plot_range) if k >= 2 else None
    _ax1.set_aspect('equal' if k >= 2 else 'auto')
    _ax1.axhline(y=0, color=CYBER_GRID, linestyle='-', alpha=0.5, linewidth=0.5)
    _ax1.axvline(x=0, color=CYBER_GRID, linestyle='-', alpha=0.5, linewidth=0.5)
    _ax1.grid(True, color=CYBER_GRID, alpha=0.3, linewidth=0.5)
    _ax1.text(0.05, 0.95, f"θ ~ N(0, {sigma_prior:.1f}²·I)", transform=_ax1.transAxes,
             fontsize=14, va='top', family='monospace', color='white', fontweight='bold',
             bbox=dict(boxstyle='round', fc=CYBER_BG, ec=CYBER_CYAN, alpha=0.9, pad=0.5))

    # === Panel 2: LIKELIHOOD (constraint from observation) ===
    _ax2 = axes[1]
    _ax2.set_title("LIKELIHOOD\n(constraint from observation)", fontsize=18, fontweight='bold', color=CYBER_YELLOW)
    _constraint_val = r_obs - mu_obs
    if k >= 2:
        if abs(L_obs[1]) > 1e-6:
            _theta1 = np.linspace(-_plot_range, _plot_range, 100)
            _theta2 = (_constraint_val - L_obs[0] * _theta1) / L_obs[1]
            _valid = np.abs(_theta2) < _plot_range * 1.5
            _ax2.plot(_theta1[_valid], _theta2[_valid], color=CYBER_YELLOW, linewidth=3.5,
                     label='Λ\'θ + μ = r')
        elif abs(L_obs[0]) > 1e-6:
            _ax2.axvline(x=_constraint_val / L_obs[0], color=CYBER_YELLOW, linewidth=3.5)
        # Loading direction arrow
        _arrow_scale = 1.8
        _ax2.arrow(0, 0, L_obs[0]*_arrow_scale, L_obs[1]*_arrow_scale,
                  head_width=0.2, head_length=0.12, fc=CYBER_ORANGE, ec=CYBER_ORANGE, lw=2.5, zorder=5)
        _ax2.text(L_obs[0]*_arrow_scale*1.15, L_obs[1]*_arrow_scale*1.15, 'Λ', fontsize=16,
                 color=CYBER_ORANGE, fontweight='bold', ha='center')
        _ax2.set_xlabel("θ₁ (Factor 1)", fontsize=14); _ax2.set_ylabel("θ₂ (Factor 2)", fontsize=14)
    else:
        _theta_constrained = _constraint_val / (L_obs[0] + 1e-10)
        _ax2.axvline(x=_theta_constrained, color=CYBER_YELLOW, linewidth=3.5)
        _ax2.scatter([_theta_constrained], [0.5], color=CYBER_YELLOW, s=200, marker='|', linewidths=3)
        _ax2.set_xlabel("θ₁ (Factor 1)", fontsize=14); _ax2.set_ylabel("", fontsize=14)
        _ax2.set_ylim(0, 1)
    _ax2.set_xlim(-_plot_range, _plot_range)
    if k >= 2: _ax2.set_ylim(-_plot_range, _plot_range)
    _ax2.set_aspect('equal' if k >= 2 else 'auto')
    _ax2.axhline(y=0, color=CYBER_GRID, linestyle='-', alpha=0.5, linewidth=0.5)
    _ax2.axvline(x=0, color=CYBER_GRID, linestyle='-', alpha=0.5, linewidth=0.5)
    _ax2.grid(True, color=CYBER_GRID, alpha=0.3, linewidth=0.5)
    _ax2.text(0.05, 0.95, f"Observed: r = {r_obs:.0f}\nΛ'θ + μ = r",
             transform=_ax2.transAxes, fontsize=14, va='top', family='monospace', color='white', fontweight='bold',
             bbox=dict(boxstyle='round', fc=CYBER_BG, ec=CYBER_YELLOW, alpha=0.9, pad=0.5))

    # === Panel 3: POSTERIOR (updated belief) ===
    _ax3 = axes[2]
    _ax3.set_title("POSTERIOR\n(updated belief)", fontsize=18, fontweight='bold', color=CYBER_MAGENTA)
    if k >= 2:
        # Posterior ellipse
        _confidence_ellipse(posterior_mean, posterior_cov, _ax3, n_std=1, edgecolor=CYBER_MAGENTA, linewidth=2.5)
        _confidence_ellipse(posterior_mean, posterior_cov, _ax3, n_std=2, edgecolor=CYBER_MAGENTA, linewidth=1.5, linestyle='--', alpha=0.6)
        # Ghost of prior
        _confidence_ellipse(prior_mean, prior_cov, _ax3, n_std=1, edgecolor=CYBER_CYAN, linewidth=1, linestyle=':', alpha=0.4)
        # Mean markers
        _ax3.scatter([0], [0], color=CYBER_CYAN, s=80, marker='+', linewidths=2, alpha=0.5, label='Prior μ')
        _ax3.scatter([posterior_mean[0]], [posterior_mean[1]], color=CYBER_MAGENTA, s=120,
                    marker='X', linewidths=2, zorder=5, label='Posterior μ')
        # Arrow showing shift
        _ax3.annotate('', xy=(posterior_mean[0], posterior_mean[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=CYBER_PINK, lw=2.5))
        _ax3.set_xlabel("θ₁ (Factor 1)", fontsize=14); _ax3.set_ylabel("θ₂ (Factor 2)", fontsize=14)
    else:
        _x = np.linspace(-_plot_range, _plot_range, 200)
        _y_prior = np.exp(-0.5 * (_x - prior_mean[0])**2 / prior_cov[0, 0])
        _y_post = np.exp(-0.5 * (_x - posterior_mean[0])**2 / (posterior_cov[0, 0] + 1e-10))
        _ax3.fill_between(_x, _y_prior, alpha=0.3, color=CYBER_CYAN, label='Prior')
        _ax3.fill_between(_x, _y_post, alpha=0.5, color=CYBER_MAGENTA, label='Posterior')
        _ax3.plot(_x, _y_post, color=CYBER_MAGENTA, linewidth=2.5)
        _ax3.axvline(x=posterior_mean[0], color=CYBER_MAGENTA, linestyle='-', alpha=0.8, linewidth=2)
        _ax3.set_xlabel("θ₁ (Factor 1)", fontsize=14); _ax3.set_ylabel("Density", fontsize=14)
        _ax3.legend(facecolor=CYBER_BG, edgecolor=CYBER_GRID, labelcolor='white', fontsize=11)
    _ax3.set_xlim(-_plot_range, _plot_range)
    if k >= 2: _ax3.set_ylim(-_plot_range, _plot_range)
    _ax3.set_aspect('equal' if k >= 2 else 'auto')
    _ax3.axhline(y=0, color=CYBER_GRID, linestyle='-', alpha=0.5, linewidth=0.5)
    _ax3.axvline(x=0, color=CYBER_GRID, linestyle='-', alpha=0.5, linewidth=0.5)
    _ax3.grid(True, color=CYBER_GRID, alpha=0.3, linewidth=0.5)
    _ax3.text(0.05, 0.95, f"Prior × Likelihood\n= Posterior",
             transform=_ax3.transAxes, fontsize=14, va='top', family='monospace', color='white', fontweight='bold',
             bbox=dict(boxstyle='round', fc=CYBER_BG, ec=CYBER_MAGENTA, alpha=0.9, pad=0.5))

    plt.tight_layout()

    # Build the layout: controls on left, plot on right
    _self_response_obs = r_self[obs_q]
    _agreement = "✓ AGREE" if abs(r_obs - _self_response_obs) <= 1 else "✗ DISAGREE"
    _agreement_color = "green" if abs(r_obs - _self_response_obs) <= 1 else "red"

    _controls = mo.vstack([
        mo.md("### Bayesian Model Controls"),
        question_selector,
        mo.hstack([response_slider, k_slider], justify="start"),
        sigma_prior_slider,
        mo.md("---"),
        mo.md("### Model Comparison"),
        lambda_slider,
        self_profile_selector,
        mo.md(f"""
---
**Observed Q{obs_q}** ({domain_labels[obs_q]}):
*{questions[obs_q][:50]}...*

Partner responded: **{r_obs:.0f}** | Your response: **{_self_response_obs:.1f}**

<span style="color:{_agreement_color}">**{_agreement}**</span> (for egocentric model)
"""),
    ], gap=0.5)

    mo.hstack([
        _controls,
        fig,
    ], widths=[1, 4], gap=1)
    return (fig,)


@app.cell
def __(mo, k, sigma_prior, obs_q, r_obs, L_obs, mu_obs, lambda_mix, self_profile_selector, domain_labels, questions):
    mo.md(f"""
## Parameter Guide

### Bayesian Factor Model Parameters

| Parameter | Current Value | Psychological Meaning |
|-----------|---------------|----------------------|
| **k** (factors) | {k} | **Dimensionality of belief space.** k=1 means all beliefs load on one factor. k=5 captures domain-specific structure (political beliefs cluster separately from religious beliefs). Higher k = more nuanced inferences. |
| **σ_prior** | {sigma_prior:.2f} | **Prior uncertainty about partner.** Larger σ = more uncertain = single observation has bigger effect on posterior. |
| **r_obs** | {r_obs:.0f} | **Partner's observed Likert response** (1-5) on the focal question. |
| **Λ** (loadings) | [{', '.join(f'{x:.2f}' for x in L_obs)}] | **Factor loadings for this question.** Determines direction of the likelihood constraint in θ-space. |
| **μ** (pop. mean) | {mu_obs:.2f} | **Population mean response.** r=5 when μ=3 → partner is above average → shifts θ in positive Λ direction. |

### Model Comparison Parameters

| Parameter | Current Value | Meaning |
|-----------|---------------|---------|
| **λ** (mixture) | {lambda_mix:.1f} | **Model blend.** λ=0 is pure Bayesian (population structure), λ=1 is pure Egocentric (self-structure). |
| **Your Profile** | {self_profile_selector.value} | **Your belief responses** used by the egocentric model. Different profiles show how self-structure affects predictions. |

---

### The Core Insight

**Bayesian model**: Observing agreement on a political question → infer similarity *in political factor space* → transfers to other political questions (same factor) more than religious questions (different factor).

**Egocentric model**: Observing agreement → "they're like me" → project YOUR responses. Transfer depends on how correlated YOUR responses are across domains.

**Human data**: People follow population structure (Bayesian captures 95% of gradient) not self-structure (Egocentric captures 12%).
""")
    return


@app.cell
def __(
    mo,
    loadings_k, means,
    posterior_mean, posterior_cov,
    ego_predictions, r_self,
    obs_q, k, lambda_mix,
    domain_labels, DOMAIN_RANGES,
    np, plt,
):
    # Cyberpunk colors
    _CYBER_BG = '#0D0221'
    _CYBER_GRID = '#1a1a3a'
    _CYBER_MAGENTA = '#FF00FF'
    _CYBER_CYAN = '#00FFFF'
    _CYBER_YELLOW = '#FFFF00'

    # Compute Bayesian predictions for all 35 questions
    _bayes_means = loadings_k @ posterior_mean + means
    _bayes_vars = np.array([loadings_k[i] @ posterior_cov @ loadings_k[i] for i in range(35)])
    _bayes_stds = np.sqrt(_bayes_vars + 1e-10)

    # Mixture predictions
    _mixture_preds = (1 - lambda_mix) * _bayes_means + lambda_mix * r_self

    # Create side-by-side comparison figure
    fig2, _axes = plt.subplots(2, 1, figsize=(14, 7), facecolor=_CYBER_BG)

    _x = np.arange(35)
    _bar_width = 0.35

    # Domain colors
    _domain_color_map = {
        "arbitrary": "#00BFFF", "background": "#FF6600", "identity": "#39FF14",
        "morality": "#FF1493", "politics": "#9D00FF", "preferences": "#FFD700", "religion": "#00FFFF",
    }
    _colors = [_domain_color_map[d] for d in domain_labels]

    # === Top panel: Bayesian Factor Model ===
    _ax1 = _axes[0]
    _ax1.set_facecolor(_CYBER_BG)
    _ax1.bar(_x, _bayes_means, yerr=_bayes_stds, color=_colors, alpha=0.8, capsize=2,
            error_kw={'linewidth': 1, 'alpha': 0.6, 'color': 'white'}, edgecolor='white', linewidth=0.5)
    _ax1.bar([obs_q], [_bayes_means[obs_q]], color=_CYBER_MAGENTA, alpha=1.0, edgecolor='white', linewidth=2)
    _ax1.scatter(_x, means, color='white', s=20, marker='_', linewidths=2, label='Pop. mean', zorder=5)

    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax1.axvline(x=_start - 0.5, color=_CYBER_GRID, linestyle='-', alpha=0.8, linewidth=1)
        _ax1.text((_start + _end) / 2 - 0.5, 0.4, _domain[:3].upper(), ha='center', fontsize=8, color='white', alpha=0.7)

    _ax1.set_ylabel("Predicted Response", color='white', fontsize=12)
    _ax1.set_title(f"BAYESIAN FACTOR MODEL (k={k}) — Uses population covariance structure",
                  fontsize=14, fontweight='bold', color=_CYBER_CYAN)
    _ax1.set_ylim(0, 6)
    _ax1.set_xlim(-0.5, 34.5)
    _ax1.tick_params(colors='white')
    _ax1.set_xticks([])
    _ax1.legend(loc='upper right', fontsize=9, facecolor=_CYBER_BG, edgecolor=_CYBER_GRID, labelcolor='white')
    _ax1.grid(True, color=_CYBER_GRID, alpha=0.3, axis='y')
    for _spine in _ax1.spines.values():
        _spine.set_color(_CYBER_GRID)

    # === Bottom panel: Egocentric Similarity Projection ===
    _ax2 = _axes[1]
    _ax2.set_facecolor(_CYBER_BG)

    # Show self-responses as reference
    _ax2.scatter(_x, r_self, color='white', s=30, marker='o', alpha=0.6, label='Your responses', zorder=3)

    # Show egocentric P(commonality) as bar heights (scaled to 1-5 range for comparison)
    # ego_predictions is P(match), scale it to show transfer pattern
    _ego_scaled = ego_predictions * 4 + 1  # Scale 0-1 to 1-5 range
    _ax2.bar(_x, _ego_scaled, color=_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    _ax2.bar([obs_q], [_ego_scaled[obs_q]], color=_CYBER_YELLOW, alpha=1.0, edgecolor='white', linewidth=2)

    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax2.axvline(x=_start - 0.5, color=_CYBER_GRID, linestyle='-', alpha=0.8, linewidth=1)
        _ax2.text((_start + _end) / 2 - 0.5, 0.4, _domain[:3].upper(), ha='center', fontsize=8, color='white', alpha=0.7)

    _ax2.set_xlabel("Question", color='white', fontsize=12)
    _ax2.set_ylabel("P(commonality) scaled", color='white', fontsize=12)
    _ax2.set_title(f"EGOCENTRIC PROJECTION — Uses YOUR belief structure only",
                  fontsize=14, fontweight='bold', color=_CYBER_YELLOW)
    _ax2.set_ylim(0, 6)
    _ax2.set_xlim(-0.5, 34.5)
    _ax2.tick_params(colors='white')
    _ax2.legend(loc='upper right', fontsize=9, facecolor=_CYBER_BG, edgecolor=_CYBER_GRID, labelcolor='white')
    _ax2.grid(True, color=_CYBER_GRID, alpha=0.3, axis='y')
    for _spine in _ax2.spines.values():
        _spine.set_color(_CYBER_GRID)

    plt.tight_layout()

    mo.vstack([
        mo.md(f"""
## Model Comparison: Predictions for All 35 Questions

**Top (Bayesian)**: Predictions based on population factor structure. Notice domain-specific gradients.

**Bottom (Egocentric)**: Predictions based on YOUR belief structure. Gradients depend on how correlated YOUR responses are.

**λ = {lambda_mix:.1f}** → {"Pure Bayesian" if lambda_mix == 0 else "Pure Egocentric" if lambda_mix == 1 else f"Mixture ({(1-lambda_mix)*100:.0f}% Bayesian, {lambda_mix*100:.0f}% Egocentric)"}
"""),
        fig2,
    ])
    return (fig2,)


@app.cell
def __(mo, questions, domain_labels, obs_q):
    # Target question selector for pairwise prediction
    _target_options = {
        f"Q{i}: [{domain_labels[i][:4].upper()}] {q[:40]}{'...' if len(q) > 40 else ''}": i
        for i, q in enumerate(questions)
    }
    # Default to a question in a different domain than the observed one
    _default_idx = (obs_q + 15) % 35  # Pick something ~15 questions away

    target_question = mo.ui.dropdown(
        options=_target_options,
        value=list(_target_options.keys())[_default_idx],
        label="Target Question (to predict)",
    )
    return (target_question,)


@app.cell
def __(
    mo,
    target_question,
    obs_q, r_obs, lambda_mix,
    r_self, ego_predictions,
    loadings_k, means,
    posterior_mean, posterior_cov,
    questions, domain_labels,
    np, plt, erf,
):
    # Cyberpunk colors
    _CYBER_BG = '#0D0221'
    _CYBER_GRID = '#1a1a3a'
    _CYBER_CYAN = '#00FFFF'
    _CYBER_MAGENTA = '#FF00FF'
    _CYBER_YELLOW = '#FFFF00'
    _CYBER_GREEN = '#39FF14'

    # Get target question index
    target_q = target_question.value

    # === BAYESIAN MODEL ===
    _L_target = loadings_k[target_q]
    _mu_target = means[target_q]
    _bayes_mean = _L_target @ posterior_mean + _mu_target
    _bayes_var = _L_target @ posterior_cov @ _L_target
    _bayes_std = np.sqrt(_bayes_var + 1e-10)

    # Discretized Bayesian probabilities
    _responses = np.array([1, 2, 3, 4, 5])
    _bayes_probs = []
    for _r in _responses:
        _p_upper = 0.5 * (1 + erf((_r + 0.5 - _bayes_mean) / (_bayes_std * np.sqrt(2))))
        _p_lower = 0.5 * (1 + erf((_r - 0.5 - _bayes_mean) / (_bayes_std * np.sqrt(2))))
        _bayes_probs.append(_p_upper - _p_lower)
    _bayes_probs = np.array(_bayes_probs)
    _bayes_probs = _bayes_probs / (_bayes_probs.sum() + 1e-10)

    # === EGOCENTRIC MODEL ===
    # The egocentric model predicts P(commonality), which we can interpret as
    # "probability partner responds similarly to me"
    _ego_p_match = ego_predictions[target_q]
    _self_response = r_self[target_q]

    # Create a simple distribution centered on self-response, scaled by P(match)
    # If P(match) is high, concentrate around self; if low, spread out
    _ego_std = 1.5 * (1 - _ego_p_match) + 0.3  # Lower P(match) -> higher variance
    _ego_probs = []
    for _r in _responses:
        _p_upper = 0.5 * (1 + erf((_r + 0.5 - _self_response) / (_ego_std * np.sqrt(2))))
        _p_lower = 0.5 * (1 + erf((_r - 0.5 - _self_response) / (_ego_std * np.sqrt(2))))
        _ego_probs.append(_p_upper - _p_lower)
    _ego_probs = np.array(_ego_probs)
    _ego_probs = _ego_probs / (_ego_probs.sum() + 1e-10)

    # === MIXTURE ===
    _mix_probs = (1 - lambda_mix) * _bayes_probs + lambda_mix * _ego_probs

    # Create 3-panel figure
    fig3, _axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=_CYBER_BG)

    # === Panel 1: Bayesian ===
    _ax1 = _axes[0]
    _ax1.set_facecolor(_CYBER_BG)
    _bars1 = _ax1.bar(_responses, _bayes_probs, color=_CYBER_CYAN, alpha=0.8, edgecolor='white', linewidth=2)
    _ax1.axvline(x=_bayes_mean, color=_CYBER_GREEN, linewidth=3, linestyle='--', label=f'E[r]={_bayes_mean:.1f}')
    _ax1.set_xlabel("Response", fontsize=12, color='white')
    _ax1.set_ylabel("Probability", fontsize=12, color='white')
    _ax1.set_title("BAYESIAN\n(population structure)", fontsize=14, fontweight='bold', color=_CYBER_CYAN)
    _ax1.set_xticks([1, 2, 3, 4, 5])
    _ax1.set_ylim(0, max(max(_bayes_probs), max(_ego_probs), max(_mix_probs)) * 1.3)
    _ax1.tick_params(colors='white', labelsize=11)
    _ax1.legend(loc='upper right', fontsize=10, facecolor=_CYBER_BG, edgecolor=_CYBER_GRID, labelcolor='white')
    _ax1.grid(True, color=_CYBER_GRID, alpha=0.3, axis='y')
    for _spine in _ax1.spines.values():
        _spine.set_color(_CYBER_GRID)
    for _bar, _p in zip(_bars1, _bayes_probs):
        _ax1.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 0.01,
                 f'{_p:.0%}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

    # === Panel 2: Egocentric ===
    _ax2 = _axes[1]
    _ax2.set_facecolor(_CYBER_BG)
    _bars2 = _ax2.bar(_responses, _ego_probs, color=_CYBER_YELLOW, alpha=0.8, edgecolor='white', linewidth=2)
    _ax2.axvline(x=_self_response, color=_CYBER_GREEN, linewidth=3, linestyle='--', label=f'Your r={_self_response:.1f}')
    _ax2.set_xlabel("Response", fontsize=12, color='white')
    _ax2.set_title("EGOCENTRIC\n(your belief structure)", fontsize=14, fontweight='bold', color=_CYBER_YELLOW)
    _ax2.set_xticks([1, 2, 3, 4, 5])
    _ax2.set_ylim(0, max(max(_bayes_probs), max(_ego_probs), max(_mix_probs)) * 1.3)
    _ax2.tick_params(colors='white', labelsize=11)
    _ax2.legend(loc='upper right', fontsize=10, facecolor=_CYBER_BG, edgecolor=_CYBER_GRID, labelcolor='white')
    _ax2.grid(True, color=_CYBER_GRID, alpha=0.3, axis='y')
    for _spine in _ax2.spines.values():
        _spine.set_color(_CYBER_GRID)
    for _bar, _p in zip(_bars2, _ego_probs):
        _ax2.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 0.01,
                 f'{_p:.0%}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

    # === Panel 3: Mixture ===
    _ax3 = _axes[2]
    _ax3.set_facecolor(_CYBER_BG)
    _bars3 = _ax3.bar(_responses, _mix_probs, color=_CYBER_MAGENTA, alpha=0.8, edgecolor='white', linewidth=2)
    _ax3.set_xlabel("Response", fontsize=12, color='white')
    _ax3.set_title(f"MIXTURE (λ={lambda_mix:.1f})\n({(1-lambda_mix)*100:.0f}% Bayes + {lambda_mix*100:.0f}% Ego)", fontsize=14, fontweight='bold', color=_CYBER_MAGENTA)
    _ax3.set_xticks([1, 2, 3, 4, 5])
    _ax3.set_ylim(0, max(max(_bayes_probs), max(_ego_probs), max(_mix_probs)) * 1.3)
    _ax3.tick_params(colors='white', labelsize=11)
    _ax3.grid(True, color=_CYBER_GRID, alpha=0.3, axis='y')
    for _spine in _ax3.spines.values():
        _spine.set_color(_CYBER_GRID)
    for _bar, _p in zip(_bars3, _mix_probs):
        _ax3.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 0.01,
                 f'{_p:.0%}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Domain info
    _obs_domain = domain_labels[obs_q]
    _target_domain = domain_labels[target_q]
    _same_domain = _obs_domain == _target_domain
    _domain_note = "**Same domain** — stronger transfer for Bayesian!" if _same_domain else "**Different domain** — weaker transfer for Bayesian"

    mo.vstack([
        mo.md(f"""
## Pairwise Prediction: Q{obs_q} → Q{target_q}

**Observed**: Q{obs_q} ({domain_labels[obs_q]}): *"{questions[obs_q][:45]}..."* → Partner responded **{r_obs:.0f}**

**Predicting**: Q{target_q} ({domain_labels[target_q]}): *"{questions[target_q][:45]}..."*

{_domain_note} | Your response on target: **{_self_response:.1f}** | P(match) from egocentric: **{_ego_p_match:.1%}**
"""),
        mo.hstack([target_question], justify="start"),
        fig3,
    ])
    return (fig3, target_q)


@app.cell
def __(mo):
    mo.md("""
---
## What is θ-space?

The Bayesian model doesn't directly model Likert responses (1-5). Instead, it models a **latent factor space** θ:

```
θ ∈ ℝᵏ  (k-dimensional vector representing partner's "belief position")

response_q = Λ_q' θ + μ_q
             ───────   ────
             factor    population
             loading   mean
```

**θ is not observable** — we infer it from responses. Each dimension captures a latent factor
(e.g., "political conservatism", "religiosity") extracted from population covariance.

---
## Why ellipses and lines (not KDEs)?

The θ-space plots show **analytical Gaussian distributions**, not empirical density estimates:

| Panel | What it shows | Why that shape |
|-------|---------------|----------------|
| **Prior** | P(θ) before observation | Spherical Gaussian in θ-space → **circle/ellipse** |
| **Likelihood** | Which θ values produce r_obs | Constraint Λ'θ + μ = r is a hyperplane → **line** in 2D |
| **Posterior** | P(θ ∣ r_obs) ∝ prior × likelihood | Prior "sliced" by the line → **compressed ellipse** |

---
## How to use this demo

### Explore Bayesian inference:
1. **Change k** — More factors = richer structure (k=1 is 1D, k≥2 shows ellipses)
2. **Change the response** — Watch the posterior shift along the loading direction (orange arrow Λ)
3. **Change the question** — Different questions have different Λ vectors → different constraint angles
4. **Change σ_prior** — Wider prior = observation has stronger relative pull on the posterior

### Compare the models:
5. **Change λ** — Slide from pure Bayesian (λ=0) to pure Egocentric (λ=1)
6. **Change "Your Belief Profile"** — See how egocentric predictions depend on your own responses
7. **Try same vs. different domain targets** — Bayesian shows domain-specific transfer; Egocentric depends on your self-structure

### Key insight:
The Bayesian model produces systematic domain gradients using *population* structure.
The Egocentric model produces gradients (if any) from *your own* belief structure.
Humans follow population structure (95% fit) not self-structure (12% fit).
    """)
    return


if __name__ == "__main__":
    app.run()
