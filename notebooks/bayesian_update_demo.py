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
        load_factor_loadings,
        load_question_means,
        load_responses,
        DOMAIN_RANGES,
    )


@app.cell
def __(load_factor_loadings, load_question_means, load_responses, DOMAIN_RANGES):
    # Load all factor loadings (up to k=10 for demo)
    all_loadings = load_factor_loadings(k=10)
    means = load_question_means()
    responses_df = load_responses()

    # Question labels with domains
    questions = list(responses_df.columns)

    # Create domain labels
    domain_labels = []
    for _i in range(35):
        for _domain, (_start, _end) in DOMAIN_RANGES.items():
            if _start <= _i < _end:
                domain_labels.append(_domain)
                break

    return all_loadings, means, questions, domain_labels


@app.cell
def __(mo, questions, domain_labels):
    # Question dropdown with domain labels
    _question_options = {
        f"Q{i}: [{domain_labels[i][:4].upper()}] {q[:45]}{'...' if len(q) > 45 else ''}": i
        for i, q in enumerate(questions)
    }

    question_selector = mo.ui.dropdown(
        options=_question_options,
        value=list(_question_options.keys())[30],  # Start with a religion question
        label="Question",
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

    return question_selector, response_slider, k_slider, sigma_prior_slider


@app.cell
def __(
    question_selector,
    response_slider,
    k_slider,
    sigma_prior_slider,
    all_loadings,
    means,
    np,
):
    # Extract current values (dropdown returns the value directly, which is the index)
    obs_q = question_selector.value
    r_obs = float(response_slider.value)
    k = int(k_slider.value)
    sigma_prior = float(sigma_prior_slider.value)

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

    return (
        obs_q, r_obs, k, sigma_prior,
        loadings_k, L_obs, mu_obs,
        prior_mean, prior_cov,
        posterior_mean, posterior_cov,
    )


@app.cell
def __(
    mo,
    question_selector,
    response_slider,
    k_slider,
    sigma_prior_slider,
    obs_q, r_obs, k, sigma_prior,
    L_obs, mu_obs,
    prior_mean, prior_cov,
    posterior_mean, posterior_cov,
    questions, domain_labels,
    np, plt, Ellipse, transforms,
):
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

    # Create the 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    _plot_range = 4.0

    # === Panel 1: PRIOR ===
    _ax1 = axes[0]
    _ax1.set_title("PRIOR", fontsize=13, fontweight='bold', color='blue')
    if k >= 2:
        _confidence_ellipse(prior_mean, prior_cov, _ax1, n_std=1, edgecolor='blue', linewidth=2, linestyle='--')
        _confidence_ellipse(prior_mean, prior_cov, _ax1, n_std=2, edgecolor='blue', linewidth=1, linestyle=':')
        _ax1.scatter([0], [0], color='blue', s=100, marker='+', linewidths=2, zorder=5)
        _ax1.set_xlabel("θ₁"); _ax1.set_ylabel("θ₂")
    else:
        _x = np.linspace(-_plot_range, _plot_range, 200)
        _y = np.exp(-0.5 * (_x - prior_mean[0])**2 / prior_cov[0, 0])
        _ax1.fill_between(_x, _y, alpha=0.3, color='blue')
        _ax1.plot(_x, _y, color='blue', linewidth=2)
        _ax1.set_xlabel("θ₁"); _ax1.set_ylabel("Density")
    _ax1.set_xlim(-_plot_range, _plot_range)
    _ax1.set_ylim(-_plot_range, _plot_range) if k >= 2 else None
    _ax1.set_aspect('equal' if k >= 2 else 'auto')
    _ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    _ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    _ax1.text(0.05, 0.95, f"θ ~ N(0, {sigma_prior:.1f}²·I)", transform=_ax1.transAxes,
             fontsize=10, va='top', family='monospace', bbox=dict(boxstyle='round', fc='lightblue', alpha=0.8))

    # === Panel 2: LIKELIHOOD ===
    _ax2 = axes[1]
    _ax2.set_title("LIKELIHOOD", fontsize=13, fontweight='bold', color='green')
    _constraint_val = r_obs - mu_obs
    if k >= 2:
        if abs(L_obs[1]) > 1e-6:
            _theta1 = np.linspace(-_plot_range, _plot_range, 100)
            _theta2 = (_constraint_val - L_obs[0] * _theta1) / L_obs[1]
            _valid = np.abs(_theta2) < _plot_range * 1.5
            _ax2.plot(_theta1[_valid], _theta2[_valid], 'g-', linewidth=3)
        elif abs(L_obs[0]) > 1e-6:
            _ax2.axvline(x=_constraint_val / L_obs[0], color='green', linewidth=3)
        _arrow_scale = 1.5
        _ax2.arrow(0, 0, L_obs[0]*_arrow_scale, L_obs[1]*_arrow_scale,
                  head_width=0.15, head_length=0.1, fc='darkgreen', ec='darkgreen', lw=2, zorder=5)
        _ax2.set_xlabel("θ₁"); _ax2.set_ylabel("θ₂")
    else:
        _theta_constrained = _constraint_val / (L_obs[0] + 1e-10)
        _ax2.axvline(x=_theta_constrained, color='green', linewidth=3)
        _ax2.scatter([_theta_constrained], [0.5], color='green', s=200, marker='|', linewidths=3)
        _ax2.set_xlabel("θ₁"); _ax2.set_ylabel("")
        _ax2.set_ylim(0, 1)
    _ax2.set_xlim(-_plot_range, _plot_range)
    if k >= 2: _ax2.set_ylim(-_plot_range, _plot_range)
    _ax2.set_aspect('equal' if k >= 2 else 'auto')
    _ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    _ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    _ax2.text(0.05, 0.95, f"r={r_obs:.0f}, μ={mu_obs:.1f}\nΛ=[{', '.join(f'{x:.2f}' for x in L_obs)}]",
             transform=_ax2.transAxes, fontsize=9, va='top', family='monospace',
             bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.8))

    # === Panel 3: POSTERIOR ===
    _ax3 = axes[2]
    _ax3.set_title("POSTERIOR", fontsize=13, fontweight='bold', color='red')
    if k >= 2:
        _confidence_ellipse(posterior_mean, posterior_cov, _ax3, n_std=1, edgecolor='red', linewidth=2)
        _confidence_ellipse(posterior_mean, posterior_cov, _ax3, n_std=2, edgecolor='red', linewidth=1, linestyle=':')
        _confidence_ellipse(prior_mean, prior_cov, _ax3, n_std=1, edgecolor='blue', linewidth=1, linestyle='--', alpha=0.3)
        _ax3.scatter([0], [0], color='blue', s=60, marker='+', linewidths=1, alpha=0.4)
        _ax3.scatter([posterior_mean[0]], [posterior_mean[1]], color='red', s=100, marker='x', linewidths=2, zorder=5)
        _ax3.annotate('', xy=(posterior_mean[0], posterior_mean[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='purple', lw=2))
        _ax3.set_xlabel("θ₁"); _ax3.set_ylabel("θ₂")
    else:
        _x = np.linspace(-_plot_range, _plot_range, 200)
        _y_prior = np.exp(-0.5 * (_x - prior_mean[0])**2 / prior_cov[0, 0])
        _y_post = np.exp(-0.5 * (_x - posterior_mean[0])**2 / (posterior_cov[0, 0] + 1e-10))
        _ax3.fill_between(_x, _y_prior, alpha=0.2, color='blue')
        _ax3.fill_between(_x, _y_post, alpha=0.4, color='red')
        _ax3.plot(_x, _y_post, color='red', linewidth=2)
        _ax3.axvline(x=posterior_mean[0], color='red', linestyle='-', alpha=0.7)
        _ax3.set_xlabel("θ₁"); _ax3.set_ylabel("Density")
    _ax3.set_xlim(-_plot_range, _plot_range)
    if k >= 2: _ax3.set_ylim(-_plot_range, _plot_range)
    _ax3.set_aspect('equal' if k >= 2 else 'auto')
    _ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    _ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    _ax3.text(0.05, 0.95, f"μ_post=[{', '.join(f'{x:.2f}' for x in posterior_mean[:min(3,k)])}]",
             transform=_ax3.transAxes, fontsize=10, va='top', family='monospace',
             bbox=dict(boxstyle='round', fc='lightcoral', alpha=0.8))

    plt.tight_layout()

    # Build the layout: controls on left, plot on right
    _controls = mo.vstack([
        mo.md("### Controls"),
        question_selector,
        mo.hstack([response_slider, k_slider], justify="start"),
        sigma_prior_slider,
        mo.md(f"""
---
**Q{obs_q}** ({domain_labels[obs_q]}):
*{questions[obs_q][:60]}...*
"""),
    ], gap=0.5)

    mo.hstack([
        _controls,
        fig,
    ], widths=[1, 4], gap=1)
    return (fig,)


@app.cell
def __(
    mo,
    loadings_k, means,
    posterior_mean, posterior_cov,
    obs_q, k,
    domain_labels, DOMAIN_RANGES,
    np, plt,
):
    # Compute predictions for all 35 questions
    _pred_means = loadings_k @ posterior_mean + means
    _pred_vars = np.array([loadings_k[i] @ posterior_cov @ loadings_k[i] for i in range(35)])
    _pred_stds = np.sqrt(_pred_vars + 1e-10)

    fig2, _ax = plt.subplots(figsize=(14, 3.5))

    _x = np.arange(35)
    _domain_color_map = {
        "arbitrary": "#1f77b4", "background": "#ff7f0e", "identity": "#2ca02c",
        "morality": "#d62728", "politics": "#9467bd", "preferences": "#8c564b", "religion": "#e377c2",
    }
    _colors = [_domain_color_map[d] for d in domain_labels]

    _ax.bar(_x, _pred_means, yerr=_pred_stds, color=_colors, alpha=0.7, capsize=2,
           error_kw={'linewidth': 1, 'alpha': 0.5})
    _ax.bar([obs_q], [_pred_means[obs_q]], color='black', alpha=0.9)
    _ax.scatter(_x, means, color='black', s=15, marker='_', linewidths=1.5, label='Pop. mean', zorder=5)

    for _domain, (_start, _end) in sorted(DOMAIN_RANGES.items(), key=lambda x: x[1][0]):
        if _start > 0:
            _ax.axvline(x=_start - 0.5, color='gray', linestyle=':', alpha=0.4)
        _ax.text((_start + _end) / 2 - 0.5, 0.3, _domain[:3].upper(), ha='center', fontsize=8, alpha=0.6)

    _ax.set_xlabel("Question"); _ax.set_ylabel("Predicted Response")
    _ax.set_title(f"Posterior Predictions (k={k}) — Black bar = observed question", fontsize=11)
    _ax.set_ylim(0, 6)
    _ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    mo.vstack([
        fig2,
        mo.md("**Bars** = predicted partner response | **Error bars** = posterior uncertainty | **Black marks** = population mean"),
    ])
    return (fig2,)


@app.cell
def __(mo):
    mo.md("""
    ---
    ## How to use this demo

    1. **Change k** — See how more factors create richer structure (k=1 is just a single dimension, k=5 captures domain structure)
    2. **Change the response** — Watch the posterior shift along the loading direction
    3. **Change the question** — Different questions have different factor loadings → different constraint directions
    4. **Change σ_prior** — Wider prior = observation has stronger relative effect

    **Key insight**: The posterior mean moves along the factor loading direction. Questions with similar loadings will show correlated predictions.
    """)
    return


if __name__ == "__main__":
    app.run()
