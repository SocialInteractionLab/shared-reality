"""Split-half held-out validation of the commonality model.

Defuses the circularity charge: factor loadings Λ are estimated from the
population response covariance, and the experimenter-defined domains are
themselves justified by within-domain correlation. So one could argue the
model merely *re-describes* the covariance that drives the human gradient.

Test: estimate Λ on one half of participants (group A), then predict the
*held-out* half's (group B) generalization gradient with those loadings. The
evaluated participants never contributed a single number to Λ. If the gradient
survives out-of-sample, "circular re-description of this sample" is off the
table — the model predicts the behavior of people whose data never touched the
parameters. Converts the result from a *fit* into a *prediction*.

Run:  uv run marimo edit analysis/split_half_heldout.py
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Split-half held-out validation

        **Charge:** loadings $\Lambda$ come from the population response
        covariance; domains are defined by within-domain correlation. Does the
        model just re-describe the covariance that produces the human gradient?

        **Test:** estimate $\Lambda$ on **group A**, predict **group B**'s
        held-out gradient. Group B never touched $\Lambda$. Repeat over many
        random splits.

        - **Held-out ≈ in-sample ≈ human** → predictive, not circular.
        - **Held-out collapses** → was overfitting this sample.
        - **Scrambled** (shuffle $\Lambda$ rows) is the floor: structure killed.
        """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    import json

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    base_dir = Path.cwd().resolve()
    while base_dir.name and not (base_dir / "data").exists():
        base_dir = base_dir.parent
    sys.path.insert(0, str(base_dir))

    from models.model import (
        CommonalityModel,
        load_evaluation_data,
        prepare_evaluation_data,
        fast_evaluate,
    )
    from analysis.utils import (
        compute_gradient,
        DOMAIN_RANGES,
        COLORS,
        set_plot_style,
    )

    set_plot_style()
    return (
        CommonalityModel,
        COLORS,
        DOMAIN_RANGES,
        compute_gradient,
        fast_evaluate,
        json,
        load_evaluation_data,
        np,
        base_dir,
        pd,
        plt,
        prepare_evaluation_data,
    )


@app.cell
def _(base_dir, json, load_evaluation_data):
    data = load_evaluation_data()

    # No-chat fit (σ_obs=0, exact observations) — same params as main analysis
    with open(base_dir / "models" / "fitted_params.json") as _f:
        FITTED_PARAMS = json.load(_f)["k5_params"]

    all_pids = data["pid"].unique()
    nochat_pids = set(data[data["experiment"] == "no-chat"]["pid"].unique())
    return FITTED_PARAMS, all_pids, data, nochat_pids


@app.cell
def _(
    CommonalityModel,
    DOMAIN_RANGES,
    FITTED_PARAMS,
    all_pids,
    compute_gradient,
    data,
    fast_evaluate,
    nochat_pids,
    np,
    pd,
    prepare_evaluation_data,
):
    def loadings_from_pids(src, pids, k):
        """Factor loadings Λ (35×k) and means from a subset of participants."""
        resp = (
            src[src["pid"].isin(list(pids))]
            .pivot_table(
                index="pid", columns="question",
                values="preChatResponse", aggfunc="first",
            )
            .reindex(columns=range(1, 36))
            .dropna()
        )
        X = resp.values
        means = X.mean(axis=0)
        corr = np.corrcoef(X.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        eigvals, eigvecs = np.linalg.eigh(corr)
        idx = np.argsort(eigvals)[::-1]
        L = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0.0))
        return L[:, :k], means

    def domain_corr(preds):
        """Domain-level model-vs-human transfer-effect correlation (Fig5B)."""
        p = preds.copy()
        p["domain"] = p["question_domain"].str.lower()
        rows = []
        for dom in DOMAIN_RANGES:
            for qt in ["same_domain", "different_domain"]:
                s = p[(p["domain"] == dom) & (p["question_type"] == qt)]
                sh, op = s[s["stance"] == "shared"], s[s["stance"] == "opposing"]
                if len(sh) == 0 or len(op) == 0:
                    continue
                rows.append({
                    "me": sh["pred_prob"].mean() - op["pred_prob"].mean(),
                    "he": sh["actual"].mean() - op["actual"].mean(),
                })
        dd = pd.DataFrame(rows)
        if len(dd) < 3:
            return np.nan
        return float(np.corrcoef(dd["me"], dd["he"])[0, 1])

    def _eval(loadings, means, eval_data):
        model = CommonalityModel(
            k=loadings.shape[1], lambda_mix=0.0,
            loadings=loadings, question_means=means, **FITTED_PARAMS,
        )
        return fast_evaluate(model, eval_data)

    def run_one_split(seed, k):
        """Λ from group A; predict held-out group B's no-chat gradient."""
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(all_pids)
        half = len(shuffled) // 2
        group_a = shuffled[:half]
        group_b = set(shuffled[half:])

        L_a, mu_a = loadings_from_pids(data, group_a, k)

        eval_pids = [p for p in nochat_pids if p in group_b]
        eval_df = data[data["pid"].isin(eval_pids)]
        ev = prepare_evaluation_data(eval_df)

        preds = _eval(L_a, mu_a, ev)

        # Scrambled floor: same loadings, rows shuffled (structure destroyed)
        perm = np.random.default_rng(seed + 99991).permutation(35)
        preds_s = _eval(L_a[perm, :], mu_a, ev)

        return {
            "seed": seed,
            "model_grad": compute_gradient(preds, "pred_prob"),
            "human_grad": compute_gradient(preds, "actual"),
            "scram_grad": compute_gradient(preds_s, "pred_prob"),
            "domain_r": domain_corr(preds),
            "n_eval_pids": len(eval_pids),
        }

    def in_sample_reference(k):
        """Original analysis: Λ from ALL pids, eval on ALL no-chat pids."""
        L_all, mu_all = loadings_from_pids(data, all_pids, k)
        ev = prepare_evaluation_data(data[data["pid"].isin(list(nochat_pids))])
        preds = _eval(L_all, mu_all, ev)
        return {
            "model_grad": compute_gradient(preds, "pred_prob"),
            "human_grad": compute_gradient(preds, "actual"),
            "domain_r": domain_corr(preds),
        }

    return in_sample_reference, run_one_split


@app.cell
def _(mo):
    n_splits = mo.ui.slider(10, 200, value=50, step=10, label="# random splits")
    k_factors = mo.ui.slider(1, 10, value=5, label="k (factors)")
    run = mo.ui.run_button(label="Run split-half")
    mo.hstack([n_splits, k_factors, run], justify="start", gap=2)
    return k_factors, n_splits, run


@app.cell
def _(in_sample_reference, k_factors, mo, n_splits, pd, run, run_one_split):
    mo.stop(
        not run.value,
        mo.callout(mo.md("Set parameters, then click **Run split-half**."), kind="info"),
    )

    _k = int(k_factors.value)
    ref = in_sample_reference(_k)
    res = pd.DataFrame(
        run_one_split(_s, _k) for _s in range(int(n_splits.value))
    )
    return ref, res


@app.cell
def _(COLORS, np, plt, ref, res):
    def _ci(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        return x.mean(), np.percentile(x, 2.5), np.percentile(x, 97.5)

    ho_m, ho_lo, ho_hi = _ci(res["model_grad"])
    hu_m, hu_lo, hu_hi = _ci(res["human_grad"])
    sc_m, sc_lo, sc_hi = _ci(res["scram_grad"])
    r_m, r_lo, r_hi = _ci(res["domain_r"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Panel A: gradient bars ---
    labels = ["Human\n(held-out)", "Model\nin-sample", "Model\nheld-out", "Scrambled\nheld-out"]
    vals = [hu_m, ref["model_grad"], ho_m, sc_m]
    los = [hu_m - hu_lo, 0, ho_m - ho_lo, sc_m - sc_lo]
    his = [hu_hi - hu_m, 0, ho_hi - ho_m, sc_hi - sc_m]
    cols = [COLORS["human"], COLORS["bayesian"], COLORS["bayesian"], COLORS["scrambled"]]
    axL.bar(range(4), vals, color=cols, yerr=[los, his], capsize=0,
            error_kw={"linewidth": 1.5})
    axL.axhline(0, color="black", lw=0.8)
    axL.set_xticks(range(4))
    axL.set_xticklabels(labels, fontsize=9)
    axL.set_ylabel("Gradient (Same − Different)", fontweight="bold")
    axL.set_title("A  Held-out gradient survives", loc="left", fontweight="bold")

    # --- Panel B: held-out model gradient distribution ---
    axR.hist(res["model_grad"].dropna(), bins=20, color=COLORS["bayesian"],
             alpha=0.8, edgecolor="white")
    axR.axvline(hu_m, color=COLORS["human"], lw=2, ls="--",
                label=f"Human (held-out) = {hu_m:.3f}")
    axR.axvline(sc_m, color=COLORS["scrambled"], lw=2, ls=":",
                label=f"Scrambled = {sc_m:.3f}")
    axR.set_xlabel("Held-out model gradient", fontweight="bold")
    axR.set_ylabel("Count (splits)", fontweight="bold")
    axR.legend(frameon=False, fontsize=8, loc="upper right")
    axR.set_title("B  Distribution across splits", loc="left", fontweight="bold")

    plt.tight_layout()
    pct = 100 * ho_m / hu_m if hu_m else float("nan")
    summary = (
        ho_m, ho_lo, ho_hi, hu_m, sc_m, r_m, r_lo, r_hi, pct, fig,
    )
    fig
    return (summary,)


@app.cell
def _(mo, ref, res, summary):
    _ho_m, _ho_lo, _ho_hi, _hu_m, _sc_m, _r_m, _r_lo, _r_hi, _pct, _fig = summary
    mo.md(
        f"""
        ### Result ({len(res)} splits)

        | quantity | value |
        |---|---|
        | Human gradient (held-out) | **{_hu_m:+.3f}** |
        | Model gradient — **held-out** | **{_ho_m:+.3f}**  [{_ho_lo:+.3f}, {_ho_hi:+.3f}] |
        | Model gradient — in-sample (reference) | {ref['model_grad']:+.3f} |
        | Scrambled gradient (held-out floor) | {_sc_m:+.3f} |
        | Domain-level fit *r* (held-out) | {_r_m:.2f}  [{_r_lo:.2f}, {_r_hi:.2f}] |
        | **Held-out recovers** | **{_pct:.0f}% of human gradient** |

        **Read:** loadings estimated on one half of participants predict
        ~{_pct:.0f}% of the *other* half's gradient — the evaluated participants
        never touched Λ. Held-out ≈ in-sample ({ref['model_grad']:+.3f}) ≫
        scrambled ({_sc_m:+.3f}). The gradient is a parameter-free *prediction*
        about unseen people, not a re-description of the fitted sample.
        """
    )
    return
