"""
Talk figure (Ch 1 model comparison, stripped to the conclusion):
Overlay three gradients across topic-distance, paneled by stance —
  (1) what people actually did  (observed / human)
  (2) shared map of how beliefs relate  (population-structure / Bayesian)
  (3) assume others are like them  (self-projection / egocentric)
Point: observed lands on the shared-map prediction, not the projection prediction.

Run from repo root:  uv run python analysis/talk_model_overlay.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.model import (
    CommonalityModel,
    load_evaluation_data,
    prepare_evaluation_data,
    fast_evaluate,
)
from analysis.utils import get_rates, set_plot_style, COLORS

FITTED = json.load(open(REPO / "models" / "fitted_params.json"))["k5_params"]

data = load_evaluation_data()
nochat = data[data["experiment"] == "no-chat"].copy()
ev = prepare_evaluation_data(nochat)

# population-structure (Bayesian) and self-projection (egocentric) predictions
bayes_df = fast_evaluate(CommonalityModel(k=5, lambda_mix=0.0, **FITTED), ev)
proj_df = fast_evaluate(
    CommonalityModel(k=0, lambda_mix=1.0, **FITTED, base_rate=0.2, projection_weight=0.8),
    ev,
)

observed = get_rates(bayes_df, "actual")       # human (same trials)
bayes = get_rates(bayes_df, "pred_prob")        # shared map of how beliefs relate
proj = get_rates(proj_df, "pred_prob")          # assume others are like them

# quick sanity print
for name, r in [("observed", observed), ("bayes", bayes), ("proj", proj)]:
    print(name, {k: round(v, 3) for k, v in r.items()})

# ---- bootstrap CIs on the agreement boost (resample participants) ----
QT = ["observed", "same_domain", "different_domain"]
CELLS = [(q, s) for q in QT for s in ("shared", "opposing")]
CELL_IX = {c: i for i, c in enumerate(CELLS)}


def per_pid_sumcount(df, col):
    """pid -> (sum[6], count[6]) over the six (question_type, stance) cells."""
    out = {}
    for pid, g in df.groupby("pid"):
        s = np.zeros(6)
        c = np.zeros(6)
        key = list(zip(g["question_type"], g["stance"]))
        vals = g[col].to_numpy()
        for (q, st), v in zip(key, vals):
            if (q, st) in CELL_IX:
                i = CELL_IX[(q, st)]
                s[i] += v
                c[i] += 1
        out[pid] = (s, c)
    return out


def boost_from(sums, counts):
    means = np.divide(sums, counts, out=np.full(6, np.nan), where=counts > 0)
    return np.array([means[CELL_IX[(q, "shared")]] - means[CELL_IX[(q, "opposing")]] for q in QT])


def bootstrap_boost_ci(series_sc, pids, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pid_arr = np.array(pids)
    boots = np.empty((n, 3))
    for b in range(n):
        samp = rng.choice(pid_arr, size=len(pid_arr), replace=True)
        s = np.zeros(6)
        c = np.zeros(6)
        for p in samp:
            ss, cc = series_sc[p]
            s += ss
            c += cc
        boots[b] = boost_from(s, c)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return lo, hi


pids = bayes_df["pid"].unique().tolist()
sc_obs = per_pid_sumcount(bayes_df, "actual")
sc_bay = per_pid_sumcount(bayes_df, "pred_prob")
sc_pro = per_pid_sumcount(proj_df, "pred_prob")
CI = {
    "observed": bootstrap_boost_ci(sc_obs, pids),
    "bayes": bootstrap_boost_ci(sc_bay, pids),
    "proj": bootstrap_boost_ci(sc_pro, pids),
}

set_plot_style()

QT_LABELS = ["focal\ntopic", "same\ndomain", "different\ndomain"]
x = np.arange(3)


def boost(rates):
    """Agreement boost = expected commonality after agreement minus after disagreement."""
    return [rates[(q, "shared")] - rates[(q, "opposing")] for q in QT]


SERIES = [
    ("human data", "observed", observed, COLORS["human"], "-", 3.2, "o"),
    ("use yourself as the template", "proj", proj, COLORS["egocentric"], "--", 2.2, "^"),
    ("uses how beliefs cluster across people", "bayes", bayes, COLORS["bayesian"], "-", 2.2, "s"),
]

out = REPO / "outputs"
out.mkdir(exist_ok=True)


def render(show_keys, stem):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for label, key, rates, color, ls, lw, mk in SERIES:
        if key not in show_keys:
            continue
        lo, hi = CI[key]
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0, zorder=1)
        ax.plot(x, boost(rates), color=color, ls=ls, lw=lw, marker=mk, ms=7,
                label=label, zorder=3 if key == "observed" else 2)
    ax.set_xticks(x)
    ax.set_xticklabels(QT_LABELS)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    # legend outside right, TOP-anchored so the first entry never moves as lines are added
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.03, 0.65))
    fig.subplots_adjust(left=0.13, right=0.62, top=0.94, bottom=0.14)
    # ylabel drawn manually so "shared"/"opposite" carry the study's stance
    # colors; two rotated lines, second composed of colored segments laid
    # end-to-end (measured in display space after layout is final)
    fig.canvas.draw()
    ax_bb = ax.get_window_extent()
    cy = (ax_bb.y0 + ax_bb.y1) / 2
    inv = fig.transFigure.inverted()
    fig.text(0.033, inv.transform((0, cy))[1], "expected commonality",
             rotation=90, va="center", ha="center", fontweight="bold",
             fontsize=12)
    segs = [("(", "black"), ("agreed", COLORS["shared"]), (" − ", "black"),
            ("disagreed", COLORS["opposing"]), (")", "black")]
    tmp = [fig.text(0.5, 0.5, s, rotation=90, fontweight="bold", fontsize=12)
           for s, _ in segs]
    fig.canvas.draw()
    heights = [t.get_window_extent().height for t in tmp]
    for t in tmp:
        t.remove()
    y_disp = cy - sum(heights) / 2
    for (s, c), h in zip(segs, heights):
        fig.text(0.063, inv.transform((0, y_disp))[1], s, rotation=90,
                 va="bottom", ha="center", color=c, fontweight="bold",
                 fontsize=12)
        y_disp += h
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)  # no bbox_inches='tight' -> identical canvas
    plt.close(fig)
    print("saved:", out / f"{stem}.png")


# build-up frames (ego-first: set up the wrong guess, then reveal the fit)
render(["observed"], "talk_model_overlay_step0")                 # human data only
render(["observed", "proj"], "talk_model_overlay_step1")         # + assume others are like them (the miss)
render(["observed", "proj", "bayes"], "talk_model_overlay")      # + uses belief structure (the fit / final)
