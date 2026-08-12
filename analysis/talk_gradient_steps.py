"""
Talk figure (Ch 1), step-frame version of talk_gradient_levels for the slide
build. Three pixel-aligned frames (same axes, same legend geometry):

    step0_chat  -- chat condition only (the phenomenon: converging gradient)
    step1_blue  -- chat only, shared line faded (disagreement reversal slide)
    step2_full  -- chat + no-chat dashed (what does talking add?)

Run from repo root:  uv run python analysis/talk_gradient_steps.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.model import load_evaluation_data
from analysis.utils import get_rates, set_plot_style, COLORS

OUTCOME = "participant_binary_prediction"
QT = ["observed", "same_domain", "different_domain"]
QT_LABELS = ["focal\ntopic", "same\ndomain", "different\ndomain"]
CELLS = [(q, s) for q in QT for s in ("shared", "opposing")]
CELL_IX = {c: i for i, c in enumerate(CELLS)}

data = load_evaluation_data()
rates_nochat = get_rates(data[data["experiment"] == "no-chat"].copy(), OUTCOME)
rates_chat = get_rates(data[data["experiment"] == "chat"].copy(), OUTCOME)


def bootstrap_ci(df, n=2000, seed=0):
    """95% CI per (question_type, stance) cell, resampling participants.
    Vectorized: per-pid sum/count matrices, then bootstrap index draws."""
    pids = df["pid"].unique()
    s_mat = np.zeros((len(pids), 6))
    c_mat = np.zeros((len(pids), 6))
    pid_row = {p: i for i, p in enumerate(pids)}
    for (pid, q, st), v in zip(
        zip(df["pid"], df["question_type"], df["stance"]),
        df[OUTCOME].to_numpy(),
    ):
        if (q, st) in CELL_IX:
            r, i = pid_row[pid], CELL_IX[(q, st)]
            s_mat[r, i] += v
            c_mat[r, i] += 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pids), size=(n, len(pids)))
    s_b = s_mat[idx].sum(axis=1)
    c_b = c_mat[idx].sum(axis=1)
    boots = np.divide(s_b, c_b, out=np.full_like(s_b, np.nan), where=c_b > 0)
    return np.nanpercentile(boots, [2.5, 97.5], axis=0)


ci_nochat = bootstrap_ci(data[data["experiment"] == "no-chat"])
ci_chat = bootstrap_ci(data[data["experiment"] == "chat"])

set_plot_style()
x = np.arange(3)

STANCES = [
    ("shared", COLORS["shared"], "o"),
    ("opposing", COLORS["opposing"], "s"),
]

# (name, show_nochat, shared_alpha, opposing_alpha, gap_frame)
# gap_frame: chat lines recede, no-chat pair comes forward with the gap
# between them shaded — transition frame into the model-comparison figure
# (which plots exactly that gap as one line, on an identical canvas)
FRAMES = [
    ("step0_pink", False, 1.0, 0.0, False),
    ("step1_both", False, 1.0, 1.0, False),
    ("step2_blue", False, 0.25, 1.0, False),
    ("step3_full", True, 1.0, 1.0, False),
    ("step4_gap", True, 1.0, 1.0, True),
]

out = REPO / "outputs"
out.mkdir(exist_ok=True)

for name, show_nochat, a_shared, a_opp, gap_frame in FRAMES:
    # canvas geometry matches analysis/talk_model_overlay.py exactly so the
    # slide transition from step4_gap into the model figure keeps the axes
    # at identical pixels (no bbox_inches='tight' anywhere)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    stance_alpha = {"shared": a_shared, "opposing": a_opp}

    def draw(rates, ci, ls, lw, base_alpha, zorder):
        for stance, color, mk in STANCES:
            a = base_alpha * stance_alpha[stance]
            y = np.array([rates[(q, stance)] for q in QT])
            ix = [CELL_IX[(q, stance)] for q in QT]
            ax.fill_between(x, ci[0, ix], ci[1, ix], color=color,
                            alpha=0.18 * a, linewidth=0, zorder=zorder - 1)
            ax.plot(x, y, color=color, ls=ls, lw=lw, marker=mk, ms=7,
                    alpha=a, zorder=zorder)

    if gap_frame:
        # no-chat pair forward, chat receded, gap between dashed lines shaded
        y_sh = [rates_nochat[(q, "shared")] for q in QT]
        y_op = [rates_nochat[(q, "opposing")] for q in QT]
        ax.fill_between(x, y_op, y_sh, color="0.75", alpha=0.5, linewidth=0,
                        zorder=1)
        draw(rates_chat, ci_chat, "-", 3.2, 0.18, 2)
        draw(rates_nochat, ci_nochat, "--", 2.2, 0.95, 3)
    else:
        if show_nochat:
            draw(rates_nochat, ci_nochat, "--", 2.0, 0.45, 2)
        draw(rates_chat, ci_chat, "-", 3.2, 1.0, 3)

    # identical legend geometry in every frame: condition entries always
    # occupy space, but are fully invisible until both conditions are drawn
    # (otherwise bbox_inches="tight" crops each frame differently and the
    # axes jump between slide builds)
    cond_alpha = 1.0 if show_nochat else 0.0
    entry_alphas = [a_shared, a_opp, cond_alpha, 0.6 * cond_alpha]
    legend_elems = [
        Line2D([0], [0], color=COLORS["shared"], lw=3, marker="o",
               alpha=a_shared, label="agreed on the topic"),
        Line2D([0], [0], color=COLORS["opposing"], lw=3, marker="s",
               alpha=a_opp, label="disagreed on the topic"),
        Line2D([0], [0], color="#555555", lw=3, ls="-", alpha=cond_alpha,
               label="talked it through"),
        Line2D([0], [0], color="#555555", lw=2, ls="--",
               alpha=0.6 * cond_alpha, label="just saw their stance"),
    ]
    leg = ax.legend(handles=legend_elems, frameon=False, fontsize=9,
                    loc="upper left", bbox_to_anchor=(1.03, 0.65), ncol=1)
    for txt, ea in zip(leg.get_texts(), entry_alphas):
        if ea == 0.0:
            txt.set_alpha(0.0)

    ax.set_xticks(x)
    ax.set_xticklabels(QT_LABELS)
    ax.set_xlabel("how related the topic is to the one they discussed",
                  fontweight="bold")
    ax.set_ylabel("expected commonality", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    # same margins as talk_model_overlay.py; fixed canvas, no tight crop
    fig.subplots_adjust(left=0.13, right=0.62, top=0.94, bottom=0.14)

    for ext in ("png", "pdf"):
        fig.savefig(out / f"talk_gradient_{name}.{ext}", dpi=200)
    plt.close(fig)
    print("saved:", out / f"talk_gradient_{name}.png")
