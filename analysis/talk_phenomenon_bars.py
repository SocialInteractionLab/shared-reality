"""
Talk figure (Ch 1 phenomenon, bars — Python/matplotlib port of the R slide):
expected commonality, faceted by topic-distance (observation / same domain /
different domain), grouped by condition (chat vs imagined), colored by stance
(shared pink / opposing blue). Bootstrap CIs over participants.

x-tick labels ('chat'/'imagined') and facet titles are text placeholders --
swap for icons in the deck later.

Run from repo root:  uv run python analysis/talk_phenomenon_bars.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from models.model import load_evaluation_data
from analysis.utils import get_rates, set_plot_style, COLORS

OUTCOME = "participant_binary_prediction"
QTS = ["observed", "same_domain", "different_domain"]
QT_TITLES = ["observation", "same domain", "different domain"]
STANCES = [("opposing", COLORS["opposing"]), ("shared", COLORS["shared"])]  # opposite left, shared right
CONDS = [("no-chat", "imagined"), ("chat", "chat")]  # imagined left (cleaner gradient, model's condition)
CELLS = [(q, s) for q in QTS for s, _ in STANCES]
CELL_IX = {c: i for i, c in enumerate(CELLS)}

data = load_evaluation_data()


def per_pid_sumcount(df):
    out = {}
    for pid, g in df.groupby("pid"):
        s = np.zeros(len(CELLS))
        c = np.zeros(len(CELLS))
        for (q, st), v in zip(zip(g["question_type"], g["stance"]), g[OUTCOME].to_numpy()):
            if (q, st) in CELL_IX:
                i = CELL_IX[(q, st)]
                s[i] += v
                c[i] += 1
        out[pid] = (s, c)
    return out


def cell_ci(df, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    sc = per_pid_sumcount(df)
    pids = np.array(list(sc.keys()))
    boots = np.empty((n, len(CELLS)))
    for b in range(n):
        s = np.zeros(len(CELLS))
        c = np.zeros(len(CELLS))
        for p in rng.choice(pids, len(pids), replace=True):
            ss, cc = sc[p]
            s += ss
            c += cc
        boots[b] = np.divide(s, c, out=np.full(len(CELLS), np.nan), where=c > 0)
    return np.percentile(boots, [2.5, 97.5], axis=0)


# rates + CIs per condition
RATES, CI = {}, {}
for cond, _ in CONDS:
    sub = data[data["experiment"] == cond]
    RATES[cond] = get_rates(sub, OUTCOME)
    CI[cond] = cell_ci(sub)

# shared-vs-opposing (within-condition) Tukey p-values, from the glmer + emmeans
# (analysis/behavioral model; see talk_stats notes). key: (question_type, experiment)
P_STANCE = {
    ("observed", "no-chat"): 1e-16,          # ***
    ("observed", "chat"): 1e-16,             # ***
    ("same_domain", "no-chat"): 4.36e-10,    # ***
    ("same_domain", "chat"): 6.98e-04,       # ***
    ("different_domain", "no-chat"): 3.66e-02,  # *
    ("different_domain", "chat"): 4.12e-01,     # n.s.
}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

set_plot_style()
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(9.2, 4.6))
width = 0.34
cond_x = {c: i for i, (c, _) in enumerate(CONDS)}  # chat=0, imagined=1

for ax, qt, title in zip(axes, QTS, QT_TITLES):
    for cond, _ in CONDS:
        base = cond_x[cond]
        for j, (stance, color) in enumerate(STANCES):
            xpos = base + (j - 0.5) * width  # shared left, opposing right
            y = RATES[cond][(qt, stance)]
            lo = CI[cond][0][CELL_IX[(qt, stance)]]
            hi = CI[cond][1][CELL_IX[(qt, stance)]]
            ax.bar(xpos, y, width, color=color, zorder=2)
            ax.errorbar(xpos, y, yerr=[[y - lo], [hi - y]], color="black",
                        lw=1.3, capsize=3, zorder=3)
    # significance brackets: shared vs opposing within each condition
    for cond, _ in CONDS:
        base = cond_x[cond]
        xL = base - 0.5 * width                  # left bar center
        xR = base + 0.5 * width                  # right bar center
        hi_sh = CI[cond][1][CELL_IX[(qt, "shared")]]
        hi_op = CI[cond][1][CELL_IX[(qt, "opposing")]]
        ytop = max(hi_sh, hi_op) + 0.045
        ax.plot([xL, xR], [ytop, ytop], color="black", lw=1.0)  # plain horizontal line
        ax.text((xL + xR) / 2, ytop + 0.008, stars(P_STANCE[(qt, cond)]),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(list(cond_x.values()))
    ax.set_xticklabels([lab for _, lab in CONDS])  # placeholder; swap for icons
    ax.set_xlim(-0.6, 1.6)
    # keep facet box (all four spines) like the R version
    for sp in ax.spines.values():
        sp.set_visible(True)

axes[0].set_ylabel("expected commonality", fontweight="bold")
axes[0].set_ylim(0, 1.08)  # headroom for brackets
axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])

# legend to the right
from matplotlib.patches import Patch
handles = [Patch(facecolor=c, label=("shared" if s == "shared" else "opposite")) for s, c in STANCES]
axes[-1].legend(handles=handles, title="stance type", frameon=False,
                loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, title_fontsize=10)

fig.subplots_adjust(left=0.09, right=0.86, top=0.90, bottom=0.12, wspace=0.12)
out = REPO / "outputs"
out.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"talk_phenomenon_bars.{ext}", dpi=200)
print("saved:", out / "talk_phenomenon_bars.png")
