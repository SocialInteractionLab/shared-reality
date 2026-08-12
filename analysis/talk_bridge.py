"""
Bridge frames from the faceted bar plot to the model-comparison boost line.
All frames share the model_overlay canvas (size, margins, x, y-ticks) so they
cross-fade cleanly in the slide build:

  bars  ->  b0 (imagined shared/opposite as LINES, un-faceted)
        ->  b1 (same + shaded GAP between the lines)
        ->  talk_model_overlay_step0 (that gap, plotted as the Human-data boost line)
        ->  step1 (+ assume-alike)  ->  final (+ uses-belief-structure)

Run from repo root:  uv run python analysis/talk_bridge.py
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
QT_LABELS = ["observation", "same\ndomain", "different\ndomain"]
CELLS = [(q, s) for q in QTS for s in ("shared", "opposing")]
CELL_IX = {c: i for i, c in enumerate(CELLS)}

data = load_evaluation_data()
imagined = data[data["experiment"] == "no-chat"].copy()
rates = get_rates(imagined, OUTCOME)


def per_pid_sumcount(df):
    out = {}
    for pid, g in df.groupby("pid"):
        s = np.zeros(len(CELLS)); c = np.zeros(len(CELLS))
        for (q, st), v in zip(zip(g["question_type"], g["stance"]), g[OUTCOME].to_numpy()):
            if (q, st) in CELL_IX:
                i = CELL_IX[(q, st)]; s[i] += v; c[i] += 1
        out[pid] = (s, c)
    return out


def cell_ci(df, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    sc = per_pid_sumcount(df); pids = np.array(list(sc.keys()))
    boots = np.empty((n, len(CELLS)))
    for b in range(n):
        s = np.zeros(len(CELLS)); c = np.zeros(len(CELLS))
        for p in rng.choice(pids, len(pids), replace=True):
            ss, cc = sc[p]; s += ss; c += cc
        boots[b] = np.divide(s, c, out=np.full(len(CELLS), np.nan), where=c > 0)
    return np.percentile(boots, [2.5, 97.5], axis=0)


lo, hi = cell_ci(imagined)
x = np.arange(3)
out = REPO / "outputs"
out.mkdir(exist_ok=True)

STANCES = [("opposing", COLORS["opposing"], "opposite", "s"),
           ("shared", COLORS["shared"], "shared", "o")]


def base_axes():
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.set_xticks(x)
    ax.set_xticklabels(QT_LABELS)
    ax.set_ylabel("expected commonality", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    return fig, ax


def draw_lines(ax):
    for stance, color, label, mk in STANCES:
        y = [rates[(q, stance)] for q in QTS]
        clo = [lo[CELL_IX[(q, stance)]] for q in QTS]
        chi = [hi[CELL_IX[(q, stance)]] for q in QTS]
        ax.fill_between(x, clo, chi, color=color, alpha=0.16, lw=0, zorder=1)
        ax.plot(x, y, color=color, lw=3, marker=mk, ms=7, label=label, zorder=3)


def finish(fig, ax, stem, legend=True):
    if legend:
        ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(1.03, 0.65))
    fig.subplots_adjust(left=0.13, right=0.62, top=0.94, bottom=0.14)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)
    print("saved:", out / f"{stem}.png")


# b0: imagined shared/opposite as lines (un-faceted phenomenon)
fig, ax = base_axes()
draw_lines(ax)
finish(fig, ax, "talk_bridge_0_lines")

# b1: same + shaded GAP between the lines (the thing that becomes the boost)
fig, ax = base_axes()
sh = [rates[(q, "shared")] for q in QTS]
op = [rates[(q, "opposing")] for q in QTS]
ax.fill_between(x, op, sh, color="#8A8A8A", alpha=0.28, lw=0, zorder=1, label="the gap")
draw_lines(ax)
# re-do legend to include the gap patch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=COLORS["opposing"], lw=3, marker="s", markersize=7, label="opposite"),
           Line2D([0], [0], color=COLORS["shared"], lw=3, marker="o", markersize=7, label="shared"),
           Patch(facecolor="#8A8A8A", alpha=0.4, label="the gap")]
ax.legend(handles=handles, frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(1.03, 0.65))
finish(fig, ax, "talk_bridge_1_gap", legend=False)
