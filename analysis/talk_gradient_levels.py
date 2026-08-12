"""
Talk figure (Ch 1 phenomenon slide): observed expected commonality by
topic-distance x stance. Two lines (agreed vs disagreed on the focal topic)
that converge as topics get less related -- the generalization gradient.
Human data only (no models). Bootstrap CIs over participants.

Run from repo root:  uv run python analysis/talk_gradient_levels.py
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
QT = ["observed", "same_domain", "different_domain"]
QT_LABELS = ["observation", "same\ndomain", "different\ndomain"]
CELLS = [(q, s) for q in QT for s in ("shared", "opposing")]
CELL_IX = {c: i for i, c in enumerate(CELLS)}

data = load_evaluation_data()
nochat = data[data["experiment"] == "no-chat"].copy()
chat = data[data["experiment"] == "chat"].copy()
rates_nochat = get_rates(nochat, OUTCOME)
rates_chat = get_rates(chat, OUTCOME)
print("no-chat:", {k: round(v, 3) for k, v in rates_nochat.items()})
print("chat:   ", {k: round(v, 3) for k, v in rates_chat.items()})


def per_pid_sumcount(df, col):
    out = {}
    for pid, g in df.groupby("pid"):
        s = np.zeros(6)
        c = np.zeros(6)
        for (q, st), v in zip(zip(g["question_type"], g["stance"]), g[col].to_numpy()):
            if (q, st) in CELL_IX:
                i = CELL_IX[(q, st)]
                s[i] += v
                c[i] += 1
        out[pid] = (s, c)
    return out


def bootstrap_cell_ci(sc, pids, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pid_arr = np.array(pids)
    boots = np.empty((n, 6))
    for b in range(n):
        s = np.zeros(6)
        c = np.zeros(6)
        for p in rng.choice(pid_arr, size=len(pid_arr), replace=True):
            ss, cc = sc[p]
            s += ss
            c += cc
        boots[b] = np.divide(s, c, out=np.full(6, np.nan), where=c > 0)
    return np.percentile(boots, [2.5, 97.5], axis=0)


set_plot_style()
x = np.arange(3)
fig, ax = plt.subplots(figsize=(6.4, 4.6))

STANCES = [
    ("shared", COLORS["shared"], "o"),
    ("opposing", COLORS["opposing"], "s"),
]
# condition: chat = solid/bold, no-chat = dashed/faded (control recedes)
CONDS = [
    ("chat", rates_chat, "-", 3.2, 1.0),
    ("no-chat", rates_nochat, "--", 2.0, 0.45),
]

for cond, rates, ls, lw, alpha in CONDS:
    for stance, color, mk in STANCES:
        y = [rates[(q, stance)] for q in QT]
        ax.plot(x, y, color=color, ls=ls, lw=lw, marker=mk, ms=7,
                alpha=alpha, zorder=3 if cond == "chat" else 2)

# custom legend: color = stance, style = condition
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], color=COLORS["shared"], lw=3, marker="o", label="Agreed on the topic"),
    Line2D([0], [0], color=COLORS["opposing"], lw=3, marker="s", label="Disagreed on the topic"),
    Line2D([0], [0], color="#555555", lw=3, ls="-", label="Talked it through"),
    Line2D([0], [0], color="#555555", lw=2, ls="--", alpha=0.6, label="Just saw their stance"),
]
ax.legend(handles=legend_elems, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 0.65), ncol=1)

ax.set_xticks(x)
ax.set_xticklabels(QT_LABELS)
ax.set_xlabel("How related the topic is to the one they discussed", fontweight="bold")
ax.set_ylabel("Expected commonality", fontweight="bold")
ax.set_ylim(0, 1)
fig.tight_layout()

out = REPO / "outputs"
out.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"talk_gradient_levels.{ext}", dpi=200, bbox_inches="tight")
print("saved:", out / "talk_gradient_levels.png")
print({k: round(v, 3) for k, v in rates.items()})
