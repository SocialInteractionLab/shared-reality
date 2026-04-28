"""
LLM-flagged post-stance commonality → expectations of commonality on unobserved topics.

Tests whether conversations containing LLM-annotated moments of commonality after the
initial stance was revealed predict higher expectations of commonality on (i) same-domain
and (ii) different-domain unobserved questions, and whether the effect differs by stance
condition.

Symmetric analog of the SI opposing-only filter: restricts to participants who
accurately perceived their partner's focal stance (|perceivedFocalResponse -
partner_response| <= 1), applied across both stance conditions.

Fits a logistic mixed-effects model via R/lme4 (subprocess; pymer4 is unreliable on this
machine) and renders a 2x2 bar figure: stance x ever_sr, split by same/different domain.

Run:
    uv run python analysis/llm_commonality_on_expectations.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import COLORS

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BIN_SECONDS = 15
N_BOOT = 2000
RNG = np.random.default_rng(42)


def load_merged(focal_err_max: int | None = 1) -> pd.DataFrame:
    """Merge per-dyad LLM annotations with participant-level inference responses.

    Args:
        focal_err_max: Max absolute error between perceivedFocalResponse and
            partner_response on the focal topic to retain a participant. This
            is the symmetric analog of the SI's "accurately perceived
            disagreement" filter: restricts analysis to participants who
            accurately perceived their partner's focal stance (|err| <= 1),
            regardless of stance condition. Set to None to disable.
    """
    ann = pd.read_csv(
        DATA_DIR / "llm_results" / f"perbin_commonality_timecourse_{BIN_SECONDS}s.csv"
    )
    resp = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)

    ann_post = ann[ann["focal_stance_revealed"]].copy()
    dyad = (
        ann_post.groupby("group_id")
        .agg(
            ever_sr=("post_stance_shared_reality", "any"),
            n_sr_bins=("post_stance_shared_reality", "sum"),
            n_bins=("post_stance_shared_reality", "count"),
            stance=("stance", "first"),
        )
        .reset_index()
    )
    dyad["ever_sr"] = dyad["ever_sr"].astype(int)

    if focal_err_max is not None:
        focal = resp[resp["question_type"] == "observed"][
            ["pid", "perceivedFocalResponse", "partner_response"]
        ].copy()
        focal["focal_err"] = (
            focal["perceivedFocalResponse"] - focal["partner_response"]
        ).abs()
        accurate_pids = set(focal[focal["focal_err"] <= focal_err_max]["pid"])
    else:
        accurate_pids = None

    resp_i = resp[
        (resp["question_type"] != "observed") & (resp["experiment"] == "chat")
    ].copy()
    if accurate_pids is not None:
        n_before = resp_i["pid"].nunique()
        resp_i = resp_i[resp_i["pid"].isin(accurate_pids)]
        n_after = resp_i["pid"].nunique()
        print(
            f"Focal-accuracy filter (|err| <= {focal_err_max}): "
            f"dropped {n_before - n_after} / {n_before} chat participants."
        )

    merged = resp_i.merge(
        dyad.rename(columns={"group_id": "groupId"}),
        on="groupId",
        how="inner",
        suffixes=("", "_annot"),
    )
    if "stance_annot" in merged.columns:
        merged["stance"] = merged["stance"].fillna(merged["stance_annot"])
        merged = merged.drop(columns=["stance_annot"])

    # Effect coding
    merged["stance_e"] = merged["stance"].map({"opposing": -0.5, "shared": +0.5})
    merged["ever_sr_e"] = merged["ever_sr"].map({0: -0.5, 1: +0.5})
    merged["same_e"] = merged["question_type"].map(
        {"same_domain": +0.5, "different_domain": -0.5}
    )

    merged = merged.dropna(
        subset=["predictShared", "stance_e", "ever_sr_e", "same_e", "groupId", "pid"]
    )
    merged["predictShared"] = merged["predictShared"].astype(int)
    return merged


def cluster_bootstrap_mean_ci(
    values: pd.Series, cluster_ids: pd.Series, n_boot: int = N_BOOT
) -> tuple[float, float, float]:
    """Bootstrap mean and 95% CI, resampling at the cluster (dyad) level."""
    unique_ids = cluster_ids.unique()
    n_clusters = len(unique_ids)
    if n_clusters == 0:
        return np.nan, np.nan, np.nan
    boot_means = np.empty(n_boot)
    values = values.to_numpy()
    cluster_ids = cluster_ids.to_numpy()
    for b in range(n_boot):
        sampled = RNG.choice(unique_ids, size=n_clusters, replace=True)
        mask = np.isin(cluster_ids, sampled)
        boot_means[b] = values[mask].mean()
    return values.mean(), float(np.percentile(boot_means, 2.5)), float(
        np.percentile(boot_means, 97.5)
    )


def fit_glmer(merged: pd.DataFrame) -> str:
    """Fit glmer models via Rscript and return captured stdout."""
    cols = ["predictShared", "stance_e", "ever_sr_e", "same_e", "groupId", "pid"]
    tmp = Path(tempfile.gettempdir())
    csv_path = tmp / "llm_commonality_glmer.csv"
    merged[cols].to_csv(csv_path, index=False)

    r_script = f"""
    suppressPackageStartupMessages({{ library(lme4); library(lmerTest) }})
    d <- read.csv('{csv_path}')
    d$groupId <- as.factor(d$groupId)
    d$pid <- as.factor(d$pid)

    ctrl <- glmerControl(optimizer='bobyqa')

    cat('=== Model 1: full 3-way (stance x ever_sr x same) ===\\n')
    m1 <- glmer(predictShared ~ stance_e * ever_sr_e * same_e + (1|groupId/pid),
                data=d, family=binomial, control=ctrl)
    print(summary(m1)$coefficients)

    cat('\\n=== Model 2: reduced (stance x ever_sr), pooled across same/diff ===\\n')
    m2 <- glmer(predictShared ~ stance_e * ever_sr_e + (1|groupId/pid),
                data=d, family=binomial, control=ctrl)
    print(summary(m2)$coefficients)

    cat('\\n=== Model 3: within opposing only ===\\n')
    m3 <- glmer(predictShared ~ ever_sr_e * same_e + (1|groupId/pid),
                data=d[d$stance_e < 0,], family=binomial, control=ctrl)
    print(summary(m3)$coefficients)

    cat('\\n=== Model 4: within shared only ===\\n')
    m4 <- glmer(predictShared ~ ever_sr_e * same_e + (1|groupId/pid),
                data=d[d$stance_e > 0,], family=binomial, control=ctrl)
    print(summary(m4)$coefficients)
    """
    r_path = tmp / "llm_commonality_glmer.R"
    r_path.write_text(r_script)

    res = subprocess.run(
        ["Rscript", str(r_path)], capture_output=True, text=True, timeout=600
    )
    if res.returncode != 0:
        raise RuntimeError(f"Rscript failed:\n{res.stderr}")
    return res.stdout


def make_figure(merged: pd.DataFrame, out_path: Path) -> None:
    """2x2 bar plot: stance (panel) x ever_sr (x-axis), same vs diff (color)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    cells = []
    for stance in ["opposing", "shared"]:
        for ever_sr in [0, 1]:
            for qtype in ["same_domain", "different_domain"]:
                sub = merged[
                    (merged["stance"] == stance)
                    & (merged["ever_sr"] == ever_sr)
                    & (merged["question_type"] == qtype)
                ]
                n_dyads = sub["groupId"].nunique()
                mean, lo, hi = cluster_bootstrap_mean_ci(
                    sub["predictShared"], sub["groupId"]
                )
                cells.append(
                    dict(
                        stance=stance,
                        ever_sr=ever_sr,
                        qtype=qtype,
                        n_dyads=n_dyads,
                        mean=mean,
                        lo=lo,
                        hi=hi,
                    )
                )
    cells_df = pd.DataFrame(cells)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), sharey=True)

    x = np.array([0, 1])
    width = 0.36
    same_color = COLORS["same"]
    diff_color = COLORS["diff"]

    for ax, stance in zip(axes, ["opposing", "shared"]):
        sub = cells_df[cells_df["stance"] == stance]
        same = sub[sub["qtype"] == "same_domain"].sort_values("ever_sr")
        diff = sub[sub["qtype"] == "different_domain"].sort_values("ever_sr")

        ax.bar(
            x - width / 2,
            same["mean"].values,
            width=width,
            color=same_color,
            label="Same Domain",
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        ax.errorbar(
            x - width / 2,
            same["mean"].values,
            yerr=[
                same["mean"].values - same["lo"].values,
                same["hi"].values - same["mean"].values,
            ],
            fmt="none",
            ecolor="#333333",
            capsize=3,
            lw=1.2,
            zorder=3,
        )
        ax.bar(
            x + width / 2,
            diff["mean"].values,
            width=width,
            color=diff_color,
            label="Different Domain",
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        ax.errorbar(
            x + width / 2,
            diff["mean"].values,
            yerr=[
                diff["mean"].values - diff["lo"].values,
                diff["hi"].values - diff["mean"].values,
            ],
            fmt="none",
            ecolor="#333333",
            capsize=3,
            lw=1.2,
            zorder=3,
        )

        n_no = sub[sub["ever_sr"] == 0]["n_dyads"].iloc[0]
        n_yes = sub[sub["ever_sr"] == 1]["n_dyads"].iloc[0]
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"No\n($n = {n_no}$ dyads)", f"Yes\n($n = {n_yes}$ dyads)"]
        )

        title_color = COLORS[stance]
        ax.set_title(
            f"{stance.capitalize()} Stance", color=title_color, fontweight="bold", pad=10
        )
        ax.set_ylim(0, 0.95)
        ax.set_yticks(np.arange(0, 0.9, 0.1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="both", length=3, width=0.8)
        ax.grid(axis="y", color="#EEEEEE", lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Expected Commonality")
    axes[1].legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        handlelength=1.4,
        handletextpad=0.6,
    )

    fig.supxlabel(
        "Conversation Contained LLM-Annotated Commonality", fontsize=11, y=0.02
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved figure: {out_path}")


def print_cell_summary(merged: pd.DataFrame) -> None:
    print("=== Dyad-level crosstab: stance × ever_sr ===")
    dyads = merged.drop_duplicates("groupId")[["stance", "ever_sr", "groupId"]]
    print(pd.crosstab(dyads["stance"], dyads["ever_sr"], margins=True))
    print()

    print("=== Cell means (P(predictShared)) by stance × ever_sr × question_type ===")
    tbl = (
        merged.groupby(["stance", "ever_sr", "question_type"])["predictShared"]
        .agg(["mean", "count"])
        .round(3)
    )
    print(tbl)
    print()


def main() -> None:
    merged = load_merged()
    print(
        f"Final analysis sample: {len(merged)} responses, "
        f"{merged['groupId'].nunique()} dyads, {merged['pid'].nunique()} participants.\n"
    )
    print_cell_summary(merged)
    print(fit_glmer(merged))
    make_figure(merged, FIG_DIR / "figure_llm_commonality_on_expectations.pdf")


if __name__ == "__main__":
    main()
