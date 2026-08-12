"""Standalone generator for figure_s_sr_annots.pdf.

Reproduces the `fig-sr-generalization` cell from `analysis/supplement.qmd`:
Expected commonality (P(predictShared)) at focal / same-domain / different-domain
questions, split by whether the LLM annotated the opposing-stance dyad's chat
transcript as containing post-stance commonality (No/Yes). Restricted to
opposing-stance participants who accurately perceived disagreement on the focal
topic (prediction error <= 1).

Writes:
  - outputs/figures/figure_s_sr_annots.pdf
  - outputs/figures/figure_s_sr_annots.png

Run:
  cd analysis && uv run python generate_figure_s_sr_annots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 10_000
RNG_SEED = 42


def main() -> None:
    resp = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)
    annot = pl.read_csv(
        DATA_DIR / "llm_results" / "perbin_commonality_timecourse_15s_aggregated.csv"
    )

    # Dyad-level SR summary from LLM annotations (post-stance only)
    df_post = annot.filter(pl.col("focal_stance_revealed") == True)
    dyad_sr = (
        df_post.group_by("group_id")
        .agg(
            pl.col("post_stance_shared_reality").any().alias("ever_sr"),
            pl.col("stance").first(),
        )
        .with_columns(pl.col("ever_sr").cast(pl.Int64))
    )
    dyad_sr_pd = dyad_sr.rename({"group_id": "groupId"}).to_pandas()

    # Prediction error on focal in opposing-chat -> accurate perceivers
    focal = resp[
        (resp["experiment"] == "chat")
        & (resp["question_type"] == "observed")
        & (resp["stance"] == "opposing")
    ].copy()
    focal["pred_error"] = (focal["postChatResponse"] - focal["partner_response"]).abs()
    accurate_pids = set(focal[focal["pred_error"] <= 1]["pid"])

    opp_all = resp[
        (resp["experiment"] == "chat")
        & (resp["stance"] == "opposing")
        & (resp["pid"].isin(accurate_pids))
    ].copy()
    opp_all = opp_all.merge(dyad_sr_pd[["groupId", "ever_sr"]], on="groupId", how="inner")
    opp_all["sr_group"] = opp_all["ever_sr"].map(
        {1: "Found commonality", 0: "Did not find commonality"}
    )

    question_type_order = ["observed", "same_domain", "different_domain"]
    question_type_labels = ["Focal Topic", "Same Domain", "Different Domain"]
    sr_groups = ["Did not find commonality", "Found commonality"]
    colors = {"Found commonality": "#1757fe", "Did not find commonality": "#b0c6ff"}
    legend_labels = {"Found commonality": "Yes", "Did not find commonality": "No"}

    rng = np.random.default_rng(RNG_SEED)
    n_boot = N_BOOT

    x = np.arange(len(question_type_order))
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))

    for si, sg in enumerate(sr_groups):
        sub = opp_all[opp_all["sr_group"] == sg]
        means, ci_los, ci_his = [], [], []
        for qt in question_type_order:
            qt_data = sub[sub["question_type"] == qt]
            vals = qt_data["predictShared"].values
            ids = qt_data["groupId"].values
            unique_ids = np.unique(ids)
            n_clusters = len(unique_ids)
            boot_means = np.empty(n_boot)
            for b in range(n_boot):
                sampled = rng.choice(unique_ids, size=n_clusters, replace=True)
                mask = np.isin(ids, sampled)
                boot_means[b] = vals[mask].mean()
            means.append(vals.mean() * 100)
            lo, hi = np.percentile(boot_means, [2.5, 97.5])
            ci_los.append(lo * 100)
            ci_his.append(hi * 100)

        offset = -bar_w / 2 if si == 0 else bar_w / 2
        ax.bar(
            x + offset, means, bar_w, color=colors[sg], edgecolor="white",
            label=legend_labels[sg],
        )
        ax.errorbar(
            x + offset, means,
            yerr=[[m - lo for m, lo in zip(means, ci_los)],
                  [hi - m for m, hi in zip(means, ci_his)]],
            fmt="none", ecolor="black", capsize=4, linewidth=1.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(question_type_labels, fontsize=12)
    ax.set_ylabel("Expected Commonality", fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="Contained LLM-annotated commonality",
        fontsize=11, title_fontsize=11, frameon=False, loc="upper right",
    )

    plt.tight_layout()
    out_pdf = FIGURES_DIR / "figure_s_sr_annots.pdf"
    out_png = FIGURES_DIR / "figure_s_sr_annots.png"
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_pdf}")
    print(f"[save] {out_png}")


if __name__ == "__main__":
    main()
