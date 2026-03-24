import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.generalized_estimating_equations import GEE

    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    FIGURES_DIR = BASE_DIR / "outputs" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    BIN_SECONDS = 15

    STANCE_COLORS = {"opposing": "#648FFF", "shared": "#DC267F"}

    # Force light plots / override marimo dark mode
    sns.set_theme(style="white")
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "legend.facecolor": "white",
            "legend.edgecolor": "0.85",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.grid": False,
        }
    )
    return (
        BIN_SECONDS,
        Binomial,
        DATA_DIR,
        Exchangeable,
        FIGURES_DIR,
        GEE,
        STANCE_COLORS,
        mo,
        np,
        pl,
        plt,
    )


@app.cell
def _(BIN_SECONDS, DATA_DIR, mo, pl):
    # Load aggregated per-bin results (modal vote across 3 runs + confidence)
    _csv_path = DATA_DIR / "llm_results" / f"perbin_commonality_timecourse_{BIN_SECONDS}s_aggregated.csv"
    df_all = pl.read_csv(_csv_path)

    if "time_seconds" not in df_all.columns:
        df_all = df_all.with_columns(
            ((pl.col("time_bin") + 1) * BIN_SECONDS).alias("time_seconds")
        )

    # Post-stance bins only (where stances have been established)
    df_post = df_all.filter(pl.col("focal_stance_revealed") == True).sort(
        ["group_id", "time_bin"]
    )

    mo.md(f"""
    ### Data loaded

    - **Total annotations**: {len(df_all)} ({df_all['group_id'].n_unique()} dyads)
    - **Post-stance annotations**: {len(df_post)} ({df_post['group_id'].n_unique()} dyads)
    """)
    return (df_post,)


@app.cell
def _(df_post, mo, pl):
    # One row per dyad: did they find shared reality post-stance?
    _scope_priority = {
        "same_values": 4, "related_subtopic": 3, "different_topic": 2,
        "rapport_only": 1, "none": 0,
    }

    dyad_sr_summary = (
        df_post.group_by("group_id")
        .agg(
            pl.col("stance").first(),
            pl.col("matchedQuestion").first().alias("focal_question"),
            pl.col("matchedDomain").first().alias("focal_domain"),
            pl.col("post_stance_shared_reality").any().alias("ever_sr"),
            pl.col("post_stance_shared_reality").sum().alias("n_sr_bins"),
            pl.col("post_stance_shared_reality").count().alias("n_post_stance_bins"),
            pl.col("post_stance_topic_scope")
            .replace_strict(_scope_priority, return_dtype=pl.Int64)
            .max()
            .alias("_best_scope_priority"),
        )
        .with_columns(
            pl.col("ever_sr").cast(pl.Int64),
            (pl.col("n_sr_bins") / pl.col("n_post_stance_bins")).alias("pct_sr_bins"),
            pl.col("_best_scope_priority")
            .replace_strict({v: k for k, v in _scope_priority.items()}, return_dtype=pl.Utf8)
            .alias("best_topic_scope"),
        )
        .drop("_best_scope_priority")
        .sort("group_id")
    )

    _opp = dyad_sr_summary.filter(pl.col("stance") == "opposing")
    _opp_sr = _opp.filter(pl.col("ever_sr") == 1)
    _opp_no_sr = _opp.filter(pl.col("ever_sr") == 0)

    mo.md(f"""
    ### Per-dyad shared reality summary

    - **Total dyads**: {len(dyad_sr_summary)}
    - **Opposing**: {len(_opp)} ({len(_opp_sr)} found SR, {len(_opp_no_sr)} did not)
    - **Shared**: {len(dyad_sr_summary.filter(pl.col("stance") == "shared"))}
    """)
    return (dyad_sr_summary,)


@app.cell
def _(DATA_DIR, FIGURES_DIR, STANCE_COLORS, dyad_sr_summary, mo, np, pl, plt):
    # ── Accurate perceivers only: generalization gradient × SR ──
    _resp = pl.read_csv(DATA_DIR / "responses.csv")

    # Per-participant prediction error on focal (observed) topic
    _focal = _resp.filter(
        (pl.col("experiment") == "chat")
        & (pl.col("question_type") == "observed")
        & (pl.col("stance") == "opposing")
    ).with_columns(
        (pl.col("postChatResponse") - pl.col("partner_response")).abs().alias("pred_error")
    )

    # Accurate perceivers: pred_error ≤ 1
    _accurate_pids = _focal.filter(pl.col("pred_error") <= 1).select("pid").unique()

    # All response rows for accurate opposing chat participants
    _opp_all = _resp.filter(
        (pl.col("experiment") == "chat")
        & (pl.col("stance") == "opposing")
    ).join(_accurate_pids, on="pid", how="inner")

    # Join with SR summary
    opp_sr = _opp_all.join(
        dyad_sr_summary.rename({"group_id": "groupId"}),
        on="groupId",
        how="inner",
        suffix="_annot",
    ).with_columns(
        pl.when(pl.col("ever_sr") == 1)
        .then(pl.lit("Found commonality during chat"))
        .otherwise(pl.lit("Did not find commonality during chat"))
        .alias("sr_group")
    )

    _question_type_order = ["observed", "same_domain", "different_domain"]
    _question_type_labels = {
        "observed": "Focal topic",
        "same_domain": "Same domain",
        "different_domain": "Different domain",
    }
    _sr_groups = ["Found commonality during chat", "Did not find commonality during chat"]
    _bar_color = STANCE_COLORS["opposing"]

    # ── Plot ──
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for _i, _sg in enumerate(_sr_groups):
        _ax = _axes[_i]
        _sub = opp_sr.filter(pl.col("sr_group") == _sg)
        _n_dyads = _sub["groupId"].n_unique()

        _means = []
        _ci_los = []
        _ci_his = []

        for _qt in _question_type_order:
            _qt_data = _sub.filter(pl.col("question_type") == _qt)
            _vals = _qt_data["predictShared"].to_numpy()
            _ids = _qt_data["groupId"].to_numpy()

            # Cluster bootstrap (resampling dyads, not individual observations)
            _unique_ids = np.unique(_ids)
            _n_clusters = len(_unique_ids)
            _boot_means = np.empty(10_000)
            _rng = np.random.default_rng(42)
            for _b in range(10_000):
                _sampled = _rng.choice(_unique_ids, size=_n_clusters, replace=True)
                _mask = np.isin(_ids, _sampled)
                _boot_means[_b] = _vals[_mask].mean()
            _m = float(_vals.mean())
            _lo, _hi = np.percentile(_boot_means, [2.5, 97.5])

            _means.append(_m * 100)
            _ci_los.append(_lo * 100)
            _ci_his.append(_hi * 100)

        _x = np.arange(len(_question_type_order))
        _ax.bar(_x, _means, color=_bar_color, edgecolor="white", width=0.6)
        _ax.errorbar(
            _x, _means,
            yerr=[
                [m - lo for m, lo in zip(_means, _ci_los)],
                [hi - m for m, hi in zip(_means, _ci_his)],
            ],
            fmt="none", ecolor="black", capsize=5, linewidth=1.5,
        )

        _ax.set_xticks(_x)
        _ax.set_xticklabels(
            [_question_type_labels[qt] for qt in _question_type_order], fontsize=12,
        )
        _ax.set_ylim(0, 100)
        _ax.set_yticks([0, 25, 50, 75, 100])
        _ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=12)
        _ax.set_title(f"{_sg}\n(N = {_n_dyads} dyads)", fontsize=14)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

        for _j, (_m, _hi) in enumerate(zip(_means, _ci_his)):
            _ax.text(_j, _hi + 2, f"{_m:.1f}%", ha="center", fontsize=11, fontweight="bold")

    _axes[0].set_ylabel("Expected Commonality", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_s_sr_annots.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # ── Summary table ──
    _n_accurate = len(_accurate_pids)
    _n_total = _focal["pid"].n_unique()
    _lines = [
        "### Generalization gradient — accurate perceivers only (opposing dyads)\n",
        f"Filtering to participants with prediction error ≤ 1 on the focal topic "
        f"({_n_accurate}/{_n_total} participants).\n",
    ]
    _lines.append("| SR Group | Focal Topic | Same Domain | Different Domain | N dyads |")
    _lines.append("|----------|------------|------------|-----------------|---------|")
    for _sg in _sr_groups:
        _sub = opp_sr.filter(pl.col("sr_group") == _sg)
        _focal_m = _sub.filter(pl.col("question_type") == "observed")["predictShared"].mean()
        _same_m = _sub.filter(pl.col("question_type") == "same_domain")["predictShared"].mean()
        _diff_m = _sub.filter(pl.col("question_type") == "different_domain")["predictShared"].mean()
        _nd = _sub["groupId"].n_unique()
        _lines.append(
            f"| {_sg} | {_focal_m * 100:.1f}% | {_same_m * 100:.1f}% "
            f"| {_diff_m * 100:.1f}% | {_nd} |"
        )

    mo.md("\n".join(_lines))
    return (opp_sr,)


@app.cell
def _(Binomial, Exchangeable, GEE, mo, opp_sr, pl):
    # ── GEE logistic: Does finding SR boost commonality? ──
    # Accurate opposing-stance dyads, excluding focal topic (where SR was established)
    _opp_gen = (
        opp_sr.filter(
            pl.col("question_type").is_in(["same_domain", "different_domain"])
        )
        .with_columns(
            pl.col("ever_sr").cast(pl.Int64).alias("ever_sr"),
            pl.when(pl.col("question_type") == "same_domain")
            .then(pl.lit(0.5))
            .otherwise(pl.lit(-0.5))
            .alias("is_same_domain"),
            (pl.col("ever_sr").cast(pl.Int64)
             * pl.when(pl.col("question_type") == "same_domain")
               .then(pl.lit(0.5))
               .otherwise(pl.lit(-0.5)))
            .alias("ever_sr_x_same_domain"),
        )
    ).to_pandas().sort_values("groupId")

    _model = GEE.from_formula(
        "predictShared ~ ever_sr + is_same_domain + ever_sr_x_same_domain",
        groups="groupId",
        data=_opp_gen,
        family=Binomial(),
        cov_struct=Exchangeable(),
    )
    _result = _model.fit()

    mo.md(f"""
    ### GEE logistic: Shared reality → commonality (accurate opposing dyads, generalization items)

    **Model:** `predictShared ~ ever_sr * is_same_domain`, clustered by dyad (exchangeable correlation)

    - **ever_sr**: Does finding SR boost expected commonality overall?
    - **is_same_domain**: Same domain (+0.5) vs different domain (−0.5)
    - **ever_sr × is_same_domain**: Is the SR boost stronger for same-domain items?

    ```
    {_result.summary()}
    ```
    """)
    return


@app.cell
def _(mo, opp_sr, pl):
    # ── Summary stats by focal topic domain ──
    _lines = ["## Summary by focal topic domain (opposing, accurate perceivers)\n"]
    _lines.append(
        "| Domain | N dyads | SR found | No SR | Focal % | Same % | Diff % |"
    )
    _lines.append(
        "|--------|---------|----------|-------|---------|--------|--------|"
    )

    _domains = sorted(opp_sr["matchedDomain"].unique().to_list())
    for _dom in _domains:
        _sub = opp_sr.filter(pl.col("matchedDomain") == _dom)
        _nd = _sub["groupId"].n_unique()
        _n_sr = _sub.filter(pl.col("ever_sr") == 1)["groupId"].n_unique()
        _n_no = _sub.filter(pl.col("ever_sr") == 0)["groupId"].n_unique()
        _focal = _sub.filter(pl.col("question_type") == "observed")["predictShared"].mean()
        _same = _sub.filter(pl.col("question_type") == "same_domain")["predictShared"].mean()
        _diff = _sub.filter(pl.col("question_type") == "different_domain")["predictShared"].mean()
        _lines.append(
            f"| {_dom} | {_nd} | {_n_sr} | {_n_no} "
            f"| {_focal * 100:.1f}% | {_same * 100:.1f}% | {_diff * 100:.1f}% |"
        )

    _total = opp_sr["groupId"].n_unique()
    _lines.append(f"\n**Total**: {_total} dyads across {len(_domains)} domains")
    mo.md("\n".join(_lines))
    return


@app.cell
def _(FIGURES_DIR, STANCE_COLORS, mo, np, opp_sr, pl, plt):
    # ── Faceted plot: generalization gradient by focal domain × SR ──
    _domains = sorted(opp_sr["matchedDomain"].unique().to_list())
    _n_domains = len(_domains)
    _ncols = 4
    _nrows = (_n_domains + _ncols - 1) // _ncols

    _question_type_order = ["observed", "same_domain", "different_domain"]
    _question_type_labels = ["Focal", "Same\ndomain", "Diff.\ndomain"]
    _sr_groups = [
        ("Found commonality during chat", 1),
        ("Did not find commonality during chat", 0),
    ]
    _colors = {1: STANCE_COLORS["opposing"], 0: "#B0B0B0"}
    _labels = {1: "Found CG", 0: "No CG"}

    _fig, _axes = plt.subplots(
        _nrows, _ncols, figsize=(_ncols * 3.5, _nrows * 3.5),
        sharey=True, squeeze=False,
    )

    for _idx, _dom in enumerate(_domains):
        _row, _col = divmod(_idx, _ncols)
        _ax = _axes[_row][_col]
        _dom_data = opp_sr.filter(pl.col("matchedDomain") == _dom)
        _n_dyads = _dom_data["groupId"].n_unique()

        _x = np.arange(len(_question_type_order))
        _bar_w = 0.35

        for _si, (_sg_label, _sr_val) in enumerate(_sr_groups):
            _sub = _dom_data.filter(pl.col("ever_sr") == _sr_val)
            _nd = _sub["groupId"].n_unique()
            _means = []
            for _qt in _question_type_order:
                _qt_data = _sub.filter(pl.col("question_type") == _qt)
                if len(_qt_data) > 0:
                    _means.append(_qt_data["predictShared"].mean() * 100)
                else:
                    _means.append(0)
            _offset = -_bar_w / 2 if _si == 0 else _bar_w / 2
            _bars = _ax.bar(
                _x + _offset, _means, _bar_w,
                color=_colors[_sr_val], edgecolor="white",
                label=f"{_labels[_sr_val]} (n={_nd})" if _idx == 0 else None,
            )
            for _j, _m in enumerate(_means):
                if _m > 0:
                    _ax.text(
                        _x[_j] + _offset, _m + 1.5, f"{_m:.0f}",
                        ha="center", fontsize=7, color="#333",
                    )

        _ax.set_xticks(_x)
        _ax.set_xticklabels(_question_type_labels, fontsize=8)
        _ax.set_ylim(0, 105)
        _ax.set_title(f"{_dom.capitalize()}\n(N={_n_dyads})", fontsize=10, fontweight="bold")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        if _col == 0:
            _ax.set_ylabel("% Expected\nCommonality", fontsize=9)

    # Hide empty subplots
    for _idx in range(_n_domains, _nrows * _ncols):
        _row, _col = divmod(_idx, _ncols)
        _axes[_row][_col].set_visible(False)

    _fig.legend(
        *_axes[0][0].get_legend_handles_labels(),
        loc="lower right", fontsize=9, frameon=False, ncol=2,
    )
    _fig.suptitle(
        "Generalization gradient by focal topic domain × commonality found",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(
        FIGURES_DIR / "figure_s_sr_annots_by_domain.pdf",
        dpi=300, bbox_inches="tight",
    )
    plt.show()
    mo.md(f"Saved domain-faceted plot ({_n_domains} domains).")
    return


@app.cell
def _(Binomial, Exchangeable, GEE, mo, opp_sr, pl):
    # ── GEE with domain as fixed effect ──
    _opp_gen = (
        opp_sr.filter(
            pl.col("question_type").is_in(["same_domain", "different_domain"])
        )
        .with_columns(
            pl.col("ever_sr").cast(pl.Int64).alias("ever_sr"),
            pl.when(pl.col("question_type") == "same_domain")
            .then(pl.lit(0.5))
            .otherwise(pl.lit(-0.5))
            .alias("is_same_domain"),
        )
    ).to_pandas().sort_values("groupId")

    # Dummy-code domain (reference = arbitrary, alphabetically first)
    _opp_gen["matchedDomain"] = _opp_gen["matchedDomain"].astype("category")

    _model = GEE.from_formula(
        "predictShared ~ ever_sr + is_same_domain + C(matchedDomain) + ever_sr:is_same_domain",
        groups="groupId",
        data=_opp_gen,
        family=Binomial(),
        cov_struct=Exchangeable(),
    )
    _result = _model.fit()

    mo.md(f"""
### GEE logistic with domain: SR → commonality (accurate opposing dyads)

**Model:** `predictShared ~ ever_sr * is_same_domain + C(matchedDomain)`, clustered by dyad

- Controls for baseline differences in commonality expectations across focal topic domains
- Tests whether the SR effect holds after accounting for domain

```
{_result.summary()}
```
""")
    return


if __name__ == "__main__":
    app.run()
