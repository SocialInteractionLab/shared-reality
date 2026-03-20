"""Chat LLM annotation analyses.

Analyzes LLM-annotated shared reality emergence over conversation time,
comparing opposing vs shared stance dyads.

Usage:
    uv run marimo edit analysis/chat_llm_annotation_analyses.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


# ── Imports & constants ───────────────────────────────────────────────────


@app.cell
def _():
    import marimo as mo
    import subprocess
    import tempfile
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl

    from matplotlib.colors import BoundaryNorm, ListedColormap
    from pymer4.models import glmer
    from scipy.stats import mannwhitneyu, permutation_test

    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    FIGURES_DIR = BASE_DIR / "outputs" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    N_BOOT = 10_000
    RNG = np.random.default_rng(42)

    BIN_SECONDS = 15

    # ── Experimental conditions ──
    STANCE_ORDER = ["opposing", "shared"]
    STANCE_COLORS = {"opposing": "#648FFF", "shared": "#DC267F"}

    # ── LLM annotation field values ──
    # Phase 1: Stance establishment
    INITIAL_ALIGNMENT_VALUES = ["agreement", "disagreement", "unclear", "not_discussed"]

    # Phase 2: Shared reality (post-stance)
    TOPIC_SCOPE_VALUES = [
        "same_values", "related_subtopic", "different_topic", "rapport_only", "none",
    ]
    TOPIC_SCOPE_LABELS = [
        "Same values", "Related subtopic", "Different topic", "Rapport only", "None",
    ]
    TOPIC_SCOPE_COLORS = {
        "same_values": "#FE6100",
        "related_subtopic": "#785EF0",
        "different_topic": "#FFB000",
        "rapport_only": "#C0C0C0",
        "none": "#E8E8E8",
    }

    INNER_STATE_DIMENSION_VALUES = [
        "shared_thoughts", "shared_feelings", "shared_beliefs", "none",
    ]
    INNER_STATE_DIMENSION_LABELS = [
        "Shared thoughts", "Shared feelings", "Shared beliefs", "None",
    ]
    INNER_STATE_DIMENSION_COLORS = {
        "shared_thoughts": "#2CA02C",
        "shared_feelings": "#D62728",
        "shared_beliefs": "#9467BD",
        "none": "#E8E8E8",
    }

    CONVERSATIONAL_GOAL_VALUES = [
        "stance_sharing", "perspective_seeking", "finding_commonality",
        "persuading", "sharing_experience", "rapport_building",
        "exploring_nuance", "other",
    ]
    CONVERSATIONAL_GOAL_LABELS = [
        "Stance sharing", "Perspective seeking", "Finding commonality",
        "Persuading", "Sharing experience", "Rapport building",
        "Exploring nuance", "Other",
    ]
    CONVERSATIONAL_GOAL_COLORS = {
        "stance_sharing": "#1F77B4",
        "perspective_seeking": "#FF7F0E",
        "finding_commonality": "#2CA02C",
        "persuading": "#D62728",
        "sharing_experience": "#9467BD",
        "rapport_building": "#C0C0C0",
        "exploring_nuance": "#8C564B",
        "other": "#7F7F7F",
    }

    return (
        BIN_SECONDS,
        BoundaryNorm,
        CONVERSATIONAL_GOAL_COLORS,
        CONVERSATIONAL_GOAL_LABELS,
        CONVERSATIONAL_GOAL_VALUES,
        DATA_DIR,
        FIGURES_DIR,
        INITIAL_ALIGNMENT_VALUES,
        INNER_STATE_DIMENSION_COLORS,
        INNER_STATE_DIMENSION_LABELS,
        INNER_STATE_DIMENSION_VALUES,
        ListedColormap,
        N_BOOT,
        Path,
        RNG,
        STANCE_COLORS,
        STANCE_ORDER,
        TOPIC_SCOPE_COLORS,
        TOPIC_SCOPE_LABELS,
        TOPIC_SCOPE_VALUES,
        glmer,
        mannwhitneyu,
        mo,
        np,
        pd,
        pl,
        plt,
        subprocess,
        tempfile,
    )


@app.cell
def _(mo):
    mo.md("""
    # Chat LLM Annotation Analyses

    Analyzes LLM-annotated **shared reality** emergence over conversation time,
    comparing **opposing** vs **shared** stance dyads.

    Each 15-second time bin is independently classified (with cumulative context) on:
    - **Stance establishment** (`focal_stance_revealed`, `stance_in_this_window`, `initial_alignment`)
    - **Shared reality** (`post_stance_shared_reality`, `post_stance_topic_scope`, `post_stance_inner_state_dimension`)
    - **Conversational goal** (`post_stance_conversational_goal`)
    - **Differences** (`post_stance_differences`)
    """)
    return


# ── Utility functions ─────────────────────────────────────────────────────


@app.cell
def _(N_BOOT, RNG, np, pd):
    def cluster_bootstrap_mean_ci(
        values: pd.Series,
        cluster_ids: pd.Series,
        n_boot: int = N_BOOT,
    ) -> tuple[float, float, float]:
        """Bootstrap mean and 95% CI, resampling at the cluster level."""
        unique_ids = cluster_ids.unique()
        n_clusters = len(unique_ids)
        if n_clusters == 0:
            return np.nan, np.nan, np.nan
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            sampled = RNG.choice(unique_ids, size=n_clusters, replace=True)
            mask = cluster_ids.isin(sampled)
            boot_means[b] = values[mask].mean()
        return values.mean(), *np.percentile(boot_means, [2.5, 97.5])

    def kaplan_meier(
        times: np.ndarray, events: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Kaplan-Meier survival curve."""
        order = np.argsort(times)
        t_sorted, e_sorted = times[order], events[order]
        unique_times = np.unique(t_sorted)
        n_at_risk = len(t_sorted)
        surv = 1.0
        km_times, km_surv = [0], [1.0]
        for t in unique_times:
            mask = t_sorted == t
            d = e_sorted[mask].sum()
            n_lost = mask.sum()
            if n_at_risk > 0 and d > 0:
                surv *= 1 - d / n_at_risk
            km_times.append(t)
            km_surv.append(surv)
            n_at_risk -= n_lost
        return np.array(km_times), np.array(km_surv)

    def km_median(km_times: np.ndarray, km_surv: np.ndarray) -> float:
        """Find median survival time (first time S(t) <= 0.5)."""
        below = np.where(km_surv <= 0.5)[0]
        return km_times[below[0]] if len(below) > 0 else np.inf

    def fmt_polars_table(df, title: str = "") -> str:
        """Format a polars DataFrame as a markdown code block."""
        lines = []
        if title:
            lines.append(f"**{title}**\n")
        lines.append(f"```\n{df}\n```")
        return "\n".join(lines)

    return cluster_bootstrap_mean_ci, fmt_polars_table, kaplan_meier, km_median


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: Load & prepare annotation data
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(BIN_SECONDS, DATA_DIR, pd, mo):
    # Load per-bin annotation results (hybrid: cumulative context, per-bin judgment)
    _csv_path = DATA_DIR / "llm_results" / f"perbin_commonality_timecourse_{BIN_SECONDS}s.csv"
    df_all = pd.read_csv(_csv_path)

    # Compute time_seconds from time_bin if not present
    if "time_seconds" not in df_all.columns:
        df_all["time_seconds"] = (df_all["time_bin"] + 1) * BIN_SECONDS

    # Filter to post-stance bins only (where stances have been established)
    df_post = df_all[df_all["focal_stance_revealed"] == True].copy()

    # All time bins (including pre-stance) for stance-establishment analyses
    time_bins_all = sorted(df_all["time_seconds"].unique())
    # Post-stance time bins only
    time_bins_post = sorted(df_post["time_seconds"].unique())

    mo.md(f"""
    ### Data loaded

    - **Total annotations**: {len(df_all)} ({df_all['group_id'].nunique()} dyads)
    - **Post-stance annotations**: {len(df_post)} ({df_post['group_id'].nunique()} dyads)
    - **Time bins (all)**: {len(time_bins_all)} ({time_bins_all[0]}s – {time_bins_all[-1]}s)
    - **Time bins (post-stance)**: {len(time_bins_post)} ({time_bins_post[0]}s – {time_bins_post[-1]}s)
    """)
    return df_all, df_post, time_bins_all, time_bins_post


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: Stance establishment timeline
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(STANCE_COLORS, STANCE_ORDER, df_all, mo, np, pd, plt, FIGURES_DIR):
    # Find the first bin where stances are established for each dyad
    _stance_rows = []
    for _gid, _gdf in df_all.groupby("group_id"):
        _revealed = _gdf[_gdf["focal_stance_revealed"] == True]
        _stance = _gdf["stance"].iloc[0] if "stance" in _gdf.columns else "unknown"
        if not _revealed.empty:
            _first_bin = _revealed["time_seconds"].min()
        else:
            _first_bin = np.nan
        _stance_rows.append({
            "group_id": _gid,
            "stance": _stance,
            "first_stance_time": _first_bin,
        })
    stance_timing_df = pd.DataFrame(_stance_rows)

    # Summary table
    _lines = ["## Stance Establishment Timeline\n"]
    for _s in STANCE_ORDER:
        _sub = stance_timing_df[stance_timing_df["stance"] == _s]
        _revealed = _sub["first_stance_time"].dropna()
        _lines.append(f"**{_s.capitalize()}** (N={len(_sub)}):")
        _lines.append(f"  - Stance revealed: {len(_revealed)}/{len(_sub)} dyads")
        if len(_revealed) > 0:
            _lines.append(f"  - Median time: {_revealed.median():.0f}s")
            _lines.append(f"  - Mean time: {_revealed.mean():.1f}s")
            _lines.append(f"  - Range: {_revealed.min():.0f}s – {_revealed.max():.0f}s")
        _lines.append("")

    # Histogram
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _s in STANCE_ORDER:
        _vals = stance_timing_df[stance_timing_df["stance"] == _s]["first_stance_time"].dropna()
        _ax.hist(_vals, bins=15, alpha=0.5, color=STANCE_COLORS[_s], label=_s.capitalize(),
                 edgecolor="white")
    _ax.set_xlabel("Time of stance establishment (seconds)", fontsize=11)
    _ax.set_ylabel("Number of dyads", fontsize=11)
    _ax.set_title("When do dyads establish their stances?", fontsize=12, fontweight="bold")
    _ax.legend(title="Stance condition")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_s_stance_timing.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("/tmp/figure_s_stance_timing.png", bbox_inches="tight", dpi=150)

    mo.md("\n".join(_lines))
    return (stance_timing_df,)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: P(Shared Reality) over time
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(STANCE_ORDER, cluster_bootstrap_mean_ci, df_post, pd, time_bins_post):
    # Compute P(shared reality) at each time bin, by stance condition
    _results = []
    for _stance in STANCE_ORDER:
        for _t in time_bins_post:
            _sub = df_post[
                (df_post["stance"] == _stance)
                & (df_post["time_seconds"] == _t)
            ]
            if len(_sub) == 0:
                continue
            _vals = _sub["post_stance_shared_reality"].astype(int)
            _ids = _sub["group_id"]
            _mean, _lo, _hi = cluster_bootstrap_mean_ci(_vals, _ids)
            _results.append({
                "stance": _stance,
                "time_seconds": _t,
                "pct_sr": _mean * 100,
                "ci_lo": _lo * 100,
                "ci_hi": _hi * 100,
                "n_dyads": len(_sub),
            })
    sr_timecourse_df = pd.DataFrame(_results)
    return (sr_timecourse_df,)


@app.cell
def _(STANCE_ORDER, mo, sr_timecourse_df):
    _lines = ["## P(Shared Reality) Over Time (cluster-bootstrapped CIs)\n"]
    for _stance in STANCE_ORDER:
        _sub = sr_timecourse_df[sr_timecourse_df["stance"] == _stance]
        _lines.append(f"**{_stance.capitalize()}:**\n")
        _lines.append("| Time | % SR | 95% CI | N |")
        _lines.append("|------|------|--------|---|")
        for _, _row in _sub.iterrows():
            _lines.append(
                f"| {_row['time_seconds']:.0f}s | {_row['pct_sr']:.1f}% "
                f"| [{_row['ci_lo']:.1f}, {_row['ci_hi']:.1f}] | {_row['n_dyads']:.0f} |"
            )
        _lines.append("")
    mo.md("\n".join(_lines))
    return


@app.cell
def _(
    FIGURES_DIR,
    STANCE_COLORS,
    STANCE_ORDER,
    plt,
    sr_timecourse_df,
    time_bins_post,
):
    # Figure: P(shared reality) over time by stance
    _fig, _ax = plt.subplots(figsize=(10, 5))

    for _stance in STANCE_ORDER:
        _sub = sr_timecourse_df[sr_timecourse_df["stance"] == _stance]
        _color = STANCE_COLORS[_stance]
        _ax.plot(
            _sub["time_seconds"], _sub["pct_sr"],
            "o-", color=_color, label=_stance.capitalize(), markersize=5, linewidth=2,
        )
        _ax.fill_between(
            _sub["time_seconds"], _sub["ci_lo"], _sub["ci_hi"],
            alpha=0.15, color=_color,
        )
        for _, _row in _sub.iterrows():
            _ax.annotate(
                f"n={_row['n_dyads']:.0f}",
                (_row["time_seconds"], _row["ci_hi"]),
                textcoords="offset points", xytext=(0, 6),
                fontsize=6, color=_color, alpha=0.7, ha="center",
            )

    _ax.set_xlabel("Time into conversation (seconds)", fontsize=11)
    _ax.set_ylabel("% of dyads with shared reality", fontsize=11)
    _ax.set_yticks([0, 25, 50, 75, 100])
    _ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    _ax.set_ylim(0, 100)
    _ax.set_xlim(time_bins_post[0] - 5, time_bins_post[-1] + 5)
    _ax.legend(title="Stance condition", fontsize=10, title_fontsize=10)
    _ax.set_title(
        "Shared reality emergence over time (post-stance bins)",
        fontsize=12, fontweight="bold", loc="left",
    )
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_s_sr_timecourse.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("/tmp/figure_s_sr_timecourse.png", bbox_inches="tight", dpi=150)
    _fig
    return


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: GLMER – P(Shared Reality) ~ Time × Stance
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(df_post, fmt_polars_table, glmer, mo, pl):
    try:
        _data = df_post.copy()
        _data["time_min"] = _data["time_seconds"] / 60.0
        _data["time_c"] = _data["time_min"] - _data["time_min"].mean()
        _data["stance_e"] = _data["stance"].map({"opposing": 0.5, "shared": -0.5})
        _data["sr_binary"] = _data["post_stance_shared_reality"].astype(int)

        _pl = pl.from_pandas(_data[["sr_binary", "time_c", "stance_e", "group_id"]])

        _sr_time_model = glmer(
            "sr_binary ~ time_c * stance_e + (1 + time_c | group_id)",
            data=_pl,
            family="binomial",
        )
        _sr_time_model.fit(verbose=False)

        _lines = [
            "## GLMER: P(Shared Reality) Over Time\n",
            "Model: `sr_binary ~ time_c * stance_e + (1 + time_c | group_id)`\n",
            fmt_polars_table(sr_time_model.result_fit, "Fixed effects"),
            "\n",
            fmt_polars_table(sr_time_model.result_fit_stats, "Fit statistics"),
        ]
        mo.md("\n".join(_lines))
    except Exception as _e:
        mo.md(f"## GLMER: P(Shared Reality) Over Time\n\n⚠️ Model fitting failed: `{_e}`")
    return


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: Topic scope over time (stacked bars)
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(
    BIN_SECONDS,
    FIGURES_DIR,
    STANCE_ORDER,
    TOPIC_SCOPE_COLORS,
    TOPIC_SCOPE_LABELS,
    TOPIC_SCOPE_VALUES,
    df_post,
    np,
    plt,
    time_bins_post,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for _i, _stance in enumerate(STANCE_ORDER):
        _ax = _axes[_i]
        _sub = df_post[df_post["stance"] == _stance]

        _substantive = [t for t in TOPIC_SCOPE_VALUES if t != "none"]
        _bottoms = np.zeros(len(time_bins_post))

        for _ts in _substantive:
            _pcts = []
            for _t in time_bins_post:
                _bin_data = _sub[_sub["time_seconds"] == _t]
                _n = len(_bin_data)
                _count = (_bin_data["post_stance_topic_scope"] == _ts).sum()
                _pcts.append(_count / _n * 100 if _n > 0 else 0)
            _pcts = np.array(_pcts)
            _idx = TOPIC_SCOPE_VALUES.index(_ts)

            _ax.bar(
                time_bins_post, _pcts, bottom=_bottoms, width=BIN_SECONDS * 0.8,
                color=TOPIC_SCOPE_COLORS[_ts],
                label=TOPIC_SCOPE_LABELS[_idx],
                edgecolor="white", linewidth=0.5,
            )
            _bottoms += _pcts

        _ax.set_xlabel("Time (seconds)", fontsize=10)
        _ax.set_ylabel("% of dyads", fontsize=10)
        _ax.set_ylim(0, 100)
        _ax.set_yticks([0, 25, 50, 75, 100])
        _ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        _ax.set_title(
            f"Topic scope — {_stance.capitalize()} stance",
            fontsize=11, fontweight="bold", loc="left",
        )
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

        if _i == 1:
            _ax.legend(fontsize=8, loc="upper left", title="Topic scope", title_fontsize=9)

    plt.suptitle(
        "Topic scope of shared reality over time (post-stance bins)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIGURES_DIR / "figure_s_topic_scope_timecourse.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("/tmp/figure_s_topic_scope_timecourse.png", bbox_inches="tight", dpi=150)
    _fig
    return


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: Inner state dimension over time (stacked bars)
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(
    BIN_SECONDS,
    FIGURES_DIR,
    INNER_STATE_DIMENSION_COLORS,
    INNER_STATE_DIMENSION_LABELS,
    INNER_STATE_DIMENSION_VALUES,
    STANCE_ORDER,
    df_post,
    np,
    plt,
    time_bins_post,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for _i, _stance in enumerate(STANCE_ORDER):
        _ax = _axes[_i]
        _sub = df_post[df_post["stance"] == _stance]

        _substantive = [d for d in INNER_STATE_DIMENSION_VALUES if d != "none"]
        _bottoms = np.zeros(len(time_bins_post))

        for _dim in _substantive:
            _pcts = []
            for _t in time_bins_post:
                _bin_data = _sub[_sub["time_seconds"] == _t]
                _n = len(_bin_data)
                _count = (_bin_data["post_stance_inner_state_dimension"] == _dim).sum()
                _pcts.append(_count / _n * 100 if _n > 0 else 0)
            _pcts = np.array(_pcts)
            _idx = INNER_STATE_DIMENSION_VALUES.index(_dim)

            _ax.bar(
                time_bins_post, _pcts, bottom=_bottoms, width=BIN_SECONDS * 0.8,
                color=INNER_STATE_DIMENSION_COLORS[_dim],
                label=INNER_STATE_DIMENSION_LABELS[_idx],
                edgecolor="white", linewidth=0.5,
            )
            _bottoms += _pcts

        _ax.set_xlabel("Time (seconds)", fontsize=10)
        _ax.set_ylabel("% of dyads", fontsize=10)
        _ax.set_ylim(0, 100)
        _ax.set_yticks([0, 25, 50, 75, 100])
        _ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        _ax.set_title(
            f"Inner state dimension — {_stance.capitalize()} stance",
            fontsize=11, fontweight="bold", loc="left",
        )
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

        if _i == 1:
            _ax.legend(fontsize=8, loc="upper left", title="Inner state", title_fontsize=9)

    plt.suptitle(
        "Inner state dimension of shared reality over time (post-stance bins)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIGURES_DIR / "figure_s_inner_state_timecourse.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("/tmp/figure_s_inner_state_timecourse.png", bbox_inches="tight", dpi=150)
    _fig
    return


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7: Conversational goal over time (stacked bars)
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(
    BIN_SECONDS,
    CONVERSATIONAL_GOAL_COLORS,
    CONVERSATIONAL_GOAL_LABELS,
    CONVERSATIONAL_GOAL_VALUES,
    FIGURES_DIR,
    STANCE_ORDER,
    df_post,
    np,
    plt,
    time_bins_post,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for _i, _stance in enumerate(STANCE_ORDER):
        _ax = _axes[_i]
        _sub = df_post[df_post["stance"] == _stance]
        _bottoms = np.zeros(len(time_bins_post))

        for _goal in CONVERSATIONAL_GOAL_VALUES:
            _pcts = []
            for _t in time_bins_post:
                _bin_data = _sub[_sub["time_seconds"] == _t]
                _n = len(_bin_data)
                _count = (_bin_data["post_stance_conversational_goal"] == _goal).sum()
                _pcts.append(_count / _n * 100 if _n > 0 else 0)
            _pcts = np.array(_pcts)
            _idx = CONVERSATIONAL_GOAL_VALUES.index(_goal)

            _ax.bar(
                time_bins_post, _pcts, bottom=_bottoms, width=BIN_SECONDS * 0.8,
                color=CONVERSATIONAL_GOAL_COLORS[_goal],
                label=CONVERSATIONAL_GOAL_LABELS[_idx],
                edgecolor="white", linewidth=0.5,
            )
            _bottoms += _pcts

        _ax.set_xlabel("Time (seconds)", fontsize=10)
        _ax.set_ylabel("% of dyads", fontsize=10)
        _ax.set_ylim(0, 100)
        _ax.set_yticks([0, 25, 50, 75, 100])
        _ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        _ax.set_title(
            f"Conversational goal — {_stance.capitalize()} stance",
            fontsize=11, fontweight="bold", loc="left",
        )
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

        if _i == 1:
            _ax.legend(fontsize=7, loc="upper left", title="Goal", title_fontsize=8)

    plt.suptitle(
        "Conversational goal over time (post-stance bins)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIGURES_DIR / "figure_s_convo_goal_timecourse.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("/tmp/figure_s_convo_goal_timecourse.png", bbox_inches="tight", dpi=150)
    _fig
    return


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8: P(Differences) over time
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(
    FIGURES_DIR,
    STANCE_COLORS,
    STANCE_ORDER,
    cluster_bootstrap_mean_ci,
    df_post,
    pd,
    plt,
    time_bins_post,
):
    # P(differences) over time by stance
    _diff_results = []
    for _stance in STANCE_ORDER:
        for _t in time_bins_post:
            _sub = df_post[
                (df_post["stance"] == _stance) & (df_post["time_seconds"] == _t)
            ]
            if len(_sub) == 0:
                continue
            _vals = _sub["post_stance_differences"].astype(int)
            _ids = _sub["group_id"]
            _mean, _lo, _hi = cluster_bootstrap_mean_ci(_vals, _ids)
            _diff_results.append({
                "stance": _stance,
                "time_seconds": _t,
                "pct_diff": _mean * 100,
                "ci_lo": _lo * 100,
                "ci_hi": _hi * 100,
                "n_dyads": len(_sub),
            })
    _diff_df = pd.DataFrame(_diff_results)

    _fig, _ax = plt.subplots(figsize=(10, 5))
    for _stance in STANCE_ORDER:
        _sub = _diff_df[_diff_df["stance"] == _stance]
        _color = STANCE_COLORS[_stance]
        _ax.plot(
            _sub["time_seconds"], _sub["pct_diff"],
            "o-", color=_color, label=_stance.capitalize(), markersize=5, linewidth=2,
        )
        _ax.fill_between(
            _sub["time_seconds"], _sub["ci_lo"], _sub["ci_hi"],
            alpha=0.15, color=_color,
        )

    _ax.set_xlabel("Time into conversation (seconds)", fontsize=11)
    _ax.set_ylabel("% of dyads with differences", fontsize=11)
    _ax.set_ylim(0, 100)
    _ax.set_yticks([0, 25, 50, 75, 100])
    _ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    _ax.legend(title="Stance condition", fontsize=10, title_fontsize=10)
    _ax.set_title(
        "Post-stance differences over time",
        fontsize=12, fontweight="bold", loc="left",
    )
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_s_differences_timecourse.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("/tmp/figure_s_differences_timecourse.png", bbox_inches="tight", dpi=150)
    _fig
    return


# ══════════════════════════════════════════════════════════════════════════
# SECTION 9: Descriptive tables
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(
    STANCE_ORDER,
    TOPIC_SCOPE_VALUES,
    INNER_STATE_DIMENSION_VALUES,
    CONVERSATIONAL_GOAL_VALUES,
    df_post,
    mo,
    time_bins_post,
):
    _lines = ["## Annotation Distributions (post-stance bins)\n"]

    # Topic scope by stance × time
    _lines.append("### Topic Scope\n")
    for _stance in STANCE_ORDER:
        _lines.append(f"**{_stance.capitalize()}**\n")
        _lines.append("| Time | " + " | ".join(TOPIC_SCOPE_VALUES) + " | N |")
        _lines.append("|------|" + "|".join(["------"] * len(TOPIC_SCOPE_VALUES)) + "|---|")
        for _t in time_bins_post:
            _sub = df_post[
                (df_post["stance"] == _stance) & (df_post["time_seconds"] == _t)
            ]
            _n = len(_sub)
            if _n == 0:
                continue
            _pcts = []
            for _ts in TOPIC_SCOPE_VALUES:
                _count = (_sub["post_stance_topic_scope"] == _ts).sum()
                _pcts.append(f"{_count / _n * 100:.0f}%")
            _lines.append(f"| {_t:.0f}s | " + " | ".join(_pcts) + f" | {_n} |")
        _lines.append("")

    # Inner state dimension
    _lines.append("### Inner State Dimension\n")
    for _stance in STANCE_ORDER:
        _lines.append(f"**{_stance.capitalize()}**\n")
        _lines.append("| Time | " + " | ".join(INNER_STATE_DIMENSION_VALUES) + " | N |")
        _lines.append("|------|" + "|".join(["------"] * len(INNER_STATE_DIMENSION_VALUES)) + "|---|")
        for _t in time_bins_post:
            _sub = df_post[
                (df_post["stance"] == _stance) & (df_post["time_seconds"] == _t)
            ]
            _n = len(_sub)
            if _n == 0:
                continue
            _pcts = []
            for _dim in INNER_STATE_DIMENSION_VALUES:
                _count = (_sub["post_stance_inner_state_dimension"] == _dim).sum()
                _pcts.append(f"{_count / _n * 100:.0f}%")
            _lines.append(f"| {_t:.0f}s | " + " | ".join(_pcts) + f" | {_n} |")
        _lines.append("")

    # Conversational goal
    _lines.append("### Conversational Goal\n")
    for _stance in STANCE_ORDER:
        _lines.append(f"**{_stance.capitalize()}**\n")
        _lines.append("| Time | " + " | ".join(CONVERSATIONAL_GOAL_VALUES) + " | N |")
        _lines.append("|------|" + "|".join(["------"] * len(CONVERSATIONAL_GOAL_VALUES)) + "|---|")
        for _t in time_bins_post:
            _sub = df_post[
                (df_post["stance"] == _stance) & (df_post["time_seconds"] == _t)
            ]
            _n = len(_sub)
            if _n == 0:
                continue
            _pcts = []
            for _goal in CONVERSATIONAL_GOAL_VALUES:
                _count = (_sub["post_stance_conversational_goal"] == _goal).sum()
                _pcts.append(f"{_count / _n * 100:.0f}%")
            _lines.append(f"| {_t:.0f}s | " + " | ".join(_pcts) + f" | {_n} |")
        _lines.append("")

    mo.md("\n".join(_lines))
    return


# ══════════════════════════════════════════════════════════════════════════
# SECTION 10: Survival analysis — time to first shared reality
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(
    FIGURES_DIR,
    N_BOOT,
    RNG,
    STANCE_COLORS,
    STANCE_ORDER,
    df_post,
    kaplan_meier,
    km_median,
    mannwhitneyu,
    mo,
    np,
    pd,
    plt,
):
    _survival_rows = []
    for _gid, _gdf in df_post.groupby("group_id"):
        _stance = _gdf["stance"].iloc[0]
        _sr_bins = _gdf[_gdf["post_stance_shared_reality"] == True]
        if not _sr_bins.empty:
            _time = _sr_bins["time_seconds"].min()
            _event = 1
        else:
            _time = _gdf["time_seconds"].max()
            _event = 0
        _survival_rows.append({
            "group_id": _gid, "stance": _stance,
            "time": _time, "event": _event,
        })
    _surv_df = pd.DataFrame(_survival_rows)

    _lines = ["## Survival Analysis: Time to First Shared Reality\n"]

    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _s in STANCE_ORDER:
        _sub = _surv_df[_surv_df["stance"] == _s]
        _km_t, _km_s = kaplan_meier(_sub["time"].values, _sub["event"].values)
        _med = km_median(_km_t, _km_s)
        _ax.step(_km_t, _km_s * 100, where="post", color=STANCE_COLORS[_s],
                 label=f"{_s.capitalize()} (median={_med:.0f}s)", linewidth=2)
        _lines.append(f"**{_s.capitalize()}**: median={_med:.0f}s, "
                      f"found SR: {_sub['event'].sum()}/{len(_sub)}")

    # Mann-Whitney test on time to first SR (among those who found it)
    _opp = _surv_df[(_surv_df["stance"] == "opposing") & (_surv_df["event"] == 1)]["time"]
    _sha = _surv_df[(_surv_df["stance"] == "shared") & (_surv_df["event"] == 1)]["time"]
    if len(_opp) > 0 and len(_sha) > 0:
        _u, _p = mannwhitneyu(_opp, _sha, alternative="two-sided")
        _lines.append(f"\nMann-Whitney U (among dyads with SR): U={_u:.0f}, p={_p:.4f}")

    _ax.set_xlabel("Time (seconds)", fontsize=11)
    _ax.set_ylabel("% of dyads without shared reality", fontsize=11)
    _ax.set_ylim(0, 105)
    _ax.legend(fontsize=10)
    _ax.set_title("Time to first shared reality", fontsize=12, fontweight="bold")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_s_sr_survival.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("/tmp/figure_s_sr_survival.png", bbox_inches="tight", dpi=150)

    mo.md("\n".join(_lines))
    return


# ══════════════════════════════════════════════════════════════════════════
# SECTION 11: Heatmap — dyads × time bins (topic scope)
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(
    BIN_SECONDS,
    BoundaryNorm,
    FIGURES_DIR,
    ListedColormap,
    STANCE_ORDER,
    TOPIC_SCOPE_COLORS,
    TOPIC_SCOPE_LABELS,
    TOPIC_SCOPE_VALUES,
    df_post,
    np,
    plt,
    time_bins_post,
):
    # Encode topic scope as integers for heatmap
    _ts_to_int = {ts: i for i, ts in enumerate(TOPIC_SCOPE_VALUES)}
    _n_types = len(TOPIC_SCOPE_VALUES)
    _cmap = ListedColormap([TOPIC_SCOPE_COLORS[ts] for ts in TOPIC_SCOPE_VALUES])
    _norm = BoundaryNorm(range(_n_types + 1), _cmap.N)

    _fig, _axes = plt.subplots(1, 2, figsize=(16, 10), sharey=False)

    for _i, _stance in enumerate(STANCE_ORDER):
        _ax = _axes[_i]
        _sub = df_post[df_post["stance"] == _stance]
        _dyads = sorted(_sub["group_id"].unique())
        _n_dyads = len(_dyads)
        _n_bins = len(time_bins_post)

        # Build matrix: rows=dyads, cols=time bins
        _matrix = np.full((_n_dyads, _n_bins), _ts_to_int["none"])
        _dyad_to_row = {d: r for r, d in enumerate(_dyads)}

        for _, _row in _sub.iterrows():
            _r = _dyad_to_row[_row["group_id"]]
            _c = time_bins_post.index(_row["time_seconds"])
            _ts = _row.get("post_stance_topic_scope", "none")
            _matrix[_r, _c] = _ts_to_int.get(_ts, _ts_to_int["none"])

        _im = _ax.imshow(
            _matrix, aspect="auto", cmap=_cmap, norm=_norm, interpolation="nearest",
        )
        _ax.set_xticks(range(_n_bins))
        _ax.set_xticklabels(
            [f"{t}s" for t in time_bins_post], fontsize=7, rotation=45,
        )
        _ax.set_xlabel("Time into conversation", fontsize=10)
        _ax.set_ylabel("Dyads", fontsize=10)
        _ax.set_title(
            f"{_stance.capitalize()} stance (N={_n_dyads})",
            fontsize=11, fontweight="bold",
        )
        _ax.set_yticks([])

    _cbar_ax = _fig.add_axes([0.15, 0.02, 0.7, 0.02])
    _cbar = _fig.colorbar(
        _im, cax=_cbar_ax, orientation="horizontal",
        ticks=np.arange(_n_types) + 0.5,
    )
    _cbar.ax.set_xticklabels(TOPIC_SCOPE_LABELS, fontsize=9)
    _cbar.set_label("Topic Scope", fontsize=10)

    plt.suptitle(
        "Topic scope per time bin (each row = one dyad)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig(FIGURES_DIR / "figure_s_topic_scope_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("/tmp/figure_s_topic_scope_heatmap.png", bbox_inches="tight", dpi=150)
    _fig
    return


# ══════════════════════════════════════════════════════════════════════════
# SECTION 12: Merge with behavioral data (predictShared)
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(DATA_DIR, df_post, pd, mo):
    # Load behavioral responses
    _resp = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)

    # Summarize annotations per dyad: did they EVER find shared reality post-stance?
    _dyad_summary = (
        df_post.groupby("group_id")
        .agg(
            ever_sr=("post_stance_shared_reality", "any"),
            n_sr_bins=("post_stance_shared_reality", "sum"),
            n_bins=("post_stance_shared_reality", "count"),
            stance=("stance", "first"),
        )
        .reset_index()
    )
    _dyad_summary["ever_sr"] = _dyad_summary["ever_sr"].astype(int)
    _dyad_summary["pct_sr_bins"] = _dyad_summary["n_sr_bins"] / _dyad_summary["n_bins"]

    # Also get the "best" topic scope per dyad (most substantive across all bins)
    # Priority: same_values > related_subtopic > different_topic > rapport_only > none
    _scope_priority = {
        "same_values": 4, "related_subtopic": 3, "different_topic": 2,
        "rapport_only": 1, "none": 0,
    }
    _scope_rows = []
    for _gid, _gdf in df_post.groupby("group_id"):
        _scopes = _gdf["post_stance_topic_scope"].map(_scope_priority)
        _best_idx = _scopes.idxmax()
        _scope_rows.append({
            "group_id": _gid,
            "best_topic_scope": _gdf.loc[_best_idx, "post_stance_topic_scope"],
        })
    _scope_df = pd.DataFrame(_scope_rows)
    _dyad_summary = _dyad_summary.merge(_scope_df, on="group_id")

    # Merge with behavioral responses (inference questions only)
    _resp_infer = _resp[
        (_resp["question_type"] != "observed") & (_resp["experiment"] == "chat")
    ].copy()
    merged_df = _resp_infer.merge(
        _dyad_summary.rename(columns={"group_id": "groupId"}),
        on="groupId",
        how="inner",
        suffixes=("", "_annot"),
    )
    # Use annotation stance if behavioral stance is missing
    if "stance_annot" in merged_df.columns:
        merged_df["stance"] = merged_df["stance"].fillna(merged_df["stance_annot"])
        merged_df.drop(columns=["stance_annot"], inplace=True)

    # Effect coding
    merged_df["stance_e"] = merged_df["stance"].map({"opposing": 0.5, "shared": -0.5})

    # No-chat baseline
    nochat_df = _resp[
        (_resp["experiment"] == "no-chat") & (_resp["question_type"] != "observed")
    ].copy()
    nochat_baseline = nochat_df.groupby("stance")["predictShared"].mean().to_dict()

    mo.md(f"""
    ### Behavioral data merged with annotations

    - **Chat inference responses**: {len(merged_df)} ({merged_df['groupId'].nunique()} dyads)
    - **Dyads with any SR**: {_dyad_summary['ever_sr'].sum()}/{len(_dyad_summary)}
    - **No-chat baseline P(predictShared)**: opposing={nochat_baseline.get('opposing', 'N/A'):.3f}, shared={nochat_baseline.get('shared', 'N/A'):.3f}

    Per-dyad summary:
    - **Best topic scope distribution**: {_dyad_summary['best_topic_scope'].value_counts().to_dict()}
    """)
    return merged_df, nochat_baseline, nochat_df


# ══════════════════════════════════════════════════════════════════════════
# SECTION 13: The gradient finding — opposing dyads who found SR
# ══════════════════════════════════════════════════════════════════════════
#
# Key hypothesis: opposing-stance dyads who discover post-stance shared
# reality are the ones driving the surprisingly high commonality expectations
# (predictShared) for opposing dyads — approaching or exceeding shared-stance
# levels, and well above the no-chat opposing baseline.


@app.cell
def _(
    FIGURES_DIR,
    STANCE_COLORS,
    cluster_bootstrap_mean_ci,
    merged_df,
    mo,
    nochat_baseline,
    np,
    pd,
    plt,
):
    # ── Descriptive stats ──
    _lines = ["## The Gradient: Post-Stance Shared Reality Drives Commonality Expectations\n"]
    _lines.append("Among **opposing-stance** dyads, do those who found shared reality ")
    _lines.append("have higher commonality expectations than those who didn't?\n")

    # Split opposing dyads by whether they found SR
    _opp = merged_df[merged_df["stance"] == "opposing"]
    _opp_sr = _opp[_opp["ever_sr"] == 1]
    _opp_no_sr = _opp[_opp["ever_sr"] == 0]
    _shared = merged_df[merged_df["stance"] == "shared"]

    # Compute P(predictShared) for each group
    _groups = {
        "No-chat opposing (baseline)": {
            "mean": nochat_baseline.get("opposing", np.nan),
            "ci_lo": np.nan, "ci_hi": np.nan,
            "n_responses": 0, "n_dyads": 0,
            "color": "#AAAAAA",
        },
        "Opposing — no SR found": {
            "mean": _opp_no_sr["predictShared"].mean() * 100 if len(_opp_no_sr) else np.nan,
            "n_responses": len(_opp_no_sr),
            "n_dyads": _opp_no_sr["groupId"].nunique(),
            "color": STANCE_COLORS["opposing"],
        },
        "Opposing — SR found": {
            "mean": _opp_sr["predictShared"].mean() * 100 if len(_opp_sr) else np.nan,
            "n_responses": len(_opp_sr),
            "n_dyads": _opp_sr["groupId"].nunique(),
            "color": "#3A5FCD",  # darker blue
        },
        "No-chat shared (baseline)": {
            "mean": nochat_baseline.get("shared", np.nan),
            "ci_lo": np.nan, "ci_hi": np.nan,
            "n_responses": 0, "n_dyads": 0,
            "color": "#CCCCCC",
        },
        "Shared — all": {
            "mean": _shared["predictShared"].mean() * 100 if len(_shared) else np.nan,
            "n_responses": len(_shared),
            "n_dyads": _shared["groupId"].nunique(),
            "color": STANCE_COLORS["shared"],
        },
    }

    # Cluster-bootstrapped CIs for chat groups
    for _label, _data, _ids in [
        ("Opposing — no SR found", _opp_no_sr, _opp_no_sr["groupId"] if len(_opp_no_sr) else pd.Series()),
        ("Opposing — SR found", _opp_sr, _opp_sr["groupId"] if len(_opp_sr) else pd.Series()),
        ("Shared — all", _shared, _shared["groupId"] if len(_shared) else pd.Series()),
    ]:
        if len(_data) > 0:
            _m, _lo, _hi = cluster_bootstrap_mean_ci(
                _data["predictShared"], _ids,
            )
            _groups[_label]["mean"] = _m * 100
            _groups[_label]["ci_lo"] = _lo * 100
            _groups[_label]["ci_hi"] = _hi * 100

    # No-chat baselines: scale to percent
    for _k in ["No-chat opposing (baseline)", "No-chat shared (baseline)"]:
        if not np.isnan(_groups[_k]["mean"]):
            _groups[_k]["mean"] *= 100

    # Table
    _lines.append("| Group | P(predictShared) | 95% CI | N responses | N dyads |")
    _lines.append("|-------|-----------------|--------|-------------|---------|")
    for _label, _vals in _groups.items():
        _ci = (f"[{_vals.get('ci_lo', np.nan):.1f}, {_vals.get('ci_hi', np.nan):.1f}]"
               if not np.isnan(_vals.get("ci_lo", np.nan)) else "—")
        _lines.append(
            f"| {_label} | {_vals['mean']:.1f}% | {_ci} "
            f"| {_vals['n_responses']} | {_vals['n_dyads']} |"
        )
    _lines.append("")

    # ── Figure ──
    _fig, _ax = plt.subplots(figsize=(10, 6))

    _bar_labels = list(_groups.keys())
    _bar_means = [_groups[k]["mean"] for k in _bar_labels]
    _bar_colors = [_groups[k]["color"] for k in _bar_labels]
    _bar_errors_lo = [
        _groups[k]["mean"] - _groups[k].get("ci_lo", _groups[k]["mean"])
        for k in _bar_labels
    ]
    _bar_errors_hi = [
        _groups[k].get("ci_hi", _groups[k]["mean"]) - _groups[k]["mean"]
        for k in _bar_labels
    ]

    _x = np.arange(len(_bar_labels))
    _ax.bar(_x, _bar_means, color=_bar_colors, edgecolor="white", linewidth=1.5, width=0.6)
    _ax.errorbar(
        _x, _bar_means,
        yerr=[_bar_errors_lo, _bar_errors_hi],
        fmt="none", ecolor="black", capsize=4, linewidth=1.5,
    )

    # Add no-chat opposing baseline as horizontal dashed line
    _opp_base = _groups["No-chat opposing (baseline)"]["mean"]
    if not np.isnan(_opp_base):
        _ax.axhline(_opp_base, color="#AAAAAA", linestyle="--", linewidth=1, alpha=0.8)
        _ax.text(
            len(_bar_labels) - 0.5, _opp_base + 1,
            f"No-chat opposing: {_opp_base:.1f}%",
            fontsize=8, color="#888888", ha="right",
        )

    _ax.set_xticks(_x)
    _ax.set_xticklabels(
        [l.replace(" — ", "\n").replace(" (baseline)", "\n(baseline)") for l in _bar_labels],
        fontsize=9, ha="center",
    )
    _ax.set_ylabel("P(predictShared)", fontsize=11)
    _ax.set_ylim(0, 100)
    _ax.set_yticks([0, 25, 50, 75, 100])
    _ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    _ax.set_title(
        "Opposing dyads who found shared reality have higher commonality expectations",
        fontsize=12, fontweight="bold", loc="left",
    )
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_s_sr_gradient.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("/tmp/figure_s_sr_gradient.png", bbox_inches="tight", dpi=150)

    mo.md("\n".join(_lines))
    return


@app.cell
def _(cluster_bootstrap_mean_ci, merged_df, mo, nochat_baseline, np, pd):
    # ── Finer gradient: by topic scope within opposing dyads ──
    _lines = ["## Gradient by Topic Scope (opposing dyads only)\n"]
    _lines.append("Does the *type* of shared reality matter for commonality expectations?\n")

    _opp = merged_df[merged_df["stance"] == "opposing"]
    _scope_order = ["none", "rapport_only", "different_topic", "related_subtopic", "same_values"]
    _scope_display = {
        "none": "None", "rapport_only": "Rapport only",
        "different_topic": "Different topic", "related_subtopic": "Related subtopic",
        "same_values": "Same values",
    }

    _lines.append("| Topic Scope | P(predictShared) | 95% CI | N dyads |")
    _lines.append("|-------------|-----------------|--------|---------|")

    # No-chat baseline row
    _opp_base = nochat_baseline.get("opposing", np.nan) * 100
    _lines.append(f"| *No-chat baseline* | {_opp_base:.1f}% | — | — |")

    for _scope in _scope_order:
        _sub = _opp[_opp["best_topic_scope"] == _scope]
        if len(_sub) == 0:
            continue
        _m, _lo, _hi = cluster_bootstrap_mean_ci(
            _sub["predictShared"], _sub["groupId"],
        )
        _lines.append(
            f"| {_scope_display[_scope]} | {_m * 100:.1f}% "
            f"| [{_lo * 100:.1f}, {_hi * 100:.1f}] | {_sub['groupId'].nunique()} |"
        )

    _lines.append("")
    _lines.append("If the gradient hypothesis holds, we should see P(predictShared) increase ")
    _lines.append("from none → rapport → different topic → related subtopic → same values.")

    mo.md("\n".join(_lines))
    return


@app.cell
def _(fmt_polars_table, glmer, merged_df, mo, pl):
    # GLMER: Does finding SR predict higher predictShared among opposing dyads?
    try:
        _opp = merged_df[merged_df["stance"] == "opposing"].copy()
        _pl = pl.from_pandas(
            _opp[["predictShared", "ever_sr", "groupId"]].rename(
                columns={"groupId": "group_id"}
            )
        )

        _model = glmer(
            "predictShared ~ ever_sr + (1 | group_id)",
            data=_pl,
            family="binomial",
        )
        _model.fit(verbose=False)

        _lines = [
            "## GLMER: Does SR Predict Commonality Expectations? (opposing only)\n",
            "Model: `predictShared ~ ever_sr + (1 | group_id)`\n",
            fmt_polars_table(_model.result_fit, "Fixed effects"),
            "\n",
            fmt_polars_table(_model.result_fit_stats, "Fit statistics"),
        ]
        mo.md("\n".join(_lines))
    except Exception as _e:
        mo.md(f"## GLMER: SR → predictShared (opposing)\n\n⚠️ Model fitting failed: `{_e}`")
    return


@app.cell
def _(
    FIGURES_DIR,
    STANCE_COLORS,
    cluster_bootstrap_mean_ci,
    merged_df,
    mo,
    nochat_baseline,
    np,
    pd,
    plt,
):
    # ── Dose-response: P(predictShared) as a function of # SR bins ──
    _lines = ["## Dose-Response: Commonality Expectations by Number of SR Bins\n"]
    _lines.append(
        "Does *more* shared reality (across more time bins) predict *higher* "
        "commonality expectations? Uses `n_sr_bins` as a continuous measure.\n"
    )

    _opp = merged_df[merged_df["stance"] == "opposing"]

    # Group by number of SR bins
    _dose_results = []
    for _n, _grp in _opp.groupby("n_sr_bins"):
        _m, _lo, _hi = cluster_bootstrap_mean_ci(
            _grp["predictShared"], _grp["groupId"],
        )
        _dose_results.append({
            "n_sr_bins": int(_n),
            "pct_predict_shared": _m * 100,
            "ci_lo": _lo * 100,
            "ci_hi": _hi * 100,
            "n_dyads": _grp["groupId"].nunique(),
            "n_responses": len(_grp),
        })
    _dose_df = pd.DataFrame(_dose_results).sort_values("n_sr_bins")

    # Table
    _lines.append("### Opposing dyads\n")
    _lines.append("| # SR bins | P(predictShared) | 95% CI | N dyads |")
    _lines.append("|-----------|-----------------|--------|---------|")
    for _, _row in _dose_df.iterrows():
        _lines.append(
            f"| {_row['n_sr_bins']:.0f} | {_row['pct_predict_shared']:.1f}% "
            f"| [{_row['ci_lo']:.1f}, {_row['ci_hi']:.1f}] | {_row['n_dyads']} |"
        )
    _lines.append("")

    # Figure
    _fig, _ax = plt.subplots(figsize=(8, 5))

    _ax.plot(
        _dose_df["n_sr_bins"], _dose_df["pct_predict_shared"],
        "o-", color=STANCE_COLORS["opposing"], markersize=8, linewidth=2,
    )
    _ax.fill_between(
        _dose_df["n_sr_bins"], _dose_df["ci_lo"], _dose_df["ci_hi"],
        alpha=0.15, color=STANCE_COLORS["opposing"],
    )

    # Annotate dyad counts
    for _, _row in _dose_df.iterrows():
        _ax.annotate(
            f"n={_row['n_dyads']:.0f}",
            (_row["n_sr_bins"], _row["ci_hi"]),
            textcoords="offset points", xytext=(0, 8),
            fontsize=8, ha="center", color=STANCE_COLORS["opposing"], alpha=0.7,
        )

    # No-chat baseline
    _opp_base = nochat_baseline.get("opposing", np.nan) * 100
    if not np.isnan(_opp_base):
        _ax.axhline(_opp_base, color="#AAAAAA", linestyle="--", linewidth=1, alpha=0.8)
        _ax.text(
            _dose_df["n_sr_bins"].max(), _opp_base + 1.5,
            f"No-chat opposing: {_opp_base:.1f}%",
            fontsize=8, color="#888888", ha="right",
        )

    _ax.set_xlabel("Number of time bins with shared reality", fontsize=11)
    _ax.set_ylabel("P(predictShared)", fontsize=11)
    _ax.set_ylim(0, 100)
    _ax.set_yticks([0, 25, 50, 75, 100])
    _ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    _ax.set_title(
        "Dose-response: more SR bins → higher commonality expectations (opposing dyads)",
        fontsize=11, fontweight="bold", loc="left",
    )
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_s_sr_dose_response.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("/tmp/figure_s_sr_dose_response.png", bbox_inches="tight", dpi=150)

    mo.md("\n".join(_lines))
    return


@app.cell
def _(fmt_polars_table, glmer, merged_df, mo, pl):
    # GLMER: n_sr_bins as continuous predictor (opposing dyads only)
    try:
        _opp = merged_df[merged_df["stance"] == "opposing"].copy()
        _opp["n_sr_bins_c"] = _opp["n_sr_bins"] - _opp["n_sr_bins"].mean()
        _pl = pl.from_pandas(
            _opp[["predictShared", "n_sr_bins_c", "groupId"]].rename(
                columns={"groupId": "group_id"}
            )
        )

        _model = glmer(
            "predictShared ~ n_sr_bins_c + (1 | group_id)",
            data=_pl,
            family="binomial",
        )
        _model.fit(verbose=False)

        _lines = [
            "## GLMER: Dose-Response (opposing only)\n",
            "Model: `predictShared ~ n_sr_bins_c + (1 | group_id)`\n",
            "Tests whether each additional SR bin predicts higher commonality expectations.\n",
            fmt_polars_table(_model.result_fit, "Fixed effects"),
            "\n",
            fmt_polars_table(_model.result_fit_stats, "Fit statistics"),
        ]
        mo.md("\n".join(_lines))
    except Exception as _e:
        mo.md(f"## GLMER: Dose-Response\n\n⚠️ Model fitting failed: `{_e}`")
    return


# ══════════════════════════════════════════════════════════════════════════
# SECTION 14: Combined figure — SR + topic scope + goal
# ══════════════════════════════════════════════════════════════════════════


@app.cell
def _(
    BIN_SECONDS,
    CONVERSATIONAL_GOAL_COLORS,
    CONVERSATIONAL_GOAL_LABELS,
    CONVERSATIONAL_GOAL_VALUES,
    FIGURES_DIR,
    STANCE_COLORS,
    STANCE_ORDER,
    TOPIC_SCOPE_COLORS,
    TOPIC_SCOPE_LABELS,
    TOPIC_SCOPE_VALUES,
    df_post,
    np,
    plt,
    sr_timecourse_df,
    time_bins_post,
):
    _fig = plt.figure(figsize=(16, 14))
    _gs = _fig.add_gridspec(3, 2, height_ratios=[2, 2, 2], hspace=0.4, wspace=0.3)

    # Row 1: P(shared reality) over time (spans both columns)
    _ax_sr = _fig.add_subplot(_gs[0, :])
    for _stance in STANCE_ORDER:
        _sub = sr_timecourse_df[sr_timecourse_df["stance"] == _stance]
        _color = STANCE_COLORS[_stance]
        _ax_sr.plot(
            _sub["time_seconds"], _sub["pct_sr"],
            "o-", color=_color, label=_stance.capitalize(), markersize=5, linewidth=2,
        )
        _ax_sr.fill_between(
            _sub["time_seconds"], _sub["ci_lo"], _sub["ci_hi"],
            alpha=0.15, color=_color,
        )
    _ax_sr.set_xlabel("Time (seconds)", fontsize=10)
    _ax_sr.set_ylabel("% of dyads with SR", fontsize=10)
    _ax_sr.set_ylim(0, 100)
    _ax_sr.set_yticks([0, 25, 50, 75, 100])
    _ax_sr.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    _ax_sr.legend(title="Stance", fontsize=9)
    _ax_sr.set_title("A. Shared reality emergence over time", fontsize=12, fontweight="bold", loc="left")
    _ax_sr.spines["top"].set_visible(False)
    _ax_sr.spines["right"].set_visible(False)

    # Row 2: Topic scope stacked bars (by stance)
    for _i, _stance in enumerate(STANCE_ORDER):
        _ax = _fig.add_subplot(_gs[1, _i])
        _sub = df_post[df_post["stance"] == _stance]
        _substantive = [t for t in TOPIC_SCOPE_VALUES if t != "none"]
        _bottoms = np.zeros(len(time_bins_post))
        for _ts in _substantive:
            _pcts = []
            for _t in time_bins_post:
                _bd = _sub[_sub["time_seconds"] == _t]
                _n = len(_bd)
                _pcts.append((_bd["post_stance_topic_scope"] == _ts).sum() / _n * 100 if _n else 0)
            _pcts = np.array(_pcts)
            _idx = TOPIC_SCOPE_VALUES.index(_ts)
            _ax.bar(time_bins_post, _pcts, bottom=_bottoms, width=BIN_SECONDS * 0.8,
                    color=TOPIC_SCOPE_COLORS[_ts], label=TOPIC_SCOPE_LABELS[_idx],
                    edgecolor="white", linewidth=0.5)
            _bottoms += _pcts
        _letter = "B" if _i == 0 else "C"
        _ax.set_title(f"{_letter}. Topic scope — {_stance.capitalize()}", fontsize=11, fontweight="bold", loc="left")
        _ax.set_xlabel("Time (s)", fontsize=9)
        _ax.set_ylabel("% of dyads", fontsize=9)
        _ax.set_ylim(0, 100)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        if _i == 1:
            _ax.legend(fontsize=7, loc="upper left", title="Topic scope", title_fontsize=8)

    # Row 3: Conversational goal stacked bars (by stance)
    for _i, _stance in enumerate(STANCE_ORDER):
        _ax = _fig.add_subplot(_gs[2, _i])
        _sub = df_post[df_post["stance"] == _stance]
        _bottoms = np.zeros(len(time_bins_post))
        for _goal in CONVERSATIONAL_GOAL_VALUES:
            _pcts = []
            for _t in time_bins_post:
                _bd = _sub[_sub["time_seconds"] == _t]
                _n = len(_bd)
                _pcts.append((_bd["post_stance_conversational_goal"] == _goal).sum() / _n * 100 if _n else 0)
            _pcts = np.array(_pcts)
            _idx = CONVERSATIONAL_GOAL_VALUES.index(_goal)
            _ax.bar(time_bins_post, _pcts, bottom=_bottoms, width=BIN_SECONDS * 0.8,
                    color=CONVERSATIONAL_GOAL_COLORS[_goal], label=CONVERSATIONAL_GOAL_LABELS[_idx],
                    edgecolor="white", linewidth=0.5)
            _bottoms += _pcts
        _letter = "D" if _i == 0 else "E"
        _ax.set_title(f"{_letter}. Convo goal — {_stance.capitalize()}", fontsize=11, fontweight="bold", loc="left")
        _ax.set_xlabel("Time (s)", fontsize=9)
        _ax.set_ylabel("% of dyads", fontsize=9)
        _ax.set_ylim(0, 100)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        if _i == 1:
            _ax.legend(fontsize=6, loc="upper left", title="Goal", title_fontsize=7)

    plt.savefig(FIGURES_DIR / "figure_s_sr_combined_timecourse.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("/tmp/figure_s_sr_combined_timecourse.png", bbox_inches="tight", dpi=150)
    _fig
    return


if __name__ == "__main__":
    app.run()
