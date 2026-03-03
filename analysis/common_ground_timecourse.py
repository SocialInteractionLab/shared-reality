"""Common ground timecourse analysis and figures.

Analyzes LLM-annotated common ground emergence over conversation time,
comparing opposing vs shared stance dyads.

Usage:
    uv run marimo edit analysis/common_ground_timecourse.py
    uv run python analysis/common_ground_timecourse.py
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import subprocess
    import tempfile
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from scipy.stats import mannwhitneyu, permutation_test

    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    FIGURES_DIR = BASE_DIR / "outputs" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    N_BOOT = 10_000
    RNG = np.random.default_rng(42)

    BIN_SECONDS = 15

    STANCE_ORDER = ["opposing", "shared"]
    STANCE_COLORS = {"opposing": "#648FFF", "shared": "#DC267F"}

    CG_TYPES = ["same_values", "related_subtopic", "different_topic", "rapport_only", "none"]
    CG_LABELS = ["Same values", "Related subtopic", "Different topic", "Rapport only", "None"]
    CG_COLORS = {
        "same_values": "#FE6100",
        "related_subtopic": "#785EF0",
        "different_topic": "#FFB000",
        "rapport_only": "#C0C0C0",
        "none": "#E8E8E8",
    }
    return (
        BIN_SECONDS,
        CG_COLORS,
        CG_LABELS,
        CG_TYPES,
        DATA_DIR,
        FIGURES_DIR,
        N_BOOT,
        Path,
        RNG,
        STANCE_COLORS,
        STANCE_ORDER,
        mannwhitneyu,
        mo,
        np,
        pd,
        plt,
        subprocess,
        tempfile,
    )


@app.cell
def _(mo):
    mo.md("""
    # Common Ground Timecourse Analysis

    Analyzes LLM-annotated common ground emergence over conversation time,
    comparing **opposing** vs **shared** stance dyads.

    Only post-stance-reveal data is included: dyads contribute to a time bin
    only after both participants have revealed their focal stance.
    """)
    return


@app.cell
def _(N_BOOT, Path, RNG, np, pd, subprocess, tempfile):
    def bootstrap_mean_ci(
        values: np.ndarray, n_boot: int = N_BOOT
    ) -> tuple[float, float, float]:
        """Bootstrap mean and 95% CI."""
        n = len(values)
        if n == 0:
            return np.nan, np.nan, np.nan
        boot = np.empty(n_boot)
        for b in range(n_boot):
            idx = RNG.integers(0, n, size=n)
            boot[b] = values[idx].mean()
        return values.mean(), *np.percentile(boot, [2.5, 97.5])

    def cluster_bootstrap_mean_ci(
        values: pd.Series,
        cluster_ids: pd.Series,
        n_boot: int = N_BOOT,
    ) -> tuple[float, float, float]:
        """Bootstrap mean and 95% CI, resampling at the cluster (dyad) level."""
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

    def run_glmer(
        df: pd.DataFrame,
        formula: str,
        r_setup: str = "",
        r_post: str = "",
    ) -> str:
        """Fit a binomial GLMER via R subprocess. Returns R output as string.

        Args:
            df: Data to pass to R.
            formula: R formula string (e.g. "y ~ x + (1 | group)").
            r_setup: Extra R code to run after loading data (e.g. releveling factors).
            r_post: Extra R code to run after fitting (e.g. emmeans contrasts).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            df.to_csv(csv_path, index=False)

            r_script = f"""
library(lme4)
d <- read.csv("{csv_path}")
{r_setup}
m <- glmer({formula}, data = d, family = binomial,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
print(summary(m))
{r_post}
"""
            r_path = Path(tmpdir) / "model.R"
            r_path.write_text(r_script)
            result = subprocess.run(
                ["Rscript", str(r_path)], capture_output=True, text=True,
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\n\nSTDERR:\n{result.stderr}"
            return output

    def run_glmer_emmeans(
        df: pd.DataFrame,
        formula: str,
        emmeans_spec: str,
        contrast_method: str = "pairwise",
        p_adjust: str = "none",
        r_setup: str = "",
    ) -> str:
        """Fit a binomial GLMER and compute emmeans contrasts via R subprocess.

        Args:
            df: Data to pass to R.
            formula: R formula string.
            emmeans_spec: Emmeans spec (e.g. "~ stance | common_ground_type").
            contrast_method: Contrast method (e.g. "pairwise", "trt.vs.ctrl").
            p_adjust: P-value adjustment method.
            r_setup: Extra R code to run after loading data.
        """
        r_post = f"""
library(emmeans)
emm <- emmeans(m, {emmeans_spec})
cat("\\n--- Estimated Marginal Means (probability scale) ---\\n")
print(summary(emm, type = "response"))
contr <- contrast(emm, method = "{contrast_method}", adjust = "{p_adjust}")
cat("\\n--- Contrasts ---\\n")
print(summary(contr))
cat("\\n--- Contrasts (odds ratios) ---\\n")
print(summary(contr, type = "response"))
"""
        return run_glmer(df, formula, r_setup=r_setup, r_post=r_post)

    return (
        bootstrap_mean_ci,
        cluster_bootstrap_mean_ci,
        kaplan_meier,
        km_median,
        run_glmer,
        run_glmer_emmeans,
    )


@app.cell
def _(BIN_SECONDS, DATA_DIR, pd):
    _path = DATA_DIR / "llm_results" / f"common_ground_timecourse_{BIN_SECONDS}s.csv"
    df_all = pd.read_csv(_path)
    df_revealed = df_all[df_all["focal_stance_revealed"] == True].copy()
    time_bins = sorted(df_revealed["time_seconds"].unique())

    print(f"Total annotations: {len(df_all)}")
    print(f"After stance reveal: {len(df_revealed)}")
    print(f"Unique dyads (revealed): {df_revealed['group_id'].nunique()}")
    print(f"Unique dyads (all): {df_all['group_id'].nunique()}")
    return df_revealed, time_bins


@app.cell
def _(STANCE_ORDER, cluster_bootstrap_mean_ci, df_revealed, pd, time_bins):
    _results = []
    for _stance in STANCE_ORDER:
        for _t in time_bins:
            _sub = df_revealed[
                (df_revealed["stance"] == _stance)
                & (df_revealed["time_seconds"] == _t)
            ]
            if len(_sub) == 0:
                continue
            _vals = _sub["post_stance_common_ground"].astype(int)
            _ids = _sub["group_id"]
            _mean, _lo, _hi = cluster_bootstrap_mean_ci(_vals, _ids)
            _results.append({
                "stance": _stance,
                "time_seconds": _t,
                "pct_cg": _mean * 100,
                "ci_lo": _lo * 100,
                "ci_hi": _hi * 100,
                "n_dyads": len(_sub),
            })

    res_df = pd.DataFrame(_results)
    return (res_df,)


@app.cell
def _(STANCE_ORDER, mo, res_df):
    _lines = ["## P(Common Ground) Over Time (cluster-bootstrapped CIs)\n"]
    for _stance in STANCE_ORDER:
        _sub = res_df[res_df["stance"] == _stance]
        _lines.append(f"**{_stance.capitalize()}:**\n")
        _lines.append("| Time | % CG | 95% CI | N |")
        _lines.append("|------|------|--------|---|")
        for _, _row in _sub.iterrows():
            _lines.append(
                f"| {_row['time_seconds']:.0f}s | {_row['pct_cg']:.1f}% "
                f"| [{_row['ci_lo']:.1f}, {_row['ci_hi']:.1f}] | {_row['n_dyads']:.0f} |"
            )
        _lines.append("")
    mo.md("\n".join(_lines))
    return


@app.cell
def _(df_revealed, mo, run_glmer):
    # Formal GLMER: does CG increase over time? Does the rate differ by stance?
    _tc_data = df_revealed.copy()
    _tc_data["time_min"] = _tc_data["time_seconds"] / 60.0
    _tc_data["time_c"] = _tc_data["time_min"] - _tc_data["time_min"].mean()
    _tc_data["stance_e"] = _tc_data["stance"].map({"opposing": 0.5, "shared": -0.5})
    _tc_data["post_stance_cg"] = _tc_data["post_stance_common_ground"].astype(int)

    _output = run_glmer(
        _tc_data[["post_stance_cg", "time_c", "stance_e", "group_id"]],
        "post_stance_cg ~ time_c * stance_e + (1 + time_c | group_id)",
    )

    mo.md(
        "## GLMER: P(Common Ground) Over Time\n\n"
        "Model: `post_stance_cg ~ time_c * stance_e + (1 + time_c | group_id)`\n\n"
        "Formal test that CG increases over time and whether the rate differs by stance.\n\n"
        f"```\n{_output}\n```"
    )
    return


@app.cell
def _(CG_TYPES, STANCE_ORDER, df_revealed, pd, time_bins):
    _type_results = []
    for _stance in STANCE_ORDER:
        for _t in time_bins:
            _sub = df_revealed[
                (df_revealed["stance"] == _stance)
                & (df_revealed["time_seconds"] == _t)
            ]
            _n = len(_sub)
            if _n == 0:
                continue
            for _cg_type in CG_TYPES:
                _count = (_sub["common_ground_type"] == _cg_type).sum()
                _type_results.append({
                    "stance": _stance,
                    "time_seconds": _t,
                    "cg_type": _cg_type,
                    "pct": _count / _n * 100,
                    "n_dyads": _n,
                })

    type_df = pd.DataFrame(_type_results)
    return (type_df,)


@app.cell
def _(
    N_BOOT,
    RNG,
    STANCE_ORDER,
    df_revealed,
    kaplan_meier,
    km_median,
    mannwhitneyu,
    mo,
    np,
    pd,
):
    _survival_rows = []
    for _group_id, _gdf in df_revealed.groupby("group_id"):
        _stance = _gdf["stance"].iloc[0]
        _cg_bins = _gdf[_gdf["post_stance_common_ground"] == True]["time_seconds"]
        if len(_cg_bins) > 0:
            _onset = _cg_bins.min()
            _event = 1
        else:
            _onset = _gdf["time_seconds"].max()
            _event = 0
        _survival_rows.append({
            "group_id": _group_id,
            "stance": _stance,
            "onset_time": _onset,
            "event": _event,
        })

    surv_df = pd.DataFrame(_survival_rows)

    _lines = ["## Survival Analysis: Time to First Common Ground\n"]
    _lines.append(
        f"Dyads: {len(surv_df)} ({surv_df['event'].sum()} found CG, "
        f"{(~surv_df['event'].astype(bool)).sum()} censored)\n"
    )

    for _stance in STANCE_ORDER:
        _sub = surv_df[surv_df["stance"] == _stance]
        _t, _s = kaplan_meier(_sub["onset_time"].values, _sub["event"].values)
        _med = km_median(_t, _s)
        _lines.append(
            f"- **{_stance.capitalize()}**: N={len(_sub)}, events={_sub['event'].sum()}, "
            f"median time to CG = {_med:.0f}s"
        )

    _opp_onset = surv_df[
        (surv_df["stance"] == "opposing") & (surv_df["event"] == 1)
    ]["onset_time"]
    _sha_onset = surv_df[
        (surv_df["stance"] == "shared") & (surv_df["event"] == 1)
    ]["onset_time"]
    _u_stat, _u_p = mannwhitneyu(_opp_onset, _sha_onset, alternative="two-sided")
    _lines.append(
        f"\n**Mann-Whitney U** (onset times, CG dyads only): "
        f"opposing median={_opp_onset.median():.0f}s, shared median={_sha_onset.median():.0f}s, "
        f"U={_u_stat:.0f}, p={_u_p:.4f}"
    )

    _boot_diffs = []
    for _ in range(N_BOOT):
        _o_samp = RNG.choice(_opp_onset.values, size=len(_opp_onset), replace=True)
        _s_samp = RNG.choice(_sha_onset.values, size=len(_sha_onset), replace=True)
        _boot_diffs.append(np.median(_o_samp) - np.median(_s_samp))
    _boot_diffs = np.array(_boot_diffs)
    _diff_ci = np.percentile(_boot_diffs, [2.5, 97.5])
    _lines.append(
        f"\nBootstrap delta-median (opposing - shared) = {np.median(_boot_diffs):.0f}s "
        f"[{_diff_ci[0]:.0f}, {_diff_ci[1]:.0f}]"
    )

    mo.md("\n".join(_lines))
    return


@app.cell
def _(df_revealed, mo, run_glmer):
    # Binomial GLMERs for each CG type over time
    _cg_data = df_revealed[df_revealed["post_stance_common_ground"] == True].copy()
    _cg_data["time_min"] = _cg_data["time_seconds"] / 60.0
    _cg_data["time_c"] = _cg_data["time_min"] - _cg_data["time_min"].mean()
    _cg_data["stance_e"] = _cg_data["stance"].map({"opposing": 0.5, "shared": -0.5})

    _substantive_types = [
        "rapport_only", "related_subtopic", "same_values", "different_topic",
    ]
    for _cg_type in _substantive_types:
        _cg_data[f"is_{_cg_type}"] = (
            _cg_data["common_ground_type"] == _cg_type
        ).astype(int)

    _lines = [
        "## Mixed-Effects Logistic Regression: CG Type ~ Time x Stance\n",
        "Model: `is_<type> ~ time_c * stance_e + (1 | group_id)`\n",
    ]

    for _cg_type in _substantive_types:
        _dv = f"is_{_cg_type}"
        _cols = [_dv, "time_c", "stance_e", "group_id"]
        _output = run_glmer(
            _cg_data[_cols],
            f"{_dv} ~ time_c * stance_e + (1 | group_id)",
        )
        _title = _cg_type.replace("_", " ").title()
        _lines.append(f"### {_title}\n")
        _lines.append(f"```\n{_output}\n```\n")

    mo.md("\n".join(_lines))
    return


@app.cell
def _(Path, df_revealed, subprocess, tempfile):
    # Multinomial logistic regression — must stay as R subprocess (no lme4 wrapper)
    _cg_data = df_revealed[df_revealed["post_stance_common_ground"] == True].copy()
    _cg_data["time_min"] = _cg_data["time_seconds"] / 60.0
    _cg_data["time_c"] = _cg_data["time_min"] - _cg_data["time_min"].mean()
    _cg_data["stance_e"] = _cg_data["stance"].map({"opposing": 0.5, "shared": -0.5})

    with tempfile.TemporaryDirectory() as _tmpdir:
        _csv_path = Path(_tmpdir) / "cg_data.csv"
        _cg_data.to_csv(_csv_path, index=False)

        _r_script = r"""
    library(nnet)
    d <- read.csv("CSV_PATH")
    d <- d[d$common_ground_type != "none", ]
    d$cg_type <- relevel(factor(d$common_ground_type), ref = "rapport_only")

    m <- multinom(cg_type ~ time_c * stance_e, data=d, trace=FALSE)
    sm <- summary(m)
    coefs <- sm$coefficients
    ses <- sm$standard.errors
    z <- coefs / ses
    p <- 2 * (1 - pnorm(abs(z)))

    types <- rownames(coefs)
    vars <- colnames(coefs)

    for (i in seq_along(types)) {
    cat(sprintf("\n--- %s vs rapport_only ---\n", types[i]))
    for (j in seq_along(vars)) {
        star <- ""
        if (p[i,j] < 0.001) star <- "***"
        else if (p[i,j] < 0.01) star <- "**"
        else if (p[i,j] < 0.05) star <- "*"
        else if (p[i,j] < 0.1) star <- "."
        cat(sprintf("  %-25s  B=%7.3f  SE=%6.3f  z=%6.3f  p=%s %s\n",
            vars[j], coefs[i,j], ses[i,j], z[i,j],
            ifelse(p[i,j] < 0.001, "<.001", sprintf("%.4f", p[i,j])),
            star))
    }
    }

    cat(sprintf("\nAIC: %.1f\n", AIC(m)))
    m0 <- multinom(cg_type ~ 1, data=d, trace=FALSE)
    cat(sprintf("Null AIC: %.1f\n", AIC(m0)))
    devdiff <- m0$deviance - m$deviance
    dfdiff <- length(coef(m)) - length(coef(m0))
    pval <- 1 - pchisq(devdiff, dfdiff)
    cat(sprintf("LR chi2 = %.2f, df = %d, p = %.6f\n", devdiff, dfdiff, pval))
    """.replace("CSV_PATH", str(_csv_path))

        _r_path = Path(_tmpdir) / "multinomial.R"
        _r_path.write_text(_r_script)
        _result = subprocess.run(
            ["Rscript", str(_r_path)], capture_output=True, text=True,
        )
        multinom_output = _result.stdout
        if _result.returncode != 0:
            multinom_output += f"\n\nSTDERR:\n{_result.stderr}"
    return (multinom_output,)


@app.cell
def _(mo, multinom_output):
    mo.md(
        "## Multinomial Logistic Regression: CG Type ~ Time x Stance\n\n"
        "Model: `cg_type ~ time_c * stance_e` (reference: rapport_only)\n\n"
        "Each coefficient compares odds of that CG type vs rapport_only. "
        "Positive `stance_e` means opposing dyads are shifted *toward* that type.\n\n"
        f"```\n{multinom_output}\n```"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Does Common Ground Type Predict Commonality Expectations?

    Linking LLM-annotated common ground types to the behavioral DV:
    P(predictShared) on inference questions (chat dyads only).
    """)
    return


@app.cell
def _(DATA_DIR, pd):
    # Load behavioral responses and CG annotations
    _resp = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)
    _cg = pd.read_csv(DATA_DIR / "llm_results" / "common_ground.csv")

    # Rename for merge
    _cg = _cg.rename(columns={"group_id": "groupId"})

    # Merge: inner join keeps only chat dyads (no-chat have groupId=NaN)
    _merged = _resp.merge(
        _cg[["groupId", "common_ground_type"]],
        on="groupId",
        how="inner",
    )

    # Filter to inference questions only
    merged_df = _merged[_merged["question_type"] != "observed"].copy()

    # Binary CG variable: substantive vs non-substantive
    merged_df["substantive_cg"] = merged_df["common_ground_type"].isin(
        ["same_values", "related_subtopic", "different_topic"]
    ).astype(int)

    # found_cg: any CG vs none
    merged_df["found_cg"] = (merged_df["common_ground_type"] != "none").astype(int)

    # Effect coding for stance
    merged_df["stance_e"] = merged_df["stance"].map(
        {"opposing": 0.5, "shared": -0.5}
    )

    # Also load no-chat data for baseline comparison
    _nochat = _resp[
        (_resp["experiment"] == "no-chat")
        & (_resp["question_type"] != "observed")
    ]
    nochat_baseline = _nochat.groupby("stance")["predictShared"].mean().to_dict()

    print(f"Merged chat inference rows: {len(merged_df)}")
    print(f"Unique dyads: {merged_df['groupId'].nunique()}")
    print(f"Unique participants: {merged_df['pid'].nunique()}")
    print(f"CG type distribution:\n{merged_df.groupby('groupId')['common_ground_type'].first().value_counts()}")
    return merged_df, nochat_baseline


@app.cell
def _(N_BOOT, RNG, STANCE_ORDER, merged_df, mo, nochat_baseline, np):
    # Descriptive: P(predictShared) by CG type x stance
    _cg_order = ["none", "rapport_only", "different_topic", "related_subtopic", "same_values"]
    _cg_display = {
        "none": "None", "rapport_only": "Rapport only",
        "different_topic": "Different topic", "related_subtopic": "Related subtopic",
        "same_values": "Same values",
    }

    _lines = ["## P(predictShared) by CG Type x Stance\n"]
    _lines.append("| CG Type | Stance | P(predictShared) | 95% CI | N participants |")
    _lines.append("|---------|--------|------------------|--------|----------------|")

    desc_results = []
    for _cg_type in _cg_order:
        for _stance in STANCE_ORDER:
            _sub = merged_df[
                (merged_df["common_ground_type"] == _cg_type)
                & (merged_df["stance"] == _stance)
            ]
            _pid_means = _sub.groupby("pid")["predictShared"].mean().values
            _n_pids = len(_pid_means)
            if _n_pids == 0:
                continue
            _mean = _pid_means.mean()
            _boots = np.array([
                RNG.choice(_pid_means, size=_n_pids, replace=True).mean()
                for _ in range(N_BOOT)
            ])
            _lo, _hi = np.percentile(_boots, [2.5, 97.5])
            _lines.append(
                f"| {_cg_display[_cg_type]} | {_stance.capitalize()} "
                f"| {_mean:.3f} | [{_lo:.3f}, {_hi:.3f}] | {_n_pids} |"
            )
            desc_results.append({
                "cg_type": _cg_type, "stance": _stance,
                "mean": _mean, "ci_lo": _lo, "ci_hi": _hi, "n": _n_pids,
            })

    # Binary split
    _lines.append("\n### Substantive vs Non-substantive CG\n")
    _lines.append("| CG Category | Stance | P(predictShared) | 95% CI | N participants |")
    _lines.append("|-------------|--------|------------------|--------|----------------|")
    for _subst, _label in [(1, "Substantive"), (0, "Non-substantive")]:
        for _stance in STANCE_ORDER:
            _sub = merged_df[
                (merged_df["substantive_cg"] == _subst)
                & (merged_df["stance"] == _stance)
            ]
            _pid_means = _sub.groupby("pid")["predictShared"].mean().values
            _n_pids = len(_pid_means)
            if _n_pids == 0:
                continue
            _mean = _pid_means.mean()
            _boots = np.array([
                RNG.choice(_pid_means, size=_n_pids, replace=True).mean()
                for _ in range(N_BOOT)
            ])
            _lo, _hi = np.percentile(_boots, [2.5, 97.5])
            _lines.append(
                f"| {_label} | {_stance.capitalize()} "
                f"| {_mean:.3f} | [{_lo:.3f}, {_hi:.3f}] | {_n_pids} |"
            )

    # No-chat baseline
    _lines.append("\n### No-chat baseline (for reference)\n")
    for _stance in STANCE_ORDER:
        _base = nochat_baseline.get(_stance, float("nan"))
        _lines.append(f"- **{_stance.capitalize()}**: P(predictShared) = {_base:.3f}")

    mo.md("\n".join(_lines))
    return


@app.cell
def _(
    FIGURES_DIR,
    N_BOOT,
    RNG,
    STANCE_COLORS,
    STANCE_ORDER,
    merged_df,
    nochat_baseline,
    np,
    plt,
):
    # Figure: P(predictShared) by CG type and stance
    _cg_order = ["none", "rapport_only", "different_topic", "related_subtopic", "same_values"]
    _cg_display = ["None", "Rapport\nonly", "Different\ntopic", "Related\nsubtopic", "Same\nvalues"]

    _fig, _ax = plt.subplots(figsize=(10, 6))
    _width = 0.35
    _x = np.arange(len(_cg_order))

    for _i, _stance in enumerate(STANCE_ORDER):
        _means, _ci_los, _ci_his = [], [], []
        for _cg_type in _cg_order:
            _sub = merged_df[
                (merged_df["common_ground_type"] == _cg_type)
                & (merged_df["stance"] == _stance)
            ]
            _pid_means = _sub.groupby("pid")["predictShared"].mean().values
            if len(_pid_means) == 0:
                _means.append(np.nan)
                _ci_los.append(np.nan)
                _ci_his.append(np.nan)
                continue
            _m = _pid_means.mean()
            _boots = np.array([
                RNG.choice(_pid_means, size=len(_pid_means), replace=True).mean()
                for _ in range(N_BOOT)
            ])
            _lo, _hi = np.percentile(_boots, [2.5, 97.5])
            _means.append(_m)
            _ci_los.append(_lo)
            _ci_his.append(_hi)

        _means = np.array(_means)
        _ci_los = np.array(_ci_los)
        _ci_his = np.array(_ci_his)
        _offset = -_width / 2 + _i * _width
        _color = STANCE_COLORS[_stance]

        _ax.bar(
            _x + _offset, _means, _width,
            color=_color, alpha=0.7, label=_stance.capitalize(),
        )
        _ax.errorbar(
            _x + _offset, _means,
            yerr=[_means - _ci_los, _ci_his - _means],
            fmt="none", color=_color, capsize=3, linewidth=1.5,
        )

        # No-chat baseline
        _base = nochat_baseline.get(_stance, np.nan)
        _ax.axhline(
            _base, color=_color, linestyle="--", alpha=0.4, linewidth=1,
        )
        _ax.text(
            len(_cg_order) - 0.5, _base + 0.005,
            f"no-chat {_stance}", fontsize=8, color=_color, alpha=0.6,
            ha="right", va="bottom",
        )

    _ax.set_xticks(_x)
    _ax.set_xticklabels(_cg_display, fontsize=10)
    _ax.set_xlabel("Common Ground Type", fontsize=11)
    _ax.set_ylabel("P(predictShared)", fontsize=11)
    _ax.set_title(
        "P(predictShared) by Common Ground Type and Stance",
        fontsize=12, fontweight="bold", loc="left",
    )
    _ax.legend(title="Stance", fontsize=10, title_fontsize=10)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.set_ylim(0.3, 0.8)

    plt.savefig(
        FIGURES_DIR / "figure_s_cg_type_predict_shared.pdf",
        bbox_inches="tight", dpi=300,
    )
    plt.savefig(
        "/tmp/figure_s_cg_type_predict_shared.png",
        bbox_inches="tight", dpi=150,
    )
    _fig
    return


@app.cell
def _(merged_df, mo, run_glmer):
    # GLMER: P(predictShared) ~ CG Type x Stance
    _model_data = merged_df[["predictShared", "substantive_cg", "common_ground_type",
                              "stance_e", "pid", "groupId"]].copy()

    _m1_output = run_glmer(
        _model_data,
        "predictShared ~ substantive_cg * stance_e + (1 | pid) + (1 | groupId)",
    )

    _m2_output = run_glmer(
        _model_data,
        "predictShared ~ common_ground_type * stance_e + (1 | pid) + (1 | groupId)",
        r_setup='d$common_ground_type <- relevel(factor(d$common_ground_type), ref = "none")',
    )

    mo.md(
        "## Mixed-Effects Logistic Regression: P(predictShared) ~ CG Type x Stance\n\n"
        "### Model 1: Binary (Substantive CG vs None/Rapport)\n"
        "`predictShared ~ substantive_cg * stance_e + (1 | pid) + (1 | groupId)`\n\n"
        f"```\n{_m1_output}\n```\n\n"
        "### Model 2: 5-level CG Type (ref = none)\n"
        "`predictShared ~ common_ground_type * stance_e + (1 | pid) + (1 | groupId)`\n\n"
        f"```\n{_m2_output}\n```"
    )
    return


@app.cell
def _(merged_df, mo, run_glmer_emmeans):
    # Simple effects: stance difference within each CG type
    _model_data = merged_df[["predictShared", "common_ground_type",
                              "stance", "pid", "groupId"]].copy()

    _output = run_glmer_emmeans(
        _model_data,
        "predictShared ~ common_ground_type * stance + (1 | pid) + (1 | groupId)",
        emmeans_spec="~ stance | common_ground_type",
        contrast_method="pairwise",
        r_setup="d$common_ground_type <- factor(d$common_ground_type)\nd$stance <- factor(d$stance)",
    )

    mo.md(
        "## Simple Effects: Stance Difference Within Each CG Type\n\n"
        "Tests whether opposing vs shared stance dyads differ in P(predictShared) "
        "within each common ground type level.\n\n"
        f"```\n{_output}\n```"
    )
    return


@app.cell
def _(
    STANCE_ORDER,
    merged_df,
    mo,
    nochat_baseline,
    run_glmer,
):
    # Does finding CG predict higher P(predictShared)?
    _lines = ["## Does Finding CG Predict Higher Expected Commonality?\n"]
    _lines.append("Mixed-effects logistic regression: "
                   "`predictShared ~ found_cg * stance_e + (1 | pid) + (1 | groupId)`\n")

    for _stance in STANCE_ORDER:
        _sub = merged_df[merged_df["stance"] == _stance]
        _dyad_means = _sub.groupby(["groupId", "found_cg"])["predictShared"].mean().reset_index()
        _cg_yes = _dyad_means[_dyad_means["found_cg"] == 1]["predictShared"].values
        _cg_no = _dyad_means[_dyad_means["found_cg"] == 0]["predictShared"].values

        _lines.append(f"### {_stance.capitalize()} stance (descriptive)\n")
        _lines.append(f"- **Found CG**: N={len(_cg_yes)} dyads, "
                       f"M={_cg_yes.mean():.3f} (SD={_cg_yes.std():.3f})")
        _lines.append(f"- **No CG**: N={len(_cg_no)} dyads, "
                       f"M={_cg_no.mean():.3f} (SD={_cg_no.std():.3f})")
        _lines.append(f"- **No-chat baseline**: "
                       f"{nochat_baseline.get(_stance, float('nan')):.3f}")
        _lines.append(f"- **Delta (found - not found)**: "
                       f"{_cg_yes.mean() - _cg_no.mean():.3f}\n")

    _model_data = merged_df[["predictShared", "found_cg", "stance_e",
                              "stance", "pid", "groupId"]].copy()

    # Full model
    _full_output = run_glmer(
        _model_data,
        "predictShared ~ found_cg * stance_e + (1 | pid) + (1 | groupId)",
    )
    _lines.append("### Full model: found_cg * stance_e\n")
    _lines.append(f"```\n{_full_output}\n```\n")

    # Opposing only
    _opp_output = run_glmer(
        _model_data[_model_data["stance"] == "opposing"],
        "predictShared ~ found_cg + (1 | pid) + (1 | groupId)",
    )
    _lines.append("### Opposing only: found_cg\n")
    _lines.append(f"```\n{_opp_output}\n```\n")

    # Shared only
    _sha_output = run_glmer(
        _model_data[_model_data["stance"] == "shared"],
        "predictShared ~ found_cg + (1 | pid) + (1 | groupId)",
    )
    _lines.append("### Shared only: found_cg\n")
    _lines.append(f"```\n{_sha_output}\n```")

    mo.md("\n".join(_lines))
    return


@app.cell
def _(DATA_DIR, N_BOOT, RNG, STANCE_ORDER, merged_df, mo, np, pd):
    # Chat vs no-chat comparison (descriptive)
    _resp = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)
    _nochat = _resp[
        (_resp["experiment"] == "no-chat")
        & (_resp["question_type"] != "observed")
    ]

    # CG categories for chat dyads
    _chat_cat = merged_df.copy()
    _chat_cat["cg_category"] = _chat_cat["common_ground_type"].map(
        lambda x: "substantive" if x in ["same_values", "related_subtopic", "different_topic"]
        else x  # keeps "rapport_only" and "none"
    )

    _lines = [
        "## Chat vs No-chat: P(predictShared) by CG Category\n",
        "| Condition | Stance | P(predictShared) | 95% CI | N participants |",
        "|-----------|--------|------------------|--------|----------------|",
    ]

    # No-chat
    for _stance in STANCE_ORDER:
        _sub = _nochat[_nochat["stance"] == _stance]
        _pid_means = _sub.groupby("pid")["predictShared"].mean().values
        _n = len(_pid_means)
        _m = _pid_means.mean()
        _boots = np.array([
            RNG.choice(_pid_means, size=_n, replace=True).mean()
            for _ in range(N_BOOT)
        ])
        _lo, _hi = np.percentile(_boots, [2.5, 97.5])
        _lines.append(
            f"| No chat | {_stance.capitalize()} "
            f"| {_m:.3f} | [{_lo:.3f}, {_hi:.3f}] | {_n} |"
        )

    # Chat by CG category
    for _cat, _label in [("none", "Chat: no CG"), ("rapport_only", "Chat: rapport only"),
                          ("substantive", "Chat: substantive CG")]:
        for _stance in STANCE_ORDER:
            _sub = _chat_cat[
                (_chat_cat["cg_category"] == _cat)
                & (_chat_cat["stance"] == _stance)
            ]
            _pid_means = _sub.groupby("pid")["predictShared"].mean().values
            _n = len(_pid_means)
            if _n == 0:
                _lines.append(f"| {_label} | {_stance.capitalize()} | -- | -- | 0 |")
                continue
            _m = _pid_means.mean()
            _boots = np.array([
                RNG.choice(_pid_means, size=_n, replace=True).mean()
                for _ in range(N_BOOT)
            ])
            _lo, _hi = np.percentile(_boots, [2.5, 97.5])
            _lines.append(
                f"| {_label} | {_stance.capitalize()} "
                f"| {_m:.3f} | [{_lo:.3f}, {_hi:.3f}] | {_n} |"
            )

    mo.md("\n".join(_lines))
    return


@app.cell
def _(DATA_DIR, STANCE_ORDER, merged_df, mo, pd, run_glmer_emmeans):
    # Formal test: each chat CG condition vs no-chat baseline (per stance)
    _resp = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)

    _all_output_lines = [
        "## GLMER: Chat (by CG type) vs No-chat Baseline\n",
        "Model: `predictShared ~ condition + (1 | pid)` per stance\n",
        "Dunnett-adjusted pairwise comparisons vs no-chat reference.\n",
    ]

    for _stance in STANCE_ORDER:
        # No-chat for this stance
        _nochat = _resp[
            (_resp["experiment"] == "no-chat")
            & (_resp["stance"] == _stance)
            & (_resp["question_type"] != "observed")
        ][["predictShared", "pid"]].copy()
        _nochat["condition"] = "no_chat"

        # Chat for this stance, with CG type
        _chat = merged_df[merged_df["stance"] == _stance][
            ["predictShared", "pid", "common_ground_type"]
        ].copy()
        _chat["condition"] = "chat_" + _chat["common_ground_type"]
        _chat = _chat.drop(columns=["common_ground_type"])

        _combined = pd.concat([_nochat, _chat], ignore_index=True)

        _output = run_glmer_emmeans(
            _combined,
            "predictShared ~ condition + (1 | pid)",
            emmeans_spec="~ condition",
            contrast_method="trt.vs.ctrl",
            p_adjust="dunnettx",
            r_setup='d$condition <- relevel(factor(d$condition), ref = "no_chat")',
        )

        _all_output_lines.append(f"### {_stance.capitalize()} stance\n")
        _all_output_lines.append(f"```\n{_output}\n```\n")

    mo.md("\n".join(_all_output_lines))
    return


@app.cell
def _(
    BIN_SECONDS,
    CG_COLORS,
    CG_LABELS,
    CG_TYPES,
    FIGURES_DIR,
    STANCE_COLORS,
    STANCE_ORDER,
    np,
    plt,
    res_df,
    time_bins,
    type_df,
):
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.35, wspace=0.3)

    # Panel A: P(CG) over time by stance
    ax_main = fig.add_subplot(gs[0, :])

    for _stance in STANCE_ORDER:
        _sub = res_df[res_df["stance"] == _stance]
        _color = STANCE_COLORS[_stance]
        _label = _stance.capitalize()

        ax_main.plot(
            _sub["time_seconds"], _sub["pct_cg"],
            "o-", color=_color, label=_label, markersize=5, linewidth=2,
        )
        ax_main.fill_between(
            _sub["time_seconds"], _sub["ci_lo"], _sub["ci_hi"],
            alpha=0.15, color=_color,
        )
        for _, _row in _sub.iterrows():
            ax_main.annotate(
                f"n={_row['n_dyads']:.0f}",
                (_row["time_seconds"], _row["ci_hi"]),
                textcoords="offset points", xytext=(0, 6),
                fontsize=6, color=_color, alpha=0.7, ha="center",
            )

    ax_main.set_xlabel("Time into conversation (seconds)", fontsize=11)
    ax_main.set_ylabel("% of dyads with post-stance\ncommon ground", fontsize=11)
    ax_main.set_yticks([0, 25, 50, 75, 100])
    ax_main.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_main.set_ylim(0, 100)
    ax_main.set_xlim(time_bins[0] - 5, time_bins[-1] + 5)
    ax_main.legend(title="Stance", fontsize=10, title_fontsize=10)
    ax_main.set_title(
        "A. Common ground emergence over time (post-stance reveal only)",
        fontsize=12, fontweight="bold", loc="left",
    )
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # Panels B & C: CG type composition over time (stacked bars, by stance)
    for _i, _stance in enumerate(STANCE_ORDER):
        _ax = fig.add_subplot(gs[1, _i])
        _sub = type_df[type_df["stance"] == _stance]

        if _sub.empty:
            _ax.text(0.5, 0.5, "No data", transform=_ax.transAxes, ha="center")
            continue

        _times = sorted(_sub["time_seconds"].unique())
        _substantive = [t for t in CG_TYPES if t != "none"]
        _bottoms = np.zeros(len(_times))

        for _cg_type in _substantive:
            _pcts = []
            for _t in _times:
                _row = _sub[(_sub["time_seconds"] == _t) & (_sub["cg_type"] == _cg_type)]
                _pcts.append(_row["pct"].values[0] if len(_row) > 0 else 0)
            _pcts = np.array(_pcts)

            _ax.bar(
                _times, _pcts, bottom=_bottoms, width=BIN_SECONDS * 0.8,
                color=CG_COLORS[_cg_type],
                label=CG_LABELS[CG_TYPES.index(_cg_type)],
                edgecolor="white", linewidth=0.5,
            )
            _bottoms += _pcts

        _ax.set_xlabel("Time (seconds)", fontsize=10)
        _ax.set_ylabel("% of dyads", fontsize=10)
        _ax.set_ylim(0, 100)
        _ax.set_yticks([0, 25, 50, 75, 100])
        _ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        _panel_letter = "B" if _i == 0 else "C"
        _ax.set_title(
            f"{_panel_letter}. CG type -- {_stance.capitalize()} stance",
            fontsize=11, fontweight="bold", loc="left",
        )
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

        if _i == 1:
            _ax.legend(fontsize=7, loc="upper left", title="CG type", title_fontsize=8)

    plt.savefig(
        FIGURES_DIR / "figure_s_common_ground_timecourse.pdf",
        bbox_inches="tight", dpi=300,
    )
    plt.savefig(
        "/tmp/figure_s_common_ground_timecourse.png",
        bbox_inches="tight", dpi=150,
    )
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # CG Type → P(predictShared): Timecourse Data

    Same analysis as above, but using the **timecourse** CG annotations instead of the
    full-conversation annotation. Takes the last time bin per dyad as the cumulative CG
    type at end of conversation.
    """)
    return


@app.cell
def _(BIN_SECONDS, DATA_DIR, pd):
    # Load timecourse CG data → take last bin per dyad
    _tc_path = DATA_DIR / "llm_results" / f"common_ground_timecourse_{BIN_SECONDS}s.csv"
    _tc = pd.read_csv(_tc_path)
    _tc_revealed = _tc[_tc["focal_stance_revealed"] == True].copy()

    # Last time bin per dyad = cumulative CG type at end of conversation
    _idx = _tc_revealed.groupby("group_id")["time_seconds"].idxmax()
    tc_final_cg = _tc_revealed.loc[_idx, ["group_id", "common_ground_type", "stance"]].copy()
    tc_final_cg = tc_final_cg.rename(columns={"common_ground_type": "tc_common_ground_type"})

    print(f"Timecourse final CG types (last bin per dyad):")
    print(tc_final_cg["tc_common_ground_type"].value_counts().to_string())
    return (tc_final_cg,)


@app.cell
def _(DATA_DIR, pd, tc_final_cg):
    # Merge timecourse CG with behavioral responses
    _resp = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)

    _tc_cg = tc_final_cg.rename(columns={"group_id": "groupId"})
    _merged_tc = _resp.merge(
        _tc_cg[["groupId", "tc_common_ground_type"]],
        on="groupId",
        how="inner",
    )
    merged_tc_df = _merged_tc[_merged_tc["question_type"] != "observed"].copy()

    merged_tc_df["stance_e"] = merged_tc_df["stance"].map(
        {"opposing": 0.5, "shared": -0.5}
    )

    print(f"Timecourse-based merged rows: {len(merged_tc_df)}")
    print(f"Unique dyads: {merged_tc_df['groupId'].nunique()}")
    return (merged_tc_df,)


@app.cell
def _(
    FIGURES_DIR,
    N_BOOT,
    RNG,
    STANCE_COLORS,
    STANCE_ORDER,
    merged_tc_df,
    nochat_baseline,
    np,
    plt,
):
    # Figure: P(predictShared) by timecourse CG type and stance
    _cg_order = ["none", "rapport_only", "different_topic", "related_subtopic", "same_values"]
    _cg_display = ["None", "Rapport\nonly", "Different\ntopic", "Related\nsubtopic", "Same\nvalues"]

    _fig, _ax = plt.subplots(figsize=(10, 6))
    _width = 0.35
    _x = np.arange(len(_cg_order))

    for _i, _stance in enumerate(STANCE_ORDER):
        _means, _ci_los, _ci_his = [], [], []
        for _cg_type in _cg_order:
            _sub = merged_tc_df[
                (merged_tc_df["tc_common_ground_type"] == _cg_type)
                & (merged_tc_df["stance"] == _stance)
            ]
            _pid_means = _sub.groupby("pid")["predictShared"].mean().values
            if len(_pid_means) == 0:
                _means.append(np.nan)
                _ci_los.append(np.nan)
                _ci_his.append(np.nan)
                continue
            _m = _pid_means.mean()
            _boots = np.array([
                RNG.choice(_pid_means, size=len(_pid_means), replace=True).mean()
                for _ in range(N_BOOT)
            ])
            _lo, _hi = np.percentile(_boots, [2.5, 97.5])
            _means.append(_m)
            _ci_los.append(_lo)
            _ci_his.append(_hi)

        _means = np.array(_means)
        _ci_los = np.array(_ci_los)
        _ci_his = np.array(_ci_his)
        _offset = -_width / 2 + _i * _width
        _color = STANCE_COLORS[_stance]

        _ax.bar(
            _x + _offset, _means, _width,
            color=_color, alpha=0.7, label=_stance.capitalize(),
        )
        _ax.errorbar(
            _x + _offset, _means,
            yerr=[_means - _ci_los, _ci_his - _means],
            fmt="none", color=_color, capsize=3, linewidth=1.5,
        )

        # No-chat baseline
        _base = nochat_baseline.get(_stance, np.nan)
        _ax.axhline(
            _base, color=_color, linestyle="--", alpha=0.4, linewidth=1,
        )
        _ax.text(
            len(_cg_order) - 0.5, _base + 0.005,
            f"no-chat {_stance}", fontsize=8, color=_color, alpha=0.6,
            ha="right", va="bottom",
        )

    _ax.set_xticks(_x)
    _ax.set_xticklabels(_cg_display, fontsize=10)
    _ax.set_xlabel("Common Ground Type (from timecourse, last bin)", fontsize=11)
    _ax.set_ylabel("P(predictShared)", fontsize=11)
    _ax.set_title(
        "P(predictShared) by CG Type and Stance (timecourse-based)",
        fontsize=12, fontweight="bold", loc="left",
    )
    _ax.legend(title="Stance", fontsize=10, title_fontsize=10)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.set_ylim(0.3, 0.8)

    plt.savefig(
        FIGURES_DIR / "figure_s_cg_type_predict_shared_timecourse.pdf",
        bbox_inches="tight", dpi=300,
    )
    plt.savefig(
        "/tmp/figure_s_cg_type_predict_shared_timecourse.png",
        bbox_inches="tight", dpi=150,
    )
    _fig
    return


@app.cell
def _(merged_tc_df, mo, run_glmer_emmeans):
    # Simple effects: stance within CG type (timecourse-based)
    _model_data = merged_tc_df[["predictShared", "tc_common_ground_type",
                                 "stance", "pid", "groupId"]].copy()
    _model_data = _model_data.rename(columns={"tc_common_ground_type": "common_ground_type"})

    _output = run_glmer_emmeans(
        _model_data,
        "predictShared ~ common_ground_type * stance + (1 | pid) + (1 | groupId)",
        emmeans_spec="~ stance | common_ground_type",
        contrast_method="pairwise",
        r_setup="d$common_ground_type <- factor(d$common_ground_type)\nd$stance <- factor(d$stance)",
    )

    mo.md(
        "## Simple Effects: Stance Within CG Type (Timecourse-Based)\n\n"
        "Same simple-effects tests but using the timecourse CG annotation "
        "(last bin per dyad) instead of the full-conversation annotation.\n\n"
        f"```\n{_output}\n```"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # Conversational Goals Over Time

    LLM-annotated primary conversational goal per dyad per 15s time bin.
    Shows how the distribution of goals evolves over the conversation,
    split by stance condition.
    """)
    return


@app.cell
def _(BIN_SECONDS, DATA_DIR, pd):
    goals_df = pd.read_csv(
        DATA_DIR / "llm_results" / f"conversation_goals_{BIN_SECONDS}s.csv"
    )
    goals_df["time_seconds"] = (goals_df["time_bin"] + 1) * BIN_SECONDS

    GOAL_ORDER = [
        "rapport_building",
        "stance_sharing",
        "perspective_seeking",
        "sharing_experience",
        "finding_commonality",
        "exploring_nuance",
        "persuading",
        "other",
    ]
    GOAL_LABELS = [
        "Rapport building",
        "Stance sharing",
        "Perspective seeking",
        "Sharing experience",
        "Finding commonality",
        "Exploring nuance",
        "Persuading",
        "Other",
    ]
    GOAL_COLORS = {
        "rapport_building": "#C0C0C0",
        "stance_sharing": "#648FFF",
        "perspective_seeking": "#785EF0",
        "sharing_experience": "#FFB000",
        "finding_commonality": "#FE6100",
        "exploring_nuance": "#49B6A6",
        "persuading": "#DC267F",
        "other": "#E8E8E8",
    }

    print(f"Goal annotations: {len(goals_df)}")
    print(f"Unique dyads: {goals_df['group_id'].nunique()}")
    print(f"Goal distribution:\n{goals_df['primary_goal'].value_counts().to_string()}")
    return GOAL_COLORS, GOAL_LABELS, GOAL_ORDER, goals_df


@app.cell
def _(
    BIN_SECONDS,
    FIGURES_DIR,
    GOAL_COLORS,
    GOAL_LABELS,
    GOAL_ORDER,
    STANCE_ORDER,
    goals_df,
    np,
    plt,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for _i, _stance in enumerate(STANCE_ORDER):
        _ax = _axes[_i]
        _sub = goals_df[goals_df["stance"] == _stance]
        _times = sorted(_sub["time_seconds"].unique())

        # Compute proportions per time bin
        _proportions = {}
        for _goal in GOAL_ORDER:
            _pcts = []
            for _t in _times:
                _bin_data = _sub[_sub["time_seconds"] == _t]
                _n = len(_bin_data)
                _count = (_bin_data["primary_goal"] == _goal).sum()
                _pcts.append(_count / _n * 100 if _n > 0 else 0)
            _proportions[_goal] = np.array(_pcts)

        # Stacked area
        _bottoms = np.zeros(len(_times))
        for _j, _goal in enumerate(GOAL_ORDER):
            _ax.fill_between(
                _times,
                _bottoms,
                _bottoms + _proportions[_goal],
                color=GOAL_COLORS[_goal],
                label=GOAL_LABELS[_j],
                alpha=0.85,
                linewidth=0.5,
                edgecolor="white",
            )
            _bottoms += _proportions[_goal]

        # N annotations
        for _t in _times:
            _n = len(_sub[_sub["time_seconds"] == _t])
            _ax.annotate(
                f"n={_n}",
                (_t, 101),
                fontsize=5.5, color="gray", ha="center", va="bottom",
            )

        _ax.set_xlabel("Time into conversation (seconds)", fontsize=11)
        _ax.set_xlim(_times[0] - BIN_SECONDS / 2, _times[-1] + BIN_SECONDS / 2)
        _ax.set_ylim(0, 105)
        _ax.set_yticks([0, 25, 50, 75, 100])
        _ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        _ax.set_title(
            f"{_stance.capitalize()} stance",
            fontsize=12, fontweight="bold", loc="left",
        )
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

    _axes[0].set_ylabel("% of dyads", fontsize=11)

    # Shared legend
    _handles, _labels = _axes[0].get_legend_handles_labels()
    _fig.legend(
        _handles, _labels,
        loc="center right",
        bbox_to_anchor=(1.15, 0.5),
        fontsize=9,
        title="Primary goal",
        title_fontsize=10,
    )

    _fig.suptitle(
        "Conversational Goals Over Time",
        fontsize=13, fontweight="bold", x=0.05, ha="left",
    )
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "figure_s_conversation_goals_timecourse.pdf",
        bbox_inches="tight", dpi=300,
    )
    plt.savefig(
        "/tmp/figure_s_conversation_goals_timecourse.png",
        bbox_inches="tight", dpi=150,
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
