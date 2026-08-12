# HANDOFF — shared-reality

_Snapshot taken 2026-08-12, before returning the lab laptop. Everything below describes state as of that date._

## What this is

The paper repo for the commonality-inference work: two studies on how people generalise from one revealed belief to a partner's other beliefs. Participants answer 35 belief questions across seven domains, then either imagine an interaction (Study 1) or have a real three-minute conversation (Study 2), then predict what their partner thinks. The finding is a *graded* generalisation gradient — strongest within domain, weaker across domains — and the theoretical claim is that the gradient tracks population-level belief covariance (a Bayesian factor model, k = 5) rather than egocentric projection from one's own beliefs. Conversation amplifies the whole effect. A set of LLM analyses reads the conversation transcripts for moments of commonality.

This is the manuscript-and-analysis repo under the `stanford-soil` org. It is the *current* repo; `shared-reality-dev` (hawkrobe org) is the older, larger predecessor that this was carved out of — see that repo's HANDOFF for its state. Unrelated to the `rare-commonality` / `rare-commonality-chat` pair, which is a different dissertation chapter.

Naming note: the internal documents call this "Ch1" (`docs/ch1_orientation.md`, `docs/ch1_queue.md`, `docs/ch1_revision_list.md`), and the mirrored figure directory in the `dissertation` repo is `figures/ch1`. If the chapter has been renumbered in the assembled dissertation, the internal labels are stale naming, not a different project.

## Where it stands

**Read this section carefully — the branch situation is the main thing that will confuse a cold start.**

- **HEAD is on `dissert-talk-figs`, not `main`.** At `fc36263` — 2026-08-11, "✨ figs from public defense talk". Working tree is completely clean and the branch is in sync with `origin/dissert-talk-figs`. Nothing is at risk.
- **`main` is four months stale**, sitting at `183dcf1` — 2026-04-13, "✨ add multi-obs model as comparison". `dissert-talk-figs` is exactly one commit ahead of `main` and branches directly off it. **It has never been merged.** If you clone fresh on the new machine you will land on `main` and silently lose sight of the most recent work; check out `dissert-talk-figs` first, or merge it.
- That single commit is substantial, not cosmetic: 1,427 insertions across nine files — `analysis/talk_bridge.py`, `talk_gradient_levels.py`, `talk_gradient_steps.py`, `talk_model_overlay.py`, `talk_phenomenon_bars.py`, `generate_figure_s_sr_annots.py`, `split_half_heldout.py`, `refit_random_slopes.R`, and a large rewrite of `models/llm/annotate_commonality.py`. Several of those files had been sitting uncommitted on disk since April–July; the 8/11 commit swept them all in. The talk figures are the visible part, but `split_half_heldout.py` and `refit_random_slopes.R` are *analysis* answering reviewer/co-author requests, not slideware.
- **`origin/develop-other-models` exists only on the remote.** At `3ccbfce` — 2026-04-28, "✨ LLM-flagged post stance commonality". Three commits ahead of `main`, branched from the same `183dcf1`. There is **no local copy of this branch on this laptop** — it was never fetched into a working checkout, so nothing of it is visible on disk. Its unique contents: `analysis/llm_commonality_on_expectations.py` (353 lines), `models/conversation_model.py` (298 lines), `models/llm/annotate_question_informativeness.py` (425 lines), plus several large model-prediction PNGs. This is real April work that lives nowhere else. Live, not stale — but orphaned. Decide whether to merge it or explicitly abandon it.
- Remote: `origin` = https://github.com/stanford-soil/shared-reality.git.

Summary of the three heads: `main` (Apr 13, oldest), `dissert-talk-figs` (Aug 11, current, +1), `origin/develop-other-models` (Apr 28, +3, remote-only). All three share `183dcf1` as their common base. Nothing has been merged into anything.

## Uncommitted work on the old laptop

`git status --porcelain` is empty. Every tracked file is committed and pushed.

But a large amount of important material on disk is **gitignored**, not untracked, so it does not show up in `git status` and will die with the laptop unless copied off:

- **`docs/` (gitignored in its entirety, ~280 KB).** This is the manuscript and its whole revision apparatus:
  - `manuscript.tex` — 123 KB, 1,091 lines, the paper itself, synced down from Overleaf.
  - `docs/pnas/manuscript_pnas.tex` — the PNAS-formatted version (author affiliations already corrected there; two contribution initials still "TBD").
  - `ch1_orientation.md` (Aug 10) — cold-start orientation; read this first if you resume the revision.
  - `ch1_queue.md` (Aug 10) — the current triaged work queue, supersedes the ordering in the revision list.
  - `ch1_revision_list.md` (Aug 4, 32 KB) — full triage of both co-authors' feedback.
  - `ch1_tier0_patches.md` (Aug 10) — 24 mechanical find/replace fixes.
  - `ch1_random_slopes_result.md` (Aug 6) — result of the one analysis already run.
  - `feedback_before_submission.md` — the raw co-author feedback (five items from Robert, ~24 from Chris).
  - plus `manuscript_changelog.md`, `model-fitting-notes.md`, `science_template.tex`, `docs/figures/`.
  **Judgment: carry over, all of it.** Losing `docs/` means losing the manuscript's working copy and the entire revision plan. Overleaf holds the paper text, but not the triage docs.
- **`outputs/` (gitignored).** Every rendered talk and supplement figure — `talk_gradient_step*.pdf/png`, `talk_model_overlay*`, `talk_phenomenon_bars`, `talk_bridge_*`, `figure_s_sr_annots`. Regenerable from the committed scripts; carry over only for convenience.
- **`.venv/`** — do not copy, rebuild with `uv sync`.

## Open threads

The paper is mid-revision, targeting **PNAS** (it had been aimed at Science; `docs/science_template.tex` is a leftover). The stated plan as of 8/10 was a revised draft to Robert and Chris before an **8/18 move**, for a **9/1 submission**. That timeline is the live constraint.

The queue in `docs/ch1_queue.md` compresses ~29 co-author comments into three decisions, four rewrites, and one mechanical sweep. Already settled: the random-slopes refit (β = 0.07, p = .065; main effect β = 0.43 robust; the three-way null in all four specifications) and the PNAS affiliation fixes. Outstanding, in the queue's order:

1. **Tier-0 sweep** — 24 exact find/replace fixes, chief among them `N = 260` → `250`.
2. **Limitations paragraph** — US-Prolific sample, and the fact that item separability was engineered by triplet validation, making the gradient magnitudes an upper bound.
3. **Disclose the n = 17 / r = 0.19 inline** — Robert's first concern is that the "population structure, not egocentric projection" story leans on that result.
4. **Decide what egocentric projection actually predicts** (Chris's C6). The manuscript states it four inconsistent ways; page 7 says the gradient is inconsistent with projection while page 10 says consistent. This needs co-author sign-off and was flagged as the item to send out first because it has the longest latency.
5. **Define "prediction error"** (C12) — the current classification rule mislabels a case Chris constructed.
6. **The "95% vs 23%" comparison oversells** (Robert #2) — item-level accuracy is 56% and the parameters were grid-searched. `analysis/split_half_heldout.py`, committed 8/11, appears to be the response to this; whether its result has been written into the paper is unclear from the repo alone.
7. Smaller: a duplicated paragraph at lines 375/377, Table S4 never referenced (C20), S9 reporting zero numbers while claiming the egocentric model wins on Likert prediction (C22), and the misperception breakdown being promoted to the main text with the dating-app framing dialled back.

One thing the orientation doc is emphatic about: **`manuscript.tex` has never been edited by an agent session** — all prior work was read-only against it, and the Overleaf project was never touched. Preserve that if you resume with an agent.

## Picking this up again

```bash
git clone https://github.com/stanford-soil/shared-reality.git
cd shared-reality
git checkout dissert-talk-figs        # do this — main is four months behind
git fetch origin develop-other-models # then decide what to do with it
uv sync
Rscript -e "install.packages(c('lme4','lmerTest','dplyr','tidyr','purrr','reticulate'))"
```

Needs Python >= 3.10 (uv, or `pip install -r requirements.txt`), R >= 4.0, and Quarto >= 1.3. The Quarto notebooks call Python through R's `reticulate`, so you must point R at the venv:

```bash
export RETICULATE_PYTHON=$(pwd)/.venv/bin/python
quarto render analysis/behavioral_analyses.qmd   # Figures 2, 3, 4
quarto render analysis/model_analyses.qmd        # Figures 5, 6 (Bayesian factor model, LLM)
quarto render analysis/supplement.qmd            # SI figures and tables
```

Standalone scripts (`analysis/talk_*.py`, `split_half_heldout.py`, `generate_figure_s_sr_annots.py`) run directly under `uv run python`. `analysis/refit_random_slopes.R` runs under `Rscript`.

What to check first:

1. Which branch you are on. This is the repo's single biggest trap.
2. Whether `docs/` was copied over. Without it there is no manuscript and no revision plan.
3. `data/` is committed here (unlike in `rare-commonality`) — `responses.csv`, `questions.csv`, `messages.csv`, `data/raw/`, `data/llm_results/` all come down with the clone. Note that `data/llm_results/*.jsonl` is gitignored by pattern but two `.jsonl` files were committed before that rule; do not assume the ignore rule reflects what is actually tracked.

## Landmines

- **Branch confusion is the top risk.** A fresh clone lands on a `main` that is missing the last four months. See above.
- **`origin/develop-other-models` has no local copy anywhere.** If that remote branch were ever deleted, ~1,000 lines of April model and LLM work would be gone. Nothing on this laptop backs it up.
- **`docs/` is gitignored and laptop-only.** The manuscript working copy, the PNAS variant, and every revision-planning document. This is the most valuable non-recoverable content in the repo.
- **The LLM pipeline needs Google Cloud credentials.** `models/llm/config.py` reads `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `GOOGLE_CLOUD_BUCKET` from a `.env` at the repo root (gitignored). There is **no `.env` on this laptop currently**, so the annotation pipeline is not runnable as-is on either machine until those are set. Vertex AI / Gemini access is required; batch requests write to `models/llm/batch_requests/`, also gitignored.
- **`outputs/` is gitignored**, so no figure you generate will ever show up in `git status`. Do not assume a clean tree means the figures are safe.
- **`*.html` is gitignored repo-wide**, which means rendered Quarto output is invisible to git by design.
- **`reticulate` breaks silently** if `RETICULATE_PYTHON` points at a stale venv path — the README's troubleshooting section covers this; use an absolute path.
- **This HANDOFF.md is itself untracked.** Commit it (onto `dissert-talk-figs`, where HEAD currently is) or copy it off before the laptop goes back.
