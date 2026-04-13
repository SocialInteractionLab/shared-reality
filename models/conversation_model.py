"""
Multi-observation conversation model.

Models conversation as a sequence of Bayesian updates in factor space,
where each informative 15-second conversation bin contributes an observation.

The topic scope of each bin determines which dimension of factor space
the observation falls on:
- same_values: focal question's factor loading (reinforces focal observation)
- related_subtopic: a same-domain question's factor loading
- different_topic: a different-domain question's factor loading

Whether shared reality was established determines the observation value:
- shared_reality=True: partner agrees with self (r_obs = r_self on that question)
- shared_reality=False: uninformative (r_obs = population mean)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm

from models.model import (
    _posterior_update_gaussian,
    _project_to_factors,
    load_factor_loadings,
    load_question_means,
    DOMAIN_RANGES,
    N_QUESTIONS,
)

DATA_DIR = Path(__file__).parent.parent / "data"

# Precompute domain membership for each question (0-indexed)
_QUESTION_TO_DOMAIN: dict[int, str] = {}
_DOMAIN_TO_QUESTIONS: dict[str, list[int]] = {}
for _domain, (_start, _end) in DOMAIN_RANGES.items():
    _DOMAIN_TO_QUESTIONS[_domain] = list(range(_start, _end))
    for _q in range(_start, _end):
        _QUESTION_TO_DOMAIN[_q] = _domain


def load_timecourse() -> pd.DataFrame:
    """Load LLM conversation timecourse annotations (run 0 only)."""
    df = pd.read_csv(
        DATA_DIR / "llm_results" / "perbin_commonality_timecourse_15s.csv"
    )
    return df[df["run"] == 0].copy()


def extract_observation_plan(
    group_id: str,
    timecourse_df: pd.DataFrame,
    focal_question: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Extract a sequence of observation specs from conversation bins.

    Each spec is: {question_idx: int, shared_reality: bool}
    indicating which question's factor loading to observe along,
    and whether the observation signals agreement.

    Args:
        group_id: group identifier
        timecourse_df: timecourse data (pre-filtered to run 0)
        focal_question: 0-indexed focal question
        rng: random generator for reproducible question selection
    """
    group = timecourse_df[timecourse_df["group_id"] == group_id].copy()
    group = group.sort_values("time_bin")

    # Only post-stance bins with informative topic scope
    post = group[group["focal_stance_revealed"] == True]
    informative = post[
        post["post_stance_topic_scope"].isin(
            ["same_values", "related_subtopic", "different_topic"]
        )
    ]

    focal_domain = _QUESTION_TO_DOMAIN[focal_question]
    same_domain_qs = [q for q in _DOMAIN_TO_QUESTIONS[focal_domain] if q != focal_question]
    diff_domain_qs = [q for q in range(N_QUESTIONS) if _QUESTION_TO_DOMAIN[q] != focal_domain]

    observations = []
    for _, row in informative.iterrows():
        scope = row["post_stance_topic_scope"]
        sr = bool(row["post_stance_shared_reality"])

        if scope == "same_values":
            q_idx = focal_question
        elif scope == "related_subtopic":
            q_idx = rng.choice(same_domain_qs) if same_domain_qs else focal_question
        elif scope == "different_topic":
            q_idx = rng.choice(diff_domain_qs)
        else:
            continue

        observations.append({"question_idx": int(q_idx), "shared_reality": sr})

    return observations


def predict_with_conversation(
    obs_plan: list[dict],
    r_self: np.ndarray,
    focal_question: int,
    focal_r_obs: float,
    loadings: np.ndarray,
    means: np.ndarray,
    sigma_prior: float,
    sigma_conv: float,
    threshold: float,
    anchor_weight: float = 0.0,
) -> np.ndarray:
    """
    Predict P(match) using chained conversation observations.

    First updates from the focal observation (what the participant actually
    perceived the partner answered), then chains updates from each
    conversation bin observation.

    Args:
        obs_plan: list of {question_idx, shared_reality} from extract_observation_plan
        r_self: participant's own responses (35,)
        focal_question: 0-indexed focal question
        focal_r_obs: partner's perceived response on focal question
        loadings: factor loadings (35, k)
        means: question means (35,)
        sigma_prior: prior std
        sigma_conv: observation noise for conversation-inferred observations
        threshold: match threshold τ
        anchor_weight: self-anchoring weight α (default 0)
    """
    k = loadings.shape[1]
    loadings_j = jnp.array(loadings)
    means_j = jnp.array(means)
    r_self_j = jnp.array(r_self)

    # Prior
    prior_cov = jnp.array(sigma_prior**2 * np.eye(k))
    if anchor_weight > 0:
        theta_self = _project_to_factors(r_self_j, loadings_j, means_j)
        prior_mean = anchor_weight * theta_self
    else:
        prior_mean = jnp.zeros(k)

    # Update 1: focal observation (same as single-obs model, but with σ_conv noise)
    L_focal = loadings_j[focal_question]
    mu_focal = means_j[focal_question]
    post_mean, post_cov = _posterior_update_gaussian(
        L_focal, focal_r_obs, mu_focal, prior_mean, prior_cov,
        jnp.array(sigma_conv**2),
    )

    # Chain conversation observations
    conv_variance = jnp.array(sigma_conv**2)
    for obs in obs_plan:
        q_idx = obs["question_idx"]
        L_obs = loadings_j[q_idx]
        mu_obs = means_j[q_idx]

        if obs["shared_reality"]:
            # Agreement: partner responded like me on this question
            r_obs = r_self_j[q_idx]
        else:
            # No shared reality: uninformative (population mean)
            r_obs = mu_obs

        post_mean, post_cov = _posterior_update_gaussian(
            L_obs, r_obs, mu_obs, post_mean, post_cov, conv_variance,
        )

    # Predict match probabilities
    pred_means = loadings_j @ post_mean + means_j
    pred_vars = jnp.sum((loadings_j @ post_cov) * loadings_j, axis=1) + sigma_conv**2
    pred_vars = jnp.maximum(pred_vars, 1e-10)
    pred_stds = jnp.sqrt(pred_vars)

    upper = (r_self_j + threshold - pred_means) / pred_stds
    lower = (r_self_j - threshold - pred_means) / pred_stds
    preds = jnp.clip(jax_norm.cdf(upper) - jax_norm.cdf(lower), 0.0, 1.0)

    return np.asarray(preds)


def predict_with_conversation_separated(
    obs_plan: list[dict],
    r_self: np.ndarray,
    focal_question: int,
    focal_r_obs: float,
    loadings: np.ndarray,
    means: np.ndarray,
    sigma_prior: float,
    sigma_focal: float,
    sigma_conv: float,
    threshold: float,
    anchor_weight: float = 0.0,
) -> np.ndarray:
    """
    Predict P(match) with separated focal and conversation noise.

    Uses σ_focal for the focal observation (fixed from single-obs model)
    and σ_conv for conversation-inferred observations (fitted separately).
    With zero conversation observations, this recovers the single-obs model.

    Args:
        obs_plan: list of {question_idx, shared_reality} from extract_observation_plan
        r_self: participant's own responses (35,)
        focal_question: 0-indexed focal question
        focal_r_obs: partner's perceived response on focal question
        loadings: factor loadings (35, k)
        means: question means (35,)
        sigma_prior: prior std
        sigma_focal: observation noise for focal observation
        sigma_conv: observation noise for conversation observations
        threshold: match threshold τ
        anchor_weight: self-anchoring weight α (default 0)
    """
    k = loadings.shape[1]
    loadings_j = jnp.array(loadings)
    means_j = jnp.array(means)
    r_self_j = jnp.array(r_self)

    # Prior
    prior_cov = jnp.array(sigma_prior**2 * np.eye(k))
    if anchor_weight > 0:
        theta_self = _project_to_factors(r_self_j, loadings_j, means_j)
        prior_mean = anchor_weight * theta_self
    else:
        prior_mean = jnp.zeros(k)

    # Update 1: focal observation with its own noise
    L_focal = loadings_j[focal_question]
    mu_focal = means_j[focal_question]
    post_mean, post_cov = _posterior_update_gaussian(
        L_focal, focal_r_obs, mu_focal, prior_mean, prior_cov,
        jnp.array(sigma_focal**2),
    )

    # Chain conversation observations with separate noise
    conv_variance = jnp.array(sigma_conv**2)
    for obs in obs_plan:
        q_idx = obs["question_idx"]
        L_obs = loadings_j[q_idx]
        mu_obs = means_j[q_idx]

        if obs["shared_reality"]:
            r_obs = r_self_j[q_idx]
        else:
            r_obs = mu_obs

        post_mean, post_cov = _posterior_update_gaussian(
            L_obs, r_obs, mu_obs, post_mean, post_cov, conv_variance,
        )

    # Predict match probabilities (use focal noise for predictive variance)
    pred_means = loadings_j @ post_mean + means_j
    pred_vars = jnp.sum((loadings_j @ post_cov) * loadings_j, axis=1) + sigma_focal**2
    pred_vars = jnp.maximum(pred_vars, 1e-10)
    pred_stds = jnp.sqrt(pred_vars)

    upper = (r_self_j + threshold - pred_means) / pred_stds
    lower = (r_self_j - threshold - pred_means) / pred_stds
    preds = jnp.clip(jax_norm.cdf(upper) - jax_norm.cdf(lower), 0.0, 1.0)

    return np.asarray(preds)


def evaluate_conversation_model_separated(
    eval_data: dict,
    loadings: np.ndarray,
    means: np.ndarray,
    sigma_prior: float,
    sigma_focal: float,
    sigma_conv: float,
    threshold: float,
    epsilon: float,
    anchor_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Evaluate the separated-σ conversation model.

    σ_focal controls the focal observation noise (can be fixed from single-obs fit).
    σ_conv controls conversation observation noise (fitted).
    """
    info = eval_data["participants"]
    pid_groups = info.groupby("pid")
    all_preds = {}

    for pid, group in pid_groups:
        first = group.iloc[0]
        preds = predict_with_conversation_separated(
            obs_plan=first["obs_plan"],
            r_self=first["r_self"],
            focal_question=first["focal_q"],
            focal_r_obs=first["focal_r_obs"],
            loadings=loadings,
            means=means,
            sigma_prior=sigma_prior,
            sigma_focal=sigma_focal,
            sigma_conv=sigma_conv,
            threshold=threshold,
            anchor_weight=anchor_weight,
        )
        preds = (1 - epsilon) * preds + epsilon * 0.5
        all_preds[pid] = preds

    result = info[["pid", "question", "question_domain", "stance", "question_type", "actual"]].copy()
    result["pred_prob"] = [
        all_preds[row["pid"]][row["question"]]
        for _, row in info.iterrows()
    ]

    return result


def prepare_chat_eval_data(
    responses_df: pd.DataFrame,
    timecourse_df: pd.DataFrame,
    seed: int = 42,
) -> dict:
    """
    Prepare evaluation data for the multi-observation conversation model.

    Returns dict with per-participant data needed for evaluation.
    """
    rng = np.random.default_rng(seed)
    chat = responses_df[responses_df["experiment"] == "chat"].copy()

    participants = []
    for pid in chat["pid"].unique():
        subj = chat[chat["pid"] == pid]
        matched = subj[subj["is_matched"] == True]
        if len(matched) == 0:
            continue

        group_id = subj["groupId"].iloc[0]
        focal_q = int(matched["matchedIdx"].iloc[0]) - 1

        # Focal observation: participant's perception of partner's response
        obs_row = subj[subj["question_type"] == "observed"]
        if len(obs_row) == 0:
            continue
        focal_r_obs = obs_row["postChatResponse"].iloc[0]
        if pd.isna(focal_r_obs):
            continue

        # Self responses
        r_self = np.zeros(N_QUESTIONS)
        for _, row in subj.iterrows():
            r_self[int(row["question"]) - 1] = row["preChatResponse"]

        # Conversation observation plan (shared per group, but extracted per call
        # since the random question selection can vary)
        obs_plan = extract_observation_plan(
            group_id, timecourse_df, focal_q, rng,
        )

        # Participant info for evaluation
        for _, row in subj.iterrows():
            participants.append({
                "pid": pid,
                "pid_idx": len(participants),
                "group_id": group_id,
                "focal_q": focal_q,
                "focal_r_obs": float(focal_r_obs),
                "question": int(row["question"]) - 1,
                "question_domain": row["preChatDomain"],
                "stance": matched["stance"].iloc[0],
                "question_type": row["question_type"],
                "actual": row["participant_binary_prediction"],
                "r_self": r_self,
                "obs_plan": obs_plan,
            })

    return {"participants": pd.DataFrame(participants)}


def evaluate_conversation_model(
    eval_data: dict,
    loadings: np.ndarray,
    means: np.ndarray,
    sigma_prior: float,
    sigma_conv: float,
    threshold: float,
    epsilon: float,
    anchor_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Evaluate the multi-observation conversation model.

    Loops over unique participants (not vmap — variable-length obs sequences).
    """
    info = eval_data["participants"]

    # Group by pid to avoid redundant prediction computation
    pid_groups = info.groupby("pid")
    all_preds = {}

    for pid, group in pid_groups:
        first = group.iloc[0]
        preds = predict_with_conversation(
            obs_plan=first["obs_plan"],
            r_self=first["r_self"],
            focal_question=first["focal_q"],
            focal_r_obs=first["focal_r_obs"],
            loadings=loadings,
            means=means,
            sigma_prior=sigma_prior,
            sigma_conv=sigma_conv,
            threshold=threshold,
            anchor_weight=anchor_weight,
        )
        # Apply lapse rate
        preds = (1 - epsilon) * preds + epsilon * 0.5
        all_preds[pid] = preds

    # Map predictions to questions
    result = info[["pid", "question", "question_domain", "stance", "question_type", "actual"]].copy()
    result["pred_prob"] = [
        all_preds[row["pid"]][row["question"]]
        for _, row in info.iterrows()
    ]

    return result
