#!/usr/bin/env python3
"""
Annotate chat transcripts for shared reality (commonality of inner states).

For each dyad, sends the conversation transcript to Gemini and asks it to classify
whether participants discovered shared inner states (thoughts, feelings, beliefs)
about the world — i.e., shared reality (Rossignac-Milon et al., 2021).

Usage:
    # Dry run: create batch file only
    uv run -m models.llm.annotate_commonality --dry-run

    # Submit batch to Vertex AI
    uv run -m models.llm.annotate_commonality

    # Download and parse results
    uv run -m models.llm.annotate_commonality --download

    # Process a small sample first
    uv run -m models.llm.annotate_commonality --dry-run --sample 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Optional

import pandas as pd
import polars as pl
from polars import col

from .config import (
    DATA_DIR,
    BATCH_DIR,
    RESULTS_DIR,
    MODEL_CONFIG,
    GCS_PROJECT,
    GCS_BUCKET,
)

GCS_PREFIX = "llm-annotate"
EXPERIMENT_NAME = "commonality"
TC_EXPERIMENT_BASE = "commonality_timecourse"
PERBIN_TC_EXPERIMENT_BASE = "perbin_commonality_timecourse"
MAX_CONVERSATION_SECONDS = 195

DEFAULT_BIN_SECONDS = 15


def tc_experiment_name(bin_seconds: int) -> str:
    """Experiment name with bin size suffix, e.g. commonality_timecourse_15s."""
    return f"{TC_EXPERIMENT_BASE}_{bin_seconds}s"


def perbin_tc_experiment_name(bin_seconds: int) -> str:
    """Experiment name for per-bin variant, e.g. perbin_commonality_timecourse_15s."""
    return f"{PERBIN_TC_EXPERIMENT_BASE}_{bin_seconds}s"


def build_full_conversation(messages_df: pd.DataFrame, group_id: str) -> str:
    """Build the full conversation transcript for a dyad."""
    group_msgs = messages_df[messages_df["group_id"] == group_id].copy()
    group_msgs = group_msgs.sort_values("absolute_timestamp")
    lines = [
        f"{'Cat' if r['author'] == '🐱' else 'Dog'}: {r['message_string']}"
        for _, r in group_msgs.iterrows()
    ]
    return "\n".join(lines)


def build_conversation_up_to(
    messages_df: pd.DataFrame, group_id: str, up_to_time: pd.Timestamp
) -> str:
    """Build conversation transcript up to a given timestamp."""
    group_msgs = messages_df[
        (messages_df["group_id"] == group_id)
        & (messages_df["absolute_timestamp"] <= up_to_time)
    ].copy()
    group_msgs = group_msgs.sort_values("absolute_timestamp")
    lines = [
        f"{'Cat' if r['author'] == '🐱' else 'Dog'}: {r['message_string']}"
        for _, r in group_msgs.iterrows()
    ]
    return "\n".join(lines)


def create_annotation_prompt(
    conversation: str, focal_question: str, focal_domain: str
) -> str:
    """Create prompt for annotating shared inner states in a conversation.

    Args:
        conversation: Full conversation transcript
        focal_question: The question the dyad was matched on
        focal_domain: Domain of the focal question
    """
    return f"""You are analyzing a conversation between two people (Cat and Dog) who were paired
for a study where they chat with a partner. They were asked to discuss a specific topic.

=== CONTEXT ===
Focal question they discussed: "{focal_question}"
Topic domain: {focal_domain}
Before the conversation, each participant independently answered this question on a 1-5 scale.
You do NOT know whether they agreed or disagreed — your job is to determine what happens
in the conversation itself.

=== CONVERSATION ===
{conversation}

=== ANNOTATION TASK ===
We are studying "shared reality" — the experience of sharing inner states (thoughts,
feelings, and beliefs) about the world with another person (Rossignac-Milon et al., 2021).
Analyze this conversation in two phases. Be specific and cite evidence.

PHASE 1 — STANCE ESTABLISHMENT:
Identify the portion of the conversation where participants first share their positions on
the focal topic.

1a. FOCAL STANCE REVEALED: Did both participants reveal their position on the focal topic?
1b. INITIAL ALIGNMENT: Based on what they said, did they appear to agree or disagree on the
    focal topic? (Only what's visible in the conversation — not your inference about what
    they "really" think.)
1c. LAST STANCE MESSAGE: Quote the last message that is part of the FIRST exchange where
    both participants establish their positions on the focal topic. Use these guidelines:
    - Look for the earliest point where both Cat and Dog have each stated a position.
    - Include any immediate reactions to those positions (e.g., "I agree" or "That's
      interesting, I feel differently").
    - If participants revisit or elaborate on their stances later, that counts as
      post-stance conversation — the boundary is the first establishment, not the last
      mention.
    - If stances are never clearly established, set last_stance_message to null.

PHASE 2 — POST-STANCE SHARED INNER STATES:
Now focus ONLY on the messages that come AFTER the initial stance establishment identified
above. Ignore the stance-establishment messages themselves. We want to know whether the
participants discovered shared reality — that they share inner states (thoughts, feelings,
or beliefs) about the world.

2a. SHARED INNER STATES: In the post-stance portion of the conversation, did the
    participants discover that they share inner states about the world? This is NOT about
    being polite or getting along — it requires a moment where both participants recognize
    they think, feel, or believe something in a similar way about the world.

2b. TOPIC SCOPE: If shared inner states were discovered, what was the topic scope?
    a) same_values: On the focal topic itself — they share underlying thoughts, feelings,
       or beliefs related to the question they were asked to discuss
    b) related_subtopic: On a closely related subtopic within the same domain
    c) different_topic: On a different topic that came up naturally in conversation
    d) rapport_only: Only general rapport or politeness without discovering substantive
       shared inner states about the world
    e) none: No shared inner states discovered in the post-stance conversation

2c. INNER STATE DIMENSION: If shared inner states were discovered (topic_scope is NOT
    "rapport_only" or "none"), what kind of inner state do they share?
    a) shared_thoughts: They discover they think about something in the same way — similar
       perspectives, judgments, or ways of seeing things
       (e.g., "we both see it that way", "we think about it the same way")
    b) shared_feelings: They discover they feel the same way about something — similar
       emotions, concerns, or affective reactions
       (e.g., "we're both frustrated by X", "we both love X", "X worries us both")
    c) shared_beliefs: They discover they hold similar beliefs about how the world is or
       should be — similar convictions, values, or principles
       (e.g., "we both believe X is important", "we both think X is wrong")
    d) none: No substantive shared inner states (use when topic_scope is "rapport_only"
       or "none")

2d. DIFFERENCES: In the post-stance portion, did the participants discover additional
    differences or disagreements beyond their initial positions?

2e. KEY EXCHANGE: Quote the most important exchange from the post-stance conversation
    that illustrates shared inner states or difference discovery.

2f. SURPRISE: Given the initial positions established in Phase 1, would an outside observer
    have predicted they would discover these shared inner states? (i.e., is it surprising
    given their stated positions on the focal topic?)

For each classification, provide your reasoning explaining WHY you chose that category,
citing specific evidence from the conversation.

IMPORTANT CONSISTENCY RULES:
- If post_stance_shared_reality is false, then topic_scope MUST be "none" and
  inner_state_dimension MUST be "none".
- If topic_scope is "rapport_only" or "none", then inner_state_dimension MUST be "none".
- If topic_scope is "same_values", "related_subtopic", or "different_topic", then
  inner_state_dimension MUST be one of: "shared_thoughts", "shared_feelings",
  or "shared_beliefs".

Return JSON with this exact structure:
{{
  "focal_stance_revealed": true/false,
  "initial_alignment": "agreement" | "disagreement" | "unclear" | "not_discussed",
  "initial_alignment_reasoning": "explain why you classified the initial alignment this way, citing specific messages",
  "initial_alignment_evidence": "brief quote showing how they established their stances",
  "last_stance_message": "quote of the last message in the stance-establishment phase, or null if stances never established",
  "n_post_stance_messages": <integer count of messages after the last stance message>,
  "post_stance_shared_reality": true/false,
  "topic_scope": "same_values" | "related_subtopic" | "different_topic" | "rapport_only" | "none",
  "topic_scope_reasoning": "explain why you chose this topic scope over alternatives, citing specific post-stance messages",
  "shared_reality_description": "what shared inner states were discovered in post-stance conversation",
  "inner_state_dimension": "shared_thoughts" | "shared_feelings" | "shared_beliefs" | "none",
  "inner_state_reasoning": "explain why you chose this inner state dimension, citing specific evidence",
  "post_stance_differences": true/false,
  "post_stance_differences_description": "what additional differences emerged after stance establishment",
  "key_exchange": "quote of key exchange from post-stance conversation",
  "surprising": true/false,
  "surprise_reasoning": "explain why these shared inner states were or were not predictable from their initial stated positions",
  "notes": "any additional observations"
}}"""


def create_timecourse_prompt(
    conversation: str, focal_question: str, focal_domain: str, time_seconds: int
) -> str:
    """Create prompt for annotating shared inner states in a PARTIAL conversation.

    Args:
        conversation: Partial conversation transcript up to a time point
        focal_question: The question the dyad was matched on
        focal_domain: Domain of the focal question
        time_seconds: How many seconds into the conversation this snapshot is
    """
    return f"""You are analyzing a PARTIAL conversation between two people (Cat and Dog) who were
paired for a study where they chat with a partner. They were asked to discuss a specific topic.
You are seeing the conversation up to the {time_seconds}-second mark. The conversation may
continue after this point, but you should only analyze what you can see.

=== CONTEXT ===
Focal question they discussed: "{focal_question}"
Topic domain: {focal_domain}
Before the conversation, each participant independently answered this question on a 1-5 scale.
You do NOT know whether they agreed or disagreed — your job is to determine what happens
in the conversation itself.

=== CONVERSATION (first {time_seconds} seconds) ===
{conversation}

=== ANNOTATION TASK ===
We are studying "shared reality" — the experience of sharing inner states (thoughts,
feelings, and beliefs) about the world with another person (Rossignac-Milon et al., 2021).
Based on what you can see SO FAR, analyze this conversation in two phases.

PHASE 1 — STANCE ESTABLISHMENT:
Has the portion of the conversation you can see included both participants sharing their
positions on the focal topic?

1a. FOCAL STANCE REVEALED: Did both participants reveal their position on the focal topic
    in the conversation so far?
1b. INITIAL ALIGNMENT: If both stances are revealed, did they appear to agree or disagree?
    (Only what's visible — not your inference about what they "really" think.)
1c. LAST STANCE MESSAGE: Quote the last message that is part of the FIRST exchange where
    both participants establish their positions. Use these guidelines:
    - Look for the earliest point where both Cat and Dog have each stated a position.
    - Include any immediate reactions to those positions.
    - If participants revisit or elaborate on their stances later, that counts as
      post-stance conversation.
    - If stances are never clearly established in what you can see, set to null.

PHASE 2 — POST-STANCE SHARED INNER STATES:
If stances have been established, focus ONLY on messages AFTER the stance establishment.
If stances have NOT been established yet, skip Phase 2 and set all Phase 2 fields to
their "not yet" defaults (post_stance_shared_reality: false, topic_scope: "none", etc.).

2a. SHARED INNER STATES: In the post-stance portion so far, did the participants discover
    shared reality — that they share inner states (thoughts, feelings, or beliefs) about
    the world? This is NOT about being polite or getting along — it requires a moment
    where both participants recognize they think, feel, or believe something in a similar
    way about the world.

2b. TOPIC SCOPE: If shared inner states were discovered, what was the topic scope?
    a) same_values: On the focal topic itself — they share underlying thoughts, feelings,
       or beliefs related to the question they were asked to discuss
    b) related_subtopic: On a closely related subtopic within the same domain
    c) different_topic: On a different topic that came up naturally
    d) rapport_only: Only general rapport or politeness without substantive shared inner
       states about the world
    e) none: No shared inner states discovered so far

2c. INNER STATE DIMENSION: If shared inner states were discovered (topic_scope is NOT
    "rapport_only" or "none"), what kind of inner state do they share?
    a) shared_thoughts: They discover they think about something in the same way — similar
       perspectives, judgments, or ways of seeing things
    b) shared_feelings: They discover they feel the same way about something — similar
       emotions, concerns, or affective reactions
    c) shared_beliefs: They discover they hold similar beliefs about how the world is or
       should be — similar convictions, values, or principles
    d) none: No substantive shared inner states (use when topic_scope is "rapport_only"
       or "none")

2d. DIFFERENCES: In the post-stance portion so far, did the participants discover additional
    differences or disagreements beyond their initial positions?

IMPORTANT CONSISTENCY RULES:
- If post_stance_shared_reality is false, then topic_scope MUST be "none" and
  inner_state_dimension MUST be "none".
- If topic_scope is "rapport_only" or "none", then inner_state_dimension MUST be "none".
- If topic_scope is "same_values", "related_subtopic", or "different_topic", then
  inner_state_dimension MUST be one of: "shared_thoughts", "shared_feelings",
  or "shared_beliefs".

Return JSON with this exact structure:
{{
  "focal_stance_revealed": true/false,
  "initial_alignment": "agreement" | "disagreement" | "unclear" | "not_discussed",
  "initial_alignment_reasoning": "explain why, citing specific messages",
  "last_stance_message": "quote or null if stances not yet established",
  "n_post_stance_messages": <integer count of messages after the last stance message, 0 if stances not yet established>,
  "post_stance_shared_reality": true/false,
  "topic_scope": "same_values" | "related_subtopic" | "different_topic" | "rapport_only" | "none",
  "topic_scope_reasoning": "explain why you chose this topic scope, citing specific post-stance messages",
  "shared_reality_description": "what shared inner states were discovered so far",
  "inner_state_dimension": "shared_thoughts" | "shared_feelings" | "shared_beliefs" | "none",
  "inner_state_reasoning": "explain why you chose this inner state dimension",
  "post_stance_differences": true/false,
  "post_stance_differences_description": "what additional differences emerged"
}}"""


def get_chat_groups() -> pd.DataFrame:
    """Get metadata for all chat dyads (both stances)."""
    responses = pl.read_csv(DATA_DIR / "responses.csv")
    messages = pd.read_csv(DATA_DIR / "messages.csv")

    # Get all chat groups with metadata
    chat = (
        responses.filter(col("experiment") == "chat")
        .filter(col("question") == col("matchedIdx"))
        .with_columns(
            pl.when(col("matchedTolerance") <= 1)
            .then(pl.lit("shared"))
            .otherwise(pl.lit("opposing"))
            .alias("stance")
        )
        .select(
            [
                "groupId",
                "matchedQuestion",
                "matchedDomain",
                "matchedTolerance",
                "stance",
                "pid",
                "predictShared",
            ]
        )
    )

    # Get unique groups (take first pid's info since matchedQuestion etc. is same for both)
    groups = (
        chat.group_by("groupId").agg(
            [
                col("matchedQuestion").first(),
                col("matchedDomain").first(),
                col("matchedTolerance").first(),
                col("stance").first(),
                # Whether either participant expected commonality on focal
                col("predictShared").mean().alias("pct_expect_commonality"),
            ]
        )
    ).to_pandas()

    # Filter to groups that have messages
    msg_groups = set(messages["group_id"].unique())
    groups = groups[groups["groupId"].isin(msg_groups)]

    return groups


def create_batch_requests(sample: Optional[int] = None) -> List[dict]:
    """Create batch requests for all chat dyads."""
    messages = pd.read_csv(DATA_DIR / "messages.csv")
    messages["absolute_timestamp"] = pd.to_datetime(
        messages["absolute_timestamp"], format="mixed"
    )
    groups = get_chat_groups()

    if sample:
        groups = groups.head(sample)

    print(f"Creating batch requests for {len(groups)} chat dyads")

    batch_requests = []
    for _, row in groups.iterrows():
        group_id = row["groupId"]
        conversation = build_full_conversation(messages, group_id)

        if not conversation.strip():
            continue

        prompt = create_annotation_prompt(
            conversation=conversation,
            focal_question=row["matchedQuestion"],
            focal_domain=row["matchedDomain"],
        )

        gen_config = {
            "temperature": MODEL_CONFIG["temperature"],
            "maxOutputTokens": MODEL_CONFIG["max_tokens"],
            "responseMimeType": "application/json",
        }
        if "thinking_level" in MODEL_CONFIG:
            gen_config["thinkingConfig"] = {
                "thinkingLevel": MODEL_CONFIG["thinking_level"],
            }

        request = {
            "custom_id": f"cg_{group_id}",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen_config,
            },
        }
        batch_requests.append(request)

    return batch_requests


def create_timecourse_batch_requests(
    sample: Optional[int] = None,
    bin_seconds: int = DEFAULT_BIN_SECONDS,
) -> List[dict]:
    """Create batch requests for common ground timecourse (all dyads × time bins).

    Args:
        sample: Only process first N groups
        bin_seconds: Size of each time bin in seconds
    """
    max_bins = MAX_CONVERSATION_SECONDS // bin_seconds
    messages = pd.read_csv(DATA_DIR / "messages.csv")
    messages["absolute_timestamp"] = pd.to_datetime(
        messages["absolute_timestamp"], format="mixed"
    )
    messages["start_time"] = pd.to_datetime(messages["start_time"], format="mixed")
    groups = get_chat_groups()

    if sample:
        groups = groups.head(sample)

    print(
        f"Creating timecourse batch: {len(groups)} dyads × {max_bins} bins "
        f"({bin_seconds}s each)"
    )

    batch_requests = []
    for _, row in groups.iterrows():
        group_id = row["groupId"]
        group_msgs = messages[messages["group_id"] == group_id]

        if group_msgs.empty:
            continue

        start_time = group_msgs["start_time"].iloc[0]

        for t in range(max_bins):
            time_seconds = (t + 1) * bin_seconds
            bin_end = start_time + pd.Timedelta(seconds=time_seconds)
            conversation = build_conversation_up_to(messages, group_id, bin_end)

            # Skip empty bins (no messages yet at this time point)
            if not conversation.strip():
                continue

            prompt = create_timecourse_prompt(
                conversation=conversation,
                focal_question=row["matchedQuestion"],
                focal_domain=row["matchedDomain"],
                time_seconds=time_seconds,
            )

            gen_config = {
                "temperature": MODEL_CONFIG["temperature"],
                "maxOutputTokens": MODEL_CONFIG["max_tokens"],
                "responseMimeType": "application/json",
            }
            if "thinking_level" in MODEL_CONFIG:
                gen_config["thinkingConfig"] = {
                    "thinkingLevel": MODEL_CONFIG["thinking_level"],
                }

            request = {
                "custom_id": f"cgtc_{group_id}_t{t}",
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": gen_config,
                },
            }
            batch_requests.append(request)

    return batch_requests


# ── Per-bin annotation (hybrid: cumulative context, independent judgment) ──


def _build_bin_messages(
    messages_df: pd.DataFrame,
    group_id: str,
    bin_start: pd.Timestamp,
    bin_end: pd.Timestamp,
) -> tuple[str, int]:
    """Build conversation transcript for messages within a specific time window.

    Returns:
        Tuple of (formatted transcript, message count)
    """
    mask = (
        (messages_df["group_id"] == group_id)
        & (messages_df["absolute_timestamp"] >= bin_start)
        & (messages_df["absolute_timestamp"] < bin_end)
    )
    bin_msgs = messages_df[mask].sort_values("absolute_timestamp")
    n_messages = len(bin_msgs)

    lines = [
        f"{'Cat' if r['author'] == '🐱' else 'Dog'}: {r['message_string']}"
        for _, r in bin_msgs.iterrows()
    ]
    return "\n".join(lines), n_messages


def _build_prior_context(
    messages_df: pd.DataFrame,
    group_id: str,
    up_to: pd.Timestamp,
) -> str:
    """Build conversation transcript for all messages BEFORE a given timestamp.

    Returns:
        Formatted transcript of prior messages (empty string if none).
    """
    mask = (
        (messages_df["group_id"] == group_id)
        & (messages_df["absolute_timestamp"] < up_to)
    )
    prior_msgs = messages_df[mask].sort_values("absolute_timestamp")
    if prior_msgs.empty:
        return ""
    lines = [
        f"{'Cat' if r['author'] == '🐱' else 'Dog'}: {r['message_string']}"
        for _, r in prior_msgs.iterrows()
    ]
    return "\n".join(lines)


def create_perbin_timecourse_prompt(
    current_bin_conversation: str,
    prior_context: str,
    focal_question: str,
    focal_domain: str,
    bin_index: int,
    bin_seconds: int,
    n_messages: int,
) -> str:
    """Create prompt for annotating shared inner states in a single time window.

    Hybrid approach: the LLM sees the full conversation history as CONTEXT
    (so it knows what stances were established, what's been discussed), but
    classifies ONLY what is happening in the current time window.

    Args:
        current_bin_conversation: Messages within the current bin only
        prior_context: All messages before the current bin (for context)
        focal_question: The question the dyad was matched on
        focal_domain: Domain of the focal question
        bin_index: 0-indexed bin number
        bin_seconds: Size of each bin in seconds
        n_messages: Number of messages in the current bin
    """
    time_start = bin_index * bin_seconds
    time_end = (bin_index + 1) * bin_seconds

    prior_section = ""
    if prior_context:
        prior_section = f"""
=== PRIOR CONVERSATION (0s – {time_start}s) ===
This is what was said BEFORE the current window. Use it to understand context \
(e.g., whether stances have been established), but do NOT classify it.
{prior_context}
"""

    return f"""You are analyzing a conversation between two people (Cat and Dog) who were \
paired for a study where they chat about a specific topic. Your task is to classify what \
is happening in ONE specific time window of this conversation.

=== STUDY CONTEXT ===
Focal question they discussed: "{focal_question}"
Topic domain: {focal_domain}
Time window to classify: {time_start}s – {time_end}s (bin {bin_index + 1}, {n_messages} messages)
{prior_section}
=== CURRENT WINDOW ({time_start}s – {time_end}s) — CLASSIFY THIS ===
{current_bin_conversation}

=== ANNOTATION TASK ===
We are studying feelings of "shared reality" — the subjective experience of being on the \
same page as another person. Shared reality occurs when both participants feel they are in \
agreement, share commonality, or see things the same way regarding their inner states \
(thoughts, feelings, or beliefs) about the world (Rossignac-Milon et al., 2021).

Classify what is happening IN THE CURRENT WINDOW ONLY ({time_start}s – {time_end}s). \
Use the prior conversation for context but do not classify it.

PHASE 1 — STANCE ESTABLISHMENT:
Based on the full conversation (prior context + current window), have both participants \
established their stances on the focal topic by the end of this window? And specifically, \
does the current window contain the moment where stances become established?

PHASE 2 — FEELINGS OF SHARED REALITY (in the current window only):
In this time window, are the participants experiencing feelings of shared reality — a sense \
that they are on the same page, in agreement, or share commonality about something? \
Look for moments where both participants recognize or express that they think, feel, or \
believe something in a similar way. This is NOT merely being polite or getting along — it \
requires a genuine sense of agreement or commonality about the world.

Note: feelings of shared reality can fluctuate. People may find agreement in one moment \
and disagree in the next. Classify what is happening RIGHT NOW in this window.

TOPIC SCOPE — What is the topic of the shared reality?
  a) same_values: On the focal topic — they feel on the same page about the question \
they were asked to discuss
  b) related_subtopic: On a closely related subtopic within the same domain
  c) different_topic: On a different topic that came up naturally
  d) rapport_only: Only general rapport or politeness without a substantive feeling of \
being on the same page about the world
  e) none: No feelings of shared reality in this window

INNER STATE DIMENSION — What kind of inner state do they feel aligned on?
  a) shared_thoughts: They feel they think about something in the same way — similar \
perspectives, judgments, or ways of seeing things
  b) shared_feelings: They feel they have the same emotional reactions — similar \
emotions, concerns, or affective responses
  c) shared_beliefs: They feel they hold similar beliefs — similar convictions, values, \
or principles about how the world is or should be
  d) none: No substantive shared inner states (when post_stance_topic_scope is "rapport_only" or "none")

CONVERSATIONAL GOAL — What are the participants primarily trying to do in this window?
  a) stance_sharing: Establishing or sharing their positions/perspectives on the topic
  b) perspective_seeking: Asking questions, trying to understand the other's viewpoint
  c) finding_commonality: Actively looking for agreement, shared ground, or common experiences
  d) persuading: Arguing, advocating, trying to change the other's mind
  e) sharing_experience: Relating personal stories or backgrounds that explain their views
  f) rapport_building: Small talk, humor, pleasantries, off-topic bonding
  g) exploring_nuance: Discussing subtleties, qualifying positions, acknowledging complexity
  h) other: A goal that does not fit the above categories (describe in post_stance_conversational_goal_reasoning)

CONSISTENCY RULES:
- If post_stance_shared_reality is false → post_stance_topic_scope MUST be "none", \
post_stance_inner_state_dimension MUST be "none"
- If post_stance_topic_scope is "rapport_only" or "none" → post_stance_inner_state_dimension MUST be "none"
- If post_stance_topic_scope is "same_values", "related_subtopic", or "different_topic" → \
post_stance_inner_state_dimension MUST be one of: "shared_thoughts", "shared_feelings", "shared_beliefs"

Return JSON with this exact structure:
{{
  "focal_stance_revealed": true/false (have BOTH stances been established by end of this window?),
  "stance_in_this_window": true/false (does THIS window contain the stance establishment moment?),
  "initial_alignment": "agreement" | "disagreement" | "unclear" | "not_discussed",
  "post_stance_shared_reality": true/false,
  "post_stance_topic_scope": "same_values" | "related_subtopic" | "different_topic" | "rapport_only" | "none",
  "post_stance_topic_scope_reasoning": "explain your categorization — why this topic scope and not another? cite specific messages from this window",
  "post_stance_shared_reality_description": "what shared inner states are emerging in this window, if any",
  "post_stance_inner_state_dimension": "shared_thoughts" | "shared_feelings" | "shared_beliefs" | "none",
  "post_stance_inner_state_reasoning": "explain your categorization — why this inner state dimension and not another?",
  "post_stance_differences": true/false,
  "post_stance_differences_description": "any differences or disagreements in this window",
  "post_stance_conversational_goal": "stance_sharing" | "perspective_seeking" | "finding_commonality" | "persuading" | "sharing_experience" | "rapport_building" | "exploring_nuance" | "other",
  "post_stance_conversational_goal_reasoning": "explain your categorization — why this goal and not another? cite specific messages"
}}"""


def create_perbin_timecourse_batch_requests(
    sample: Optional[int] = None,
    bin_seconds: int = DEFAULT_BIN_SECONDS,
    n_runs: int = 1,
) -> List[dict]:
    """Create batch requests for per-bin annotation (hybrid: cumulative context, independent judgment).

    Each request includes the full prior conversation as context, but asks the
    LLM to classify only the current time bin. The LLM determines stance
    establishment itself from the conversation history.

    Args:
        sample: Only process first N groups
        bin_seconds: Size of each time bin in seconds
        n_runs: Number of duplicate runs per request (for reliability analysis).
            When > 1, each request is duplicated with custom_ids like
            {group_id}_t{bin}_r{run} (run = 0..n_runs-1).
    """
    max_bins = MAX_CONVERSATION_SECONDS // bin_seconds
    messages = pd.read_csv(DATA_DIR / "messages.csv")
    messages["absolute_timestamp"] = pd.to_datetime(
        messages["absolute_timestamp"], format="mixed"
    )
    messages["start_time"] = pd.to_datetime(messages["start_time"], format="mixed")
    groups = get_chat_groups()

    if sample:
        groups = groups.head(sample)

    print(
        f"Creating per-bin batch: {len(groups)} dyads × up to {max_bins} bins "
        f"({bin_seconds}s each)"
    )

    batch_requests = []
    skipped_empty = 0

    for _, row in groups.iterrows():
        group_id = row["groupId"]
        group_msgs = messages[messages["group_id"] == group_id]

        if group_msgs.empty:
            continue

        start_time = group_msgs["start_time"].iloc[0]

        for t in range(max_bins):
            bin_start = start_time + pd.Timedelta(seconds=t * bin_seconds)
            bin_end = start_time + pd.Timedelta(seconds=(t + 1) * bin_seconds)

            current_conv, n_messages = _build_bin_messages(
                messages, group_id, bin_start, bin_end
            )

            if n_messages == 0:
                skipped_empty += 1
                continue

            # Build prior conversation context (everything before this bin)
            prior_context = _build_prior_context(
                messages, group_id, bin_start
            )

            prompt = create_perbin_timecourse_prompt(
                current_bin_conversation=current_conv,
                prior_context=prior_context,
                focal_question=row["matchedQuestion"],
                focal_domain=row["matchedDomain"],
                bin_index=t,
                bin_seconds=bin_seconds,
                n_messages=n_messages,
            )

            gen_config = {
                "temperature": MODEL_CONFIG["temperature"],
                "maxOutputTokens": MODEL_CONFIG["max_tokens"],
                "responseMimeType": "application/json",
            }
            if "thinking_level" in MODEL_CONFIG:
                gen_config["thinkingConfig"] = {
                    "thinkingLevel": MODEL_CONFIG["thinking_level"],
                }

            for run in range(n_runs):
                custom_id = f"{group_id}_t{t}"
                if n_runs > 1:
                    custom_id += f"_r{run}"

                request = {
                    "custom_id": custom_id,
                    "request": {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": gen_config,
                    },
                }
                batch_requests.append(request)

    print(f"  {len(batch_requests)} requests created, {skipped_empty} empty bins skipped")
    if n_runs > 1:
        print(f"  ({n_runs} runs per request)")
    return batch_requests


def download_and_parse_perbin_timecourse(
    bin_seconds: int = DEFAULT_BIN_SECONDS,
) -> pd.DataFrame:
    """Download and parse per-bin CG timecourse results into a DataFrame."""
    experiment_name = perbin_tc_experiment_name(bin_seconds)
    _, records = _download_raw(experiment_name)

    # Extract group_id, time_bin, and optional run from custom_id
    # Formats: {group_id}_t{bin} or {group_id}_t{bin}_r{run}
    import re

    for rec in records:
        custom_id = rec.pop("custom_id")
        # Try multi-run format first: {group_id}_t{bin}_r{run}
        m = re.match(r"^(.+)_t(\d+)_r(\d+)$", custom_id)
        if m:
            rec["group_id"] = m.group(1)
            rec["time_bin"] = int(m.group(2))
            rec["run"] = int(m.group(3))
        else:
            # Single-run format: {group_id}_t{bin}
            parts = custom_id.rsplit("_t", 1)
            rec["group_id"] = parts[0]
            rec["time_bin"] = int(parts[1])
            rec["run"] = 0
        rec["time_seconds"] = (rec["time_bin"] + 1) * bin_seconds

    df = pd.DataFrame(records)
    has_multi_run = df["run"].nunique() > 1

    # Join with group metadata
    groups = get_chat_groups()
    df = df.merge(
        groups.rename(columns={"groupId": "group_id"}),
        on="group_id",
        how="left",
    )

    output_file = RESULTS_DIR / f"{experiment_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total annotations: {len(df)}")
    print(f"Unique dyads: {df['group_id'].nunique()}")
    print(f"Time bins: {sorted(df['time_bin'].unique())}")
    if has_multi_run:
        print(f"Runs: {sorted(df['run'].unique())}")

    if "post_stance_topic_scope" in df.columns:
        print(f"\n=== TOPIC SCOPE DISTRIBUTION (per-bin) ===")
        print(df["post_stance_topic_scope"].value_counts().to_string())

    if "post_stance_topic_scope" in df.columns and "stance" in df.columns:
        print(f"\n=== TOPIC SCOPE OVER TIME BY STANCE ===")
        ct = pd.crosstab(
            [df["time_seconds"], df["stance"]],
            df["post_stance_topic_scope"],
            normalize="index",
        )
        print(ct.round(2).to_string())

    # Multi-run reliability statistics and aggregation
    if has_multi_run:
        n_runs = df["run"].nunique()
        group_keys = ["group_id", "time_bin"]

        # Axes to compute reliability on
        reliability_axes = [
            col for col in [
                "post_stance_shared_reality",
                "post_stance_topic_scope",
                "post_stance_inner_state_dimension",
                "post_stance_conversational_goal",
            ]
            if col in df.columns
        ]

        if reliability_axes:
            print(f"\n=== MULTI-RUN RELIABILITY ({n_runs} runs) ===")

            for axis in reliability_axes:
                # Modal label per item
                axis_mode = (
                    df.groupby(group_keys)[axis]
                    .agg(lambda x: x.mode().iloc[0])
                    .reset_index()
                    .rename(columns={axis: f"{axis}_majority"})
                )
                df_with_mode = df.merge(axis_mode, on=group_keys)
                df_with_mode["_agrees"] = (
                    df_with_mode[axis] == df_with_mode[f"{axis}_majority"]
                )

                # Per-item agreement count
                item_agree = (
                    df_with_mode.groupby(group_keys)["_agrees"]
                    .sum()
                    .astype(int)
                )
                n_items = len(item_agree)
                n_unanimous = (item_agree == n_runs).sum()
                n_majority = (item_agree >= 2).sum()  # at least 2/3
                n_no_consensus = n_items - n_majority

                overall_rate = df_with_mode["_agrees"].mean()
                print(f"\n  {axis}:")
                print(f"    Overall agreement rate: {overall_rate:.1%}")
                print(f"    Unanimous ({n_runs}/{n_runs}): "
                      f"{n_unanimous}/{n_items} ({n_unanimous/n_items:.0%})")
                if n_runs >= 3:
                    print(f"    Majority (≥2/{n_runs}): "
                          f"{n_majority}/{n_items} ({n_majority/n_items:.0%})")
                    print(f"    No consensus: "
                          f"{n_no_consensus}/{n_items} ({n_no_consensus/n_items:.0%})")

            # Build aggregated dataframe with modal vote + confidence
            agg_df = df.groupby(group_keys).first().reset_index()
            # Keep only metadata columns, drop per-run annotation columns
            meta_cols = [
                c for c in agg_df.columns
                if c not in reliability_axes
                and c not in ["run", "post_stance_topic_scope_reasoning",
                              "post_stance_shared_reality_description",
                              "post_stance_inner_state_reasoning",
                              "post_stance_differences_description",
                              "post_stance_conversational_goal_reasoning",
                              "initial_alignment_reasoning",
                              "initial_alignment_evidence",
                              "last_stance_message", "key_exchange",
                              "surprise_reasoning", "notes",
                              "n_post_stance_messages"]
            ]
            agg_df = agg_df[meta_cols].copy()

            for axis in reliability_axes:
                # Modal label
                mode_df = (
                    df.groupby(group_keys)[axis]
                    .agg(lambda x: x.mode().iloc[0])
                    .reset_index()
                    .rename(columns={axis: axis})
                )
                agg_df = agg_df.merge(mode_df, on=group_keys)

                # Agreement count → confidence
                agree_count = (
                    df.merge(mode_df, on=group_keys, suffixes=("", "_mode"))
                    .assign(_match=lambda d: d[axis] == d[f"{axis}_mode"])
                    .groupby(group_keys)["_match"]
                    .sum()
                    .astype(int)
                    .reset_index()
                    .rename(columns={"_match": f"{axis}_agree_count"})
                )
                agg_df = agg_df.merge(agree_count, on=group_keys)

                # Confidence label
                agg_df[f"{axis}_confidence"] = agg_df[f"{axis}_agree_count"].map(
                    lambda x: (
                        "unanimous" if x == n_runs
                        else "majority" if x >= 2
                        else "no_consensus"
                    )
                )

            agg_file = RESULTS_DIR / f"{experiment_name}_aggregated.csv"
            agg_df.to_csv(agg_file, index=False)
            print(f"\nSaved aggregated results to {agg_file}")
            print(f"  {len(agg_df)} items (modal vote + confidence)")

    return df


def submit_batch(batch_requests: List[dict], experiment_name: str = EXPERIMENT_NAME) -> str:
    """Submit batch to Vertex AI via REST API (supports global endpoint)."""
    import google.auth
    import google.auth.transport.requests

    batch_file = BATCH_DIR / f"{experiment_name}.jsonl"
    with open(batch_file, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    print(f"Created {batch_file} ({len(batch_requests)} requests)")

    gcs_input = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/input/{experiment_name}.jsonl"
    gcs_output = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/output/{experiment_name}/"

    subprocess.run(["gsutil", "cp", str(batch_file), gcs_input], check=True)
    print(f"Uploaded to {gcs_input}")

    # Use REST API directly to support global endpoint
    credentials, _ = google.auth.default()
    credentials.refresh(google.auth.transport.requests.Request())

    import requests

    url = (
        f"https://aiplatform.googleapis.com/v1/"
        f"projects/{GCS_PROJECT}/locations/global/"
        f"batchPredictionJobs"
    )
    body = {
        "displayName": f"{experiment_name}_batch",
        "model": f"publishers/google/models/{MODEL_CONFIG['model']}",
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [gcs_input]},
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": gcs_output},
        },
    }

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        json=body,
    )
    resp.raise_for_status()
    job = resp.json()

    job_name = job["name"]
    state = job.get("state", "UNKNOWN")
    print(f"Job: {job_name}")
    print(f"State: {state}")

    job_info = {
        experiment_name: {
            "resource_name": job_name,
            "input_uri": gcs_input,
            "output_uri": gcs_output,
        }
    }
    ids_file = BATCH_DIR / f"{experiment_name}_batch_ids.json"
    with open(ids_file, "w") as f:
        json.dump(job_info, f, indent=2)

    return job_name


def _download_raw(experiment_name: str) -> tuple[Path, List[dict]]:
    """Download and parse raw JSONL results for an experiment.

    Returns:
        Tuple of (raw_file_path, list_of_parsed_records)
    """
    ids_file = BATCH_DIR / f"{experiment_name}_batch_ids.json"
    if not ids_file.exists():
        raise FileNotFoundError(f"No batch job found for {experiment_name}. Run submit first.")

    with open(ids_file) as f:
        batch_ids = json.load(f)

    info = batch_ids[experiment_name]
    output_uri = info["output_uri"]

    result = subprocess.run(
        ["gsutil", "ls", f"{output_uri}**predictions.jsonl"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FileNotFoundError("No predictions file found. Job may still be running.")

    predictions_uri = result.stdout.strip().split("\n")[-1]  # latest batch
    print(f"Downloading {predictions_uri}...")

    raw_file = RESULTS_DIR / f"{experiment_name}_raw.jsonl"
    subprocess.run(["gsutil", "cp", predictions_uri, str(raw_file)], check=True)

    # Parse results
    records = []
    errors = 0
    with open(raw_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                custom_id = d["custom_id"]

                response = d.get("response", {})
                candidates = response.get("candidates", [])
                if not candidates:
                    errors += 1
                    continue

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])

                annotation = None
                for part in parts:
                    if "text" in part:
                        try:
                            annotation = json.loads(part["text"])
                            break
                        except json.JSONDecodeError:
                            continue

                if annotation is None:
                    errors += 1
                    continue

                annotation["custom_id"] = custom_id
                records.append(annotation)

            except Exception:
                errors += 1

    print(f"Parsed {len(records)} results, {errors} errors")

    if not records:
        raise RuntimeError(
            f"All {errors} responses failed to parse. "
            "Check the raw file for error messages."
        )

    return raw_file, records


def download_and_parse() -> pd.DataFrame:
    """Download results and parse into a DataFrame (full-conversation annotation)."""
    _, records = _download_raw(EXPERIMENT_NAME)

    # Extract group_id from custom_id
    for rec in records:
        rec["group_id"] = rec.pop("custom_id").replace("cg_", "")

    df = pd.DataFrame(records)

    # Join with group metadata
    groups = get_chat_groups()
    df = df.merge(
        groups.rename(columns={"groupId": "group_id"}),
        on="group_id",
        how="left",
    )

    output_file = RESULTS_DIR / f"{EXPERIMENT_NAME}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total dyads annotated: {len(df)}")
    if "initial_alignment" in df.columns:
        print(f"\nInitial alignment (LLM-perceived):")
        print(df["initial_alignment"].value_counts().to_string())
        if "stance" in df.columns:
            print(f"\nInitial alignment x actual stance:")
            print(pd.crosstab(df["stance"], df["initial_alignment"], margins=True))
    if "post_stance_shared_reality" in df.columns:
        n_sr = df["post_stance_shared_reality"].sum()
        print(
            f"\nPost-stance shared reality: {n_sr} ({df['post_stance_shared_reality'].mean():.0%})"
        )
    if "topic_scope" in df.columns:
        print(f"\nTopic scope:")
        print(df["topic_scope"].value_counts().to_string())

    return df


def download_and_parse_timecourse(bin_seconds: int = DEFAULT_BIN_SECONDS) -> pd.DataFrame:
    """Download and parse timecourse results into a DataFrame."""
    experiment_name = tc_experiment_name(bin_seconds)
    _, records = _download_raw(experiment_name)

    # Extract group_id and time_bin from custom_id (format: cgtc_{group_id}_t{bin})
    for rec in records:
        custom_id = rec.pop("custom_id")
        # Split from the right to handle group_ids that contain underscores
        parts = custom_id.rsplit("_t", 1)
        rec["group_id"] = parts[0].replace("cgtc_", "")
        rec["time_bin"] = int(parts[1])
        rec["time_seconds"] = (rec["time_bin"] + 1) * bin_seconds

    df = pd.DataFrame(records)

    # Join with group metadata
    groups = get_chat_groups()
    df = df.merge(
        groups.rename(columns={"groupId": "group_id"}),
        on="group_id",
        how="left",
    )

    output_file = RESULTS_DIR / f"{experiment_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total annotations: {len(df)}")
    print(f"Unique dyads: {df['group_id'].nunique()}")
    print(f"Time bins: {sorted(df['time_bin'].unique())}")

    # Shared reality emergence over time by stance
    if "post_stance_shared_reality" in df.columns and "stance" in df.columns:
        print(f"\n=== SHARED REALITY OVER TIME BY STANCE ===")
        sr_by_time = (
            df.groupby(["time_seconds", "stance"])["post_stance_shared_reality"]
            .agg(["mean", "count"])
            .reset_index()
        )
        for stance in ["opposing", "shared"]:
            sub = sr_by_time[sr_by_time["stance"] == stance]
            print(f"\n{stance}:")
            for _, row in sub.iterrows():
                print(
                    f"  t={row['time_seconds']:3.0f}s: "
                    f"{row['mean']:.0%} SR (N={row['count']:.0f})"
                )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Annotate chats for shared reality (commonality of inner states)"
    )
    parser.add_argument(
        "mode",
        choices=["full", "timecourse", "perbin-timecourse"],
        nargs="?",
        default="full",
        help="Annotation mode: 'full' (default), 'timecourse' (cumulative), or 'perbin-timecourse' (independent bins)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Create batch file only (no submission)")
    parser.add_argument(
        "--test", action="store_true",
        help="Send a small sample directly to Gemini API (not batch) and print results",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download and parse results"
    )
    parser.add_argument("--sample", type=int, help="Only process first N groups")
    parser.add_argument(
        "--bin-size", type=int, default=DEFAULT_BIN_SECONDS,
        help=f"Time bin size in seconds for timecourse mode (default: {DEFAULT_BIN_SECONDS})",
    )
    parser.add_argument(
        "--n-runs", type=int, default=1,
        help="Number of duplicate runs per request for reliability analysis (default: 1)",
    )
    args = parser.parse_args()

    if args.mode == "timecourse":
        bin_seconds = args.bin_size
        experiment_name = tc_experiment_name(bin_seconds)

        if args.download:
            download_and_parse_timecourse(bin_seconds=bin_seconds)
            return

        batch_requests = create_timecourse_batch_requests(
            sample=args.sample, bin_seconds=bin_seconds,
        )
    elif args.mode == "perbin-timecourse":
        bin_seconds = args.bin_size
        experiment_name = perbin_tc_experiment_name(bin_seconds)

        if args.download:
            download_and_parse_perbin_timecourse(bin_seconds=bin_seconds)
            return

        batch_requests = create_perbin_timecourse_batch_requests(
            sample=args.sample, bin_seconds=bin_seconds, n_runs=args.n_runs,
        )
    else:
        if args.download:
            download_and_parse()
            return

        batch_requests = create_batch_requests(sample=args.sample)
        experiment_name = EXPERIMENT_NAME

    print(f"Created {len(batch_requests)} batch requests")

    if args.test:
        # Send a small sample directly to Gemini API via REST for inspection
        import google.auth
        import google.auth.transport.requests
        import requests

        credentials, _ = google.auth.default()
        credentials.refresh(google.auth.transport.requests.Request())

        n_test = min(len(batch_requests), args.sample or 3)
        print(f"\n[TEST] Sending {n_test} requests directly to Gemini API...\n")

        url = (
            f"https://aiplatform.googleapis.com/v1/"
            f"projects/{GCS_PROJECT}/locations/global/"
            f"publishers/google/models/{MODEL_CONFIG['model']}:generateContent"
        )

        for i, req in enumerate(batch_requests[:n_test]):
            custom_id = req["custom_id"]
            prompt = req["request"]["contents"][0]["parts"][0]["text"]

            print(f"{'=' * 60}")
            print(f"Request {i + 1}/{n_test}: {custom_id}")
            print(f"{'=' * 60}")
            print(f"PROMPT (first 500 chars):\n{prompt[:500]}...\n")

            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": req["request"]["generationConfig"],
            }

            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {credentials.token}",
                    "Content-Type": "application/json",
                },
                json=body,
            )

            if resp.status_code != 200:
                print(f"ERROR {resp.status_code}: {resp.text[:500]}")
                continue

            result = resp.json()
            candidates = result.get("candidates", [])
            if not candidates:
                print("No candidates in response")
                continue

            text = candidates[0].get("content", {}).get("parts", [{}])
            # Find the text part (skip thinking parts)
            response_text = None
            for part in text:
                if "text" in part:
                    response_text = part["text"]

            if response_text:
                print("RESPONSE:")
                try:
                    parsed = json.loads(response_text)
                    print(json.dumps(parsed, indent=2))
                except (json.JSONDecodeError, ValueError):
                    print(response_text)
            else:
                print("No text in response")
            print()

        return

    if args.dry_run:
        batch_file = BATCH_DIR / f"{experiment_name}.jsonl"
        with open(batch_file, "w") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")
        print(f"[DRY RUN] Saved to {batch_file}")

        # Print a sample prompt
        if batch_requests:
            sample_prompt = batch_requests[0]["request"]["contents"][0]["parts"][0][
                "text"
            ]
            print(f"\n=== SAMPLE PROMPT ===\n{sample_prompt[:2000]}...")
        return

    submit_batch(batch_requests, experiment_name=experiment_name)


if __name__ == "__main__":
    main()
