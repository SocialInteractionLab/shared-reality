#!/usr/bin/env python3
"""
Annotate conversational goals over time for each dyad.

For each dyad × time bin, classifies the primary conversational goal of the
exchange within that window (e.g., stance_sharing, persuading, perspective_seeking,
finding_commonality, sharing_experience, rapport_building, exploring_nuance).

Usage:
    # Dry run: create batch file, print sample prompt
    uv run -m models.llm.annotate_conversation_goals --dry-run

    # Submit batch to Vertex AI
    uv run -m models.llm.annotate_conversation_goals

    # Download and parse results
    uv run -m models.llm.annotate_conversation_goals --download

    # Small sample first
    uv run -m models.llm.annotate_conversation_goals --dry-run --sample 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .annotate_common_ground import (
    get_chat_groups,
    submit_batch,
    _download_raw,
    DEFAULT_BIN_SECONDS,
    MAX_CONVERSATION_SECONDS,
)
from .config import (
    BATCH_DIR,
    RESULTS_DIR,
    MODEL_CONFIG,
    DATA_DIR,
)

EXPERIMENT_BASE = "conversation_goals"


def experiment_name(bin_seconds: int) -> str:
    """Experiment name with bin size suffix, e.g. conversation_goals_15s."""
    return f"{EXPERIMENT_BASE}_{bin_seconds}s"


# ── Goal taxonomy ──────────────────────────────────────────────────────────

GOAL_CATEGORIES = {
    "stance_sharing": (
        "Establishing or elaborating their initial positions on the focal topic"
    ),
    "persuading": (
        "Advocating for their position, arguing, trying to change the other's mind"
    ),
    "perspective_seeking": (
        "Asking questions, trying to understand the other's viewpoint or reasoning"
    ),
    "finding_commonality": (
        "Identifying shared values, beliefs, experiences, or points of agreement"
    ),
    "sharing_experience": (
        "Relating personal stories, backgrounds, or contexts that explain their views"
    ),
    "rapport_building": (
        "Small talk, humor, off-topic bonding, pleasantries"
    ),
    "exploring_nuance": (
        "Discussing subtleties, acknowledging complexity, qualifying positions"
    ),
    "other": (
        "A goal that does not fit the above categories (describe in notes)"
    ),
}


def _format_goal_list() -> str:
    """Format the goal categories for inclusion in the prompt."""
    lines = []
    for cat, desc in GOAL_CATEGORIES.items():
        lines.append(f"  - {cat}: {desc}")
    return "\n".join(lines)


# ── Conversation builders ──────────────────────────────────────────────────


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


def _build_prior_messages(
    messages_df: pd.DataFrame,
    group_id: str,
    before_time: pd.Timestamp,
) -> str:
    """Build conversation transcript for all messages before a given time."""
    mask = (
        (messages_df["group_id"] == group_id)
        & (messages_df["absolute_timestamp"] < before_time)
    )
    prior_msgs = messages_df[mask].sort_values("absolute_timestamp")

    if prior_msgs.empty:
        return ""

    lines = [
        f"{'Cat' if r['author'] == '🐱' else 'Dog'}: {r['message_string']}"
        for _, r in prior_msgs.iterrows()
    ]
    return "\n".join(lines)


# ── Prompt ─────────────────────────────────────────────────────────────────


def create_goal_prompt(
    prior_conversation: str,
    current_bin_conversation: str,
    focal_question: str,
    focal_domain: str,
    bin_index: int,
    bin_seconds: int,
    n_messages: int,
) -> str:
    """Create prompt for classifying the conversational goal in a time window.

    Args:
        prior_conversation: Messages before the current bin (context)
        current_bin_conversation: Messages within the current bin (to classify)
        focal_question: The question the dyad was matched on
        focal_domain: Domain of the focal question
        bin_index: 0-indexed bin number
        bin_seconds: Size of each bin in seconds
        n_messages: Number of messages in the current bin
    """
    time_start = bin_index * bin_seconds
    time_end = (bin_index + 1) * bin_seconds

    prior_section = (
        prior_conversation
        if prior_conversation
        else "(This is the start of the conversation.)"
    )

    goal_list = _format_goal_list()

    return f"""You are analyzing a conversation between two people (Cat and Dog) who were paired \
for a study where they chat about a specific topic. You will see the conversation in two parts: \
prior context and the current time window to classify.

=== CONTEXT ===
Focal question they discussed: "{focal_question}"
Topic domain: {focal_domain}
Time window: {time_start}s – {time_end}s (bin {bin_index + 1})

=== PRIOR CONVERSATION (before {time_start}s) ===
{prior_section}

=== CURRENT WINDOW ({time_start}s – {time_end}s, {n_messages} messages) ===
{current_bin_conversation}

=== TASK ===
Classify the PRIMARY conversational goal of the exchange in the CURRENT WINDOW above. \
Consider what the participants are primarily *trying to do* in this segment, taking into \
account the prior conversation for context.

Goal categories:
{goal_list}

Rules:
- Choose the single best-fitting category as primary_goal.
- If a secondary goal is clearly present, include it; otherwise set to null.
- If "other", describe the goal in the notes field.
- Base your classification on the current window messages only, using prior context \
to understand the flow.

Return JSON with this exact structure:
{{
  "primary_goal": "<category>",
  "goal_reasoning": "Brief explanation citing specific messages from the current window",
  "secondary_goal": "<category>" | null,
  "notes": "anything notable, or description if primary_goal is 'other'"
}}"""


# ── Batch creation ─────────────────────────────────────────────────────────


def create_goal_timecourse_batch_requests(
    sample: Optional[int] = None,
    bin_seconds: int = DEFAULT_BIN_SECONDS,
) -> List[dict]:
    """Create batch requests for conversational goal annotation (all dyads × time bins).

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
        f"Creating goal timecourse batch: {len(groups)} dyads × {max_bins} bins "
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

            # Get current bin messages
            current_conv, n_messages = _build_bin_messages(
                messages, group_id, bin_start, bin_end
            )

            # Skip bins with no messages
            if n_messages == 0:
                skipped_empty += 1
                continue

            # Get prior context
            prior_conv = _build_prior_messages(messages, group_id, bin_start)

            prompt = create_goal_prompt(
                prior_conversation=prior_conv,
                current_bin_conversation=current_conv,
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

            request = {
                "custom_id": f"goal_{group_id}_t{t}",
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": gen_config,
                },
            }
            batch_requests.append(request)

    print(f"  {len(batch_requests)} requests created, {skipped_empty} empty bins skipped")
    return batch_requests


# ── Download & parse ───────────────────────────────────────────────────────


def download_and_parse_goals(bin_seconds: int = DEFAULT_BIN_SECONDS) -> pd.DataFrame:
    """Download and parse conversational goal results into a DataFrame."""
    exp_name = experiment_name(bin_seconds)
    _, records = _download_raw(exp_name)

    # Extract group_id and time_bin from custom_id (format: goal_{group_id}_t{bin})
    for rec in records:
        custom_id = rec.pop("custom_id")
        # Split from the right to handle group_ids that contain underscores
        parts = custom_id.rsplit("_t", 1)
        rec["group_id"] = parts[0].replace("goal_", "")
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

    output_file = RESULTS_DIR / f"{exp_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total annotations: {len(df)}")
    print(f"Unique dyads: {df['group_id'].nunique()}")
    print(f"Time bins: {sorted(df['time_bin'].unique())}")

    # Goal distribution
    if "primary_goal" in df.columns:
        print(f"\n=== GOAL DISTRIBUTION ===")
        print(df["primary_goal"].value_counts().to_string())

        # Goals by stance
        if "stance" in df.columns:
            print(f"\n=== GOALS BY STANCE ===")
            ct = pd.crosstab(
                df["primary_goal"], df["stance"], normalize="columns"
            )
            print(ct.round(3).to_string())

        # Goals over time
        print(f"\n=== GOALS OVER TIME ===")
        goal_by_time = (
            df.groupby(["time_seconds", "primary_goal"])
            .size()
            .unstack(fill_value=0)
        )
        # Normalize to proportions per time bin
        goal_by_time_pct = goal_by_time.div(goal_by_time.sum(axis=1), axis=0)
        print(goal_by_time_pct.round(2).to_string())

    return df


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Annotate conversational goals over time"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Create batch file only"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download and parse results"
    )
    parser.add_argument("--sample", type=int, help="Only process first N groups")
    parser.add_argument(
        "--bin-size",
        type=int,
        default=DEFAULT_BIN_SECONDS,
        help=f"Time bin size in seconds (default: {DEFAULT_BIN_SECONDS})",
    )
    args = parser.parse_args()

    bin_seconds = args.bin_size
    exp_name = experiment_name(bin_seconds)

    if args.download:
        download_and_parse_goals(bin_seconds=bin_seconds)
        return

    batch_requests = create_goal_timecourse_batch_requests(
        sample=args.sample, bin_seconds=bin_seconds
    )

    print(f"Created {len(batch_requests)} batch requests")

    if args.dry_run:
        batch_file = BATCH_DIR / f"{exp_name}.jsonl"
        with open(batch_file, "w") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")
        print(f"[DRY RUN] Saved to {batch_file}")

        # Print a sample prompt
        if batch_requests:
            sample_prompt = batch_requests[0]["request"]["contents"][0]["parts"][0][
                "text"
            ]
            print(f"\n=== SAMPLE PROMPT ===\n{sample_prompt[:3000]}...")
        return

    submit_batch(batch_requests, experiment_name=exp_name)


if __name__ == "__main__":
    main()
