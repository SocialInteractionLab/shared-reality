#!/usr/bin/env python3
"""
Annotate opposing-stance chat transcripts for common ground discovery.

For each opposing-stance dyad, sends the full conversation transcript to Gemini
and asks it to classify whether participants established disagreement and then
discovered common ground despite that disagreement.

Usage:
    # Dry run: create batch file only
    uv run -m models.llm.annotate_common_ground --dry-run

    # Submit batch to Vertex AI
    uv run -m models.llm.annotate_common_ground

    # Download and parse results
    uv run -m models.llm.annotate_common_ground --download

    # Process a small sample first
    uv run -m models.llm.annotate_common_ground --dry-run --sample 5
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
EXPERIMENT_NAME = "common_ground"
TC_EXPERIMENT_NAME = "common_ground_timecourse"

BIN_SECONDS = 15
MAX_BINS = 13


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
    """Create prompt for annotating common ground in a conversation.

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

PHASE 2 — POST-STANCE CONVERSATION:
Now focus ONLY on the messages that come AFTER the initial stance establishment identified
above. Ignore the stance-establishment messages themselves. We only care about what
happens next.

2a. COMMON GROUND: In the post-stance portion of the conversation, did the participants
    discover shared values, beliefs, experiences, or perspectives? Classify the type:
    a) Same underlying values
    b) Common ground on a closely related subtopic within the same domain
    c) Common ground on a different topic that came up naturally
    d) Only general rapport/politeness without substantive common ground
    e) No common ground found in the post-stance conversation

2b. DIFFERENCES: In the post-stance portion, did the participants discover additional
    differences or disagreements beyond their initial positions?

2c. KEY EXCHANGE: Quote the most important exchange from the post-stance conversation
    that illustrates common ground or difference discovery.

2d. SURPRISE: Given the initial positions established in Phase 1, would an outside observer
    have predicted they would find this common ground? (i.e., is the common ground surprising
    given their stated positions on the focal topic?)

For each classification, provide your reasoning explaining WHY you chose that category,
citing specific evidence from the conversation.

IMPORTANT CONSISTENCY RULE: If post_stance_common_ground is false, then common_ground_type
MUST be "none". If common_ground_type is anything other than "none", then
post_stance_common_ground MUST be true.

Return JSON with this exact structure:
{{
  "focal_stance_revealed": true/false,
  "initial_alignment": "agreement" | "disagreement" | "unclear" | "not_discussed",
  "initial_alignment_reasoning": "explain why you classified the initial alignment this way, citing specific messages",
  "initial_alignment_evidence": "brief quote showing how they established their stances",
  "last_stance_message": "quote of the last message in the stance-establishment phase, or null if stances never established",
  "n_post_stance_messages": <integer count of messages after the last stance message>,
  "post_stance_common_ground": true/false,
  "common_ground_type": "same_values" | "related_subtopic" | "different_topic" | "rapport_only" | "none",
  "common_ground_reasoning": "explain why you chose this common ground type over alternatives, citing specific post-stance messages",
  "common_ground_description": "what common ground was found in post-stance conversation",
  "post_stance_differences": true/false,
  "post_stance_differences_description": "what additional differences emerged after stance establishment",
  "key_exchange": "quote of key exchange from post-stance conversation",
  "surprising": true/false,
  "surprise_reasoning": "explain why this common ground was or was not predictable from their initial stated positions",
  "notes": "any additional observations"
}}"""


def create_timecourse_prompt(
    conversation: str, focal_question: str, focal_domain: str, time_seconds: int
) -> str:
    """Create prompt for annotating common ground in a PARTIAL conversation.

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

PHASE 2 — POST-STANCE CONVERSATION:
If stances have been established, focus ONLY on messages AFTER the stance establishment.
If stances have NOT been established yet, skip Phase 2 and set all Phase 2 fields to
their "not yet" defaults (post_stance_common_ground: false, common_ground_type: "none", etc.).

2a. COMMON GROUND: In the post-stance portion so far, did the participants discover shared
    values, beliefs, experiences, or perspectives? Classify the type:
    a) Same underlying values
    b) Common ground on a closely related subtopic within the same domain
    c) Common ground on a different topic that came up naturally
    d) Only general rapport/politeness without substantive common ground
    e) No common ground found in the post-stance conversation so far

2b. DIFFERENCES: In the post-stance portion so far, did the participants discover additional
    differences or disagreements beyond their initial positions?

IMPORTANT CONSISTENCY RULE: If post_stance_common_ground is false, then common_ground_type
MUST be "none". If common_ground_type is anything other than "none", then
post_stance_common_ground MUST be true.

Return JSON with this exact structure:
{{
  "focal_stance_revealed": true/false,
  "initial_alignment": "agreement" | "disagreement" | "unclear" | "not_discussed",
  "initial_alignment_reasoning": "explain why, citing specific messages",
  "last_stance_message": "quote or null if stances not yet established",
  "n_post_stance_messages": <integer count of messages after the last stance message, 0 if stances not yet established>,
  "post_stance_common_ground": true/false,
  "common_ground_type": "same_values" | "related_subtopic" | "different_topic" | "rapport_only" | "none",
  "common_ground_reasoning": "explain why you chose this type, citing specific post-stance messages",
  "common_ground_description": "what common ground was found so far",
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
    bin_seconds: int = BIN_SECONDS,
    max_bins: int = MAX_BINS,
) -> List[dict]:
    """Create batch requests for common ground timecourse (all dyads × time bins).

    Args:
        sample: Only process first N groups
        bin_seconds: Size of each time bin in seconds
        max_bins: Maximum number of time bins
    """
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
    if "post_stance_common_ground" in df.columns:
        n_cg = df["post_stance_common_ground"].sum()
        print(
            f"\nPost-stance common ground: {n_cg} ({df['post_stance_common_ground'].mean():.0%})"
        )
    if "common_ground_type" in df.columns:
        print(f"\nCommon ground types:")
        print(df["common_ground_type"].value_counts().to_string())

    return df


def download_and_parse_timecourse() -> pd.DataFrame:
    """Download and parse timecourse results into a DataFrame."""
    _, records = _download_raw(TC_EXPERIMENT_NAME)

    # Extract group_id and time_bin from custom_id (format: cgtc_{group_id}_t{bin})
    for rec in records:
        custom_id = rec.pop("custom_id")
        # Split from the right to handle group_ids that contain underscores
        parts = custom_id.rsplit("_t", 1)
        rec["group_id"] = parts[0].replace("cgtc_", "")
        rec["time_bin"] = int(parts[1])
        rec["time_seconds"] = (rec["time_bin"] + 1) * BIN_SECONDS

    df = pd.DataFrame(records)

    # Join with group metadata
    groups = get_chat_groups()
    df = df.merge(
        groups.rename(columns={"groupId": "group_id"}),
        on="group_id",
        how="left",
    )

    output_file = RESULTS_DIR / f"{TC_EXPERIMENT_NAME}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total annotations: {len(df)}")
    print(f"Unique dyads: {df['group_id'].nunique()}")
    print(f"Time bins: {sorted(df['time_bin'].unique())}")

    # CG emergence over time by stance
    if "post_stance_common_ground" in df.columns and "stance" in df.columns:
        print(f"\n=== CG OVER TIME BY STANCE ===")
        cg_by_time = (
            df.groupby(["time_seconds", "stance"])["post_stance_common_ground"]
            .agg(["mean", "count"])
            .reset_index()
        )
        for stance in ["opposing", "shared"]:
            sub = cg_by_time[cg_by_time["stance"] == stance]
            print(f"\n{stance}:")
            for _, row in sub.iterrows():
                print(
                    f"  t={row['time_seconds']:3.0f}s: "
                    f"{row['mean']:.0%} CG (N={row['count']:.0f})"
                )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Annotate chats for common ground discovery"
    )
    parser.add_argument(
        "mode",
        choices=["full", "timecourse"],
        nargs="?",
        default="full",
        help="Annotation mode: 'full' (default) or 'timecourse'",
    )
    parser.add_argument("--dry-run", action="store_true", help="Create batch file only")
    parser.add_argument(
        "--download", action="store_true", help="Download and parse results"
    )
    parser.add_argument("--sample", type=int, help="Only process first N groups")
    args = parser.parse_args()

    if args.mode == "timecourse":
        if args.download:
            download_and_parse_timecourse()
            return

        batch_requests = create_timecourse_batch_requests(sample=args.sample)
        experiment_name = TC_EXPERIMENT_NAME
    else:
        if args.download:
            download_and_parse()
            return

        batch_requests = create_batch_requests(sample=args.sample)
        experiment_name = EXPERIMENT_NAME

    print(f"Created {len(batch_requests)} batch requests")

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
