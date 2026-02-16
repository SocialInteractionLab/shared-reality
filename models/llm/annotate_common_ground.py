#!/usr/bin/env python3
"""
Annotate opposing-stance chat transcripts for common ground discovery.

For each opposing-stance dyad, sends the full conversation transcript to Gemini
and asks it to classify whether participants established disagreement and then
discovered common ground despite that disagreement.

Usage:
    # Dry run: create batch file only
    python -m models.llm.annotate_common_ground --dry-run

    # Submit batch to Vertex AI
    python -m models.llm.annotate_common_ground

    # Download and parse results
    python -m models.llm.annotate_common_ground --download

    # Process a small sample first
    python -m models.llm.annotate_common_ground --dry-run --sample 5
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

# Import config directly to avoid __init__.py pulling in vertexai
from pathlib import Path as _Path
import os as _os

_PAPER_DIR = _Path(__file__).parent.parent.parent
DATA_DIR = _PAPER_DIR / "data"
BATCH_DIR = _Path(__file__).parent / "batch_requests"
BATCH_DIR.mkdir(exist_ok=True)

MODEL_CONFIG = {
    "model": "gemini-2.5-pro",
    "temperature": 1.0,
    "max_tokens": 65536,
}

RESULTS_DIR = DATA_DIR / "llm_results"
RESULTS_DIR.mkdir(exist_ok=True)

GCS_PROJECT = "709275529646"
GCS_BUCKET = "hs-social-interaction-llm-batches"
GCS_PREFIX = "llm-stance-detection"

EXPERIMENT_NAME = "common_ground"


def build_full_conversation(messages_df: pd.DataFrame, group_id: str) -> str:
    """Build the full conversation transcript for a dyad."""
    group_msgs = messages_df[messages_df["group_id"] == group_id].copy()
    group_msgs = group_msgs.sort_values("absolute_timestamp")
    lines = [
        f"{'Cat' if r['author'] == '🐱' else 'Dog'}: {r['message_string']}"
        for _, r in group_msgs.iterrows()
    ]
    return "\n".join(lines)


def create_annotation_prompt(
    conversation: str,
    focal_question: str,
    focal_domain: str,
    matched_tolerance: float,
    stance: str,
) -> str:
    """Create prompt for annotating common ground in a conversation.

    Args:
        conversation: Full conversation transcript
        focal_question: The question the dyad was matched on
        focal_domain: Domain of the focal question
        matched_tolerance: Absolute difference in their focal responses
        stance: "opposing" or "shared"
    """
    return f"""You are analyzing a conversation between two people (Cat and Dog) who were paired
for a study on social perception. They were asked to discuss a specific topic.

=== CONTEXT ===
Focal question they discussed: "{focal_question}"
Topic domain: {focal_domain}
Their actual responses to this question differed by {matched_tolerance:.0f} point(s) on a 1-5 scale ({stance}).

=== CONVERSATION ===
{conversation}

=== ANNOTATION TASK ===
Analyze this conversation and answer the following questions. Be specific and cite evidence from the conversation.

1. DISAGREEMENT SURFACED: Did the participants explicitly or implicitly establish that they
   have different views on the focal topic? (Not whether they *actually* disagree — we know they
   do — but whether this difference became apparent during the conversation.)

2. COMMON GROUND FOUND: After any disagreement (or despite their different stances), did the
   participants discover shared values, beliefs, experiences, or perspectives? This could be:
   a) Same underlying values despite different surface answer (e.g., both care about the issue
      even if they answer differently)
   b) Common ground on a closely related subtopic within the same domain
   c) Common ground on a different topic that came up naturally
   d) Only general rapport/politeness (e.g., "nice talking to you") without substantive common ground

3. DISCOVERY MOMENT: If common ground was found, which message(s) mark the turning point where
   it was discovered? Quote the key exchange.

4. SURPRISE/UNEXPECTEDNESS: Was the common ground discovery surprising given their initial
   disagreement? (i.e., would an observer have predicted they'd find this commonality?)

Return JSON with this exact structure:
{{
  "disagreement_surfaced": true/false,
  "disagreement_evidence": "brief quote or description",
  "common_ground_found": true/false,
  "common_ground_type": "same_values" | "related_subtopic" | "different_topic" | "rapport_only" | "none",
  "common_ground_description": "what common ground was found",
  "discovery_moment": "quote of key exchange",
  "surprising": true/false,
  "surprise_explanation": "why or why not surprising",
  "conversation_arc": "disagreement_then_discovery" | "disagreement_no_resolution" | "no_disagreement_surfaced" | "immediate_agreement" | "other",
  "notes": "any additional observations"
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
        .select([
            "groupId", "matchedQuestion", "matchedDomain",
            "matchedTolerance", "stance", "pid", "predictShared",
        ])
    )

    # Get unique groups (take first pid's info since matchedQuestion etc. is same for both)
    groups = (
        chat.group_by("groupId")
        .agg([
            col("matchedQuestion").first(),
            col("matchedDomain").first(),
            col("matchedTolerance").first(),
            col("stance").first(),
            # Whether either participant expected commonality on focal
            col("predictShared").mean().alias("pct_expect_commonality"),
        ])
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
            matched_tolerance=row["matchedTolerance"],
            stance=row["stance"],
        )

        request = {
            "custom_id": f"cg_{group_id}",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": MODEL_CONFIG["temperature"],
                    "maxOutputTokens": MODEL_CONFIG["max_tokens"],
                    "responseMimeType": "application/json",
                },
            },
        }
        batch_requests.append(request)

    return batch_requests


def submit_batch(batch_requests: List[dict]) -> str:
    """Submit batch to Vertex AI."""
    import vertexai
    from vertexai.preview.batch_prediction import BatchPredictionJob

    vertexai.init(project=GCS_PROJECT, location="us-central1")

    batch_file = BATCH_DIR / f"{EXPERIMENT_NAME}.jsonl"
    with open(batch_file, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    print(f"Created {batch_file} ({len(batch_requests)} requests)")

    gcs_input = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/input/{EXPERIMENT_NAME}.jsonl"
    gcs_output = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/output/{EXPERIMENT_NAME}/"

    subprocess.run(["gsutil", "cp", str(batch_file), gcs_input], check=True)
    print(f"Uploaded to {gcs_input}")

    job = BatchPredictionJob.submit(
        source_model=f"publishers/google/models/{MODEL_CONFIG['model']}",
        input_dataset=gcs_input,
        output_uri_prefix=gcs_output,
    )

    state_names = {1: "PENDING", 2: "RUNNING", 3: "SUCCEEDED", 4: "FAILED", 5: "CANCELLED"}
    print(f"Job: {job.resource_name}")
    print(f"State: {state_names.get(job.state, job.state)}")

    job_info = {
        EXPERIMENT_NAME: {
            "resource_name": job.resource_name,
            "input_uri": gcs_input,
            "output_uri": gcs_output,
        }
    }
    ids_file = BATCH_DIR / f"{EXPERIMENT_NAME}_batch_ids.json"
    with open(ids_file, "w") as f:
        json.dump(job_info, f, indent=2)

    return job.resource_name


def download_and_parse() -> pd.DataFrame:
    """Download results and parse into a DataFrame."""
    ids_file = BATCH_DIR / f"{EXPERIMENT_NAME}_batch_ids.json"
    if not ids_file.exists():
        raise FileNotFoundError(f"No batch job found. Run submit first.")

    with open(ids_file) as f:
        batch_ids = json.load(f)

    info = batch_ids[EXPERIMENT_NAME]
    output_uri = info["output_uri"]

    result = subprocess.run(
        ["gsutil", "ls", f"{output_uri}**predictions.jsonl"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FileNotFoundError("No predictions file found. Job may still be running.")

    predictions_uri = result.stdout.strip().split("\n")[0]
    print(f"Downloading {predictions_uri}...")

    raw_file = RESULTS_DIR / f"{EXPERIMENT_NAME}_raw.jsonl"
    subprocess.run(["gsutil", "cp", predictions_uri, str(raw_file)], check=True)

    # Parse results
    records = []
    errors = 0
    with open(raw_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                custom_id = d["custom_id"]
                group_id = custom_id.replace("cg_", "")

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

                annotation["group_id"] = group_id
                records.append(annotation)

            except Exception:
                errors += 1

    print(f"Parsed {len(records)} results, {errors} errors")

    if not records:
        raise RuntimeError(
            f"All {errors} responses failed to parse. "
            "Check the raw file for error messages."
        )

    df = pd.DataFrame(records)

    # Join with group metadata
    groups = get_chat_groups()
    df = df.merge(
        groups.rename(columns={"groupId": "group_id"}),
        on="group_id", how="left",
    )

    output_file = RESULTS_DIR / f"{EXPERIMENT_NAME}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total dyads annotated: {len(df)}")
    if "disagreement_surfaced" in df.columns:
        print(f"Disagreement surfaced: {df['disagreement_surfaced'].sum()} ({df['disagreement_surfaced'].mean():.0%})")
    if "common_ground_found" in df.columns:
        print(f"Common ground found: {df['common_ground_found'].sum()} ({df['common_ground_found'].mean():.0%})")
    if "common_ground_type" in df.columns:
        print(f"\nCommon ground types:")
        print(df["common_ground_type"].value_counts().to_string())
    if "conversation_arc" in df.columns:
        print(f"\nConversation arcs:")
        print(df["conversation_arc"].value_counts().to_string())

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Annotate opposing-stance chats for common ground discovery"
    )
    parser.add_argument("--dry-run", action="store_true", help="Create batch file only")
    parser.add_argument("--download", action="store_true", help="Download and parse results")
    parser.add_argument("--sample", type=int, help="Only process first N groups")
    args = parser.parse_args()

    if args.download:
        download_and_parse()
        return

    batch_requests = create_batch_requests(sample=args.sample)
    print(f"Created {len(batch_requests)} batch requests")

    if args.dry_run:
        batch_file = BATCH_DIR / f"{EXPERIMENT_NAME}.jsonl"
        with open(batch_file, "w") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")
        print(f"[DRY RUN] Saved to {batch_file}")

        # Print a sample prompt
        if batch_requests:
            sample_prompt = batch_requests[0]["request"]["contents"][0]["parts"][0]["text"]
            print(f"\n=== SAMPLE PROMPT ===\n{sample_prompt[:2000]}...")
        return

    submit_batch(batch_requests)


if __name__ == "__main__":
    main()
