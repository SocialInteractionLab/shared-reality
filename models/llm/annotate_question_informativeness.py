#!/usr/bin/env python3
"""
Annotate chat transcripts for question-level informativeness.

For each dyad, sends the full conversation transcript to Gemini and asks it to
identify — separately for each participant — which of the 35 pre-chat survey
questions their conversation contributions revealed their stance on, with an
inferred 1-5 stance and confidence score.

Output: long-format CSV `data/llm_results/question_informativeness.csv` with
one row per (dyad_id, pid, question_idx) inference. Columns:
    dyad_id, pid, assigned_question_idx, stance_type,
    question_idx, inferred_stance, confidence, rationale
Only confidence ≥ 0.3 rows are kept — questions not in the CSV had no
informative chat evidence. `stance_type` ("shared" or "opposing") is joined
from dyad metadata at parse time; the LLM is kept blind to it to avoid
biasing its per-question stance inferences.

Downstream: the multi-observation Bayesian model loads these rows and chains
Gaussian updates per inference, with σ_q = σ_base / confidence.

Usage:
    # Inspect the prompt for the first dyad (no file write, no API call)
    uv run -m models.llm.annotate_question_informativeness --inspect-prompt

    # Dry run: create batch JSONL locally, no submission
    uv run -m models.llm.annotate_question_informativeness --dry-run

    # Submit batch to Vertex AI
    uv run -m models.llm.annotate_question_informativeness

    # Download and parse results
    uv run -m models.llm.annotate_question_informativeness --download
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

import pandas as pd

from .annotate_commonality import (
    _download_raw,
    build_full_conversation,
    get_chat_groups,
    submit_batch,
)
from .config import BATCH_DIR, DATA_DIR, MODEL_CONFIG, RESULTS_DIR

EXPERIMENT_NAME = "question_informativeness"
CONFIDENCE_THRESHOLD = 0.3


def build_questions_reference() -> str:
    """Build the static 35-question reference table injected into every prompt."""
    df = pd.read_csv(DATA_DIR / "questions.csv")
    lines = ["| idx | domain | question |", "|-----|--------|----------|"]
    for _, row in df.iterrows():
        lines.append(f"| {int(row['num'])} | {row['domain']} | {row['text']} |")
    return "\n".join(lines)


def get_dyad_pids(messages_df: pd.DataFrame, group_id: str) -> dict[str, str]:
    """Return {'cat': pid, 'dog': pid} for a dyad (from messages.csv)."""
    g = messages_df[messages_df["group_id"] == group_id]
    cat_pid = g.loc[g["author"] == "🐱", "prolific_id"].iloc[0]
    dog_pid = g.loc[g["author"] == "🐶", "prolific_id"].iloc[0]
    return {"cat": str(cat_pid), "dog": str(dog_pid)}


def get_focal_idx(messages_df: pd.DataFrame, group_id: str) -> int:
    """Return the 1-indexed focal question for a dyad."""
    g = messages_df[messages_df["group_id"] == group_id]
    return int(g["matched_idx"].iloc[0])


def create_informativeness_prompt(
    conversation: str,
    focal_question: str,
    focal_domain: str,
    focal_idx: int,
    questions_table: str,
    dyad_id: str,
    cat_pid: str,
    dog_pid: str,
) -> str:
    """Create the prompt for per-participant question-level stance inference.

    The LLM is asked, for each participant independently, to identify which
    of the 35 survey questions that participant's chat contributions gave
    evidence about, and to estimate the revealed stance (1-5) + confidence.
    """
    return f"""You are analyzing a conversation between two people (Cat and Dog) who were \
paired for a study. Before the chat, each of them independently answered 35 yes/no \
survey questions on a 1-5 Likert scale, where 1 = "Definitely Not / Strongly Disagree" \
and 5 = "Definitely Yes / Strongly Agree". The experimenter then assigned the pair a \
single focal question from that 35-item survey and instructed them to chat about it — \
participants did NOT choose the topic and were NOT told why they were chatting with this person specifically.

Your task: for EACH participant (Cat and Dog) separately, identify which of the 35 \
survey questions their conversation contributions revealed their stance on. For each \
such question, estimate what they would have answered (integer 1-5) and how confident \
you are that the chat supports that inference.

=== THE 35 SURVEY QUESTIONS ===
{questions_table}

=== STANCE SCALE (how a participant would have answered that yes/no question) ===
1 = Definitely Not / Strongly Disagree
2 = Probably Not / Disagree
3 = Unsure / Neutral
4 = Probably Yes / Agree
5 = Definitely Yes / Strongly Agree

=== STUDY CONTEXT ===
Dyad id: {dyad_id}
Cat's participant id: {cat_pid}
Dog's participant id: {dog_pid}
Assigned focal question: "{focal_question}"
Assigned focal question idx: {focal_idx}
Assigned focal question domain: {focal_domain}

=== CONVERSATION ===
{conversation}

=== ANNOTATION INSTRUCTIONS ===
For EACH participant separately, go through the 35 questions and determine whether \
that participant's own words in the conversation revealed their stance on that \
question. Judge based on what THAT participant said, not what their partner said.

INCLUDE a question only if the participant's words give real evidence of a \
directional stance (leaning toward "Yes" or toward "No"). Generic backchannels \
and politeness do NOT count:
  - "I love being outdoors, I hike every weekend" → informative about Q2 \
(Do you enjoy being outdoors?) with high confidence, stance ~5
  - "I work out maybe once a week, I'm not super into it" → informative about \
Q1 (Do you exercise regularly?) with moderate confidence, stance ~2
  - "yeah, cool" / "that makes sense" / "right" → not informative about anything \
(just backchannel acknowledgment)

DO NOT EMIT `inferred_stance = 3`. Only emit rows where the participant revealed \
a directional stance. Valid `inferred_stance` values are {{1, 2, 4, 5}} ONLY. \
If a participant expressed pure ambivalence or did not substantively address \
the topic, do NOT emit a row for that question — the absence of a row means \
"no directional evidence from chat", which is what the downstream model expects.

The focal question (idx {focal_idx}) MAY be included for a participant if they \
expressed a position on it during the chat. Include only if the chat itself is \
informative — not just because the question was the assigned topic.

CONFIDENCE guidelines (float 0.3–1.0; use 0.0 to omit):
  - 0.9+     : direct, explicit statement of position
  - 0.7–0.9  : clear implication from what they said
  - 0.5–0.7  : reasonable inference from broader context
  - 0.3–0.5  : weak or indirect signal
  - < 0.3    : DO NOT include (too speculative)

INFERRED_STANCE (integer, restricted to {{1, 2, 4, 5}}): what you think they \
would have answered on that question, based ONLY on what they said in the \
conversation. 3 is NOT a permitted output — if the participant's stance is \
ambivalent, omit the row instead.

RATIONALE: one sentence citing the relevant moment(s) in the chat.

CONSISTENCY RULES:
- Include each (question_idx) at most ONCE per participant.
- Only include questions where confidence ≥ 0.3.
- If a participant's chat contributions were not informative about any question, \
return an empty list for that participant.
- Do NOT infer a participant's stance from their partner's words.

=== OUTPUT FORMAT ===
Return JSON with this exact structure. Use the participant ids shown above \
(Cat's pid "{cat_pid}" and Dog's pid "{dog_pid}") as the keys inside \
"inferences_by_pid". Echo the dyad id and assigned focal question idx at the top \
level so the record is self-identifying:

{{
  "dyad_id": "{dyad_id}",
  "assigned_question_idx": {focal_idx},
  "inferences_by_pid": {{
    "{cat_pid}": [
      {{
        "question_idx": <int 1-35>,
        "inferred_stance": <int in {{1, 2, 4, 5}}>,
        "confidence": <float 0.3-1.0>,
        "rationale": "<one sentence>"
      }},
      ...
    ],
    "{dog_pid}": [
      {{
        "question_idx": <int 1-35>,
        "inferred_stance": <int in {{1, 2, 4, 5}}>,
        "confidence": <float 0.3-1.0>,
        "rationale": "<one sentence>"
      }},
      ...
    ]
  }}
}}
"""


def create_informativeness_batch_requests(
    sample: Optional[int] = None,
) -> List[dict]:
    """Create one batch request per chat dyad."""
    messages = pd.read_csv(DATA_DIR / "messages.csv")
    messages["absolute_timestamp"] = pd.to_datetime(
        messages["absolute_timestamp"], format="mixed"
    )
    groups = get_chat_groups()

    if sample:
        groups = groups.head(sample)

    questions_table = build_questions_reference()
    print(f"Creating informativeness batch: {len(groups)} dyads")

    batch_requests = []
    for _, row in groups.iterrows():
        group_id = row["groupId"]
        conversation = build_full_conversation(messages, group_id)
        if not conversation.strip():
            continue

        focal_idx = get_focal_idx(messages, group_id)
        pids = get_dyad_pids(messages, group_id)

        prompt = create_informativeness_prompt(
            conversation=conversation,
            focal_question=row["matchedQuestion"],
            focal_domain=row["matchedDomain"],
            focal_idx=focal_idx,
            questions_table=questions_table,
            dyad_id=group_id,
            cat_pid=pids["cat"],
            dog_pid=pids["dog"],
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
            "custom_id": f"qi_{group_id}",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen_config,
            },
        }
        batch_requests.append(request)

    return batch_requests


def download_and_parse_informativeness() -> pd.DataFrame:
    """Download and parse results into a long-format informativeness CSV.

    Output columns:
        dyad_id, pid, assigned_question_idx, stance_type,
        question_idx, inferred_stance, confidence, rationale

    stance_type is joined from get_chat_groups() metadata; it is NOT produced
    by the LLM (which is kept blind to shared vs. opposing assignment to
    avoid biasing its per-question stance inferences).
    """
    _, records = _download_raw(EXPERIMENT_NAME)

    messages = pd.read_csv(DATA_DIR / "messages.csv")
    groups_meta = get_chat_groups().rename(
        columns={"groupId": "dyad_id", "stance": "stance_type"}
    )[["dyad_id", "stance_type"]]

    rows = []
    n_dropped_lowconf = 0
    n_dropped_badschema = 0
    n_unknown_pid = 0

    for rec in records:
        custom_id = rec.pop("custom_id")
        dyad_id = custom_id.replace("qi_", "")

        try:
            pids = get_dyad_pids(messages, dyad_id)
            focal_idx = get_focal_idx(messages, dyad_id)
        except (IndexError, KeyError):
            continue

        expected_pids = {pids["cat"], pids["dog"]}
        inferences_by_pid = rec.get("inferences_by_pid", {}) or {}

        for pid, inferences in inferences_by_pid.items():
            if pid not in expected_pids:
                n_unknown_pid += 1
                continue
            for inf in inferences or []:
                try:
                    conf = float(inf["confidence"])
                    q_idx = int(inf["question_idx"])
                    stance = int(inf["inferred_stance"])
                except (KeyError, TypeError, ValueError):
                    n_dropped_badschema += 1
                    continue

                if conf < CONFIDENCE_THRESHOLD:
                    n_dropped_lowconf += 1
                    continue
                if not (1 <= q_idx <= 35) or stance not in (1, 2, 4, 5):
                    n_dropped_badschema += 1
                    continue

                rows.append({
                    "dyad_id": dyad_id,
                    "pid": pid,
                    "assigned_question_idx": focal_idx,
                    "question_idx": q_idx,
                    "inferred_stance": stance,
                    "confidence": conf,
                    "rationale": inf.get("rationale", ""),
                })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.merge(groups_meta, on="dyad_id", how="left")
        df = df[[
            "dyad_id", "pid", "assigned_question_idx", "stance_type",
            "question_idx", "inferred_stance", "confidence", "rationale",
        ]]

    output_file = RESULTS_DIR / f"{EXPERIMENT_NAME}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    print(f"\n=== SUMMARY ===")
    print(f"Total informative rows: {len(df)}")
    print(f"Dropped (low confidence): {n_dropped_lowconf}")
    print(f"Dropped (bad schema): {n_dropped_badschema}")
    print(f"Dropped (unknown pid in LLM output): {n_unknown_pid}")
    if len(df) > 0:
        print(f"Unique dyads: {df['dyad_id'].nunique()}")
        print(f"Unique participants: {df['pid'].nunique()}")
        print(f"\nInferences per participant:")
        print(df.groupby("pid").size().describe().round(2).to_string())
        print(f"\nConfidence distribution:")
        print(df["confidence"].describe().round(3).to_string())
        print(f"\nInferred-stance distribution:")
        print(df["inferred_stance"].value_counts().sort_index().to_string())
        print(f"\nFocal-question inclusion:")
        is_focal = df["question_idx"] == df["assigned_question_idx"]
        print(f"  Focal: {is_focal.sum()} ({is_focal.mean():.1%})")
        print(f"  Non-focal: {(~is_focal).sum()}")
        print(f"\nStance-type split:")
        print(df["stance_type"].value_counts().to_string())

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Annotate chats for question-level informativeness"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Create batch JSONL locally, no submission",
    )
    parser.add_argument(
        "--inspect-prompt", action="store_true",
        help="Print the prompt for the first dyad and exit (no file write, no API)",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download and parse submitted batch results",
    )
    parser.add_argument(
        "--sample", type=int, help="Only process first N dyads",
    )
    args = parser.parse_args()

    if args.download:
        download_and_parse_informativeness()
        return

    batch_requests = create_informativeness_batch_requests(sample=args.sample)
    print(f"Created {len(batch_requests)} batch requests")

    if args.inspect_prompt:
        if not batch_requests:
            print("No batch requests produced (empty groups?).")
            return
        sample = batch_requests[0]
        prompt = sample["request"]["contents"][0]["parts"][0]["text"]
        print("=" * 60)
        print(f"Sample request: {sample['custom_id']}")
        print("=" * 60)
        print(prompt)
        return

    if args.dry_run:
        batch_file = BATCH_DIR / f"{EXPERIMENT_NAME}.jsonl"
        with open(batch_file, "w") as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")
        print(f"[DRY RUN] Saved to {batch_file}")
        if batch_requests:
            sample_prompt = batch_requests[0]["request"]["contents"][0]["parts"][0][
                "text"
            ]
            print(f"\n=== SAMPLE PROMPT (first 2000 chars) ===\n{sample_prompt[:2000]}...")
        return

    submit_batch(batch_requests, experiment_name=EXPERIMENT_NAME)


if __name__ == "__main__":
    main()
