"""
Prompt templates for LLM stance detection.

These prompts instruct Gemini to predict probability distributions over
Likert responses (1-5) for each participant across all 35 survey questions.
"""

import random
import pandas as pd


def create_chat_prompt(conversation: str, questions_df: pd.DataFrame, chat_topic: str) -> str:
    """Create prompt for predicting responses from conversation content.

    Args:
        conversation: Interleaved conversation text with speaker labels (Cat/Dog)
        questions_df: DataFrame with columns [questionText, domain]
        chat_topic: The question they discussed (marked [DISCUSSED] in output)

    Returns:
        Formatted prompt string
    """
    conversation = str(conversation) if not pd.isna(conversation) else ""

    # Randomize question order (seeded by conversation hash for reproducibility)
    seed = hash(conversation) % (2**32)
    rng = random.Random(seed)
    indices = list(range(len(questions_df)))
    rng.shuffle(indices)

    # Build question list with [DISCUSSED] marker
    questions_list = "\n".join([
        f"{idx}. {questions_df.iloc[idx]['questionText']}" +
        (" [DISCUSSED]" if questions_df.iloc[idx]['questionText'] == chat_topic else "")
        for idx in indices
    ])

    return f"""Predict how two people (Cat and Dog) would respond to survey questions based on their conversation.

Likert scale:
1 = Definitely Not / Strongly Disagree
2 = Probably Not / Disagree
3 = Unsure / Neutral
4 = Probably Yes / Agree
5 = Definitely Yes / Strongly Agree

One question is marked [DISCUSSED] - they talked about this topic. For other questions, infer from the conversation.

Base your predictions on:
- General population tendencies (as a starting point)
- Evidence from the conversation (update based on what they discuss)
- When conversation evidence is weak or absent, rely more on population priors
- If participants haven't spoken yet or messages are empty, just use your general population knowledge

=== CONVERSATION ===
{conversation if conversation else "(No conversation yet)"}

=== QUESTIONS ===
These participants answered the following questions on a Likert scale (1-5) BEFORE their conversation:

{questions_list}

=== TASK ===
Based on their conversation, predict probability distributions for BOTH participants' responses to ALL {len(questions_df)} questions.

For each question, return a probability distribution over Likert values 1-5 that sums to 1.0.

Return JSON with this structure (using question indices 0-{len(questions_df)-1}):
{{
  "cat_predictions": {{
    "0": {{"1": p1, "2": p2, "3": p3, "4": p4, "5": p5}},
    "1": {{"1": p1, "2": p2, "3": p3, "4": p4, "5": p5}},
    ...
  }},
  "dog_predictions": {{
    "0": {{"1": p1, "2": p2, "3": p3, "4": p4, "5": p5}},
    ...
  }}
}}"""


def create_prior_prompt(questions_df: pd.DataFrame, chat_topic: str) -> str:
    """Create baseline prompt (no conversation information).

    Used to measure LLM's prior knowledge about population response distributions.

    Args:
        questions_df: DataFrame with columns [questionText, domain]
        chat_topic: Used only for consistent random seeding

    Returns:
        Formatted prompt string
    """
    seed = hash(chat_topic) % (2**32)
    rng = random.Random(seed)
    indices = list(range(len(questions_df)))
    rng.shuffle(indices)

    questions_list = "\n".join([
        f"{idx}. {questions_df.iloc[idx]['questionText']}"
        for idx in indices
    ])

    return f"""You are making predictions about people's beliefs on various topics.

For each question, predict how a randomly selected person would respond on a Likert scale:
1 = Definitely Not / Strongly Disagree
2 = Probably Not / Disagree
3 = Unsure / Neutral
4 = Probably Yes / Agree
5 = Definitely Yes / Strongly Agree

You have NO information about this person beyond the question itself. Base your prediction on:
- General population tendencies
- The nature of the question
- Common sense about belief distributions

=== QUESTIONS ===
{questions_list}

=== TASK ===
For each question, return a probability distribution over Likert values 1-5 that sums to 1.0.

Return JSON with this structure (using question indices 0-{len(questions_df)-1}):
{{
  "predictions": {{
    "0": {{"1": p1, "2": p2, "3": p3, "4": p4, "5": p5}},
    "1": {{"1": p1, "2": p2, "3": p3, "4": p4, "5": p5}},
    ...
  }}
}}"""


def create_nochat_prompt(
    observed_question: str,
    partner_response: int,
    questions_df: pd.DataFrame,
    predict_for: str = "partner"
) -> str:
    """Create prompt for predicting responses from a single observed Likert response.

    This is information-matched to the Bayesian factor models, which condition only
    on observing the partner's response to one question (no conversation content).

    Args:
        observed_question: The question for which we observed the partner's response
        partner_response: Partner's Likert response (1-5) to that question
        questions_df: DataFrame with columns [questionText, domain]
        predict_for: "partner" to predict remaining partner responses, or "self" for self

    Returns:
        Formatted prompt string
    """
    # Randomize question order (seeded for reproducibility)
    seed = hash(observed_question) % (2**32)
    rng = random.Random(seed)
    indices = list(range(len(questions_df)))
    rng.shuffle(indices)

    # Build question list marking which one was observed
    questions_list = "\n".join([
        f"{idx}. {questions_df.iloc[idx]['questionText']}" +
        (" [OBSERVED]" if questions_df.iloc[idx]['questionText'] == observed_question else "")
        for idx in indices
    ])

    # Map Likert response to descriptive label
    likert_labels = {
        1: "1 (Definitely Not / Strongly Disagree)",
        2: "2 (Probably Not / Disagree)",
        3: "3 (Unsure / Neutral)",
        4: "4 (Probably Yes / Agree)",
        5: "5 (Definitely Yes / Strongly Agree)"
    }
    response_label = likert_labels.get(partner_response, str(partner_response))

    return f"""You are predicting a person's beliefs based on observing their response to ONE question.

=== OBSERVED RESPONSE ===
Question: "{observed_question}"
This person responded: {response_label}

This is the ONLY information you have about this person. You have NO other context - no conversation, no demographics, no other responses.

=== LIKERT SCALE ===
1 = Definitely Not / Strongly Disagree
2 = Probably Not / Disagree
3 = Unsure / Neutral
4 = Probably Yes / Agree
5 = Definitely Yes / Strongly Agree

=== QUESTIONS ===
Predict how this same person would respond to these questions:

{questions_list}

One question is marked [OBSERVED] - you already know their response to this question (use it as your anchor).

=== TASK ===
Based ONLY on the single observed response, predict probability distributions for this person's responses to ALL {len(questions_df)} questions.

Consider:
- What does this response reveal about the person's beliefs, values, or personality?
- How might someone who gave this response answer related questions?
- For unrelated questions, rely on population base rates
- The observed question provides evidence about this person - use it to update from population priors

For each question, return a probability distribution over Likert values 1-5 that sums to 1.0.

Return JSON with this structure (using question indices 0-{len(questions_df)-1}):
{{
  "predictions": {{
    "0": {{"1": p1, "2": p2, "3": p3, "4": p4, "5": p5}},
    "1": {{"1": p1, "2": p2, "3": p3, "4": p4, "5": p5}},
    ...
  }}
}}"""


def create_pshared_prompt(conversation: str, questions_df: pd.DataFrame, chat_topic: str) -> str:
    """Create prompt for directly predicting P(shared) - probability of agreement.

    Instead of predicting full probability distributions, this prompt asks the LLM
    to directly predict whether two people would agree on each question based on
    their conversation.

    Args:
        conversation: Interleaved conversation text with speaker labels (Cat/Dog)
        questions_df: DataFrame with columns [questionText, domain]
        chat_topic: The question they discussed (marked [DISCUSSED] in output)

    Returns:
        Formatted prompt string
    """
    conversation = str(conversation) if not pd.isna(conversation) else ""

    # Randomize question order (seeded by conversation hash for reproducibility)
    seed = hash(conversation) % (2**32)
    rng = random.Random(seed)
    indices = list(range(len(questions_df)))
    rng.shuffle(indices)

    # Build question list with [DISCUSSED] marker and domain labels
    questions_list = "\n".join([
        f"{idx}. [{questions_df.iloc[idx]['domain'].upper()}] {questions_df.iloc[idx]['questionText']}" +
        (" [DISCUSSED]" if questions_df.iloc[idx]['questionText'] == chat_topic else "")
        for idx in indices
    ])

    return f"""You are predicting whether two people (Cat and Dog) would AGREE on various survey questions.

Two people "share" a response if their Likert answers are within 2 points of each other (e.g., one says 3 and the other says 4 or 5).

Based on their conversation, predict the probability that Cat and Dog would agree on each question.

=== CONVERSATION ===
{conversation if conversation else "(No conversation yet)"}

=== QUESTIONS ===
The questions are organized by domain. People who agree on one topic within a domain often (but not always) agree on related topics.

{questions_list}

=== COGNITIVE TASK ===
This is a social cognition task. Use the conversation to:

1. INFER each person's underlying beliefs, values, and worldview
2. GENERALIZE: People with similar views on one topic often share views on related topics
3. PREDICT: Based on inferred profiles, estimate agreement probability for each question

Key insight: The question marked [DISCUSSED] gives you direct evidence. Questions in the SAME DOMAIN should show correlated beliefs. Different domains are less predictive.

=== OUTPUT FORMAT ===
For each question, return your probability estimate that Cat and Dog would agree (0.0 to 1.0).

Return JSON with this structure (using question indices 0-{len(questions_df)-1}):
{{
  "agreement_probabilities": {{
    "0": 0.85,
    "1": 0.42,
    "2": 0.71,
    ...
  }}
}}"""
