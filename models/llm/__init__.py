"""
LLM Stance Detection Package

Self-contained pipeline for predicting participant Likert responses
from conversation transcripts using Gemini 3 Pro.

Usage:
    # Submit batch
    python -m models.llm.pipeline submit pshared --max-bins 13

    # Check status
    python -m models.llm.pipeline status pshared

    # Download results
    python -m models.llm.pipeline download pshared

See README.md for full documentation.
"""

from .prompts import (
    create_chat_prompt,
    create_prior_prompt,
    create_nochat_prompt,
    create_pshared_prompt,
)
from .data import (
    load_questions,
    load_unified_data,
    load_ground_truth,
    load_nochat_observations,
    load_dialogue_data,
)
from .pipeline import (
    create_chat_timecourse_batch,
    submit_batch,
    check_status,
    download_results,
    parse_raw_results,
    compute_pshared_metrics,
)

__all__ = [
    # Prompts
    'create_chat_prompt',
    'create_prior_prompt',
    'create_nochat_prompt',
    'create_pshared_prompt',
    # Data
    'load_questions',
    'load_unified_data',
    'load_ground_truth',
    'load_nochat_observations',
    'load_dialogue_data',
    # Pipeline
    'create_chat_timecourse_batch',
    'submit_batch',
    'check_status',
    'download_results',
    'parse_raw_results',
    'compute_pshared_metrics',
]
