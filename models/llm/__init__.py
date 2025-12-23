"""
LLM Stance Detection Package

Self-contained pipeline for predicting participant Likert responses
from conversation transcripts using Gemini 3 Pro.

See README.md for usage instructions.
"""

from .prompts import create_chat_prompt, create_prior_prompt, create_nochat_prompt
from .data import (
    load_questions,
    load_unified_data,
    load_ground_truth,
    load_nochat_observations,
    load_dialogue_data,
)

__all__ = [
    'create_chat_prompt',
    'create_prior_prompt',
    'create_nochat_prompt',
    'load_questions',
    'load_unified_data',
    'load_ground_truth',
    'load_nochat_observations',
    'load_dialogue_data',
]
