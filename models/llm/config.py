"""
Configuration for LLM stance detection experiments.
"""

from pathlib import Path
import os

# Paths - self-contained within /paper/
PAPER_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PAPER_DIR / "data"  # Local paper/data directory

BATCH_DIR = Path(__file__).parent / "batch_requests"
BATCH_DIR.mkdir(exist_ok=True)  # Ensure batch directory exists

RESULTS_DIR = DATA_DIR / "llm_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv(PAPER_DIR / ".env")
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# Google Cloud configuration (set in environment or .env file)
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
GOOGLE_CLOUD_BUCKET = os.getenv("GOOGLE_CLOUD_BUCKET")

GCS_PROJECT = "709275529646"
GCS_BUCKET = "hs-social-interaction-llm-batches"

# Model configuration
MODEL_CONFIG = {
    "model": "gemini-3-pro-preview",
    "temperature": 1.0,
    "max_tokens": 65536,
    "thinking_level": "high",
}
