"""
Gemini Batch API submission utilities.

Uses Google Cloud Vertex AI Batch Prediction API.
Requires environment variables:
  - GOOGLE_CLOUD_PROJECT
  - GOOGLE_CLOUD_BUCKET
  - GOOGLE_CLOUD_LOCATION (optional, defaults to us-central1)
"""

import json
from pathlib import Path
from .config import (
    BATCH_DIR, MODEL_CONFIG,
    GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_CLOUD_BUCKET
)

# Lazy imports for Google Cloud (only needed when submitting batches)
aiplatform = None
storage = None


def _init_google_cloud():
    """Initialize Google Cloud clients."""
    global aiplatform, storage

    if aiplatform is None:
        from google.cloud import aiplatform as _aiplatform
        from google.cloud import storage as _storage
        aiplatform = _aiplatform
        storage = _storage

        if GOOGLE_CLOUD_PROJECT:
            aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)


def create_batch_request(custom_id: str, prompt: str) -> dict:
    """Create a single batch request in Gemini format.

    Args:
        custom_id: Unique identifier for this request
        prompt: The full prompt text

    Returns:
        Gemini batch request dict
    """
    return {
        "custom_id": custom_id,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": MODEL_CONFIG["temperature"],
                "maxOutputTokens": MODEL_CONFIG["max_tokens"],
                "responseMimeType": "application/json",
                "thinkingConfig": {
                    "thinkingLevel": MODEL_CONFIG.get("thinking_level", "high")
                }
            }
        }
    }


def submit_batch(batch_requests: list, batch_name: str, experiment_type: str) -> str:
    """Submit batch requests to Gemini via Vertex AI.

    Args:
        batch_requests: List of batch request dicts from create_batch_request()
        batch_name: Name for this batch (used in filenames)
        experiment_type: Type of experiment (for tracking)

    Returns:
        Batch job resource name

    Raises:
        ValueError: If Google Cloud credentials not configured
    """
    if not GOOGLE_CLOUD_PROJECT or not GOOGLE_CLOUD_BUCKET:
        raise ValueError(
            "Gemini batch processing requires environment variables:\n"
            "  GOOGLE_CLOUD_PROJECT\n"
            "  GOOGLE_CLOUD_BUCKET\n"
            "Set these in your .env file or environment."
        )

    _init_google_cloud()

    # Save batch file locally
    batch_file = BATCH_DIR / f"{batch_name}.jsonl"
    with open(batch_file, 'w') as f:
        for req in batch_requests:
            f.write(json.dumps(req) + '\n')

    print(f"Created {batch_file} ({len(batch_requests)} requests)")

    # Upload to GCS
    storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
    bucket = storage_client.bucket(GOOGLE_CLOUD_BUCKET)

    input_blob_name = f"llm-stance-detection/input/{batch_name}.jsonl"
    blob = bucket.blob(input_blob_name)
    blob.upload_from_filename(batch_file)

    input_uri = f"gs://{GOOGLE_CLOUD_BUCKET}/{input_blob_name}"
    output_uri = f"gs://{GOOGLE_CLOUD_BUCKET}/llm-stance-detection/output/{batch_name}/"

    print(f"Uploaded to: {input_uri}")

    # Submit batch job
    model_name = MODEL_CONFIG["model"]
    job = aiplatform.BatchPredictionJob.create(
        model_name=f"publishers/google/models/{model_name}",
        job_display_name=f"stance-detection-{batch_name}",
        gcs_source=input_uri,
        gcs_destination_prefix=output_uri,
    )

    print(f"Batch Job: {job.resource_name}")
    print(f"Status: {job.state}")

    # Save job info
    batch_ids_file = BATCH_DIR / f"{experiment_type}_batch_ids.json"
    batch_ids = {}
    if batch_ids_file.exists():
        with open(batch_ids_file) as f:
            batch_ids = json.load(f)

    batch_ids[batch_name] = {
        "resource_name": job.resource_name,
        "input_uri": input_uri,
        "output_uri": output_uri,
    }

    with open(batch_ids_file, 'w') as f:
        json.dump(batch_ids, f, indent=2)

    return job.resource_name


def download_results(output_uri: str) -> str:
    """Download results from a completed batch job.

    Args:
        output_uri: GCS output URI from batch job

    Returns:
        Results as JSONL text
    """
    _init_google_cloud()

    # Parse GCS URI
    path_parts = output_uri[5:].split("/", 1)  # Remove "gs://"
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Find predictions file
    for blob in blobs:
        if blob.name.endswith("predictions.jsonl"):
            print(f"Downloading: gs://{bucket_name}/{blob.name}")
            return blob.download_as_text()

    raise FileNotFoundError(f"No predictions.jsonl found in {output_uri}")
