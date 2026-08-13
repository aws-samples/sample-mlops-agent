"""Functional test for the "Quick fine-tune" starter tile.

Submits a tiny (max_steps=1, max_samples=8) SFT training job for
Qwen 2.5-0.5B-Instruct on HuggingFaceH4/ultrachat_200k by invoking the
SageMaker skill Lambda handler directly, then polls until the SageMaker
training job reaches a terminal state. Pass criterion: job status
"Completed" and a non-empty ModelArtifacts S3 URI.
"""
import json
import time
import uuid

import pytest


_TRAINING_TIMEOUT_SEC = 25 * 60
_POLL_INTERVAL_SEC = 30


def _wait_for_training(sm_client, job_name: str) -> dict:
    """Poll describe_training_job every 30 s until terminal or timeout.

    Raises AssertionError on timeout with the latest secondary status history
    so a stuck job is debuggable from the pytest output.
    """
    terminal = {"Completed", "Failed", "Stopped"}
    deadline = time.time() + _TRAINING_TIMEOUT_SEC
    last_desc: dict = {}
    while time.time() < deadline:
        last_desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = last_desc["TrainingJobStatus"]
        if status in terminal:
            return last_desc
        time.sleep(_POLL_INTERVAL_SEC)
    transitions = last_desc.get("SecondaryStatusTransitions", [])
    raise AssertionError(
        f"Training job {job_name} did not reach terminal state in "
        f"{_TRAINING_TIMEOUT_SEC}s. Last status={last_desc.get('TrainingJobStatus')!r}. "
        f"Transitions: {transitions}"
    )


@pytest.mark.functional
def test_fine_tune_sft_submits_and_reaches_terminal(
    skill_handler, gateway_ctx, sm_client, ddb_table
):
    sm_h = skill_handler("sagemaker")

    thread_id = f"functest-finetune-{uuid.uuid4()}"
    event = {
        "thread_id": thread_id,
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_name": "HuggingFaceH4/ultrachat_200k",
        "training_type": "sft",
        "instance_type": "ml.g5.2xlarge",
        "max_steps": 1,
        "learning_rate": 1e-5,
        "max_samples": 8,
        # ultrachat_200k uses non-canonical split names; passing plain
        # "train"/"test" fails inside the training container.
        "train_split": "train_sft",
        "test_split": "test_sft",
        "_user_id": "functest",
    }

    resp = sm_h.handler(event, gateway_ctx("sagemaker-skill___submit_training_job"))
    assert resp.get("isError") is not True, resp
    body = json.loads(resp["content"][0]["text"])
    assert body["status"] == "SUBMITTING"
    assert body["sagemaker_job_name"].startswith(
        f"{sm_h.PROJECT_NAME}-job-"
    )
    job_name = body["sagemaker_job_name"]
    job_id = body["job_id"]

    # The submit handler calls _invoke_background; locally the fallback runs
    # inline, in Lambda it self-invokes async. Give CreateTrainingJob a short
    # grace period either way before polling.
    time.sleep(10)

    desc = _wait_for_training(sm_client, job_name)
    assert desc["TrainingJobStatus"] == "Completed", (
        f"Training failed: {desc.get('FailureReason')!r}. "
        f"Secondary transitions: {desc.get('SecondaryStatusTransitions')}"
    )
    assert desc["ModelArtifacts"]["S3ModelArtifacts"].startswith("s3://")

    # Confirm the thread row + job sub-record was written by the handler.
    row = ddb_table.get_item(Key={"task_id": thread_id}).get("Item") or {}
    job_entry = (row.get("jobs") or {}).get(job_id) or {}
    assert job_entry.get("kind") == "training"
    assert job_entry.get("sagemaker_job_name") == job_name
    assert job_entry.get("training_type") == "sft"
