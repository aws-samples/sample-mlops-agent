"""Functional test for the "LLM evaluation with MLflow" starter tile.

Runs the two-step sequence the agent executes:
  1. HuggingFace skill: prepare_eval_dataset → spec_s3_uri
  2. SageMaker skill: submit_eval_job against the spec

Uses Nova Lite (`bedrock:/us.amazon.nova-lite-v1:0`, US cross-region
inference profile — the actual Bedrock model ID, not the UI display alias)
on a 2-row slice of PatronusAI/financebench with a single `correctness`
scorer judged by Claude 3.5 Haiku. Polls until the SageMaker processing
job reaches a terminal state.
"""
import json
import os
import time
import uuid

import pytest


_EVAL_TIMEOUT_SEC = 15 * 60
_POLL_INTERVAL_SEC = 30


def _wait_for_processing(sm_client, job_name: str) -> dict:
    """Poll describe_processing_job every 30 s until terminal or timeout."""
    terminal = {"Completed", "Failed", "Stopped"}
    deadline = time.time() + _EVAL_TIMEOUT_SEC
    last_desc: dict = {}
    while time.time() < deadline:
        last_desc = sm_client.describe_processing_job(ProcessingJobName=job_name)
        status = last_desc["ProcessingJobStatus"]
        if status in terminal:
            return last_desc
        time.sleep(_POLL_INTERVAL_SEC)
    raise AssertionError(
        f"Processing job {job_name} did not reach terminal state in "
        f"{_EVAL_TIMEOUT_SEC}s. Last status={last_desc.get('ProcessingJobStatus')!r}. "
        f"ExitMessage: {last_desc.get('ExitMessage')!r}"
    )


@pytest.mark.functional
def test_eval_nova_lite_on_financebench_submits_and_reaches_terminal(
    skill_handler, gateway_ctx, sm_client, ddb_table
):
    # Step 1 — prepare the eval spec (writes a tiny JSON to S3, no dataset load).
    hf_h = skill_handler("huggingface")
    prep_event = {
        "task_type": "question_answering",
        "dataset_name": "PatronusAI/financebench",
        # financebench only exposes a `train` split on HF; requesting `test`
        # made load_dataset raise "Unknown split" inside the container.
        "split": "train",
        "max_rows": 2,
    }
    prep_resp = hf_h.handler(
        prep_event, gateway_ctx("huggingface-skill___prepare_eval_dataset")
    )
    assert prep_resp.get("isError") is not True, prep_resp
    prep_body = json.loads(prep_resp["content"][0]["text"])
    spec_s3_uri = prep_body["spec_s3_uri"]
    assert spec_s3_uri.startswith("s3://")

    # Step 2 — submit the processing job against the spec.
    sm_h = skill_handler("sagemaker")
    thread_id = f"functest-eval-{uuid.uuid4()}"
    event = {
        "thread_id": thread_id,
        "eval_dataset_s3_uri": spec_s3_uri,
        # Nova Lite on-demand via US cross-region inference profile. The tile's
        # `global.amazon.nova-2-lite-v1:0` is a UI alias; Bedrock InvokeModel
        # expects the canonical model ID below.
        "target_model": "bedrock:/us.amazon.nova-lite-v1:0",
        # Claude 3.5 Haiku (claude-3-5-haiku-20241022-v1:0) was marked
        # Legacy by Bedrock — invocations return 404 "Access denied. This
        # Model is marked by provider as Legacy". Use the active Haiku 4.5
        # cross-region inference profile instead.
        "judge_model": "bedrock:/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        # Canonical MLflow scorer name (PascalCase). The container's
        # _resolve_scorer_name rejects lowercase aliases — the agent is
        # expected to call mlflow-skill___list_scorers and forward the
        # canonical name verbatim.
        "scorers": ["Correctness"],
        "task": "question_answering",
        "instance_type": "ml.m5.large",
        "_user_id": "functest",
    }
    resp = sm_h.handler(event, gateway_ctx("sagemaker-skill___submit_eval_job"))
    assert resp.get("isError") is not True, resp
    body = json.loads(resp["content"][0]["text"])
    assert body["status"] == "SUBMITTING"
    proc_name = body["processing_job_name"]
    assert proc_name.startswith(f"{sm_h.PROJECT_NAME}-eval-")
    job_id = body["job_id"]
    mlflow_run_id = body["mlflow_run_id"]
    assert mlflow_run_id, "handler must pre-create an MLflow run before submit"

    # Short grace period for CreateProcessingJob to land.
    time.sleep(10)

    desc = _wait_for_processing(sm_client, proc_name)
    assert desc["ProcessingJobStatus"] == "Completed", (
        f"Eval processing failed: exit={desc.get('ExitMessage')!r}, "
        f"failure={desc.get('FailureReason')!r}"
    )

    row = ddb_table.get_item(Key={"task_id": thread_id}).get("Item") or {}
    job_entry = (row.get("jobs") or {}).get(job_id) or {}
    assert job_entry.get("kind") == "eval"
    assert job_entry.get("processing_job_name") == proc_name
    assert job_entry.get("task") == "question_answering"
    assert job_entry.get("scorers") == ["Correctness"]

    # Guard against silent scorer failure: a previous run saw
    # ProcessingJobStatus=Completed while every Correctness invocation died
    # inside MLflow with pydantic "AWSIdAndKey(aws_access_key_id=None, …)".
    # Assert the run actually produced a numeric Correctness metric.
    import mlflow  # local import so the module still skips cleanly when infra is absent
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    run = MlflowClient().get_run(mlflow_run_id)
    metric_keys = list(run.data.metrics.keys())
    correctness_keys = [k for k in metric_keys if k.lower().startswith("correctness")]
    assert correctness_keys, (
        f"MLflow run {mlflow_run_id} has no Correctness metric. "
        f"Judge scorer likely failed silently. All metrics: {metric_keys}"
    )
    for key in correctness_keys:
        value = run.data.metrics[key]
        assert isinstance(value, (int, float)), (
            f"MLflow metric {key}={value!r} is not numeric — judge returned null."
        )
