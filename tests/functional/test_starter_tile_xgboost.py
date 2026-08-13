"""Functional test for the "XGBoost training" starter tile.

Submits a 100-round XGBoost job on the bundled sklearn iris dataset
(150 rows, 4 features, 3 classes) via the SageMaker skill Lambda handler
directly, then polls until the SageMaker training job reaches a terminal
state. Goal: catch regressions in the xgboost training-script dataset
dispatch (s3://, HF repo, sklearn bundled) and MLflow instrumentation.

Unlike the SFT test, this runs on ml.m5.xlarge (no GPU) so a full run is
<10 min wall clock.
"""
import json
import os
import time
import uuid

import pytest


_TRAINING_TIMEOUT_SEC = 15 * 60
_POLL_INTERVAL_SEC = 20


def _wait_for_training(sm_client, job_name: str) -> dict:
    """Poll describe_training_job every 20 s until terminal or timeout."""
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
def test_xgboost_iris_submits_completes_and_logs_metrics(
    skill_handler, gateway_ctx, sm_client, ddb_table
):
    sm_h = skill_handler("sagemaker")

    thread_id = f"functest-xgboost-{uuid.uuid4()}"
    event = {
        "thread_id": thread_id,
        # model_id is a slug for tabular training — it only shows up in
        # the SageMaker job name.
        "model_id": "xgboost",
        # Bundled sklearn dataset — loaded in-process from
        # sklearn.datasets.load_iris, no network I/O.
        "dataset_name": "iris",
        "target_column": "target",
        "training_type": "xgboost",
        "instance_type": "ml.m5.xlarge",
        "learning_rate": 0.1,
        "max_depth": 4,
        "n_estimators": 100,
        "_user_id": "functest",
    }

    resp = sm_h.handler(event, gateway_ctx("sagemaker-skill___submit_training_job"))
    assert resp.get("isError") is not True, resp
    body = json.loads(resp["content"][0]["text"])
    assert body["status"] == "SUBMITTING"
    job_name = body["sagemaker_job_name"]
    assert job_name.startswith(f"{sm_h.PROJECT_NAME}-job-")
    job_id = body["job_id"]
    mlflow_run_id = body["mlflow_run_id"]
    assert mlflow_run_id, "handler must pre-create an MLflow run before submit"

    # Short grace period for CreateTrainingJob to land.
    time.sleep(10)

    desc = _wait_for_training(sm_client, job_name)
    assert desc["TrainingJobStatus"] == "Completed", (
        f"Training failed: {desc.get('FailureReason')!r}. "
        f"Transitions: {desc.get('SecondaryStatusTransitions')}"
    )
    assert desc["ModelArtifacts"]["S3ModelArtifacts"].startswith("s3://")

    # DDB row + job sub-record should reflect tabular submission.
    row = ddb_table.get_item(Key={"task_id": thread_id}).get("Item") or {}
    job_entry = (row.get("jobs") or {}).get(job_id) or {}
    assert job_entry.get("kind") == "training"
    assert job_entry.get("training_type") == "xgboost"
    assert job_entry.get("sagemaker_job_name") == job_name

    # Guard against a silent MLflow regression: the run must carry
    # per-round metrics (eval-mlogloss, train-mlogloss) logged by the
    # xgb_train.py MLflow callback. Iris is 3-class so the objective
    # defaults to multi:softprob / eval_metric=mlogloss.
    import mlflow  # noqa: PLC0415 — local import so suite still skips cleanly when infra absent
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    run = MlflowClient().get_run(mlflow_run_id)
    metric_keys = list(run.data.metrics.keys())
    assert any("mlogloss" in k.lower() or "logloss" in k.lower() for k in metric_keys), (
        f"MLflow run {mlflow_run_id} has no logloss-family metric. "
        f"xgb_train.py MLflow callback may not be running. All metrics: {metric_keys}"
    )
    # Sanity-check hyperparams were logged.
    params = run.data.params
    assert params.get("dataset_name") == "iris"
    assert params.get("target_column") == "target"
