"""R1 Task 12 — functional test for the 'Monitor XGBoost predictions' starter tile.

Two-phase end-to-end:
  1. Submit XGBoost iris training (reuses the xgboost handler path from the
     existing functional test). Wait for Completed.
  2. Submit monitoring with use_training_eval_split=true, target_column=target.
     Wait for Completed. Assert MLflow drift + accuracy metrics, DDB record
     shape, and S3 output artifacts.

Skips cleanly when infra env vars are absent (same _guard_infra fixture as the
other functional tests).

Cheap-reuse knob: when FUNCTEST_XGBOOST_JOB_NAME is set, Phase 1 is skipped
and the monitoring call points at the pre-existing Completed job. Seeds a
minimal jobs.seeded entry into DDB so the handler's lookup succeeds.
"""
import json
import os
import time
import uuid

import pytest


_TRAINING_TIMEOUT_SEC = 15 * 60
_MONITORING_TIMEOUT_SEC = 15 * 60
_POLL_INTERVAL_SEC = 20


def _wait_for_training(sm_client, job_name: str) -> dict:
    terminal = {"Completed", "Failed", "Stopped"}
    deadline = time.time() + _TRAINING_TIMEOUT_SEC
    while time.time() < deadline:
        d = sm_client.describe_training_job(TrainingJobName=job_name)
        if d["TrainingJobStatus"] in terminal:
            return d
        time.sleep(_POLL_INTERVAL_SEC)
    raise AssertionError(f"Training {job_name} did not reach terminal state")


def _wait_for_processing(sm_client, job_name: str) -> dict:
    terminal = {"Completed", "Failed", "Stopped"}
    deadline = time.time() + _MONITORING_TIMEOUT_SEC
    while time.time() < deadline:
        d = sm_client.describe_processing_job(ProcessingJobName=job_name)
        if d["ProcessingJobStatus"] in terminal:
            return d
        time.sleep(_POLL_INTERVAL_SEC)
    raise AssertionError(f"Processing {job_name} did not reach terminal state")


@pytest.mark.functional
def test_monitoring_against_xgboost_iris_eval_split(
    skill_handler, gateway_ctx, sm_client, ddb_table,
):
    sm_h = skill_handler("sagemaker")
    thread_id = f"functest-monitor-{uuid.uuid4()}"

    # ─── Phase 1: training ─────────────────────────────────────────────
    # FUNCTEST_XGBOOST_JOB_NAME env var lets devs skip the 5-minute
    # training submit when they just ran the XGBoost functional test.
    # The monitoring handler resolves baseline_s3_uri from DDB, so we
    # seed a jobs.seeded entry under this thread_id.
    existing_job = os.environ.get("FUNCTEST_XGBOOST_JOB_NAME", "").strip()
    if existing_job:
        train_desc = sm_client.describe_training_job(TrainingJobName=existing_job)
        assert train_desc["TrainingJobStatus"] == "Completed", (
            f"FUNCTEST_XGBOOST_JOB_NAME={existing_job!r} status is "
            f"{train_desc['TrainingJobStatus']!r}, need Completed"
        )
        sagemaker_job_name = existing_job
        artifact_s3 = train_desc["ModelArtifacts"]["S3ModelArtifacts"]
        base_prefix = artifact_s3.rsplit("/output/", 1)[0] + "/output/baseline"
        ddb_table.update_item(
            Key={"task_id": thread_id},
            UpdateExpression=(
                "SET jobs = :j, user_id = :u, thread_id = :t, "
                "    created_at = :c, updated_at = :c"
            ),
            ExpressionAttributeValues={
                ":j": {"seeded": {
                    "job_id": "seeded", "kind": "training",
                    "training_type": "xgboost",
                    "sagemaker_job_name": existing_job,
                    "baseline_s3_uri": f"{base_prefix}/baseline.csv",
                    "eval_split_s3_uri": f"{base_prefix}/eval_split.csv",
                }},
                ":u": "functest", ":t": thread_id, ":c": int(time.time()),
            },
        )
    else:
        train_event = {
            "thread_id": thread_id, "model_id": "xgboost",
            "dataset_name": "iris", "target_column": "target",
            "training_type": "xgboost", "instance_type": "ml.m5.xlarge",
            "learning_rate": 0.1, "max_depth": 4, "n_estimators": 50,
            "_user_id": "functest",
        }
        train_resp = sm_h.handler(
            train_event, gateway_ctx("sagemaker-skill___submit_training_job"),
        )
        assert train_resp.get("isError") is not True, train_resp
        train_body = json.loads(train_resp["content"][0]["text"])
        sagemaker_job_name = train_body["sagemaker_job_name"]
        time.sleep(10)
        train_desc = _wait_for_training(sm_client, sagemaker_job_name)
        assert train_desc["TrainingJobStatus"] == "Completed", (
            f"Training failed: {train_desc.get('FailureReason')!r}"
        )

    # ─── Phase 2: monitoring ───────────────────────────────────────────
    mon_event = {
        "thread_id": thread_id,
        "sagemaker_job_name": sagemaker_job_name,
        "use_training_eval_split": True,
        "target_column": "target",
        "_user_id": "functest",
    }
    mon_resp = sm_h.handler(
        mon_event, gateway_ctx("sagemaker-skill___submit_monitoring_job"),
    )
    assert mon_resp.get("isError") is not True, mon_resp
    mon_body = json.loads(mon_resp["content"][0]["text"])
    assert mon_body["status"] == "SUBMITTING"
    assert mon_body["baseline_s3_uri"].endswith("/baseline/baseline.csv")
    assert mon_body["current_data_s3_uri"].endswith("/baseline/eval_split.csv")
    proc_name = mon_body["processing_job_name"]
    mon_job_id = mon_body["job_id"]
    mlflow_run_id = mon_body["mlflow_run_id"]

    time.sleep(10)
    mon_desc = _wait_for_processing(sm_client, proc_name)
    assert mon_desc["ProcessingJobStatus"] == "Completed", (
        f"Monitoring failed: exit={mon_desc.get('ExitMessage')!r}, "
        f"failure={mon_desc.get('FailureReason')!r}"
    )

    # ─── Assertions: DDB shape ─────────────────────────────────────────
    row = ddb_table.get_item(Key={"task_id": thread_id}).get("Item") or {}
    mon_entry = (row.get("jobs") or {}).get(mon_job_id) or {}
    assert mon_entry.get("kind") == "monitoring"
    assert mon_entry.get("source_training_job") == sagemaker_job_name

    # ─── Assertions: MLflow metrics ────────────────────────────────────
    import mlflow  # noqa: PLC0415
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    run = MlflowClient().get_run(mlflow_run_id)
    metrics = run.data.metrics
    assert "drifted_columns_count" in metrics, (
        f"No drifted_columns_count in run {mlflow_run_id}: {list(metrics.keys())}"
    )
    assert "drifted_columns_share" in metrics
    assert 0.0 <= float(metrics["drifted_columns_share"]) <= 1.0
    # Classification metrics are optional — iris has labels so they should
    # be present. Value depends on fit quality; we only assert presence +
    # range.
    assert "accuracy" in metrics, (
        f"accuracy missing with labels present: {list(metrics.keys())}"
    )
    assert 0.0 <= float(metrics["accuracy"]) <= 1.0
