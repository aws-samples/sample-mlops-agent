"""Functional test for the 'Benchmark fine-tuned LLM' starter tile.

Two-phase end-to-end:
  1. Submit SFT fine-tune on Qwen 2.5-0.5B (reuses the fine-tune handler);
     honours `FUNCTEST_SFT_JOB_NAME` env var to skip training and reuse
     an already-Completed SFT job (~$0.80 + ~10 min savings per run).
  2. Submit recommendation on ml.g6.xlarge with default workload.
     Poll `get_recommendation_results` every 60 s (simulates the agent-
     driven polling cadence) until terminal, up to 60 min. Assert metrics
     are non-empty, teardown_complete is True, and the temporary endpoint
     is actually gone.

Skips cleanly when infra env vars are absent (same `_guard_infra` as the
other functional tests)."""
import json
import os
import time
import uuid
import pytest


_TRAINING_TIMEOUT_SEC  = 30 * 60   # SFT is slower than iris XGBoost
_BENCHMARK_TIMEOUT_SEC = 60 * 60   # endpoint provision + benchmark
_POLL_INTERVAL_SEC     = 60        # one get_recommendation_results call / min


def _wait_for_training(sm_client, job_name):
    terminal = {"Completed", "Failed", "Stopped"}
    deadline = time.time() + _TRAINING_TIMEOUT_SEC
    while time.time() < deadline:
        d = sm_client.describe_training_job(TrainingJobName=job_name)
        if d["TrainingJobStatus"] in terminal:
            return d
        time.sleep(_POLL_INTERVAL_SEC)
    raise AssertionError(f"Training {job_name} did not reach terminal state")


def _poll_get_recommendation_results(sm_handler, gateway_ctx_fn, *,
                                     thread_id, recommender_job_name):
    """Simulates the agent-driven polling cadence: call get_recommendation_results
    every _POLL_INTERVAL_SEC until it returns COMPLETED / FAILED or we time out.
    Mirrors what the starter-tile prompt tells the agent to do (one call per
    user message), just compressed to a tight loop for CI."""
    terminal = {"COMPLETED", "FAILED"}
    deadline = time.time() + _BENCHMARK_TIMEOUT_SEC
    last_body = None
    while time.time() < deadline:
        resp = sm_handler.handler(
            {"thread_id": thread_id,
             "recommender_job_name": recommender_job_name,
             "_user_id": "functest"},
            gateway_ctx_fn("sagemaker-skill___get_recommendation_results"))
        assert resp.get("isError") is not True, resp
        last_body = json.loads(resp["content"][0]["text"])
        if last_body.get("status") in terminal:
            return last_body
        time.sleep(_POLL_INTERVAL_SEC)
    raise AssertionError(
        f"Recommendation {recommender_job_name} did not reach terminal "
        f"in {_BENCHMARK_TIMEOUT_SEC}s. Last body: {last_body}")


@pytest.mark.functional
def test_recommendation_against_sft_qwen(
    skill_handler, gateway_ctx, sm_client, ddb_table,
):
    sm_h = skill_handler("sagemaker")
    thread_id = f"functest-rec-{uuid.uuid4()}"

    # ---- Phase 1: training (reuse path) ----
    existing_job = os.environ.get("FUNCTEST_SFT_JOB_NAME", "").strip()
    if existing_job:
        d = sm_client.describe_training_job(TrainingJobName=existing_job)
        assert d["TrainingJobStatus"] == "Completed", \
            f"FUNCTEST_SFT_JOB_NAME={existing_job!r} is " \
            f"{d['TrainingJobStatus']!r}, need Completed"
        sagemaker_job_name = existing_job
        # Seed a minimal jobs.seeded entry so the handler lookup finds
        # training_type=sft under this thread.
        ddb_table.update_item(
            Key={"task_id": thread_id},
            UpdateExpression=("SET jobs = :j, user_id = :u, thread_id = :t, "
                              "created_at = :c, updated_at = :c"),
            ExpressionAttributeValues={
                ":j": {"seeded": {
                    "job_id": "seeded", "kind": "training",
                    "training_type": "sft",
                    "sagemaker_job_name": existing_job,
                }},
                ":u": "functest", ":t": thread_id, ":c": int(time.time()),
            },
        )
    else:
        train_event = {
            "thread_id": thread_id, "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "dataset_name": "HuggingFaceH4/ultrachat_200k",
            "train_split": "train_sft", "test_split": "test_sft",
            "training_type": "sft", "instance_type": "ml.g5.2xlarge",
            # F-8 from eng review: match test_starter_tile_fine_tune.py
            # params exactly — minimum-viable SFT that still produces a
            # model artifact the benchmark path can load.
            "max_steps": 1, "max_samples": 8,
            "learning_rate": 1e-5, "_user_id": "functest",
        }
        train_resp = sm_h.handler(
            train_event, gateway_ctx("sagemaker-skill___submit_training_job"))
        assert train_resp.get("isError") is not True, train_resp
        train_body = json.loads(train_resp["content"][0]["text"])
        sagemaker_job_name = train_body["sagemaker_job_name"]
        time.sleep(10)
        train_desc = _wait_for_training(sm_client, sagemaker_job_name)
        assert train_desc["TrainingJobStatus"] == "Completed", \
            f"Training failed: {train_desc.get('FailureReason')!r}"

    # ---- Phase 2: recommendation ----
    # NOTE: the starter-tile prompt defaults to ml.g6.xlarge, but this test
    # account has ml.g6.xlarge endpoint quota = 0 (verified via
    # service-quotas). Use ml.g5.xlarge which has quota 4 in this account.
    rec_event = {
        "thread_id": thread_id, "sagemaker_job_name": sagemaker_job_name,
        "instance_type": "ml.g5.xlarge", "_user_id": "functest",
    }
    rec_resp = sm_h.handler(
        rec_event, gateway_ctx("sagemaker-skill___submit_recommendation_job"))
    assert rec_resp.get("isError") is not True, rec_resp
    rec_body = json.loads(rec_resp["content"][0]["text"])
    assert rec_body["status"] == "SUBMITTING"
    recommender_job_name = rec_body["recommender_job_name"]
    endpoint_name = rec_body["endpoint_name"]
    rec_job_id = rec_body["job_id"]

    # Give the background Lambda ~10 s to create the endpoint, then drive
    # the agent-polling cadence (call get_recommendation_results repeatedly)
    # until terminal. Because AWS does NOT emit a terminal EventBridge
    # event for AI Benchmark jobs, this tool call is the ONLY place that
    # discovers completion — which also parses metrics + tears down resources.
    time.sleep(10)
    res_body = _poll_get_recommendation_results(
        sm_h, gateway_ctx,
        thread_id=thread_id, recommender_job_name=recommender_job_name)

    # ---- Assertions ----
    assert res_body["status"] == "COMPLETED", \
        f"recommendation status={res_body['status']!r}; summary={res_body.get('summary')!r}"
    metrics = res_body.get("metrics", {})
    assert isinstance(metrics.get("ttft_ms_p99"), (int, float)) and metrics["ttft_ms_p99"] > 0, \
        f"ttft_ms_p99 missing/zero: {metrics}"
    assert isinstance(metrics.get("throughput_tokens_per_sec"), (int, float)) \
        and metrics["throughput_tokens_per_sec"] > 0, \
        f"throughput_tokens_per_sec missing/zero: {metrics}"
    assert res_body["teardown_complete"] is True, \
        "teardown_complete=False — endpoint may still be billing"

    # Endpoint really gone
    with pytest.raises(Exception) as exc_info:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
    assert "Could not find endpoint" in str(exc_info.value) or \
           "ValidationException" in str(exc_info.value)

    # DDB shape
    row = ddb_table.get_item(Key={"task_id": thread_id}).get("Item") or {}
    rec_entry = (row.get("jobs") or {}).get(rec_job_id) or {}
    assert rec_entry.get("kind") == "recommendation"
    assert rec_entry.get("source_training_job") == sagemaker_job_name
    assert rec_entry.get("teardown_complete") is True
