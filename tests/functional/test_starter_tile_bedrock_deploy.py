"""Functional test for R5 — deploy_model target="bedrock".

Discovers the most recent Completed SFT fine-tune (cross-session, via the
list_recent_training_jobs tool), kicks off a Bedrock Custom Model Import
against its artifact, and polls the import job to a terminal state.

Wall-clock ≈ 10-25 min (import time for a ~1 GB 0.5B checkpoint). The
imported model is deleted afterwards — the test verifies the pipeline, not
the model. Skips when no Completed SFT job exists in the account.
"""
import json
import time
import uuid

import boto3
import pytest

_IMPORT_TIMEOUT_SEC = 35 * 60
_POLL_INTERVAL_SEC = 30


def _latest_completed_sft(sm_h, gateway_ctx) -> str:
    """Resolve the newest Completed SFT job via the discovery tool (BUG-004)."""
    resp = sm_h.handler(
        {"status_equals": "Completed", "_user_id": "functest"},
        gateway_ctx("sagemaker-skill___list_recent_training_jobs"),
    )
    assert resp.get("isError") is not True, resp
    jobs = json.loads(resp["content"][0]["text"])["jobs"]
    for job in jobs:  # newest first
        if job.get("training_type") == "sft":
            return job["sagemaker_job_name"]
    pytest.skip("no Completed SFT training job in the account — run the Quick fine-tune tile first")


@pytest.mark.functional
def test_bedrock_deploy_imports_sft_artifact(skill_handler, gateway_ctx, aws_region):
    sm_h = skill_handler("sagemaker")
    source_job = _latest_completed_sft(sm_h, gateway_ctx)

    # Deliberately NO thread_id: seeding one makes the 15-min poller resume a
    # phantom agent session when the import completes (observed live: the
    # resumed agent then autonomously submitted evals against the imported
    # model). The test polls the Bedrock API directly instead.
    model_name = f"functest-import-{uuid.uuid4().hex[:8]}"
    resp = sm_h.handler(
        {
            "sagemaker_job_name": source_job,
            "target": "bedrock",
            "bedrock_model_name": model_name,
            "_user_id": "functest",
        },
        gateway_ctx("sagemaker-skill___deploy_model"),
    )
    assert resp.get("isError") is not True, resp
    body = json.loads(resp["content"][0]["text"])
    assert body["target"] == "bedrock"
    assert body["bedrock_model_name"] == model_name
    # Async contract (R5 closure): the tool returns SUBMITTING immediately;
    # the background worker unpacks the artifact to an HF prefix and creates
    # the import job named after bedrock_model_name.
    assert body["status"] == "SUBMITTING"
    import_job = model_name

    # Poll the import job to terminal. Bedrock emits no EventBridge event for
    # import jobs (the 15-min poller Lambda handles thread resume in prod);
    # the functional test polls the API directly for determinism. The first
    # get may 404 briefly while the (inline-fallback) worker finishes.
    bedrock = boto3.client("bedrock", region_name=aws_region)
    deadline = time.time() + _IMPORT_TIMEOUT_SEC
    status, desc = "", {}
    while time.time() < deadline:
        try:
            desc = bedrock.get_model_import_job(jobIdentifier=import_job)
        except bedrock.exceptions.ResourceNotFoundException:
            time.sleep(_POLL_INTERVAL_SEC)
            continue
        status = desc["status"]
        if status in ("Completed", "Failed"):
            break
        time.sleep(_POLL_INTERVAL_SEC)
    assert status == "Completed", (
        f"import job {import_job} status={status!r} "
        f"failureMessage={desc.get('failureMessage')!r}"
    )

    # Cleanup: drop the imported model — the pipeline is what's under test.
    imported_arn = desc.get("importedModelArn", "")
    if imported_arn:
        try:
            bedrock.delete_imported_model(modelIdentifier=imported_arn)
        except Exception:  # nosec B110
            pass  # best-effort; leftover imported models cost only storage
