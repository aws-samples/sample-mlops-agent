"""Functional test for R6 — submit_eval_job with a custom scorer Lambda.

Runs a 2-row financebench eval with the built-in Correctness judge PLUS the
deployed reference scorer (`<project>-math-scorer-ref`), then asserts the
custom scorer produced a per-row value in the eval results artifact.

Wall-clock ≈ 8-12 min. Requires the reference scorer Lambda (deployed by the
gateway stack) — skips with a clear message if it is absent.
"""
import json
import time
import uuid

import boto3
import pytest

_EVAL_TIMEOUT_SEC = 15 * 60
_POLL_INTERVAL_SEC = 30


def _scorer_arn(aws_region: str, project_name: str) -> str:
    lam = boto3.client("lambda", region_name=aws_region)
    name = f"{project_name}-math-scorer-ref"
    try:
        return lam.get_function(FunctionName=name)["Configuration"]["FunctionArn"]
    except lam.exceptions.ResourceNotFoundException:
        pytest.skip(f"reference scorer Lambda {name!r} not deployed")


def _wait_terminal(sm_client, job_name: str) -> str:
    terminal = {"Completed", "Failed", "Stopped"}
    deadline = time.time() + _EVAL_TIMEOUT_SEC
    status = ""
    while time.time() < deadline:
        status = sm_client.describe_processing_job(
            ProcessingJobName=job_name)["ProcessingJobStatus"]
        if status in terminal:
            return status
        time.sleep(_POLL_INTERVAL_SEC)
    raise AssertionError(f"{job_name} not terminal after {_EVAL_TIMEOUT_SEC}s ({status=})")


def _fetch_eval_results(aws_region: str, project_name: str, run_id: str) -> list[dict]:
    """Locate eval_results.json for the run in the MLflow artifact store."""
    s3 = boto3.client("s3", region_name=aws_region)
    sts = boto3.client("sts", region_name=aws_region)
    account = sts.get_caller_identity()["Account"]
    bucket = f"{project_name}-sessions-{account}-{aws_region}"
    token, key = None, ""
    while True:
        kwargs = {"Bucket": bucket, "Prefix": "mlflow-artifacts/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        for obj in page.get("Contents", []):
            if run_id in obj["Key"] and obj["Key"].endswith("eval_results.json"):
                key = obj["Key"]
                break
        if key or not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    assert key, f"eval_results.json for run {run_id} not found in s3://{bucket}"
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


@pytest.mark.functional
def test_eval_with_custom_math_scorer_logs_metric(
    skill_handler, gateway_ctx, sm_client, aws_region, project_name
):
    arn = _scorer_arn(aws_region, project_name)

    # Step 1 — eval spec (2 rows keeps judge cost trivial).
    hf_h = skill_handler("huggingface")
    prep = hf_h.handler(
        {"task_type": "question_answering", "dataset_name": "PatronusAI/financebench",
         "split": "train", "max_rows": 2},
        gateway_ctx("huggingface-skill___prepare_eval_dataset"),
    )
    assert prep.get("isError") is not True, prep
    spec_s3_uri = json.loads(prep["content"][0]["text"])["spec_s3_uri"]

    # Step 2 — submit with the custom scorer alongside a built-in.
    sm_h = skill_handler("sagemaker")
    resp = sm_h.handler(
        {
            "thread_id": f"functest-scorer-{uuid.uuid4()}",
            "eval_dataset_s3_uri": spec_s3_uri,
            "target_model": "bedrock:/us.amazon.nova-lite-v1:0",
            "judge_model": "bedrock:/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "scorers": ["Correctness"],
            "custom_scorer_lambda_arns": [arn],
            "task": "question_answering",
            "instance_type": "ml.m5.large",
            "_user_id": "functest",
        },
        gateway_ctx("sagemaker-skill___submit_eval_job"),
    )
    assert resp.get("isError") is not True, resp
    body = json.loads(resp["content"][0]["text"])
    proc_name = body["processing_job_name"]
    run_id = body["mlflow_run_id"]

    status = _wait_terminal(sm_client, proc_name)
    assert status == "Completed", f"eval job {proc_name} ended {status!r}"

    # Step 3 — the custom scorer must have produced per-row values.
    rows = _fetch_eval_results(aws_region, project_name, run_id)
    cols = set().union(*(r.keys() for r in rows))
    # The MLflow metric key is the scorer callable's __name__, which the
    # wrapper derives from the Lambda ARN's function-name segment.
    mc_cols = [c for c in cols if "math-scorer-ref" in c]
    assert mc_cols, f"no custom-scorer columns in eval results; columns={sorted(cols)}"
    valued = [r for r in rows if any(r.get(c) is not None for c in mc_cols)]
    assert valued, "custom scorer produced no values on any row"
