"""Phase 3 Gateway + Cedar tests.

Covers:
  T3.2 — Interceptor extracts _user_id from tool arguments and injects
          _injected_user_id into the transformed Gateway request
  T3.3 — Authorized call: SageMaker skill handler dispatches submit_training_job
          and creates DynamoDB row with user_id
  T3.4 — Unauthorized tool: handler returns isError for unknown tool names
          (Cedar deny path produces an error response — not a Lambda exception)
  T3.5 — Per-skill dispatch: each of the 4 skill Lambdas dispatches only its
          own tools; cross-skill tool names return isError
  T3.6 — M2M token exchange shape: the Secrets Manager secret ARN stored in SSM
          is a non-empty string (configuration hygiene check)

T3.0 (health check URL), T3.1 (agent MCP connection), T3.6 (live curl to
Cognito token endpoint) are live-infra checks that cannot run in unit tests.
"""
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3 as real_boto3
import pytest
from moto import mock_aws

# ── paths for deployed Lambda code ───────────────────────────────────────────
_INTERCEPTOR_DIR = Path("/tmp/interceptor_code")  # nosec B108
_SM_SKILL_DIR = Path("/tmp/sagemaker_skill_code")  # nosec B108
_HF_SKILL_DIR = Path("/tmp/hf_skill_code")  # nosec B108
_GIT_SKILL_DIR = Path("/tmp/git_skill_code")  # nosec B108
_MLFLOW_SKILL_DIR = Path("/tmp/mlflow_skill_code")  # nosec B108

for _p in (_INTERCEPTOR_DIR, _SM_SKILL_DIR, _HF_SKILL_DIR, _GIT_SKILL_DIR, _MLFLOW_SKILL_DIR):
    if not _p.exists():
        pytest.skip(
            reason=(
                f"Gateway Lambda code not found at {_p}. "
                "Download and extract the Lambda deployment packages first — "
                "see the Pre-Test Checklist in docs/plans/2026-04-15-composite-identity-test-plan.md"
            ),
            allow_module_level=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# T3.2 — Interceptor: _user_id → _injected_user_id
# ─────────────────────────────────────────────────────────────────────────────

def _load_interceptor():
    sys.path.insert(0, str(_INTERCEPTOR_DIR))
    import handler as interceptor
    importlib.reload(interceptor)
    return interceptor


def test_interceptor_injects_user_id_into_transformed_request():
    """Interceptor must extract _user_id from tool arguments and inject it as
    _injected_user_id in the transformed Gateway request params."""
    interceptor = _load_interceptor()

    event = {
        "mcp": {
            "gatewayRequest": {
                "body": {
                    "method": "tools/call",
                    "params": {
                        "name": "submit_training_job",
                        "arguments": {
                            "session_id": "sess-1",
                            "model_id": "Qwen/Q2.5",
                            "dataset_name": "ds",
                            "_user_id": "us-east-1:user-abc",
                        },
                    },
                }
            }
        }
    }

    result = interceptor.handler(event, {})

    transformed = (
        result.get("mcp", {})
              .get("transformedGatewayRequest", {})
              .get("body", {})
    )
    assert transformed, "T3.2: transformedGatewayRequest body must be present"
    assert transformed.get("params", {}).get("_injected_user_id") == "us-east-1:user-abc", (
        "T3.2: _injected_user_id must be set to the value of _user_id from arguments"
    )


def test_interceptor_injects_empty_string_when_user_id_absent():
    """When _user_id is absent from arguments, _injected_user_id must be an
    empty string — not omitted or set to None."""
    interceptor = _load_interceptor()

    event = {
        "mcp": {
            "gatewayRequest": {
                "body": {
                    "params": {
                        "name": "submit_training_job",
                        "arguments": {"session_id": "sess-2"},
                    }
                }
            }
        }
    }

    result = interceptor.handler(event, {})
    injected = (
        result.get("mcp", {})
              .get("transformedGatewayRequest", {})
              .get("body", {})
              .get("params", {})
              .get("_injected_user_id")
    )
    assert injected == "", (
        "T3.2: _injected_user_id must be empty string when _user_id is absent, not None"
    )


def test_interceptor_preserves_original_body_fields():
    """Interceptor must not drop any fields from the original request body."""
    interceptor = _load_interceptor()

    body = {
        "method": "tools/call",
        "jsonrpc": "2.0",
        "id": 42,
        "params": {
            "name": "upload_model",
            "arguments": {
                "repo_id": "user/model",
                "_user_id": "uid-xyz",
            },
        },
    }
    event = {"mcp": {"gatewayRequest": {"body": body}}}
    result = interceptor.handler(event, {})
    transformed_body = (
        result["mcp"]["transformedGatewayRequest"]["body"]
    )

    # All original top-level keys must be preserved
    for key in ("method", "jsonrpc", "id"):
        assert key in transformed_body, (
            f"T3.2: interceptor must preserve '{key}' from original request body"
        )


def test_interceptor_output_version():
    """Interceptor response must include interceptorOutputVersion='1.0'."""
    interceptor = _load_interceptor()
    event = {"mcp": {"gatewayRequest": {"body": {"params": {"arguments": {}}}}}}
    result = interceptor.handler(event, {})
    assert result.get("interceptorOutputVersion") == "1.0", (
        "T3.2: interceptorOutputVersion must be '1.0'"
    )


def test_interceptor_handles_missing_mcp_key_gracefully():
    """Interceptor must not raise when the 'mcp' key is absent from the event.
    It should still produce a valid (empty-body) transformed response."""
    interceptor = _load_interceptor()
    result = interceptor.handler({}, {})
    # Should return a valid structure without crashing
    assert "mcp" in result
    assert "interceptorOutputVersion" in result


# ─────────────────────────────────────────────────────────────────────────────
# T3.3 — SageMaker skill: authorized submit_training_job creates DDB row
# ─────────────────────────────────────────────────────────────────────────────

def _load_sm_handler():
    sys.path.insert(0, str(_SM_SKILL_DIR))
    # Stub heavy deps before import
    with patch.dict("sys.modules", {
        "datasets": MagicMock(),
    }):
        import handler as sm_handler
        importlib.reload(sm_handler)
    return sm_handler


@mock_aws
def test_sm_submit_training_job_writes_user_id_to_dynamodb():
    """submit_training_job must create a DynamoDB row that includes user_id."""
    # Create moto DynamoDB table
    ddb = real_boto3.resource("dynamodb", region_name="us-east-1")
    ddb.create_table(
        TableName="sample-mlops-agent-metadata",
        KeySchema=[{"AttributeName": "task_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "task_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    table = ddb.Table("sample-mlops-agent-metadata")

    # Create moto S3 bucket for training scripts
    real_boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="sbkt-sm")

    sm_handler = _load_sm_handler()

    mock_sm_client = MagicMock()
    mock_sm_client.create_training_job.return_value = {}

    with patch.dict("os.environ", {
        "JOBS_TABLE": "sample-mlops-agent-metadata",
        "SESSION_BUCKET": "sbkt-sm",
        "SAGEMAKER_EXECUTION_ROLE_ARN": "arn:aws:iam::123:role/sm",
        "AWS_REGION": "us-east-1",
        "PROJECT_NAME": "sample-mlops-agent",
    }), patch("handler.boto3", real_boto3), \
       patch.object(sm_handler, "_upload_training_script", return_value="s3://sbkt-sm/training-scripts/sft_train.py"), \
       patch("handler.boto3.client") as mock_boto_client:

        mock_boto_client.return_value = mock_sm_client

        event = {
            "params": {
                "name": "submit_training_job",
                "arguments": {
                    "session_id": "sess-t33",
                    "model_id": "Qwen/Qwen2.5",
                    "dataset_name": "trl-lib/tldr",
                    "training_type": "sft",
                    "_user_id": "us-east-1:user-t33",
                },
            }
        }

        # Re-import with real DDB for this test
        with patch("handler.boto3.resource", real_boto3.resource):
            result = sm_handler.handler(event, {})

    assert result.get("isError") is not True, (
        f"T3.3: submit_training_job returned an error: {result}"
    )
    content_text = result.get("content", [{}])[0].get("text", "")
    out = json.loads(content_text)
    # Unified-thread-row schema: one row keyed by task_id (= thread_id),
    # with each submission recorded under jobs.<job_id>.
    assert out.get("thread_id") == "sess-t33", "T3.3: result must contain thread_id"
    assert "job_id" in out, "T3.3: result must contain job_id"
    assert "sagemaker_job_name" in out, "T3.3: result must contain sagemaker_job_name"

    row = table.get_item(Key={"task_id": "sess-t33"}).get("Item") or {}
    assert row.get("user_id") == "us-east-1:user-t33", (
        "T3.3: thread row must carry the user_id passed in _user_id"
    )
    jobs = row.get("jobs") or {}
    assert out["job_id"] in jobs, (
        "T3.3: submitted job must be written under jobs.<job_id>"
    )
    entry = jobs[out["job_id"]]
    assert entry.get("sagemaker_job_name") == out["sagemaker_job_name"]
    assert entry.get("model_id") == "Qwen/Qwen2.5"
    assert entry.get("dataset_name") == "trl-lib/tldr"


def _capture_update_items(mock_table) -> list[dict]:
    """Capture update_item kwargs on a mock DynamoDB Table."""
    captured: list[dict] = []
    mock_table.update_item.side_effect = lambda **kw: captured.append(kw) or {}
    return captured


def test_sm_submit_training_job_includes_user_id_in_dynamodb_item():
    """submit_training_job must seed user_id on the thread row via the initial
    update_item pre-pass. Unified-thread-row schema uses update_item, not put_item."""
    sm_handler = _load_sm_handler()

    mock_boto = MagicMock()
    mock_ddb = MagicMock()
    mock_table = MagicMock()
    mock_ddb.Table.return_value = mock_table
    mock_boto.resource.return_value = mock_ddb
    mock_boto.client.return_value = MagicMock()

    updates = _capture_update_items(mock_table)

    with patch.dict("os.environ", {
        "JOBS_TABLE": "tbl",
        "SESSION_BUCKET": "bkt",
        "SAGEMAKER_EXECUTION_ROLE_ARN": "arn:aws:iam::123:role/sm",
        "AWS_REGION": "us-east-1",
        "PROJECT_NAME": "sample-mlops-agent",
    }), patch("handler.boto3", mock_boto), \
       patch.object(sm_handler, "_upload_training_script", return_value="s3://bkt/script.py"):

        sm_handler._submit_training_job({
            "session_id": "sess-1",
            "model_id": "Qwen/Qwen2.5",
            "dataset_name": "ds",
            "training_type": "sft",
            "_user_id": "us-east-1:user-per-skill",
        })

    assert updates, "T3.3: update_item must be called"
    # First update is the pre-pass that seeds user_id via if_not_exists
    pre_pass = updates[0]
    assert pre_pass["Key"] == {"task_id": "sess-1"}
    assert pre_pass["ExpressionAttributeValues"].get(":uid") == "us-east-1:user-per-skill", (
        "T3.3: pre-pass update_item must seed user_id with the _user_id argument"
    )


def test_sm_submit_training_job_stores_empty_user_id_when_absent():
    """When _user_id is absent from arguments, user_id seeded onto the thread
    row must be an empty string — not omitted."""
    sm_handler = _load_sm_handler()

    mock_boto = MagicMock()
    mock_ddb = MagicMock()
    mock_table = MagicMock()
    mock_ddb.Table.return_value = mock_table
    mock_boto.resource.return_value = mock_ddb
    mock_boto.client.return_value = MagicMock()

    updates = _capture_update_items(mock_table)

    with patch.dict("os.environ", {
        "JOBS_TABLE": "tbl",
        "SESSION_BUCKET": "bkt",
        "SAGEMAKER_EXECUTION_ROLE_ARN": "arn:aws:iam::123:role/sm",
        "AWS_REGION": "us-east-1",
        "PROJECT_NAME": "sample-mlops-agent",
    }), patch("handler.boto3", mock_boto), \
       patch.object(sm_handler, "_upload_training_script", return_value="s3://bkt/script.py"):

        sm_handler._submit_training_job({
            "session_id": "sess-no-uid",
            "model_id": "Qwen/Qwen2.5",
            "dataset_name": "ds",
        })

    assert updates, "update_item must be called"
    pre_pass = updates[0]
    assert ":uid" in pre_pass["ExpressionAttributeValues"], (
        "user_id must always be seeded (empty string if absent)"
    )
    assert pre_pass["ExpressionAttributeValues"][":uid"] == "", (
        "user_id must be empty string when _user_id is absent from arguments"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T3.4 — Unknown tool returns isError (Cedar deny path analogue)
# ─────────────────────────────────────────────────────────────────────────────

def test_sm_handler_unknown_tool_returns_is_error():
    """SageMaker handler must return isError for unrecognised tool names."""
    sm_handler = _load_sm_handler()
    event = {"params": {"name": "delete_training_job", "arguments": {}}}
    result = sm_handler.handler(event, {})
    assert result.get("isError") is True, (
        "T3.4: Unknown tool must return isError=True"
    )


def test_hf_handler_unknown_tool_returns_is_error():
    """HuggingFace handler must return isError for unrecognised tool names."""
    sys.path.insert(0, str(_HF_SKILL_DIR))
    with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}):
        import handler as hf_handler
        importlib.reload(hf_handler)

    event = {"params": {"name": "submit_training_job", "arguments": {}}}
    result = hf_handler.handler(event, {})
    assert result.get("isError") is True, (
        "T3.4: HuggingFace handler must not dispatch SageMaker tool names"
    )


def test_git_handler_unknown_tool_returns_is_error():
    """Git handler must return isError for unrecognised tool names."""
    sys.path.insert(0, str(_GIT_SKILL_DIR))
    with patch.dict("sys.modules", {"git": MagicMock()}):
        import handler as git_handler
        importlib.reload(git_handler)

    event = {"params": {"name": "upload_model", "arguments": {}}}
    result = git_handler.handler(event, {})
    assert result.get("isError") is True, (
        "T3.4: Git handler must not dispatch HuggingFace tool names"
    )


def test_mlflow_handler_unknown_tool_returns_is_error():
    """MLflow handler must return isError for unrecognised tool names."""
    sys.path.insert(0, str(_MLFLOW_SKILL_DIR))
    with patch.dict("sys.modules", {
        "mlflow": MagicMock(),
        "s3fs": MagicMock(),
        "pandas": MagicMock(),
    }):
        import handler as mlflow_handler
        importlib.reload(mlflow_handler)

    event = {"params": {"name": "commit_experiment", "arguments": {}}}
    result = mlflow_handler.handler(event, {})
    assert result.get("isError") is True, (
        "T3.4: MLflow handler must not dispatch Git tool names"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T3.5 — Per-skill dispatch: each Lambda owns its own tool set
# ─────────────────────────────────────────────────────────────────────────────

def test_sm_handler_dispatches_all_four_sm_tools():
    """SageMaker handler must register exactly these 4 tools."""
    sm_handler = _load_sm_handler()
    expected = {"submit_training_job", "complete_training_job", "deploy_model", "submit_eval_job"}
    actual = set(sm_handler._DISPATCH.keys())
    assert actual == expected, (
        f"T3.5: SageMaker handler tools mismatch. Expected {expected}, got {actual}"
    )


def test_hf_handler_dispatches_all_six_hf_tools():
    """HuggingFace handler must register exactly these 6 tools."""
    sys.path.insert(0, str(_HF_SKILL_DIR))
    with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}):
        import handler as hf_handler
        importlib.reload(hf_handler)
    expected = {
        "upload_model",
        "hf_snapshot_download",
        "retrieve_dataset_metadata",
        "prepare_eval_dataset",
        "update_model_card",
        "manage_tags",
    }
    actual = set(hf_handler._DISPATCH.keys())
    assert actual == expected, (
        f"T3.5: HuggingFace handler tools mismatch. Expected {expected}, got {actual}"
    )


def test_git_handler_dispatches_commit_experiment():
    """Git handler must register exactly the commit_experiment tool."""
    sys.path.insert(0, str(_GIT_SKILL_DIR))
    with patch.dict("sys.modules", {"git": MagicMock()}):
        import handler as git_handler
        importlib.reload(git_handler)
    assert set(git_handler._DISPATCH.keys()) == {"commit_experiment"}, (
        "T3.5: Git handler must only expose commit_experiment"
    )


def test_mlflow_handler_dispatches_all_four_mlflow_tools():
    """MLflow handler must register exactly these 4 tools."""
    sys.path.insert(0, str(_MLFLOW_SKILL_DIR))
    with patch.dict("sys.modules", {
        "mlflow": MagicMock(),
        "s3fs": MagicMock(),
        "pandas": MagicMock(),
    }):
        import handler as mlflow_handler
        importlib.reload(mlflow_handler)
    expected = {"query_metrics", "retrieve_traces", "analyze_trace", "generate_compliance_report"}
    actual = set(mlflow_handler._DISPATCH.keys())
    assert actual == expected, (
        f"T3.5: MLflow handler tools mismatch. Expected {expected}, got {actual}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T3.6 — M2M secret ARN configuration hygiene
# ─────────────────────────────────────────────────────────────────────────────

def test_m2m_secret_arn_ssm_parameter_is_configured():
    """The M2M client secret ARN SSM parameter must point to a real Secrets
    Manager secret (non-empty, non-PLACEHOLDER). This is a configuration
    hygiene check — the actual token exchange is verified live in T3.6."""
    ssm = real_boto3.client("ssm", region_name="us-east-1")
    try:
        resp = ssm.get_parameter(
            Name="/sample-mlops-agent/dev/gateway/m2m-client-secret-arn"
        )
        value = resp["Parameter"]["Value"]
    except Exception as exc:
        pytest.skip(f"T3.6: SSM parameter not accessible: {exc}")

    assert value and value != "PLACEHOLDER", (
        "T3.6: /sample-mlops-agent/dev/gateway/m2m-client-secret-arn "
        "must be a real Secrets Manager ARN, not PLACEHOLDER. "
        "Deploy the gateway stack or populate the parameter manually."
    )
    assert "secretsmanager" in value, (
        "T3.6: M2M secret ARN must be a Secrets Manager ARN "
        "(should contain 'secretsmanager')"
    )


def test_m2m_secret_ssm_arn_is_reachable():
    """The Secrets Manager secret referenced by the SSM parameter must exist
    and be retrievable (basic connectivity check for T3.6)."""
    ssm = real_boto3.client("ssm", region_name="us-east-1")
    try:
        arn = ssm.get_parameter(
            Name="/sample-mlops-agent/dev/gateway/m2m-client-secret-arn"
        )["Parameter"]["Value"]
    except Exception as exc:
        pytest.skip(f"T3.6: SSM parameter not accessible: {exc}")

    if not arn or arn == "PLACEHOLDER":
        pytest.skip("T3.6: M2M secret ARN is PLACEHOLDER — Phase 3 not yet configured")

    sm_client = real_boto3.client("secretsmanager", region_name="us-east-1")
    try:
        resp = sm_client.get_secret_value(SecretId=arn)
        secret = resp.get("SecretString", "")
    except Exception as exc:
        pytest.fail(
            f"T3.6: Secrets Manager secret is unreachable ({exc}). "
            "Verify the secret exists and the caller has secretsmanager:GetSecretValue permission."
        )

    assert secret, (
        "T3.6: M2M client secret must be non-empty. "
        "Populate it via: aws secretsmanager put-secret-value "
        f"--secret-id {arn} --secret-string '<client-secret>'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Complete training job tool
# ─────────────────────────────────────────────────────────────────────────────

def test_sm_complete_training_job_returns_status_and_artifact():
    """complete_training_job must return the SageMaker job status and artifact S3 URI."""
    sm_handler = _load_sm_handler()

    mock_boto = MagicMock()
    mock_sm_client = MagicMock()
    mock_sm_client.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "s3://bucket/output/model.tar.gz"},
    }
    mock_boto.client.return_value = mock_sm_client

    with patch("handler.boto3", mock_boto):
        result = sm_handler._complete_training_job({"sagemaker_job_name": "job-abc"})

    assert result["status"] == "Completed"
    assert result["artifact_s3"] == "s3://bucket/output/model.tar.gz"
    assert result["sagemaker_job_name"] == "job-abc"


# ─────────────────────────────────────────────────────────────────────────────
# MLflow skill: query_metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_mlflow_query_metrics_returns_run_metrics():
    """query_metrics must return metrics dict for the given run_id."""
    sys.path.insert(0, str(_MLFLOW_SKILL_DIR))
    mock_mlflow = MagicMock()
    mock_client = MagicMock()
    mock_run = MagicMock()
    mock_run.data.metrics = {"loss": 0.42, "accuracy": 0.88}
    mock_client.get_run.return_value = mock_run

    with patch.dict("sys.modules", {
        "mlflow": mock_mlflow,
        "s3fs": MagicMock(),
        "pandas": MagicMock(),
    }):
        import handler as mlflow_handler
        importlib.reload(mlflow_handler)
        with patch.object(mlflow_handler, "_mlflow_client", return_value=mock_client):
            result = mlflow_handler._query_metrics({"run_id": "run-abc"})

    assert result["run_id"] == "run-abc"
    assert result["metrics"]["loss"] == pytest.approx(0.42)
    assert result["metrics"]["accuracy"] == pytest.approx(0.88)


# ─────────────────────────────────────────────────────────────────────────────
# Git skill: commit_experiment raises when SSM fails
# ─────────────────────────────────────────────────────────────────────────────

def test_git_get_github_token_raises_when_ssm_fails():
    """_get_github_token must raise RuntimeError when SSM is inaccessible."""
    sys.path.insert(0, str(_GIT_SKILL_DIR))
    mock_boto = MagicMock()
    mock_ssm = MagicMock()
    mock_ssm.get_parameter.side_effect = Exception("AccessDenied")
    mock_boto.client.return_value = mock_ssm

    with patch.dict("sys.modules", {"git": MagicMock()}):
        import handler as git_handler
        importlib.reload(git_handler)

    with patch("handler.boto3", mock_boto), \
         patch.dict("os.environ", {"PROJECT_NAME": "sample-mlops-agent", "AWS_REGION": "us-east-1"}):
        with pytest.raises(RuntimeError, match="Cannot retrieve GitHub token"):
            git_handler._get_github_token()
