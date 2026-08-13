import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add lambda/callback to path since 'lambda' is a reserved keyword
sys.path.insert(0, str(Path(__file__).parent.parent / "callback"))


def _load_handler():
    """Load lambda/callback/handler.py by file path under the module name
    ``handler`` (so ``patch("handler.boto3")`` keeps working).

    Other test modules (skill-handler tests) also import a module named
    ``handler`` and prepend their own directories to ``sys.path``, so a plain
    ``import handler`` can resolve to the *sagemaker skill* handler when the
    full suite runs. Loading by explicit path makes this module order-proof.
    """
    path = Path(__file__).parent.parent / "callback" / "handler.py"
    spec = importlib.util.spec_from_file_location("handler", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["handler"] = mod
    spec.loader.exec_module(mod)
    return mod


def _event(job_name, status):
    # The callback handler routes on EventBridge detail-type
    # (lambda/callback/handler.py handler()) — events without one are dropped.
    return {
        "detail-type": "SageMaker Training Job State Change",
        "detail": {"TrainingJobName": job_name, "TrainingJobStatus": status},
    }


def _describe_training_resp(status):
    """describe_training_job response carrying the ThreadId/JobId tags the
    handler resolves the DynamoDB row from."""
    return {
        "TrainingJobStatus": status,
        "TrainingJobArn": "arn:aws:sagemaker:us-east-1:1:training-job/my-job",
        "ModelArtifacts": {"S3ModelArtifacts": ""},
        "Tags": [{"Key": "ThreadId", "Value": "t1"},
                 {"Key": "JobId", "Value": "j1"}],
        "FailureReason": "",
    }


def test_updates_dynamodb_on_completion():
    handler = _load_handler()
    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_ddb_resource = MagicMock()
        mock_ddb_resource.Table.return_value = mock_table
        mock_boto.resource.return_value = mock_ddb_resource
        mock_table.get_item.return_value = {"Item": {
            "task_id": "t1", "session_id": "sess-1", "user_id": "u",
            "jobs": {"j1": {"job_id": "j1", "kind": "training"}}}}

        def _client(service, **kw):
            if service == "ssm":
                ssm = MagicMock()
                ssm.get_parameter.return_value = {"Parameter": {"Value": "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/arn:aws:bedrock-agentcore:us-east-1:123:runtime/r"}}
                return ssm
            c = MagicMock()
            c.invoke_agent_runtime.return_value = {"response": iter([])}
            c.describe_training_job.return_value = _describe_training_resp("Completed")
            return c

        mock_boto.client.side_effect = _client
        handler.handler(_event("my-job", "Completed"), {})
        mock_table.update_item.assert_called_once()


def test_posts_to_agentcore_on_terminal_state():
    handler = _load_handler()

    mock_agentcore = MagicMock()
    mock_agentcore.invoke_agent_runtime.return_value = {"response": iter([])}

    def _client(service, **kw):
        if service == "ssm":
            ssm = MagicMock()
            ssm.get_parameter.return_value = {
                "Parameter": {"Value": "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/arn:aws:bedrock-agentcore:us-east-1:123:runtime/r"}
            }
            return ssm
        if service == "bedrock-agentcore":
            return mock_agentcore
        c = MagicMock()
        c.describe_training_job.return_value = _describe_training_resp("Completed")
        return c

    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_ddb_resource = MagicMock()
        mock_ddb_resource.Table.return_value = mock_table
        mock_boto.resource.return_value = mock_ddb_resource
        mock_table.get_item.return_value = {"Item": {
            "task_id": "t1", "session_id": "sess-1", "user_id": "u",
            "jobs": {"j1": {"job_id": "j1", "kind": "training"}}}}
        mock_boto.client.side_effect = _client
        handler.handler(_event("my-job", "Completed"), {})

    assert mock_agentcore.invoke_agent_runtime.called, (
        "invoke_agent_runtime must be called for a terminal job state"
    )


def test_skips_non_terminal_status():
    """Callback Lambda must NOT call invoke_agent_runtime for non-terminal statuses."""
    handler = _load_handler()

    mock_agentcore = MagicMock()

    def _client(service, **kw):
        if service == "ssm":
            ssm = MagicMock()
            ssm.get_parameter.return_value = {"Parameter": {"Value": "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/arn:aws:bedrock-agentcore:us-east-1:123:runtime/r"}}
            return ssm
        if service == "bedrock-agentcore":
            return mock_agentcore
        c = MagicMock()
        c.describe_training_job.return_value = _describe_training_resp("InProgress")
        return c

    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_ddb_resource = MagicMock()
        mock_ddb_resource.Table.return_value = mock_table
        mock_boto.resource.return_value = mock_ddb_resource
        mock_boto.client.side_effect = _client
        result = handler.handler(_event("my-job", "InProgress"), {})

    mock_agentcore.invoke_agent_runtime.assert_not_called()
    assert result["statusCode"] == 200


def test_returns_404_when_job_has_no_thread_tags():
    """Handler must return 404 when the SageMaker job carries no
    ThreadId/JobId tags — it cannot resolve the DynamoDB row without them."""
    handler = _load_handler()

    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_ddb_resource = MagicMock()
        mock_ddb_resource.Table.return_value = mock_table
        mock_boto.resource.return_value = mock_ddb_resource
        sm = MagicMock()
        sm.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:1:training-job/unknown-job",
            "Tags": [],
        }
        sm.list_tags.return_value = {"Tags": []}
        mock_boto.client.return_value = sm
        result = handler.handler(_event("unknown-job", "Completed"), {})

    assert result["statusCode"] == 404
    mock_table.update_item.assert_not_called()


def test_returns_400_for_missing_job_name():
    """Handler must return 400 when event detail has no TrainingJobName."""
    handler = _load_handler()

    with patch("handler.boto3"):
        result = handler.handler({"detail": {}}, {})

    assert result["statusCode"] == 400


def test_parse_profile_export_extracts_aggregate_metrics(tmp_path):
    handler = _load_handler()
    # Build a minimal profile_export.jsonl with the shape the AWS AI Benchmark
    # service emits. The last line is the aggregate row.
    jsonl = (
        '{"concurrency":1,"ttft_ms_p50":230,"ttft_ms_p99":800,'
        '"inter_token_ms_p50":28,"inter_token_ms_p99":55,'
        '"request_latency_p50_ms":1200,"request_latency_p99_ms":3600,'
        '"throughput_requests_per_sec":9.8,"throughput_tokens_per_sec":1480}\n'
        '{"concurrency":"aggregate","ttft_ms_p50":245,"ttft_ms_p99":812,'
        '"inter_token_ms_p50":31,"inter_token_ms_p99":58,'
        '"request_latency_p50_ms":1420,"request_latency_p99_ms":3900,'
        '"throughput_requests_per_sec":11.4,"throughput_tokens_per_sec":1710}\n'
    )
    p = tmp_path / "profile_export.jsonl"
    p.write_text(jsonl)
    out = handler._parse_profile_export_jsonl_from_path(str(p))
    assert out["ttft_ms_p99"] == 812
    assert out["throughput_tokens_per_sec"] == 1710


def test_teardown_treats_already_deleted_as_success():
    """F-D: on idempotent re-invocation every delete hits 'resource not
    found' — helper must still return True so teardown_complete stays True."""
    handler = _load_handler()
    not_found = Exception("Could not find resource")
    with patch("handler.boto3") as mock_boto:
        mock_sm = MagicMock()
        mock_sm.delete_inference_component.side_effect = not_found
        mock_sm.delete_endpoint.side_effect = not_found
        mock_sm.delete_endpoint_config.side_effect = not_found
        mock_sm.delete_model.side_effect = not_found
        # waiter.wait also raises "already gone"; treated as non-fatal
        mock_sm.get_waiter.return_value.wait.side_effect = not_found
        mock_boto.client.return_value = mock_sm
        ok = handler._teardown_recommendation_resources({
            "inference_component_name": "ic",
            "endpoint_name": "e", "endpoint_config_name": "ec",
            "model_name": "m",
        })
        assert ok is True
        assert mock_sm.delete_inference_component.called
        assert mock_sm.delete_endpoint.called
        assert mock_sm.delete_endpoint_config.called
        assert mock_sm.delete_model.called


def test_teardown_real_error_flips_all_ok_false():
    """A non-'not-found' error (e.g. AccessDenied) must still flip the
    success flag so ops can see something is wrong."""
    handler = _load_handler()
    denied = Exception("AccessDeniedException: not authorized")
    with patch("handler.boto3") as mock_boto:
        mock_sm = MagicMock()
        mock_sm.delete_inference_component.return_value = None
        mock_sm.delete_endpoint.side_effect = denied
        mock_boto.client.return_value = mock_sm
        ok = handler._teardown_recommendation_resources({
            "inference_component_name": "ic",
            "endpoint_name": "e", "endpoint_config_name": "ec",
            "model_name": "m",
        })
        assert ok is False


def test_teardown_waits_for_ic_deletion_before_endpoint():
    """F-F: delete_endpoint must run AFTER the IC waiter returns."""
    handler = _load_handler()
    call_order = []
    with patch("handler.boto3") as mock_boto:
        mock_sm = MagicMock()
        mock_sm.delete_inference_component.side_effect = \
            lambda **kw: call_order.append("del_ic")
        mock_sm.get_waiter.return_value.wait.side_effect = \
            lambda **kw: call_order.append("wait_ic")
        mock_sm.delete_endpoint.side_effect = \
            lambda **kw: call_order.append("del_endpoint")
        mock_boto.client.return_value = mock_sm
        handler._teardown_recommendation_resources({
            "inference_component_name": "ic",
            "endpoint_name": "e", "endpoint_config_name": "ec",
            "model_name": "m",
        })
        assert call_order.index("wait_ic") < call_order.index("del_endpoint"), \
            f"waiter must run before delete_endpoint. Order: {call_order}"


def test_endpoint_in_service_starts_benchmark_for_kind_recommendation():
    """EventBridge 'SageMaker Endpoint State Change' with Kind=recommendation
    tags and EndpointStatus=InService must fire create_inference_component,
    create_ai_workload_config, create_ai_benchmark_job and mark
    status=BENCHMARKING."""
    handler = _load_handler()
    os.environ.setdefault("SESSION_BUCKET", "test-bucket")
    os.environ.setdefault("SAGEMAKER_EXECUTION_ROLE_ARN", "arn:aws:iam::123:role/test")
    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_boto.resource.return_value.Table.return_value = mock_table
        mock_table.query.return_value = {"Items": [{"task_id": "t1"}]}
        mock_table.get_item.return_value = {"Item": {
            "task_id": "t1", "user_id": "u",
            "jobs": {"r1": {
                "job_id": "r1", "kind": "recommendation",
                "endpoint_name": "e1", "model_name": "m1",
                "endpoint_config_name": "ec1",
                "inference_component_name": "ic1",
                "ai_workload_config_name": "wl1",
                "recommender_job_name": "rec-1",
                "instance_type": "ml.g6.xlarge",
                "workload_spec": {"input_tokens": 500, "output_tokens": 150,
                                  "concurrency_levels": [1, 4, 16],
                                  "max_latency_p99_ms": 5000,
                                  "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct"},
            }}}}
        mock_sm = MagicMock()
        mock_sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService", "EndpointArn": "arn:e1",
            "Tags": [{"Key": "Kind", "Value": "recommendation"},
                     {"Key": "RecId", "Value": "r1"},
                     {"Key": "ThreadId", "Value": "t1"}],
        }
        mock_boto.client.return_value = mock_sm
        handler.handler({"detail-type": "SageMaker Endpoint State Change",
                         "detail": {"EndpointName": "e1",
                                    "EndpointStatus": "InService"}}, {})
        # BUG-020: no inference component — model rides the variant.
        mock_sm.create_inference_component.assert_not_called()
        assert mock_sm.create_ai_workload_config.called
        target = mock_sm.create_ai_benchmark_job.call_args.kwargs["BenchmarkTarget"]
        assert target == {"Endpoint": {"Identifier": "e1"}}, \
            "benchmark must target the bare endpoint (no InferenceComponents)"
        assert mock_sm.create_ai_benchmark_job.called
        # status must be stamped BENCHMARKING
        set_writes = []
        for call in mock_table.update_item.call_args_list:
            set_writes.append(call.kwargs.get("ExpressionAttributeValues", {}))
        assert any("BENCHMARKING" in str(v.values()) for v in set_writes), \
            f"status=BENCHMARKING never written; got {set_writes}"


def test_endpoint_event_real_eventbridge_casing_in_service():
    """Regression for QA BUG-012: real EventBridge events carry
    ``EndpointStatus: IN_SERVICE`` (UPPER_SNAKE), not the API's ``InService``.
    The benchmark must still start. Describe returns the API casing, and the
    handler must prefer/normalize it rather than string-compare the raw
    event value."""
    handler = _load_handler()
    os.environ.setdefault("SESSION_BUCKET", "test-bucket")
    os.environ.setdefault("SAGEMAKER_EXECUTION_ROLE_ARN", "arn:aws:iam::123:role/test")
    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_boto.resource.return_value.Table.return_value = mock_table
        mock_table.get_item.return_value = {"Item": {
            "task_id": "t1", "user_id": "u",
            "jobs": {"r1": {
                "job_id": "r1", "kind": "recommendation",
                "endpoint_name": "e1", "model_name": "m1",
                "endpoint_config_name": "ec1",
                "inference_component_name": "ic1",
                "ai_workload_config_name": "wl1",
                "recommender_job_name": "rec-1",
                "instance_type": "ml.g6.xlarge",
                "workload_spec": {"input_tokens": 500, "output_tokens": 150,
                                  "concurrency_levels": [1, 4, 16],
                                  "max_latency_p99_ms": 5000,
                                  "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct"},
            }}}}
        mock_sm = MagicMock()
        mock_sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService", "EndpointArn": "arn:e1",
            "Tags": [{"Key": "Kind", "Value": "recommendation"},
                     {"Key": "RecId", "Value": "r1"},
                     {"Key": "ThreadId", "Value": "t1"}],
        }
        mock_boto.client.return_value = mock_sm
        # Real EventBridge shape — UPPER_SNAKE status.
        handler.handler({"detail-type": "SageMaker Endpoint State Change",
                         "detail": {"EndpointName": "e1",
                                    "EndpointStatus": "IN_SERVICE"}}, {})
        assert mock_sm.create_ai_benchmark_job.called, \
            "IN_SERVICE (EventBridge casing) must start the benchmark"


def test_endpoint_event_real_eventbridge_casing_failed_tears_down():
    """Regression for QA BUG-012 (failure leg): a real ``FAILED`` event must
    tear down partial resources and stamp the job FAILED — observed in prod:
    a Failed endpoint sat leaked and the job stayed DEPLOYING forever."""
    handler = _load_handler()
    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_boto.resource.return_value.Table.return_value = mock_table
        mock_table.get_item.return_value = {"Item": {
            "task_id": "t1", "user_id": "u",
            "jobs": {"r1": {
                "job_id": "r1", "kind": "recommendation",
                "endpoint_name": "e1", "model_name": "m1",
                "endpoint_config_name": "ec1",
                "inference_component_name": "ic1",
            }}}}
        mock_sm = MagicMock()
        mock_sm.describe_endpoint.return_value = {
            "EndpointStatus": "Failed", "EndpointArn": "arn:e1",
            "FailureReason": "InsufficientInstanceCapacity",
            "Tags": [{"Key": "Kind", "Value": "recommendation"},
                     {"Key": "RecId", "Value": "r1"},
                     {"Key": "ThreadId", "Value": "t1"}],
        }
        mock_boto.client.return_value = mock_sm
        handler.handler({"detail-type": "SageMaker Endpoint State Change",
                         "detail": {"EndpointName": "e1",
                                    "EndpointStatus": "FAILED"}}, {})
        assert mock_sm.delete_endpoint.called, "FAILED event must tear down the endpoint"
        all_values = []
        for call in mock_table.update_item.call_args_list:
            all_values.extend(call.kwargs.get("ExpressionAttributeValues", {}).values())
        assert any("FAILED" == str(v) for v in all_values), \
            f"status=FAILED never stamped; wrote {all_values}"


def test_normalize_endpoint_status_maps_both_casings():
    """Unit coverage for the BUG-012 normalizer."""
    handler = _load_handler()
    assert handler._normalize_endpoint_status("IN_SERVICE") == "InService"
    assert handler._normalize_endpoint_status("InService") == "InService"
    assert handler._normalize_endpoint_status("FAILED") == "Failed"
    assert handler._normalize_endpoint_status("Failed") == "Failed"
    assert handler._normalize_endpoint_status("CREATING") == "Creating"
    assert handler._normalize_endpoint_status("weird") == "weird"


def test_endpoint_in_service_ignores_non_recommendation_endpoint():
    """An endpoint without Kind=recommendation tag must NOT trigger any
    create_* calls — other services in the account own those endpoints."""
    handler = _load_handler()
    with patch("handler.boto3") as mock_boto:
        mock_boto.resource.return_value.Table.return_value = MagicMock()
        mock_sm = MagicMock()
        mock_sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService",
            "Tags": [{"Key": "Kind", "Value": "production-inference"}],
        }
        mock_boto.client.return_value = mock_sm
        handler.handler({"detail-type": "SageMaker Endpoint State Change",
                         "detail": {"EndpointName": "some-prod-ep",
                                    "EndpointStatus": "InService"}}, {})
        assert not mock_sm.create_inference_component.called
        assert not mock_sm.create_ai_workload_config.called
        assert not mock_sm.create_ai_benchmark_job.called


# ── R1 Task 3: baseline URI stamping for tabular training ────────────────


def test_stamps_baseline_and_eval_split_uris_for_tabular_training():
    """When a tabular (xgboost/sklearn) training job Completes, the callback
    must derive baseline + eval_split S3 URIs from ModelArtifacts.S3ModelArtifacts
    and write them onto jobs.<job_id>.{baseline_s3_uri,eval_split_s3_uri}.
    BUG-003: URIs are only stamped after the baseline objects were actually
    materialized — extraction is mocked as succeeding here."""
    handler = _load_handler()
    handler._materialize_baseline_objects = MagicMock()

    artifact_s3 = "s3://bkt/training-output/my-job/output/model.tar.gz"
    expected_baseline = "s3://bkt/training-output/my-job/output/baseline/baseline.csv"
    expected_eval = "s3://bkt/training-output/my-job/output/baseline/eval_split.csv"

    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_boto.resource.return_value.Table.return_value = mock_table
        mock_table.query.return_value = {"Items": [{"task_id": "t1"}]}
        # training_type read for the baseline-stamp branch.
        mock_table.get_item.return_value = {"Item": {
            "task_id": "t1", "user_id": "u",
            "jobs": {"j1": {"job_id": "j1", "kind": "training",
                            "training_type": "xgboost"}}}}

        describe_resp = {
            "TrainingJobStatus": "Completed",
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:1:training-job/my-job",
            "ModelArtifacts": {"S3ModelArtifacts": artifact_s3},
            "Tags": [{"Key": "ThreadId", "Value": "t1"},
                     {"Key": "JobId", "Value": "j1"}],
            "FailureReason": "",
        }

        def _client(service, **kw):
            if service == "ssm":
                ssm = MagicMock()
                ssm.get_parameter.return_value = {"Parameter": {
                    "Value": "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/"
                             "arn:aws:bedrock-agentcore:us-east-1:123:runtime/r"}}
                return ssm
            c = MagicMock()
            c.describe_training_job.return_value = describe_resp
            c.invoke_agent_runtime.return_value = {"response": iter([])}
            return c

        mock_boto.client.side_effect = _client
        handler.handler({"detail-type": "SageMaker Training Job State Change",
                         "detail": {"TrainingJobName": "my-job",
                                    "TrainingJobStatus": "Completed"}}, {})

        # Collect every ExpressionAttributeValues dict across every update_item
        # call; assert both URIs landed somewhere. Robust to the one-vs-two
        # update_item implementation choice.
        all_values = []
        for call in mock_table.update_item.call_args_list:
            all_values.extend(call.kwargs.get("ExpressionAttributeValues", {}).values())
        assert expected_baseline in all_values, \
            f"baseline URI not stamped; values written = {all_values}"
        assert expected_eval in all_values, \
            f"eval_split URI not stamped; values written = {all_values}"


def test_stamps_no_baseline_uri_for_llm_training():
    """Negative test: LLM training_type (grpo/sft/dpo) must NOT get baseline
    URIs stamped. Guards against regressions that'd inject meaningless URIs
    into the thread row."""
    handler = _load_handler()

    artifact_s3 = "s3://bkt/training-output/llm-job/output/model.tar.gz"
    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_boto.resource.return_value.Table.return_value = mock_table
        mock_table.query.return_value = {"Items": [{"task_id": "t2"}]}
        mock_table.get_item.return_value = {"Item": {
            "task_id": "t2", "user_id": "u",
            "jobs": {"j2": {"job_id": "j2", "kind": "training",
                            "training_type": "sft"}}}}
        describe_resp = {
            "TrainingJobStatus": "Completed",
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:1:training-job/llm-job",
            "ModelArtifacts": {"S3ModelArtifacts": artifact_s3},
            "Tags": [{"Key": "ThreadId", "Value": "t2"},
                     {"Key": "JobId", "Value": "j2"}],
            "FailureReason": "",
        }

        def _client(service, **kw):
            if service == "ssm":
                ssm = MagicMock()
                ssm.get_parameter.return_value = {"Parameter": {
                    "Value": "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/"
                             "arn:aws:bedrock-agentcore:us-east-1:123:runtime/r"}}
                return ssm
            c = MagicMock()
            c.describe_training_job.return_value = describe_resp
            c.invoke_agent_runtime.return_value = {"response": iter([])}
            return c

        mock_boto.client.side_effect = _client
        handler.handler({"detail-type": "SageMaker Training Job State Change",
                         "detail": {"TrainingJobName": "llm-job",
                                    "TrainingJobStatus": "Completed"}}, {})

        all_values = []
        for call in mock_table.update_item.call_args_list:
            all_values.extend(call.kwargs.get("ExpressionAttributeValues", {}).values())
        assert not any("baseline.csv" in str(v) for v in all_values), \
            f"baseline URI was stamped for LLM training; values = {all_values}"


# ── R1 Task 9: Kind=monitoring branch on Processing-job events ────────────


def test_monitoring_completed_resumes_agent_with_monitoring_message():
    """When Kind=monitoring Processing job Completes, callback must POST a
    resume message to AgentCore with monitoring-specific wording — and
    must NOT tell the agent to invoke compliance-documentation."""
    handler = _load_handler()

    mock_agentcore = MagicMock()
    mock_agentcore.invoke_agent_runtime.return_value = {"response": iter([])}

    def _client(service, **kw):
        if service == "ssm":
            s = MagicMock()
            s.get_parameter.return_value = {"Parameter": {
                "Value": "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/"
                         "arn:aws:bedrock-agentcore:us-east-1:123:runtime/r"}}
            return s
        if service == "bedrock-agentcore":
            return mock_agentcore
        if service == "sagemaker":
            c = MagicMock()
            c.describe_processing_job.return_value = {
                "ProcessingJobStatus": "Completed",
                "ProcessingJobArn": "arn:aws:sagemaker:us-east-1:1:processing-job/monitor-1",
                "FailureReason": "",
                "Tags": [
                    {"Key": "ThreadId", "Value": "t1"},
                    {"Key": "JobId", "Value": "j1"},
                    {"Key": "Kind", "Value": "monitoring"},
                ],
            }
            return c
        return MagicMock()

    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_boto.resource.return_value.Table.return_value = mock_table
        # _process_state_change calls query() to find the task_id then
        # get_item() to fetch the full row (used by _build_resume_message).
        thread_row = {
            "task_id": "t1", "session_id": "s1", "user_id": "u",
            "jobs": {"j1": {
                "job_id": "j1", "kind": "monitoring",
                "mlflow_run_id": "abc",
                "mlflow_run_url": "https://mlflow/42/runs/abc",
            }},
        }
        mock_table.query.return_value = {"Items": [thread_row]}
        mock_table.get_item.return_value = {"Item": thread_row}
        mock_boto.client.side_effect = _client
        handler.handler({
            "detail-type": "SageMaker Processing Job State Change",
            "detail": {
                "ProcessingJobName": "sample-mlops-agent-monitor-1",
                "ProcessingJobStatus": "Completed",
            },
        }, {})

    assert mock_agentcore.invoke_agent_runtime.called
    payload = mock_agentcore.invoke_agent_runtime.call_args.kwargs.get("payload", b"")
    body = payload.decode() if isinstance(payload, bytes) else str(payload)
    assert "Monitoring job" in body
    # R1 explicit divergence from eval — message must tell the agent
    # NOT to invoke compliance-documentation (the eval branch sends the
    # opposite instruction).
    assert "do not invoke the compliance-documentation" in body.lower()
    # And the eval path's "invoke the compliance-documentation skill"
    # wording must NOT appear.
    assert "invoke the compliance-documentation skill" not in body.lower() \
        or "do not invoke the compliance-documentation skill" in body.lower()


# ── BUG-003: baseline materialization from model.tar.gz ───────────────────


def _build_model_tarball(tmp_path, with_baseline=True):
    """Create a model.tar.gz shaped like a real tabular training artifact."""
    import tarfile
    src = tmp_path / "src"
    src.mkdir()
    (src / "model.xgb").write_text("weights")
    if with_baseline:
        b = src / "baseline"
        b.mkdir()
        (b / "baseline.csv").write_text("a,b,target\n1,2,0\n")
        (b / "eval_split.csv").write_text("a,b,target\n3,4,1\n")
        (b / "baseline_stats.json").write_text('{"target_column": "target"}')
    tar_path = tmp_path / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(src)), recursive=False)
    return tar_path


def test_materialize_baseline_extracts_from_tarball_and_uploads(tmp_path):
    """BUG-003 regression: baseline files live INSIDE model.tar.gz (packed
    from /opt/ml/model). The callback must extract and upload them as the
    standalone objects the monitoring handler pre-flights."""
    handler = _load_handler()
    tar_path = _build_model_tarball(tmp_path)

    uploads = []
    s3 = MagicMock()
    s3.head_object.return_value = {"ContentLength": tar_path.stat().st_size}
    s3.download_file.side_effect = lambda b, k, dst: __import__("shutil").copy(tar_path, dst)
    s3.upload_fileobj.side_effect = lambda fobj, bucket, key: uploads.append((bucket, key))

    with patch("handler.boto3") as mock_boto:
        mock_boto.client.return_value = s3
        handler._materialize_baseline_objects(
            artifact_s3="s3://bkt/training-output/my-job/output/model.tar.gz",
            base_prefix="s3://bkt/training-output/my-job/output/baseline",
        )

    keys = sorted(k for _, k in uploads)
    assert keys == [
        "training-output/my-job/output/baseline/baseline.csv",
        "training-output/my-job/output/baseline/baseline_stats.json",
        "training-output/my-job/output/baseline/eval_split.csv",
    ], f"unexpected upload keys: {keys}"
    assert all(b == "bkt" for b, _ in uploads)


def test_materialize_baseline_raises_when_tarball_has_no_baseline(tmp_path):
    """A tarball without baseline/* (e.g. legacy training script) must raise
    so the caller does NOT stamp dead URIs."""
    handler = _load_handler()
    tar_path = _build_model_tarball(tmp_path, with_baseline=False)
    s3 = MagicMock()
    s3.head_object.return_value = {"ContentLength": tar_path.stat().st_size}
    s3.download_file.side_effect = lambda b, k, dst: __import__("shutil").copy(tar_path, dst)
    with patch("handler.boto3") as mock_boto:
        mock_boto.client.return_value = s3
        try:
            handler._materialize_baseline_objects(
                artifact_s3="s3://bkt/j/output/model.tar.gz",
                base_prefix="s3://bkt/j/output/baseline",
            )
        except RuntimeError as e:
            assert "no baseline" in str(e)
        else:
            raise AssertionError("expected RuntimeError for baseline-less tarball")


def test_stamp_skipped_when_materialization_fails():
    """If extraction/upload fails, baseline URIs must NOT be stamped —
    a stamped-but-missing URI reproduces the original BUG-003 403."""
    handler = _load_handler()
    handler._materialize_baseline_objects = MagicMock(side_effect=RuntimeError("boom"))
    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_boto.resource.return_value.Table.return_value = mock_table
        handler._stamp_tabular_baseline_uris(
            thread_id="t1", job_id="j1",
            artifact_s3="s3://bkt/j/output/model.tar.gz",
            training_type="xgboost",
        )
        mock_table.update_item.assert_not_called()




def test_in_service_workload_spec_serializes_ddb_decimals():
    """QA BUG-016: DynamoDB numbers arrive as Decimal; the workload-spec
    json.dumps must not raise — a real run failed with 'Object of type
    Decimal is not JSON serializable' after the IC was created."""
    from decimal import Decimal
    handler = _load_handler()
    os.environ.setdefault("SESSION_BUCKET", "test-bucket")
    os.environ.setdefault("SAGEMAKER_EXECUTION_ROLE_ARN", "arn:aws:iam::123:role/test")
    with patch("handler.boto3") as mock_boto:
        mock_table = MagicMock()
        mock_boto.resource.return_value.Table.return_value = mock_table
        mock_table.get_item.return_value = {"Item": {
            "task_id": "t1", "user_id": "u",
            "jobs": {"r1": {
                "job_id": "r1", "kind": "recommendation",
                "endpoint_name": "e1", "model_name": "m1",
                "endpoint_config_name": "ec1",
                "inference_component_name": "ic1",
                "ai_workload_config_name": "wl1",
                "recommender_job_name": "rec-1",
                "instance_type": "ml.g5.xlarge",
                # exactly what boto3's DDB deserializer hands back
                "workload_spec": {"input_tokens": Decimal("500"),
                                  "output_tokens": Decimal("150"),
                                  "concurrency_levels": [Decimal("1"), Decimal("4"), Decimal("16")],
                                  "max_latency_p99_ms": Decimal("5000"),
                                  "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct"},
            }}}}
        mock_sm = MagicMock()
        mock_sm.describe_endpoint.return_value = {
            "EndpointStatus": "InService", "EndpointArn": "arn:e1",
            "Tags": [{"Key": "Kind", "Value": "recommendation"},
                     {"Key": "RecId", "Value": "r1"},
                     {"Key": "ThreadId", "Value": "t1"}],
        }
        mock_boto.client.return_value = mock_sm
        handler.handler({"detail-type": "SageMaker Endpoint State Change",
                         "detail": {"EndpointName": "e1",
                                    "EndpointStatus": "IN_SERVICE"}}, {})
        assert mock_sm.create_ai_benchmark_job.called, \
            "Decimal workload_spec must not abort the benchmark start"
        inline = json.loads(mock_sm.create_ai_workload_config.call_args.kwargs[
            "AIWorkloadConfigs"]["WorkloadSpec"]["Inline"])
        # QA BUG-019: the service validates the AIPerf spec shape — the flat
        # spec was rejected with "benchmark: Field required".
        assert inline["benchmark"] == {"type": "aiperf"}
        p = inline["parameters"]
        assert p["tokenizer"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert p["concurrency"] == 16  # highest requested level
        assert p["prompt_input_tokens_mean"] == 500
        assert p["output_tokens_mean"] == 150
        assert inline["tooling"]["api_standard"] == "openai"


def test_aiperf_spec_requires_tokenizer():
    """QA BUG-019: no tokenizer → fail loudly rather than benchmark with a
    wrong tokenizer (silently corrupt token counts)."""
    handler = _load_handler()
    try:
        handler._aiperf_workload_spec({"input_tokens": 500, "concurrency_levels": [1]})
    except RuntimeError as e:
        assert "tokenizer" in str(e)
    else:
        raise AssertionError("expected RuntimeError without tokenizer")
