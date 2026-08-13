"""Regression tests for Gateway Lambda tool-name dispatch.

AgentCore Gateway delivers the MCP tool name via
`context.client_context.custom['bedrockAgentCoreToolName']` prefixed with
`${target_name}___`, and passes the tool arguments map directly as `event`.
A prior bug read them from `event['params']['name']` / `event['params']['arguments']`
— that produced "Unknown tool: " (empty) on every invocation. These tests pin
the fixed contract so the bug cannot silently regress.
"""
import importlib
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock


def _ensure_stub(name: str) -> None:
    """Register a MagicMock module under `name` so `import name` succeeds.

    Skill Lambdas import heavy SDKs (mlflow, huggingface_hub) at module-load
    time; the Lambda image ships them but the test environment does not.
    We only exercise the dispatch wrapper, so stubs are enough.

    Prefer the real module when it is installed: a fake left in sys.modules
    shadows the genuine package for every later test module in the pytest
    session (a stub `requests` broke moto's `import requests.adapters` in
    test_phase3_gateway).
    """
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass
    mod = ModuleType(name)
    mod.__getattr__ = lambda attr, _m=mod: MagicMock(name=f"{name}.{attr}")  # type: ignore[attr-defined]
    sys.modules[name] = mod


for _mod in ("mlflow", "mlflow.tracking", "mlflow.exceptions", "huggingface_hub", "requests"):
    _ensure_stub(_mod)
# huggingface_hub.HfApi / snapshot_download are referenced as attributes on
# import — set them only on OUR stub (a stub has no __spec__); never mutate a
# real installed huggingface_hub in place.
if getattr(sys.modules["huggingface_hub"], "__spec__", None) is None:
    sys.modules["huggingface_hub"].HfApi = MagicMock(name="HfApi")
    sys.modules["huggingface_hub"].snapshot_download = MagicMock(name="snapshot_download")


def _import_handler(skill: str):
    """Import lambda/skills/<skill>/handler.py freshly for each test.

    Skill handlers all expose the module name `handler`, so we must purge the
    cached entry in sys.modules between imports to prevent cross-contamination.
    """
    lam_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "lambda", "skills", skill)
    )
    # Drop any previous skill's sys.path entry so the right handler.py wins
    for p in list(sys.path):
        if "/lambda/skills/" in p:
            sys.path.remove(p)
    sys.path.insert(0, lam_dir)
    sys.modules.pop("handler", None)
    return importlib.import_module("handler")


def _ctx(tool_name: str) -> SimpleNamespace:
    """Build a minimal Lambda context shaped like the AgentCore Gateway payload."""
    return SimpleNamespace(
        client_context=SimpleNamespace(custom={"bedrockAgentCoreToolName": tool_name})
    )


def _stub_dispatch(h, tool: str, return_value: dict) -> list:
    """Replace handler._DISPATCH with a single stub that records its arg."""
    captured: list = []

    def _stub(args):
        captured.append(args)
        return return_value

    h._DISPATCH = {tool: _stub}
    return captured


def test_sagemaker_handler_strips_target_prefix_and_dispatches():
    """Gateway sends `sagemaker-skill___submit_training_job` — handler must strip prefix and dispatch."""
    h = _import_handler("sagemaker")
    captured = _stub_dispatch(
        h,
        "submit_training_job",
        {"thread_id": "t1", "job_id": "jb1", "sagemaker_job_name": "j1"},
    )
    resp = h.handler(
        {"model_id": "m", "dataset_name": "d", "session_id": "s"},
        _ctx("sagemaker-skill___submit_training_job"),
    )
    assert captured == [{"model_id": "m", "dataset_name": "d", "session_id": "s"}]
    assert resp.get("isError") is not True
    assert "t1" in resp["content"][0]["text"]
    assert "jb1" in resp["content"][0]["text"]


def test_sagemaker_handler_unknown_tool_returns_error():
    """Unknown tool name → isError=True response with the parsed tool name."""
    h = _import_handler("sagemaker")
    h._DISPATCH = {}
    resp = h.handler({}, _ctx("sagemaker-skill___does_not_exist"))
    assert resp["isError"] is True
    assert "does_not_exist" in resp["content"][0]["text"]


def test_sagemaker_handler_dispatches_submit_recommendation_job():
    """Handler strips sagemaker-skill___ prefix and routes to
    submit_recommendation_job in _DISPATCH."""
    h = _import_handler("sagemaker")
    captured = _stub_dispatch(h, "submit_recommendation_job",
        {"thread_id": "t1", "job_id": "r1",
         "recommender_job_name": "rec-1", "status": "SUBMITTING"})
    resp = h.handler(
        {"thread_id": "t1", "sagemaker_job_name": "src",
         "instance_type": "ml.g6.xlarge", "_user_id": "u"},
        _ctx("sagemaker-skill___submit_recommendation_job"))
    assert captured == [{"thread_id": "t1", "sagemaker_job_name": "src",
                         "instance_type": "ml.g6.xlarge", "_user_id": "u"}]
    assert resp.get("isError") is not True
    assert "rec-1" in resp["content"][0]["text"]

def test_sagemaker_handler_dispatches_get_recommendation_results():
    h = _import_handler("sagemaker")
    captured = _stub_dispatch(h, "get_recommendation_results",
        {"recommender_job_name": "rec-1", "status": "COMPLETED",
         "metrics": {"ttft_ms_p99": 812.0}})
    resp = h.handler(
        {"thread_id": "t1", "recommender_job_name": "rec-1", "_user_id": "u"},
        _ctx("sagemaker-skill___get_recommendation_results"))
    assert captured == [{"thread_id": "t1", "recommender_job_name": "rec-1", "_user_id": "u"}]
    assert resp.get("isError") is not True
    assert "rec-1" in resp["content"][0]["text"]


def test_mlflow_handler_strips_target_prefix():
    h = _import_handler("mlflow")
    captured = _stub_dispatch(h, "query_metrics", {"run_id": "r1", "metrics": {"acc": 0.9}})
    resp = h.handler({"run_id": "r1"}, _ctx("mlflow-skill___query_metrics"))
    assert captured == [{"run_id": "r1"}]
    assert resp.get("isError") is not True


def test_huggingface_handler_strips_target_prefix():
    h = _import_handler("huggingface")
    captured = _stub_dispatch(h, "hf_snapshot_download", {"s3_uri": "s3://b/p"})
    resp = h.handler(
        {"repo_id": "org/model", "s3_prefix": "models/x"},
        _ctx("huggingface-skill___hf_snapshot_download"),
    )
    assert captured == [{"repo_id": "org/model", "s3_prefix": "models/x"}]
    assert resp.get("isError") is not True


def test_git_handler_strips_target_prefix():
    h = _import_handler("git")
    captured = _stub_dispatch(h, "commit_experiment", {"commit_sha": "abc", "repo_url": "https://x"})
    resp = h.handler(
        {"files": {"a.txt": "hi"}, "commit_message": "test"},
        _ctx("git-skill___commit_experiment"),
    )
    assert captured == [{"files": {"a.txt": "hi"}, "commit_message": "test"}]
    assert resp.get("isError") is not True


def test_get_recommendation_results_pre_terminal_returns_benchmarking():
    """When AWS reports status in {InProgress, Starting, Stopping}, the tool
    must return status=BENCHMARKING and NOT call parse/teardown."""
    from unittest.mock import patch
    h = _import_handler("sagemaker")
    importlib.reload(h)
    mock_sm = MagicMock()
    mock_sm.describe_ai_benchmark_job.return_value = {
        "AIBenchmarkJobStatus": "InProgress"}
    mock_table = MagicMock()
    mock_table.get_item.return_value = {"Item": {
        "task_id": "t1",
        "jobs": {"r1": {
            "job_id": "r1", "kind": "recommendation",
            "status": "BENCHMARKING",
            "recommender_job_name": "rec-1",
            "instance_type": "ml.g6.xlarge",
        }}}}
    with patch.object(h, "boto3") as mock_boto, \
         patch.object(h, "_ddb_table", return_value=mock_table), \
         patch.object(h, "_parse_profile_export_jsonl") as mock_parse, \
         patch.object(h, "_teardown_recommendation_resources") as mock_td:
        mock_boto.client.return_value = mock_sm
        out = h._get_recommendation_results({
            "thread_id": "t1", "recommender_job_name": "rec-1",
            "_user_id": "functest"})
        assert out["status"] == "BENCHMARKING", out
        assert not mock_parse.called
        assert not mock_td.called


def test_get_recommendation_results_completed_parses_and_tears_down():
    """Terminal Completed → parse profile_export.jsonl, tear down,
    DDB gets status=COMPLETED + metrics."""
    from unittest.mock import patch
    h = _import_handler("sagemaker")
    importlib.reload(h)
    mock_sm = MagicMock()
    mock_sm.describe_ai_benchmark_job.return_value = {
        "AIBenchmarkJobStatus": "Completed",
        "OutputConfig": {"S3OutputLocation": "s3://b/rec/"}}
    mock_table = MagicMock()
    # First get_item (pre-update): BENCHMARKING; second (post-update re-read): COMPLETED
    mock_table.get_item.side_effect = [
        {"Item": {
            "task_id": "t1",
            "jobs": {"r1": {
                "job_id": "r1", "kind": "recommendation",
                "status": "BENCHMARKING",
                "recommender_job_name": "rec-1",
                "endpoint_name": "e1", "endpoint_config_name": "ec1",
                "inference_component_name": "ic1", "model_name": "m1",
            }}}},
        {"Item": {
            "task_id": "t1",
            "jobs": {"r1": {
                "job_id": "r1", "kind": "recommendation",
                "status": "COMPLETED",
                "recommender_job_name": "rec-1",
                "results": {"ttft_ms_p99": 812, "throughput_tokens_per_sec": 1710},
                "teardown_complete": True,
            }}}},
    ]
    with patch.object(h, "boto3") as mock_boto, \
         patch.object(h, "_ddb_table", return_value=mock_table), \
         patch.object(h, "_parse_profile_export_jsonl",
               return_value={"ttft_ms_p99": 812,
                             "throughput_tokens_per_sec": 1710}) as mock_parse, \
         patch.object(h, "_teardown_recommendation_resources",
               return_value=True) as mock_td:
        mock_boto.client.return_value = mock_sm
        out = h._get_recommendation_results({
            "thread_id": "t1", "recommender_job_name": "rec-1",
            "_user_id": "functest"})
        assert mock_parse.called
        assert mock_td.called
        assert out["status"] == "COMPLETED", out
        all_values = []
        for call in mock_table.update_item.call_args_list:
            all_values.extend(call.kwargs.get("ExpressionAttributeValues", {}).values())
        assert any(isinstance(v, dict) and v.get("ttft_ms_p99") == 812
                   for v in all_values)


def test_get_recommendation_results_failed_captures_failure_reason_and_tears_down():
    """Terminal Failed → teardown still runs, status=FAILED,
    status_message carries FailureReason."""
    from unittest.mock import patch
    h = _import_handler("sagemaker")
    importlib.reload(h)
    mock_sm = MagicMock()
    mock_sm.describe_ai_benchmark_job.return_value = {
        "AIBenchmarkJobStatus": "Failed",
        "FailureReason": "OOM at concurrency=16",
        "OutputConfig": {"S3OutputLocation": "s3://b/rec/"}}
    mock_table = MagicMock()
    # First get_item (pre-update): BENCHMARKING; second (post-update re-read): FAILED
    mock_table.get_item.side_effect = [
        {"Item": {
            "task_id": "t1",
            "jobs": {"r1": {
                "job_id": "r1", "kind": "recommendation",
                "status": "BENCHMARKING",
                "recommender_job_name": "rec-1",
                "endpoint_name": "e1", "endpoint_config_name": "ec1",
                "inference_component_name": "ic1", "model_name": "m1",
            }}}},
        {"Item": {
            "task_id": "t1",
            "jobs": {"r1": {
                "job_id": "r1", "kind": "recommendation",
                "status": "FAILED",
                "recommender_job_name": "rec-1",
                "status_message": "Benchmark Failed; failure: OOM at concurrency=16",
                "teardown_complete": True,
            }}}},
    ]
    with patch.object(h, "boto3") as mock_boto, \
         patch.object(h, "_ddb_table", return_value=mock_table), \
         patch.object(h, "_teardown_recommendation_resources",
               return_value=True) as mock_td:
        mock_boto.client.return_value = mock_sm
        out = h._get_recommendation_results({
            "thread_id": "t1", "recommender_job_name": "rec-1",
            "_user_id": "functest"})
        assert mock_td.called
        assert out["status"] == "FAILED", out
        assert "OOM at concurrency=16" in out.get("summary", "")


def test_get_recommendation_results_fast_path_skips_sagemaker_on_already_terminal():
    """If DDB row is already COMPLETED (a previous call wrote it), we must
    NOT call describe_ai_benchmark_job / parse / teardown again."""
    from unittest.mock import patch
    h = _import_handler("sagemaker")
    importlib.reload(h)
    mock_sm = MagicMock()
    mock_table = MagicMock()
    mock_table.get_item.return_value = {"Item": {
        "task_id": "t1",
        "jobs": {"r1": {
            "job_id": "r1", "kind": "recommendation",
            "status": "COMPLETED",
            "recommender_job_name": "rec-1",
            "instance_type": "ml.g6.xlarge",
            "results": {"ttft_ms_p99": 812,
                        "throughput_tokens_per_sec": 1710},
            "teardown_complete": True,
            "status_message": "Benchmark Completed; TTFT p99=812ms, throughput=1710 tok/s",
        }}}}
    with patch.object(h, "boto3") as mock_boto, \
         patch.object(h, "_ddb_table", return_value=mock_table), \
         patch.object(h, "_parse_profile_export_jsonl") as mock_parse, \
         patch.object(h, "_teardown_recommendation_resources") as mock_td:
        mock_boto.client.return_value = mock_sm
        out = h._get_recommendation_results({
            "thread_id": "t1", "recommender_job_name": "rec-1",
            "_user_id": "functest"})
        assert not mock_sm.describe_ai_benchmark_job.called
        assert not mock_parse.called
        assert not mock_td.called
        assert out["status"] == "COMPLETED"
        assert out["metrics"]["ttft_ms_p99"] == 812


def test_get_recommendation_results_completed_but_tarball_missing_defers_teardown():
    """F-B: AWS flips status Completed before uploading output.tar.gz.
    First call sees Completed + empty parse → returns BENCHMARKING and
    does NOT tear down. Second call (after tarball lands) can parse."""
    from unittest.mock import patch
    h = _import_handler("sagemaker")
    importlib.reload(h)
    mock_sm = MagicMock()
    mock_sm.describe_ai_benchmark_job.return_value = {
        "AIBenchmarkJobStatus": "Completed",
        "OutputConfig": {"S3OutputLocation": "s3://b/rec/"}}
    mock_table = MagicMock()
    mock_table.get_item.return_value = {"Item": {
        "task_id": "t1",
        "jobs": {"r1": {
            "job_id": "r1", "kind": "recommendation",
            "status": "BENCHMARKING",
            "recommender_job_name": "rec-1",
            "endpoint_name": "e1", "endpoint_config_name": "ec1",
            "inference_component_name": "ic1", "model_name": "m1",
        }}}}
    with patch.object(h, "boto3") as mock_boto, \
         patch.object(h, "_ddb_table", return_value=mock_table), \
         patch.object(h, "_parse_profile_export_jsonl",
               return_value={}) as mock_parse, \
         patch.object(h, "_teardown_recommendation_resources") as mock_td, \
         patch.object(h.time, "sleep"):
        mock_boto.client.return_value = mock_sm
        out = h._get_recommendation_results({
            "thread_id": "t1", "recommender_job_name": "rec-1",
            "_user_id": "functest"})
        assert mock_parse.call_count == 3  # 3 retries per F-B
        assert not mock_td.called           # teardown deferred
        assert out["status"] == "BENCHMARKING"
        assert "not yet in S3" in out["message"]


def test_get_recommendation_results_condition_check_failed_falls_through():
    """F-C: concurrent terminal-write race — the loser must NOT raise and
    must return the winner's DDB state via the re-read path."""
    from unittest.mock import patch
    from botocore.exceptions import ClientError
    h = _import_handler("sagemaker")
    importlib.reload(h)
    mock_sm = MagicMock()
    mock_sm.describe_ai_benchmark_job.return_value = {
        "AIBenchmarkJobStatus": "Completed",
        "OutputConfig": {"S3OutputLocation": "s3://b/rec/"}}
    # DDB table's meta.client.exceptions.ConditionalCheckFailedException
    # surfaces as a ClientError with that specific code.
    cond_fail = ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException",
                   "Message": "The conditional request failed"}},
        "UpdateItem")
    mock_table = MagicMock()
    mock_table.meta.client.exceptions.ConditionalCheckFailedException = \
        cond_fail.__class__
    # First get_item (pre-update) returns BENCHMARKING; second returns
    # whatever the winning caller wrote (COMPLETED).
    mock_table.get_item.side_effect = [
        {"Item": {"task_id": "t1",
                  "jobs": {"r1": {"job_id": "r1", "kind": "recommendation",
                                  "status": "BENCHMARKING",
                                  "recommender_job_name": "rec-1",
                                  "endpoint_name": "e1",
                                  "endpoint_config_name": "ec1",
                                  "inference_component_name": "ic1",
                                  "model_name": "m1"}}}},
        {"Item": {"task_id": "t1",
                  "jobs": {"r1": {"job_id": "r1", "kind": "recommendation",
                                  "status": "COMPLETED",
                                  "recommender_job_name": "rec-1",
                                  "results": {"ttft_ms_p99": 812},
                                  "teardown_complete": True}}}},
    ]
    mock_table.update_item.side_effect = cond_fail
    with patch.object(h, "boto3") as mock_boto, \
         patch.object(h, "_ddb_table", return_value=mock_table), \
         patch.object(h, "_parse_profile_export_jsonl",
               return_value={"ttft_ms_p99": 812}), \
         patch.object(h, "_teardown_recommendation_resources",
               return_value=True):
        mock_boto.client.return_value = mock_sm
        out = h._get_recommendation_results({
            "thread_id": "t1", "recommender_job_name": "rec-1",
            "_user_id": "functest"})
        assert out["status"] == "COMPLETED"       # from winner's write
        assert out["teardown_complete"] is True


# ─── R4: list_hub_models ──────────────────────────────────────────────────
def test_sagemaker_handler_dispatches_list_hub_models():
    """Handler strips sagemaker-skill___ prefix and routes to list_hub_models."""
    h = _import_handler("sagemaker")
    captured = _stub_dispatch(
        h, "list_hub_models",
        {"hub_name": "SageMakerPublicHub", "total": 2, "models": []},
    )
    resp = h.handler(
        {"filter": "Llama", "_user_id": "u"},
        _ctx("sagemaker-skill___list_hub_models"),
    )
    assert captured == [{"filter": "Llama", "_user_id": "u"}]
    assert resp.get("isError") is not True
    assert "SageMakerPublicHub" in resp["content"][0]["text"]


def test_list_hub_models_filters_and_shapes_response():
    """_list_hub_models paginates list_hub_contents, applies the filter
    substring (case-insensitive), and shapes each item with EULA +
    recipe tags parsed from HubContentSearchKeywords."""
    from unittest.mock import patch, MagicMock

    h = _import_handler("sagemaker")
    sm = MagicMock()

    # list_hub_contents → paginator.paginate() yields pages of summaries.
    llama_summary = {
        "HubContentName": "meta-llama/Llama-3.1-8B-Instruct",
        "HubContentVersion": "1.0.0",
        "HubContentDescription": "Llama 3.1 8B Instruct",
    }
    qwen_summary = {
        "HubContentName": "Qwen/Qwen2.5-0.5B-Instruct",
        "HubContentVersion": "1.0.0",
        "HubContentDescription": "Qwen 2.5 0.5B Instruct",
    }
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"HubContentSummaries": [llama_summary, qwen_summary]},
    ]
    sm.get_paginator.return_value = paginator

    # describe_hub_content returns different tag bundles per model.
    def _describe(HubName, HubContentType, HubContentName, HubContentVersion):  # noqa: N803
        if "Llama" in HubContentName:
            return {
                "HubContentArn": "arn:aws:sagemaker:us-east-1:x:hub-content/llama",
                "HubContentSearchKeywords": [
                    {"Key": "requires_eula", "Value": "true"},
                    {"Key": "eula_url", "Value": "https://llama.meta.com/llama3/license/"},
                    {"Key": "supported_recipes", "Value": "sft, dpo"},
                    {"Key": "license", "Value": "Llama-3 Community License"},
                ],
            }
        return {
            "HubContentArn": "arn:aws:sagemaker:us-east-1:x:hub-content/qwen",
            "HubContentSearchKeywords": [
                {"Key": "supported_recipes", "Value": "sft"},
                {"Key": "license", "Value": "Apache-2.0"},
            ],
        }

    sm.describe_hub_content.side_effect = _describe

    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.return_value = sm

        # No filter — both models returned.
        out = h._list_hub_models({"_user_id": "u"})
        assert out["hub_name"] == "SageMakerPublicHub"
        assert out["total"] == 2
        names = {m["name"] for m in out["models"]}
        assert names == {
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
        }

        # Filter narrows to Llama (case-insensitive).
        out = h._list_hub_models({"filter": "llama", "_user_id": "u"})
        assert out["total"] == 1
        llama = out["models"][0]
        assert llama["name"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert llama["requires_eula"] is True
        assert llama["eula_url"] == "https://llama.meta.com/llama3/license/"
        assert llama["supported_recipes"] == ["sft", "dpo"]
        assert llama["license"] == "Llama-3 Community License"
        assert llama["arn"].endswith("/llama")


def test_list_hub_models_swallows_per_item_describe_errors():
    """One malformed Hub entry must not nuke the whole listing."""
    from unittest.mock import patch, MagicMock

    h = _import_handler("sagemaker")
    sm = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{
        "HubContentSummaries": [
            {"HubContentName": "good-model", "HubContentVersion": "1.0", "HubContentDescription": ""},
            {"HubContentName": "broken-model", "HubContentVersion": "1.0", "HubContentDescription": ""},
        ]
    }]
    sm.get_paginator.return_value = paginator

    def _describe(HubName, HubContentType, HubContentName, HubContentVersion):  # noqa: N803
        if HubContentName == "broken-model":
            raise RuntimeError("simulated DescribeHubContent failure")
        return {"HubContentArn": f"arn::hub-content/{HubContentName}",
                "HubContentSearchKeywords": []}

    sm.describe_hub_content.side_effect = _describe

    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.return_value = sm
        out = h._list_hub_models({"_user_id": "u"})

    # The good model survives; the broken one is skipped, not propagated.
    assert out["total"] == 1
    assert out["models"][0]["name"] == "good-model"


# ─── R5: deploy_model target=sagemaker|bedrock ───────────────────────────
def test_deploy_model_default_target_is_sagemaker():
    """Omitting `target` preserves the legacy SageMaker endpoint path."""
    from unittest.mock import patch, MagicMock

    h = _import_handler("sagemaker")
    sm = MagicMock()
    sm.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "s3://b/k/model.tar.gz"},
        "AlgorithmSpecification": {"TrainingImage": "IMG"},
    }
    bedrock = MagicMock()
    def _client(name, region_name=None):
        return sm if name == "sagemaker" else bedrock
    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.side_effect = _client
        out = h._deploy_model({
            "sagemaker_job_name": "jb",
            "endpoint_name": "ep-1",
        })
    assert out["target"] == "sagemaker"
    assert out["endpoint_name"] == "ep-1"
    # Bedrock client must NOT have been called.
    bedrock.create_model_import_job.assert_not_called()
    sm.create_endpoint.assert_called_once()


def test_deploy_model_bedrock_target_dispatches_background_import():
    """R5 async contract: target=bedrock returns SUBMITTING and hands the
    unpack+import to the deploy_model_bedrock background worker (Bedrock CMI
    needs an unpacked HF prefix, too slow for the tool-call budget)."""
    from unittest.mock import patch, MagicMock

    h = _import_handler("sagemaker")
    sm = MagicMock()
    sm.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "s3://b/k/output/model.tar.gz"},
        "AlgorithmSpecification": {"TrainingImage": "IMG"},
    }
    bg: list = []
    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.return_value = sm
        with patch.object(h, "_invoke_background", side_effect=bg.append):
            out = h._deploy_model({
                "sagemaker_job_name": "my-train-job_2026",
                "target": "bedrock",
            })
    assert out["target"] == "bedrock"
    assert out["status"] == "SUBMITTING"
    # Default bedrock_model_name = sanitised training job name (alnum + -_. ok).
    assert out["bedrock_model_name"] == "my-train-job_2026"
    assert bg and bg[0]["_bg_tool"] == "deploy_model_bedrock"


def test_deploy_model_bedrock_refuses_when_training_not_completed():
    from unittest.mock import patch, MagicMock

    h = _import_handler("sagemaker")
    sm = MagicMock()
    sm.describe_training_job.return_value = {
        "TrainingJobStatus": "InProgress",
        "ModelArtifacts": {"S3ModelArtifacts": ""},
        "AlgorithmSpecification": {"TrainingImage": "IMG"},
    }
    bedrock = MagicMock()
    def _client(name, region_name=None):
        return sm if name == "sagemaker" else bedrock
    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.side_effect = _client
        import pytest  # noqa: PLC0415
        with pytest.raises(RuntimeError) as exc:
            h._deploy_model({
                "sagemaker_job_name": "jb",
                "target": "bedrock",
            })
        assert "Completed" in str(exc.value)
    bedrock.create_model_import_job.assert_not_called()


def test_deploy_model_rejects_unknown_target():
    h = _import_handler("sagemaker")
    import pytest  # noqa: PLC0415
    with pytest.raises(ValueError) as exc:
        h._deploy_model({
            "sagemaker_job_name": "jb",
            "target": "vertex-ai",
        })
    assert "vertex-ai" in str(exc.value)


def test_deploy_model_bedrock_sanitises_model_name():
    """Training job names with slashes / colons become Bedrock-safe."""
    from unittest.mock import patch, MagicMock

    h = _import_handler("sagemaker")
    sm = MagicMock()
    sm.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "s3://b/k/m.tar.gz"},
        "AlgorithmSpecification": {"TrainingImage": "IMG"},
    }
    bedrock = MagicMock()
    bedrock.create_model_import_job.return_value = {"jobArn": "", "jobIdentifier": ""}
    def _client(name, region_name=None):
        return sm if name == "sagemaker" else bedrock
    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.side_effect = _client
        # Async contract: keep the background unpack worker out of this test
        # (its inline fallback would tar-read a MagicMock body forever).
        with patch.object(h, "_invoke_background"):
            out = h._deploy_model({
                "sagemaker_job_name": "meta/llama-3.1-8b train/run #42",
                "target": "bedrock",
            })
    # Spaces, slashes, and '#' become '-' (dot and underscore are kept).
    assert out["bedrock_model_name"] == "meta-llama-3.1-8b-train-run--42"
    assert all(c.isalnum() or c in "-_." for c in out["bedrock_model_name"])


# ─── R6: custom_scorer_lambda_arns validation ────────────────────────────
def test_validate_custom_scorer_arns_accepts_valid_arn():
    h = _import_handler("sagemaker")
    # Should not raise.
    h._validate_custom_scorer_arns([
        "arn:aws:lambda:us-east-1:123456789012:function:math-scorer-v1",
        "arn:aws:lambda:us-west-2:000000000000:function:code-scorer-prod:PROD",
    ])


def test_validate_custom_scorer_arns_rejects_malformed():
    h = _import_handler("sagemaker")
    import pytest  # noqa: PLC0415
    bad_cases = [
        "not-an-arn",
        "arn:aws:s3:::bucket",                          # wrong service
        "arn:aws:lambda:us-east-1:abc:function:x",      # non-numeric account
        "arn:aws:lambda:us-east-1:123456789012:function:",  # empty name
        ["not", "a", "string"],                         # wrong type
        None,                                           # wrong type
    ]
    for arn in bad_cases:
        with pytest.raises(ValueError):
            h._validate_custom_scorer_arns([arn])


def test_validate_custom_scorer_arns_rejects_non_list():
    h = _import_handler("sagemaker")
    import pytest  # noqa: PLC0415
    with pytest.raises(ValueError):
        h._validate_custom_scorer_arns("arn:aws:lambda:us-east-1:123456789012:function:x")


def test_validate_custom_scorer_arns_empty_list_is_ok():
    h = _import_handler("sagemaker")
    # Legacy callers that omit the arg → we default to [] in _submit_eval_job;
    # the validator must accept that.
    h._validate_custom_scorer_arns([])


# ─── R1 Tasks 7+8: submit_monitoring_job dispatch ────────────────────────
def test_sagemaker_handler_dispatches_submit_monitoring_job():
    """New monitoring tool: handler strips target prefix and dispatches."""
    h = _import_handler("sagemaker")
    captured = _stub_dispatch(
        h, "submit_monitoring_job",
        {"thread_id": "t1", "job_id": "jb1", "processing_job_name": "p1"},
    )
    resp = h.handler(
        {"thread_id": "t1", "sagemaker_job_name": "src-job",
         "use_training_eval_split": True, "_user_id": "u"},
        _ctx("sagemaker-skill___submit_monitoring_job"),
    )
    assert captured == [{"thread_id": "t1", "sagemaker_job_name": "src-job",
                         "use_training_eval_split": True, "_user_id": "u"}]
    assert resp.get("isError") is not True
    assert "p1" in resp["content"][0]["text"]


def test_sagemaker_handler_monitoring_env_missing_raises():
    """MONITORING_IMAGE_URI unset → MCP error (CDK wiring incomplete)."""
    h = _import_handler("sagemaker")
    h.MONITORING_IMAGE_URI = ""
    resp = h.handler(
        {"thread_id": "t1", "sagemaker_job_name": "src", "_user_id": "u"},
        _ctx("sagemaker-skill___submit_monitoring_job"),
    )
    assert resp.get("isError") is True
    assert "MONITORING_IMAGE_URI" in resp["content"][0]["text"]


def test_dispatch_rejects_env_var_placeholders():
    """QA BUG-014: literal CURRENT_THREAD_ID / ${CURRENT_USER_ID} placeholders
    must be rejected with a self-correcting error, not forwarded to AWS."""
    handler = _import_handler("sagemaker")
    ctx = MagicMock()
    ctx.client_context.custom = {"bedrockAgentCoreToolName": "sagemaker-skill___submit_monitoring_job"}
    for bad in ("CURRENT_THREAD_ID", "${CURRENT_THREAD_ID}", "$CURRENT_THREAD_ID"):
        out = handler.handler({"thread_id": bad, "sagemaker_job_name": "j"}, ctx)
        assert out.get("isError"), f"placeholder {bad!r} must be rejected"
        assert "placeholder" in out["content"][0]["text"]
    out = handler.handler({"thread_id": "real-uuid", "_user_id": "CURRENT_USER_ID",
                           "sagemaker_job_name": "j"}, ctx)
    assert out.get("isError") and "placeholder" in out["content"][0]["text"]
