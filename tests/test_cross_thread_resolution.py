"""QA BUG-004 — cross-thread source-job resolution in the sagemaker skill.

Starter tiles open a fresh thread, so `_find_source_job_entry` must fall back
to the job's own thread (via its SageMaker ThreadId/JobId resource tags) when
the current thread row has no matching entry. `list_recent_training_jobs`
gives the agent cross-session discovery.
"""
import importlib.util
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _ensure_stub(name: str) -> None:
    """Stub heavy deps only when not installed (session-safe)."""
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


import importlib  # noqa: E402


def _load_handler():
    for m in ("mlflow", "mlflow.tracking", "mlflow.exceptions"):
        _ensure_stub(m)
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "lambda", "skills", "sagemaker", "handler.py"))
    spec = importlib.util.spec_from_file_location("_sm_handler_xthread", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sm_handler_xthread"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("_sm_handler_xthread", None)
    return mod


def _ddb_with_rows(rows: dict):
    """MagicMock DDB table whose get_item serves from a task_id→Item map."""
    table = MagicMock()
    table.get_item.side_effect = lambda Key: {"Item": rows.get(Key["task_id"], {})}
    return table


def test_find_source_entry_in_current_thread():
    h = _load_handler()
    entry = {"sagemaker_job_name": "job-A", "training_type": "xgboost"}
    h._ddb_table = lambda: _ddb_with_rows({"t-cur": {"jobs": {"j1": entry}}})
    got = h._find_source_job_entry(
        sm=MagicMock(), thread_id="t-cur", source_job="job-A",
        desc={"TrainingJobArn": "arn:job-A"},
    )
    assert got == entry


def test_find_source_entry_falls_back_to_tagged_thread():
    """The BUG-004 case: job lives in ANOTHER thread; resolve via tags."""
    h = _load_handler()
    entry = {"sagemaker_job_name": "job-B", "training_type": "xgboost",
             "baseline_s3_uri": "s3://b/base.csv"}
    h._ddb_table = lambda: _ddb_with_rows({
        "t-cur": {"jobs": {}},
        "t-src": {"jobs": {"j9": entry}},
    })
    sm = MagicMock()
    sm.list_tags.return_value = {"Tags": [
        {"Key": "ThreadId", "Value": "t-src"},
        {"Key": "JobId", "Value": "j9"},
    ]}
    got = h._find_source_job_entry(
        sm=sm, thread_id="t-cur", source_job="job-B",
        desc={"TrainingJobArn": "arn:job-B"},  # no inline Tags → list_tags fallback
    )
    assert got == entry


def test_find_source_entry_returns_none_without_tags():
    h = _load_handler()
    h._ddb_table = lambda: _ddb_with_rows({"t-cur": {"jobs": {}}})
    sm = MagicMock()
    sm.list_tags.return_value = {"Tags": []}
    got = h._find_source_job_entry(
        sm=sm, thread_id="t-cur", source_job="job-C",
        desc={"TrainingJobArn": "arn:job-C"},
    )
    assert got is None


def test_list_recent_training_jobs_enriches_from_tagged_threads():
    h = _load_handler()
    entry = {"training_type": "sft", "dataset_name": "ultrachat"}
    h._ddb_table = lambda: _ddb_with_rows({"t-src": {"jobs": {"j1": entry}}})
    sm = MagicMock()
    sm.list_training_jobs.return_value = {"TrainingJobSummaries": [{
        "TrainingJobName": "sample-mlops-agent-job-qwen-1",
        "TrainingJobStatus": "Completed",
        "TrainingJobArn": "arn:1",
        "CreationTime": "2026-07-24",
    }]}
    sm.list_tags.return_value = {"Tags": [
        {"Key": "ThreadId", "Value": "t-src"},
        {"Key": "JobId", "Value": "j1"},
    ]}
    h.boto3 = MagicMock()
    h.boto3.client.return_value = sm
    out = h._list_recent_training_jobs({"status_equals": "Completed"})
    assert out["jobs"][0]["sagemaker_job_name"] == "sample-mlops-agent-job-qwen-1"
    assert out["jobs"][0]["training_type"] == "sft"
    assert out["jobs"][0]["dataset_name"] == "ultrachat"
    assert out["jobs"][0]["thread_id"] == "t-src"
    # SageMaker list must be project-scoped and newest-first
    kwargs = sm.list_training_jobs.call_args.kwargs
    assert kwargs["SortBy"] == "CreationTime" and kwargs["SortOrder"] == "Descending"
    assert kwargs["StatusEquals"] == "Completed"
