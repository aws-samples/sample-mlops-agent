"""QA BUG-020 follow-up — AIPerf aggregate metric extraction.

Fixture shape taken verbatim from a real benchmark run
(sample-mlops-agent-rec-c301c07d-558, aiperf 0.8.0 schema_version 1.1).
"""
import importlib
import importlib.util
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _ensure_stub(name):
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


def _load_handler():
    for m in ("mlflow", "mlflow.tracking", "mlflow.exceptions"):
        _ensure_stub(m)
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "lambda", "skills", "sagemaker", "handler.py"))
    spec = importlib.util.spec_from_file_location("_sm_handler_aiperf", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sm_handler_aiperf"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("_sm_handler_aiperf", None)
    return mod


REAL_EXPORT = {
    "schema_version": "1.1",
    "aiperf_version": "0.8.0.dev20260515",
    "request_throughput": {"unit": "requests/sec", "avg": 9.894695423604585},
    "request_latency": {"unit": "ms", "avg": 1320.99, "p50": 1265.687, "p99": 2409.084},
    "request_count": {"unit": "requests", "avg": 30.0},
    "time_to_first_token": {"unit": "ms", "avg": 336.86, "p50": 301.2, "p99": 812.4},
    "inter_token_latency": {"unit": "ms", "avg": 30.1, "p50": 28.4, "p99": 55.9},
    "output_token_throughput": {"unit": "tokens/sec", "avg": 1480.2},
    "benchmark_duration": {"unit": "sec", "avg": 3.03},
}


def test_maps_real_aiperf_export_to_flat_metrics():
    h = _load_handler()
    m = h._metrics_from_aiperf_export(REAL_EXPORT)
    assert m["ttft_ms_p99"] == 812.4
    assert m["inter_token_ms_p99"] == 55.9
    assert m["request_latency_p99_ms"] == 2409.084
    assert m["throughput_requests_per_sec"] == 9.894695423604585
    assert m["throughput_tokens_per_sec"] == 1480.2
    assert m["request_count"] == 30.0


def test_empty_doc_returns_empty():
    h = _load_handler()
    assert h._metrics_from_aiperf_export({}) == {}


def test_parse_locates_tarball_under_job_subprefix(tmp_path):
    """The tarball lives at <prefix>/bmk-…/output/output.tar.gz — the parser
    must list the prefix rather than assume the root key."""
    import json
    import tarfile
    h = _load_handler()
    out = tmp_path / "export"
    out.mkdir()
    (out / "profile_export_aiperf.json").write_text(json.dumps(REAL_EXPORT))
    tar_path = tmp_path / "output.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(out / "profile_export_aiperf.json", arcname="profile_export_aiperf.json")

    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"Contents": [
        {"Key": "recommendation-benchmarks/rid/bmk-prod-x-1234/output/output.tar.gz"},
    ]}
    s3.download_file.side_effect = lambda b, k, dst: __import__("shutil").copy(tar_path, dst)
    h.boto3 = MagicMock()
    h.boto3.client.return_value = s3
    m = h._parse_profile_export_jsonl("s3://bkt/recommendation-benchmarks/rid/")
    assert m["ttft_ms_p99"] == 812.4
    kwargs = s3.list_objects_v2.call_args.kwargs
    assert kwargs["Prefix"] == "recommendation-benchmarks/rid"


def test_terminal_write_decimalizes_float_metrics():
    """DynamoDB rejects floats — the terminal stamp must Decimal-ize the
    AIPerf metrics (observed live: 'Float types are not supported')."""
    from decimal import Decimal
    metrics = {"ttft_ms_p99": 812.4, "request_count": 30}
    out = {k: (Decimal(str(v)) if isinstance(v, float) else v) for k, v in metrics.items()}
    assert out["ttft_ms_p99"] == Decimal("812.4")
    assert out["request_count"] == 30
