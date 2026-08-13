"""Unit tests for the SageMaker skill's deploy_model tool (R5).

Pins the dispatch contract and the bedrock target's side-effects:
  * target=sagemaker (default) — unchanged legacy path
  * target=bedrock — calls bedrock.create_model_import_job with jobTags
    carrying ThreadId/JobId/Kind, and seeds DDB with kind=bedrock_import
  * target=<garbage> — raises ValueError (no silent fallback to sagemaker)

Heavy SDKs are stubbed out the same way test_lambda_handler_dispatch.py does
it; we only exercise pure-Python branching.
"""
import importlib
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _ensure_stub(name: str) -> None:
    if name not in sys.modules:
        mod = ModuleType(name)
        mod.__getattr__ = lambda attr, _m=mod: MagicMock(name=f"{name}.{attr}")  # type: ignore[attr-defined]
        sys.modules[name] = mod


for _mod in ("mlflow", "mlflow.tracking", "mlflow.exceptions"):
    _ensure_stub(_mod)


def _import_sagemaker_handler():
    """Import lambda/skills/sagemaker/handler.py freshly each call."""
    lam_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "lambda", "skills", "sagemaker")
    )
    for p in list(sys.path):
        if "/lambda/skills/" in p:
            sys.path.remove(p)
    sys.path.insert(0, lam_dir)
    sys.modules.pop("handler", None)
    return importlib.import_module("handler")


# ---------- target=sagemaker (legacy path) ----------

def test_deploy_model_default_target_calls_sagemaker_path():
    """Caller without target= must hit _deploy_model_sagemaker (unchanged)."""
    h = _import_sagemaker_handler()
    captured: dict = {}

    def _stub(args):
        captured["args"] = args
        return {"target": "sagemaker", "endpoint_name": args["endpoint_name"]}

    h._deploy_model_sagemaker = _stub  # type: ignore[attr-defined]
    out = h._deploy_model({"sagemaker_job_name": "j1", "endpoint_name": "ep1"})
    assert out == {"target": "sagemaker", "endpoint_name": "ep1"}
    assert captured["args"]["endpoint_name"] == "ep1"


def test_deploy_model_explicit_sagemaker_target():
    """target='sagemaker' must route to the same legacy path."""
    h = _import_sagemaker_handler()
    h._deploy_model_sagemaker = lambda args: {"target": "sagemaker", "endpoint_name": "x"}  # type: ignore[attr-defined]
    out = h._deploy_model({"target": "sagemaker", "sagemaker_job_name": "j", "endpoint_name": "x"})
    assert out["target"] == "sagemaker"


# ---------- target=bedrock ----------



def test_deploy_model_bedrock_rejects_non_completed_training_job():
    """Bedrock import requires the source training job to be Completed."""
    h = _import_sagemaker_handler()
    sm_client = MagicMock()
    sm_client.describe_training_job.return_value = {"TrainingJobStatus": "InProgress"}
    h.boto3.client = lambda n, region_name=None: sm_client  # type: ignore[attr-defined]
    try:
        h._deploy_model_bedrock({"sagemaker_job_name": "still-running"})
    except RuntimeError as e:
        assert "Completed" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for non-Completed job")


# ---------- target=<garbage> ----------

def test_deploy_model_unknown_target_raises():
    """Fail loudly: unknown target must NOT silently fall back to sagemaker."""
    h = _import_sagemaker_handler()
    try:
        h._deploy_model({"target": "vertex", "sagemaker_job_name": "j", "endpoint_name": "ep"})
    except ValueError as e:
        assert "vertex" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown target")


# ── R5 closure: async unpack-then-import contract ──────────────────────────


def test_deploy_model_bedrock_seeds_unpacking_and_fires_background():
    """target=bedrock returns immediately (SUBMITTING), seeds a DDB record in
    UNPACKING state, and hands off to the deploy_model_bedrock background
    worker — Bedrock CMI needs an unpacked HF prefix, which takes minutes."""
    h = _import_sagemaker_handler()
    sm_client = MagicMock()
    sm_client.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "s3://b/k/output/model.tar.gz"},
    }
    h.boto3.client = lambda n, region_name=None: sm_client  # type: ignore[attr-defined]
    table = MagicMock()
    h._ddb_table = lambda: table  # type: ignore[attr-defined]
    bg_calls: list = []
    h._invoke_background = lambda payload: bg_calls.append(payload)  # type: ignore[attr-defined]

    out = h._deploy_model_bedrock({
        "sagemaker_job_name": "j", "thread_id": "t1", "_user_id": "u1",
    })
    assert out["target"] == "bedrock" and out["status"] == "SUBMITTING"
    seeded = [c.kwargs["ExpressionAttributeValues"]
              for c in table.update_item.call_args_list
              if ":job" in c.kwargs.get("ExpressionAttributeValues", {})]
    assert seeded and seeded[0][":job"]["kind"] == "bedrock_import"
    assert seeded[0][":job"]["status"] == "UNPACKING"
    assert bg_calls and bg_calls[0]["_bg_tool"] == "deploy_model_bedrock"
    assert bg_calls[0]["artifact_s3"] == "s3://b/k/output/model.tar.gz"


def test_deploy_model_bedrock_without_thread_id_skips_ddb_but_dispatches():
    """Without thread_id there is no poller resume — skip the seed but still
    run the background import."""
    h = _import_sagemaker_handler()
    sm_client = MagicMock()
    sm_client.describe_training_job.return_value = {
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "s3://b/k/output/model.tar.gz"},
    }
    h.boto3.client = lambda n, region_name=None: sm_client  # type: ignore[attr-defined]
    ddb_called: list = []
    h._ddb_table = lambda: ddb_called.append(1) or MagicMock()  # type: ignore[attr-defined]
    bg_calls: list = []
    h._invoke_background = lambda payload: bg_calls.append(payload)  # type: ignore[attr-defined]
    out = h._deploy_model_bedrock({"sagemaker_job_name": "j"})
    assert out["status"] == "SUBMITTING"
    assert ddb_called == []
    assert bg_calls


def _build_sft_tarball(tmp_path):
    """model.tar.gz shaped like a real SFT artifact (root HF files + debris)."""
    import tarfile
    src = tmp_path / "src"
    (src / "checkpoint-100").mkdir(parents=True)
    (src / "baseline").mkdir()
    (src / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    (src / "model.safetensors").write_text("weights")
    (src / "tokenizer_config.json").write_text("{}")
    (src / "training_args.bin").write_text("argsbin")
    (src / "checkpoint-100" / "optimizer.pt").write_text("opt")
    (src / "baseline" / "baseline.csv").write_text("a,b\n")
    tar_path = tmp_path / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                tf.add(f, arcname=str(f.relative_to(src)), recursive=False)
    return tar_path


def test_background_deploy_bedrock_unpacks_hf_prefix_and_imports(tmp_path):
    """R5 live-verify regression: CMI rejects model.tar.gz ("could not find
    …/model.tar.gz/config.json"). The worker must upload the root HF files to
    an hf-import prefix (excluding checkpoints/baseline/optimizer debris) and
    point the import at the PREFIX, then stamp IN_PROGRESS with the job id."""
    h = _import_sagemaker_handler()
    tar_path = _build_sft_tarball(tmp_path)
    uploads: list = []
    s3 = MagicMock()
    s3.get_object.side_effect = lambda Bucket, Key: {"Body": open(tar_path, "rb")}
    s3.upload_fileobj.side_effect = lambda fobj, b, k: uploads.append(k)
    bedrock = MagicMock()
    bedrock.create_model_import_job.return_value = {
        "jobArn": "arn:aws:bedrock:us-east-1:1:model-import-job/x", "jobIdentifier": "x"}
    h.boto3.client = lambda n, region_name=None: {"s3": s3, "bedrock": bedrock}[n]  # type: ignore[attr-defined]
    table = MagicMock()
    h._ddb_table = lambda: table  # type: ignore[attr-defined]

    h._background_deploy_bedrock({
        "thread_id": "t1", "job_id": "j1", "user_id": "u1",
        "artifact_s3": "s3://bkt/training-output/job/output/model.tar.gz",
        "bedrock_model_name": "m1", "sagemaker_job_name": "job",
    })
    assert sorted(uploads) == [
        "training-output/job/output/hf-import/config.json",
        "training-output/job/output/hf-import/model.safetensors",
        "training-output/job/output/hf-import/tokenizer_config.json",
    ], f"unexpected uploads: {uploads}"
    kwargs = bedrock.create_model_import_job.call_args.kwargs
    assert kwargs["modelDataSource"]["s3DataSource"]["s3Uri"] == \
        "s3://bkt/training-output/job/output/hf-import/"
    assert {t["key"] for t in kwargs["jobTags"]} == {"ThreadId", "JobId", "Kind"}
    vals = table.update_item.call_args.kwargs["ExpressionAttributeValues"]
    assert vals[":s"] == "IN_PROGRESS" and vals[":i"] == "x"


def test_unpack_raises_without_config_json(tmp_path):
    """A tarball with no root config.json is not an HF model — fail loudly
    before creating a doomed import job."""
    import tarfile
    h = _import_sagemaker_handler()
    src = tmp_path / "src"
    src.mkdir()
    (src / "model.xgb").write_text("boosted")
    tar_path = tmp_path / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src / "model.xgb", arcname="model.xgb", recursive=False)
    s3 = MagicMock()
    s3.get_object.side_effect = lambda Bucket, Key: {"Body": open(tar_path, "rb")}
    h.boto3.client = lambda n, region_name=None: s3  # type: ignore[attr-defined]
    try:
        h._unpack_model_to_hf_prefix("s3://b/j/output/model.tar.gz", "s3://b/j/output/hf-import")
    except RuntimeError as e:
        assert "config.json" in str(e)
    else:
        raise AssertionError("expected RuntimeError for non-HF tarball")
