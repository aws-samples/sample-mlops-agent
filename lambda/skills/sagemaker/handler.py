"""SageMaker skill Lambda — Gateway MCP target.

Exposes 8 tools to the AgentCore Gateway:
  - submit_training_job
  - complete_training_job
  - deploy_model
  - list_hub_models             (read-only: Hub model discovery; see R4 plan)
  - submit_eval_job
  - submit_monitoring_job       (R1: batch drift + quality via Evidently)
  - submit_recommendation_job
  - get_recommendation_results

Each tool maps to the equivalent logic previously in agent container scripts.
The Lambda handler dispatches on the MCP tool name from the Gateway event.
"""
import json
import logging
import os
import time
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import re
import boto3


# R6: custom_scorer_lambda_arns validation regex. Matches a Lambda ARN
# with an optional :alias or :version suffix. Pinned at module scope so
# handler + tests share one definition.
_LAMBDA_ARN_RE = re.compile(
    r"^arn:aws:lambda:[a-z0-9-]+:\d{12}:function:[A-Za-z0-9-_]+(:[A-Za-z0-9-_$]+)?$"
)


def _validate_custom_scorer_arns(arns: list[str]) -> None:
    """Reject malformed ARNs up-front so we fail in the Lambda rather than
    at Processing-job boot time. Does NOT check that the scorer Lambdas
    exist — that's a v1.1 follow-up (lambda:GetFunction probe).
    """
    if not isinstance(arns, list):
        raise ValueError(
            f"custom_scorer_lambda_arns must be a list, got {type(arns).__name__}."
        )
    for arn in arns:
        if not isinstance(arn, str) or not _LAMBDA_ARN_RE.match(arn):
            raise ValueError(
                f"Invalid Lambda ARN in custom_scorer_lambda_arns: {arn!r}. "
                f"Expected arn:aws:lambda:<region>:<account>:function:<name>[:alias]."
            )


def _json_default(obj: Any) -> Any:
    """json.dumps default hook — converts DDB Decimals to native numerics."""
    if isinstance(obj, Decimal):
        i = int(obj)
        return i if i == obj else float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROJECT_NAME = os.environ.get("PROJECT_NAME", "sample-mlops-agent")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "sample-mlops-agent-metadata")
SESSION_BUCKET = os.environ.get("SESSION_BUCKET", "")
SM_ROLE = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
EVAL_IMAGE_URI = os.environ.get("EVAL_IMAGE_URI", "")
# R1 Task 8: monitoring container image URI. CDK sets this on the
# sagemaker skill Lambda via sagemakerSkillFn.addEnvironment().
MONITORING_IMAGE_URI = os.environ.get("MONITORING_IMAGE_URI", "")
# Default judge model when submit_eval_job callers don't supply one. Uses
# the US cross-region inference profile so it routes across us-east/us-west
# without requiring on-demand access in the caller's region.
DEFAULT_JUDGE_MODEL = os.environ.get(
    "DEFAULT_JUDGE_MODEL",
    "bedrock:/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)
# Deduplicate submissions-in-flight within this window. If the agent retries
# a submit because the MCP round-trip appeared to time out, the retry should
# return the already-in-flight job rather than creating a second one.
_IDEMPOTENCY_WINDOW_SEC = 600

# boto3's default log level spams "Found credentials in environment variables"
# on every client creation. Suppress once at module load.
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

ENTRYPOINTS = {
    "grpo": "grpo_train.py",
    "sft": "sft_train.py",
    "dpo": "dpo_train.py",
    "xgboost": "xgb_train.py",
    "sklearn": "sklearn_train.py",
}

# Training scripts are packaged alongside this Lambda in the training/ subdirectory
_TRAINING_DIR = Path(__file__).parent / "training"


def _ddb_table() -> Any:
    """Return DynamoDB Table resource."""
    return boto3.resource("dynamodb", region_name=AWS_REGION).Table(JOBS_TABLE)


def _get_hf_token_optional() -> str:
    """Return HuggingFace token from SSM, or empty string if unavailable.

    Non-fatal fallback: public HF datasets don't require a token, so missing
    SSM parameter is not an error. Gated datasets/models will raise 401 at
    dataset-load time, which surfaces as a clear training-script failure.
    """
    try:
        resp = boto3.client("ssm", region_name=AWS_REGION).get_parameter(
            Name=f"/{PROJECT_NAME}/dev/hf-token", WithDecryption=True
        )
        return resp["Parameter"]["Value"]
    except Exception as exc:
        # Logs only the exception TYPE — the token value never reaches the logger.
        logger.info(  # nosemgrep: python-logger-credential-disclosure
            "[submit_training_job] HF token not in SSM (ok for public datasets): %s",
                    type(exc).__name__)
        return ""


def _resolve_entrypoint(training_type: str) -> str:
    """Return the entrypoint script filename for the requested training type.

    Raises:
        ValueError: If training_type is not recognised.
    """
    script_name = ENTRYPOINTS.get(training_type)
    if not script_name:
        raise ValueError(f"Unknown training_type: {training_type!r}. Valid: {list(ENTRYPOINTS)}")
    return script_name


def _start_mlflow_run(
    *,
    thread_id: str,
    job_id: str,
    job_name: str,
    model_id: str,
    dataset_name: str,
    training_type: str,
    instance_type: str,
    max_steps: int,
    learning_rate: float,
    user_id: str,
) -> tuple[str, str, str]:
    """Create a RUNNING MLflow run so the training container resumes it via MLFLOW_RUN_ID.

    Mirrors the pre-Gateway pattern (commit a40f8f8 / faffb95): the orchestration
    layer pre-creates the run in RUNNING state via MlflowClient; HuggingFace
    MLflowCallback (or the training script's ``_init_mlflow``) resumes it via
    ``MLFLOW_RUN_ID`` and closes it when training finishes.

    Returns:
        tuple[str, str, str]: (experiment_id, run_id, run_url). Empty strings
        if MLFLOW_TRACKING_URI is not configured — caller must treat that as a
        "MLflow disabled" signal rather than abort.
    """
    if not MLFLOW_TRACKING_URI:
        return "", "", ""

    # Imported lazily so the Lambda cold-start cost is only paid when MLflow is wired.
    import mlflow  # noqa: PLC0415
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # One experiment per thread so every retry/submission on the same thread
    # lands in the same MLflow experiment view. Idempotent lookup-then-create.
    experiment_name = f"{PROJECT_NAME}/{thread_id}"
    exp = client.get_experiment_by_name(experiment_name)
    experiment_id = exp.experiment_id if exp else client.create_experiment(experiment_name)

    run = client.create_run(
        experiment_id=experiment_id,
        run_name=job_name,
        tags={
            "thread_id":     thread_id,
            "job_id":        job_id,
            "user_id":       user_id,
            "model_id":      model_id,
            "dataset_name":  dataset_name,
            "training_type": training_type,
            "instance_type": instance_type,
        },
    )
    run_id = run.info.run_id
    # Log params individually — training container will add more via MLflowCallback.
    for key, value in {
        "model_id":      model_id,
        "dataset_name":  dataset_name,
        "training_type": training_type,
        "max_steps":     max_steps,
        "learning_rate": learning_rate,
    }.items():
        client.log_param(run_id, key, str(value))

    run_url = f"{MLFLOW_TRACKING_URI.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
    return experiment_id, run_id, run_url


# ── Idempotency + async self-invoke helpers ────────────────────────────────
#
# The SageMaker Python SDK + image_uris.retrieve cold-path takes ~60 s. That
# exceeds the MCP tool round-trip budget, so the agent sees a timeout and
# retries. Meanwhile the first submission succeeded quietly — the retry
# creates a duplicate job name and hits ResourceLimitExceeded because the
# first job is holding the quota.
#
# Fix: the sync path writes a "SUBMITTING" row to DDB, fires a background
# self-invocation that does the slow SageMaker work, and returns in ≤ 5 s
# with job_id + mlflow_run_url. The background invocation updates DDB to
# PENDING/FAILED once CreateTrainingJob / CreateProcessingJob returns.


def _find_in_flight_duplicate(
    *, thread_id: str, fingerprint: dict, now: int,
) -> dict | None:
    """Return an existing job record matching `fingerprint` if one was submitted
    on this thread within _IDEMPOTENCY_WINDOW_SEC and is still pre-terminal.

    Fingerprint is the tuple of user-visible fields that would cause two
    submissions to represent "the same request" — for training: (kind,
    model_id, dataset_name, training_type); for eval: (kind, target_model,
    dataset_s3_uri, task).
    """
    ddb = _ddb_table()
    resp = ddb.get_item(Key={"task_id": thread_id})
    row = resp.get("Item")
    if not row:
        return None
    pre_terminal = {"SUBMITTING", "PENDING", "IN_PROGRESS"}
    for entry in (row.get("jobs") or {}).values():
        if entry.get("status") not in pre_terminal:
            continue
        if (now - int(entry.get("created_at") or 0)) > _IDEMPOTENCY_WINDOW_SEC:
            continue
        if all(entry.get(k) == v for k, v in fingerprint.items()):
            return entry
    return None


def _invoke_background(payload: dict) -> None:
    """Self-invoke this Lambda asynchronously to run the slow SageMaker SDK
    work outside the synchronous MCP tool-call response path.
    """
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
    if not function_name:
        # Local/test execution — fall through to inline exec so tests still cover the path.
        _background_dispatcher(payload)
        return
    client = boto3.client("lambda", region_name=AWS_REGION)
    client.invoke(
        FunctionName=function_name,
        InvocationType="Event",
        Payload=json.dumps(payload).encode(),
    )


def _mark_job(thread_id: str, job_id: str, *, status: str, message: str, extras: dict | None = None) -> None:
    """Patch the job entry's status + message + optional extra fields in DDB.
    Used by the background path to report PENDING/FAILED after the real
    CreateTrainingJob / CreateProcessingJob call returns.
    """
    set_clauses = [
        "jobs.#jid.#s         = :s",
        "jobs.#jid.status_message = :m",
        "jobs.#jid.updated_at = :t",
        "updated_at           = :t",
    ]
    names: dict[str, str] = {"#jid": job_id, "#s": "status"}
    values: dict[str, Any] = {":s": status, ":m": message, ":t": int(time.time())}
    if extras:
        for idx, (k, v) in enumerate(extras.items()):
            placeholder = f":extra{idx}"
            name_placeholder = f"#e{idx}"
            set_clauses.append(f"jobs.#jid.{name_placeholder} = {placeholder}")
            names[name_placeholder] = k
            values[placeholder] = v
    _ddb_table().update_item(
        Key={"task_id": thread_id},
        UpdateExpression="SET " + ", ".join(set_clauses),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


class _NonSeekableReader:
    """Minimal file-like adapter for boto3 upload_fileobj over tarfile
    stream-mode members (QA R5 closure): exposes read() and an explicit
    seekable() -> False so boto3 takes its non-seekable multipart path
    instead of raising AttributeError on the probe."""

    def __init__(self, fobj: Any) -> None:
        self._fobj = fobj

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes from the underlying tar member stream."""
        return self._fobj.read(size)

    def seekable(self) -> bool:
        """Declare the stream non-seekable (sequential tar member)."""
        return False


def _unpack_model_to_hf_prefix(artifact_s3: str, dest_prefix: str) -> int:
    """Extract root-level HuggingFace model files from model.tar.gz to an S3
    prefix (QA R5 closure — Bedrock Custom Model Import requires unpacked
    HF-format objects, not a tarball).

    Skips training debris: checkpoint-*/ subfolders, the tabular baseline/
    folder, and optimizer/rng state files — Bedrock only needs the model
    weights + tokenizer + config JSONs.

    Args:
        artifact_s3: s3:// URI of model.tar.gz.
        dest_prefix: s3:// prefix to upload the unpacked files under.

    Returns:
        Number of files uploaded.

    Raises:
        RuntimeError: If the tarball contains no config.json (not an HF model).
    """
    import tarfile  # noqa: PLC0415
    from urllib.parse import urlparse  # noqa: PLC0415

    src, dst = urlparse(artifact_s3), urlparse(dest_prefix)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    uploaded, saw_config = 0, False
    # Stream the tarball straight from S3 (mode "r|gz", sequential) — SFT
    # artifacts include checkpoint-*/optimizer states and reach 6.8+ GiB,
    # far beyond Lambda /tmp; streaming needs no local disk at any size.
    body = s3.get_object(Bucket=src.netloc, Key=src.path.lstrip("/"))["Body"]
    with tarfile.open(fileobj=body, mode="r|gz") as tf:
        for member in tf:
            name = member.name.lstrip("./")
            if (not member.isfile() or "/" in name  # root-level files only
                    or name in ("training_args.bin",)
                    or ".." in name):
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            if name == "config.json":
                saw_config = True
            # boto3 probes fileobj.seekable(); tarfile's stream-mode member
            # object (_Stream-backed) doesn't implement it. Wrap with an
            # explicit non-seekable reader — boto3 then multipart-buffers
            # small chunks in memory, which is exactly what we want here.
            s3.upload_fileobj(_NonSeekableReader(fobj), dst.netloc,
                              f"{dst.path.strip('/')}/{name}")
            uploaded += 1
    if not saw_config:
        raise RuntimeError(
            f"{artifact_s3} contains no root-level config.json — not a "
            "HuggingFace-format model; Bedrock import would be rejected."
        )
    return uploaded


def _background_deploy_bedrock(payload: dict) -> None:
    """Async worker: unpack the artifact to HF format, then start the
    Bedrock Custom Model Import (QA R5 closure)."""
    thread_id = payload.get("thread_id", "")
    job_id = payload["job_id"]
    artifact_s3 = payload["artifact_s3"]
    bedrock_model_name = payload["bedrock_model_name"]
    hf_prefix = artifact_s3.rsplit("/output/", 1)[0] + "/output/hf-import"
    try:
        n = _unpack_model_to_hf_prefix(artifact_s3, hf_prefix)
        logger.info("[deploy_model:bg] unpacked %d files to %s", n, hf_prefix)
        bedrock = boto3.client("bedrock", region_name=AWS_REGION)
        create_kwargs: dict = dict(
            jobName=bedrock_model_name,
            importedModelName=bedrock_model_name,
            # Bedrock assumes the DEDICATED import role (bedrock.amazonaws.com
            # trust); the SageMaker execution role cannot be assumed by it.
            roleArn=os.environ.get("BEDROCK_IMPORT_ROLE_ARN") or SM_ROLE,
            modelDataSource={"s3DataSource": {"s3Uri": hf_prefix + "/"}},
        )
        if thread_id:
            create_kwargs["jobTags"] = [
                {"key": "ThreadId", "value": thread_id},
                {"key": "JobId", "value": job_id},
                {"key": "Kind", "value": "bedrock_import"},
            ]
        resp = bedrock.create_model_import_job(**create_kwargs)
        if thread_id:
            _ddb_table().update_item(
                Key={"task_id": thread_id},
                UpdateExpression=("SET jobs.#jid.#s = :s, jobs.#jid.status_message = :m, "
                                  "jobs.#jid.import_job_arn = :a, "
                                  "jobs.#jid.import_job_identifier = :i, "
                                  "jobs.#jid.hf_prefix_s3_uri = :p, updated_at = :t"),
                ExpressionAttributeNames={"#jid": job_id, "#s": "status"},
                ExpressionAttributeValues={
                    ":s": "IN_PROGRESS",
                    ":m": f"Bedrock import job {bedrock_model_name} submitted",
                    ":a": resp.get("jobArn", ""),
                    ":i": resp.get("jobIdentifier", "") or resp.get("jobArn", ""),
                    ":p": hf_prefix,
                    ":t": int(time.time()),
                },
            )
    except Exception as exc:
        logger.exception("[deploy_model:bg] bedrock import setup failed")
        if thread_id:
            _mark_job(thread_id, job_id, status="FAILED",
                      message=f"Bedrock import setup failed: {type(exc).__name__}: {exc}")


def _background_dispatcher(payload: dict) -> None:
    """Route a self-invoke event to the correct background worker. Payload
    shape: {"_bg_tool": "submit_training_job" | "submit_eval_job" | "submit_recommendation_job" | "submit_monitoring_job", ...}.
    """
    bg_tool = payload.get("_bg_tool")
    if bg_tool == "submit_training_job":
        _background_submit_training(payload)
    elif bg_tool == "submit_eval_job":
        _background_submit_eval(payload)
    elif bg_tool == "submit_recommendation_job":
        _background_submit_recommendation(payload)
    elif bg_tool == "submit_monitoring_job":
        _background_submit_monitoring(payload)
    elif bg_tool == "deploy_model_bedrock":
        _background_deploy_bedrock(payload)
    else:
        logger.error("[_background_dispatcher] unknown _bg_tool: %r", bg_tool)


# Region → HuggingFace TGI DLC URI lookup (F-3 from eng review: regex-
# derivation off the training image was producing an ECR URI that may not
# exist. Hardcoded known-good pin is more operationally predictable; users
# with non-HF training stacks override via serving_image_uri instead.
# Tag source: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
_TGI_DLC_BY_REGION = {
    "us-east-1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
                 "huggingface-pytorch-tgi-inference:"
                 "2.7.0-tgi3.3.6-gpu-py311-cu124-ubuntu22.04-v1.1",
    "us-west-2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                 "huggingface-pytorch-tgi-inference:"
                 "2.7.0-tgi3.3.6-gpu-py311-cu124-ubuntu22.04-v1.1",
    # Add more as we onboard regions. Absent region → None so the caller
    # falls back to the serving_image_uri override.
}


def _derive_serving_from_training(training_image: str) -> str | None:
    """Return the pinned TGI serving DLC for the current region if the
    training image looks like a PyTorch SageMaker DLC. Regex-derivation
    off the training tag (replaced by this pin per F-3) was brittle; we
    now only confirm the training image is a recognisable PyTorch DLC
    and return the region-pinned TGI URI verbatim."""
    if not training_image or "pytorch-training:" not in training_image:
        return None
    return _TGI_DLC_BY_REGION.get(AWS_REGION)


def _tgi_default_env(args: dict) -> dict:
    """Default Environment dict for the TGI container's PrimaryContainer.

    TGI constraints: MAX_INPUT_LENGTH < MAX_TOTAL_TOKENS. MAX_INPUT_LENGTH
    is the prompt cap in tokens; MAX_TOTAL_TOKENS = prompt + completion cap.
    We give the prompt a 256-token margin over the requested input_tokens,
    then TOTAL = MAX_INPUT_LENGTH + output_tokens + 256 (safety buffer)."""
    input_tokens = int(args.get("input_tokens", 500))
    output_tokens = int(args.get("output_tokens", 150))
    max_input_length = input_tokens + 256
    max_total_tokens = max_input_length + output_tokens + 256
    return {
        "HF_MODEL_ID":      "/opt/ml/model",
        "SM_NUM_GPUS":      "1",
        "MAX_INPUT_LENGTH": str(max_input_length),
        "MAX_TOTAL_TOKENS": str(max_total_tokens),
        # QA BUG-020 follow-up: serve the OpenAI Messages API on /invocations.
        # The AI Benchmark's AIPerf client speaks api_standard=openai; against
        # TGI's native generate API every request failed to parse
        # (InvalidInferenceResultError 30/30, benchmark error rate 100%).
        "MESSAGES_API_ENABLED": "true",
    }

def _compute_workload_fingerprint(workload_spec: dict) -> str:
    """SHA-256 of a canonicalised workload spec, for idempotency dedup."""
    import hashlib
    import json  # noqa: PLC0415
    blob = json.dumps(workload_spec, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()

def _estimate_wall_clock_minutes(instance_type: str) -> int:
    """Coarse wall-clock estimate surfaced in the submit response so the
    agent can tell the user how long to wait. Endpoint provisioning +
    benchmark run. Order-of-magnitude only."""
    # g5 family provisions ~6 min; g6 ~5 min; p-class ~10 min.
    # Benchmark run typically 10–20 min at default concurrency.
    if instance_type.startswith("ml.p"):
        return 35
    if instance_type.startswith("ml.g6"):
        return 20
    return 25  # g5 and everything else


def _submit_training_job(args: dict) -> dict:
    """Submit a SageMaker training job and write its entry into the thread row.

    Unified-thread-row model: one DynamoDB item per thread (PK=task_id holds
    the thread_id value). Each training job is stored under ``jobs.<job_id>``
    as a nested map. SageMaker ThreadId/JobId resource tags travel with the
    job so the EventBridge callback can resolve them without a GSI.

    An MLflow run is pre-created in RUNNING state and its run_id is injected
    into the training container via the ``MLFLOW_RUN_ID`` env var so
    HuggingFace's MLflowCallback resumes it rather than creating a new run.

    Args:
        args: Tool arguments from Gateway. Required keys:
            thread_id, model_id, dataset_name.
            Optional: training_type, instance_type, max_steps,
                      learning_rate, max_samples, train_split,
                      test_split, _user_id.

    `train_split` / `test_split` default to "train" / "test" but must be
    overridden for HuggingFace datasets that use non-default split names
    (e.g. HuggingFaceH4/ultrachat_200k → "train_sft" / "test_sft").
    The HF Hub's error ("Unknown split 'train'. Should be one of [...]")
    is the signal to re-submit with explicit split names.

    Returns:
        dict: thread_id, job_id, sagemaker_job_name, mlflow_run_id, mlflow_run_url.
    """
    thread_id = args["thread_id"]
    model_id = args["model_id"]
    dataset_name = args["dataset_name"]
    training_type = args.get("training_type", "grpo")
    instance_type = args.get("instance_type", "ml.g5.2xlarge")
    max_steps = int(args.get("max_steps", 500))
    learning_rate = float(args.get("learning_rate", 1e-5))
    max_samples = int(args.get("max_samples", 0))
    train_split = args.get("train_split", "train")
    test_split = args.get("test_split", "test")
    # XGBoost-only hyperparameters; kept None so the training script can fall
    # back to its own defaults when the caller didn't override them.
    xgb_max_depth = args.get("max_depth")
    xgb_n_estimators = args.get("n_estimators")
    # Tabular-only (xgboost/sklearn): the label column in the loaded
    # DataFrame. The training script raises immediately if this is
    # missing for s3:// or HF inputs, defaults to "target" for bundled
    # sklearn datasets.
    target_column = args.get("target_column")
    user_id = args.get("_user_id", "")
    # Log the incoming args keys + user_id/thread_id to make Gateway arg-stripping
    # regressions easy to spot (if _user_id is missing from the keys list, it's
    # being filtered out by the Gateway tool schema).
    logger.info(
        "[submit_training_job] arg_keys=%s thread_id=%r user_id=%r",
        sorted(args.keys()), thread_id, user_id,
    )

    now = int(time.time())
    # Idempotency: if an agent retry lands while the first submission is still
    # in-flight, return the already-submitted job rather than double-submitting.
    fingerprint = {
        "kind":          "training",
        "model_id":      model_id,
        "dataset_name":  dataset_name,
        "training_type": training_type,
    }
    existing = _find_in_flight_duplicate(
        thread_id=thread_id, fingerprint=fingerprint, now=now,
    )
    if existing:
        logger.info(
            "[submit_training_job] returning in-flight duplicate job_id=%r status=%r",
            existing.get("job_id"), existing.get("status"),
        )
        return {
            "thread_id":          thread_id,
            "job_id":             existing["job_id"],
            "sagemaker_job_name": existing.get("sagemaker_job_name", ""),
            "mlflow_run_id":      existing.get("mlflow_run_id", ""),
            "mlflow_run_url":     existing.get("mlflow_run_url", ""),
            "status":             existing.get("status", "SUBMITTING"),
            "deduplicated":       True,
        }

    # Extract last path segment, lowercase, strip special chars, truncate to 12 chars for job name
    slug = model_id.split("/")[-1].lower().replace("-", "").replace(".", "")[:12]
    job_id = str(uuid.uuid4())
    job_name = f"{PROJECT_NAME}-job-{slug}-{int(time.time())}"

    # Pre-create the MLflow run so the training container resumes it, not creates its own.
    experiment_id, mlflow_run_id, mlflow_run_url = _start_mlflow_run(
        thread_id=thread_id,
        job_id=job_id,
        job_name=job_name,
        model_id=model_id,
        dataset_name=dataset_name,
        training_type=training_type,
        instance_type=instance_type,
        max_steps=max_steps,
        learning_rate=learning_rate,
        user_id=user_id,
    )

    ddb = _ddb_table()
    # Pre-pass: ensure the jobs map exists so SET jobs.#jid = :job won't fail
    # on a thread that has had no chat turn yet. Also seed created_at / user_id
    # so a submission-first thread carries timestamps.
    ddb.update_item(
        Key={"task_id": thread_id},
        UpdateExpression=(
            "SET jobs       = if_not_exists(jobs, :empty_map), "
            "    created_at = if_not_exists(created_at, :t), "
            "    user_id    = if_not_exists(user_id, :uid), "
            "    thread_id  = if_not_exists(thread_id, :tid)"
        ),
        ExpressionAttributeValues={
            ":empty_map": {},
            ":t":         now,
            ":uid":       user_id,
            ":tid":       thread_id,
        },
    )

    job_record = {
        "job_id":          job_id,
        "kind":            "training",
        "model_id":        model_id,
        "dataset_name":    dataset_name,
        "instance_type":   instance_type,
        "training_type":   training_type,
        "sagemaker_job_name": job_name,
        "status":          "SUBMITTING",
        "status_message":  f"Queueing job {job_name} for SageMaker CreateTrainingJob",
        "created_at":      now,
        "updated_at":      now,
        "mlflow_experiment_id": experiment_id,
        "mlflow_run_id":   mlflow_run_id,
        "mlflow_run_url":  mlflow_run_url,
    }
    ddb.update_item(
        Key={"task_id": thread_id},
        UpdateExpression="SET jobs.#jid = :job, updated_at = :t",
        ExpressionAttributeNames={"#jid": job_id},
        ExpressionAttributeValues={":job": job_record, ":t": now},
    )

    # Fire the slow CreateTrainingJob on a self-invoked background Lambda so
    # this sync path returns well inside the MCP tool-call budget.
    _invoke_background({
        "_bg_tool":         "submit_training_job",
        "thread_id":        thread_id,
        "job_id":           job_id,
        "job_name":         job_name,
        "model_id":         model_id,
        "dataset_name":     dataset_name,
        "training_type":    training_type,
        "instance_type":    instance_type,
        "max_steps":        max_steps,
        "learning_rate":    learning_rate,
        "max_samples":      max_samples,
        "train_split":      train_split,
        "test_split":       test_split,
        "xgb_max_depth":    xgb_max_depth,
        "xgb_n_estimators": xgb_n_estimators,
        "target_column":    target_column,
        "mlflow_run_id":    mlflow_run_id,
    })

    return {
        "thread_id":          thread_id,
        "job_id":              job_id,
        "sagemaker_job_name": job_name,
        "mlflow_run_id":       mlflow_run_id,
        "mlflow_run_url":      mlflow_run_url,
        "status":              "SUBMITTING",
    }


def _background_submit_training(payload: dict) -> None:
    """Run the slow SageMaker SDK path out-of-band. Updates DDB to PENDING on
    success or FAILED on any exception with the real error text — so the
    frontend and the agent see the ground truth rather than a generic timeout.
    """
    thread_id     = payload["thread_id"]
    job_id        = payload["job_id"]
    job_name      = payload["job_name"]
    model_id      = payload["model_id"]
    dataset_name  = payload["dataset_name"]
    training_type = payload["training_type"]
    instance_type = payload["instance_type"]
    max_steps     = payload["max_steps"]
    learning_rate = payload["learning_rate"]
    max_samples   = payload["max_samples"]
    # Back-compat: older in-flight payloads won't have split keys. Default
    # to "train" / "test" so those still work for HF datasets that use
    # the canonical split names.
    train_split   = payload.get("train_split", "train")
    test_split    = payload.get("test_split", "test")
    # XGBoost-only overrides — None means "use xgb_train.py's defaults".
    xgb_max_depth    = payload.get("xgb_max_depth")
    xgb_n_estimators = payload.get("xgb_n_estimators")
    target_column    = payload.get("target_column")
    mlflow_run_id = payload.get("mlflow_run_id", "")

    try:
        # Use the SageMaker Python SDK's PyTorch estimator (not boto3
        # create_training_job) so the container's built-in sagemaker-training
        # bootloader correctly reads SAGEMAKER_PROGRAM / SAGEMAKER_SUBMIT_DIRECTORY
        # and launches our entry point. Matches commit a40f8f8.
        import sagemaker  # noqa: PLC0415
        from sagemaker.pytorch import PyTorch  # noqa: PLC0415

        script_name = _resolve_entrypoint(training_type)

        # Upload training config JSON to S3. Training scripts read
        # CONFIG_S3_BUCKET / CONFIG_S3_KEY at runtime to fetch this blob,
        # which carries the nested model/data/training/grpo/reward config the
        # scripts expect. Without this, scripts raise
        # KeyError: 'CONFIG_S3_BUCKET' before training starts.
        # XGBoost section — xgb_train.py reads cfg["xgb"] and falls back to
        # its own defaults for any key we omit. Forward learning_rate as
        # eta (xgboost's canonical name for the shrinkage parameter) so the
        # same top-level learning_rate argument works for LLM and tree models.
        xgb_section: dict[str, Any] = {"eta": learning_rate}
        if xgb_max_depth is not None:
            xgb_section["max_depth"] = int(xgb_max_depth)
        if xgb_n_estimators is not None:
            xgb_section["num_round"] = int(xgb_n_estimators)

        data_section: dict[str, Any] = {
            "dataset_name": dataset_name,
            "train_split": train_split,
            "test_split": test_split,
            "max_samples": max_samples,
        }
        # Tabular training reads `target_column` from the data section.
        # LLM training ignores it, but including it unconditionally keeps
        # the config shape consistent across training_type values.
        if target_column:
            data_section["target_column"] = target_column
        config_payload = {
            "config": {
                "model":    {"name": model_id, "trust_remote_code": True},
                "data":     data_section,
                "training": {"max_steps": max_steps,
                             "learning_rate": learning_rate,
                             "bf16": True,
                             "gradient_checkpointing": True,
                             "gradient_accumulation_steps": 8},
                "grpo":     {"num_generations": 2, "max_completion_length": 512},
                "reward":   {"target_length": 200},
                "xgb":      xgb_section,
            },
            "training_type": training_type,
        }
        config_key = f"training-configs/{job_name}.json"
        boto3.client("s3", region_name=AWS_REGION).put_object(
            Bucket=SESSION_BUCKET,
            Key=config_key,
            Body=json.dumps(config_payload),
        )

        training_env: dict[str, str] = {
            "CONFIG_S3_BUCKET": SESSION_BUCKET,
            "CONFIG_S3_KEY":    config_key,
        }
        if MLFLOW_TRACKING_URI and mlflow_run_id:
            training_env["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
            training_env["MLFLOW_RUN_ID"] = mlflow_run_id
        # HF_TOKEN is optional — only needed for gated HuggingFace models or
        # datasets. Non-fatal if the SSM parameter is missing; public datasets
        # (e.g. HuggingFaceH4/ultrachat_200k) don't require auth.
        hf_token = _get_hf_token_optional()
        if hf_token:
            training_env["HF_TOKEN"] = hf_token

        boto_session = boto3.session.Session(region_name=AWS_REGION)
        sm_session = sagemaker.Session(boto_session=boto_session, default_bucket=SESSION_BUCKET)

        estimator = PyTorch(
            entry_point=script_name,
            source_dir=str(_TRAINING_DIR),
            role=SM_ROLE,
            instance_type=instance_type,
            instance_count=1,
            # PyTorch 2.4 (py311) — matches the training scripts' pinned stack
            # (trl==0.21.0, accelerate==1.7.0, transformers==4.56.0). The old
            # 2.1.0/py310 image shipped torch 2.1, which lacks
            # torch.utils._pytree.register_pytree_node — a public API
            # transformers 4.56+ imports at module load time. A training run on
            # 2026-04-25 (job sample-mlops-agent-job-qwen2505bins-1777082331)
            # died on exactly that AttributeError before reaching user code.
            framework_version="2.4.0",
            py_version="py311",
            max_run=86400,
            volume_size=30,
            output_path=f"s3://{SESSION_BUCKET}/training-output/",
            hyperparameters={
                "model_id":       model_id,
                "dataset_name":   dataset_name,
                "max_steps":      max_steps,
                "learning_rate":  learning_rate,
                "max_samples":    max_samples,
            },
            environment=training_env,
            tags=[
                {"Key": "ThreadId",    "Value": thread_id},
                {"Key": "JobId",       "Value": job_id},
                {"Key": "ProjectName", "Value": PROJECT_NAME},
            ],
            sagemaker_session=sm_session,
        )
        estimator.fit(job_name=job_name, wait=False)
    except Exception as exc:
        logger.exception("[submit_training_job:bg] CreateTrainingJob failed")
        _mark_job(
            thread_id, job_id,
            status="FAILED",
            message=f"CreateTrainingJob failed: {type(exc).__name__}: {exc}",
        )
        return

    _mark_job(
        thread_id, job_id,
        status="PENDING",
        message=f"SageMaker accepted {job_name}; waiting for instance provisioning",
    )


def _complete_training_job(args: dict) -> dict:
    """Retrieve final status and artifacts for a completed training job.

    Args:
        args: Tool arguments. Required: sagemaker_job_name.

    Returns:
        dict: status, artifact_s3, sagemaker_job_name.
    """
    job_name = args["sagemaker_job_name"]
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    desc = sm.describe_training_job(TrainingJobName=job_name)
    status = desc["TrainingJobStatus"]
    artifact_s3 = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
    return {"sagemaker_job_name": job_name, "status": status, "artifact_s3": artifact_s3}


def _deploy_model_sagemaker(args: dict) -> dict:
    """Deploy a trained model artifact to a SageMaker real-time endpoint.

    Original single-target deploy path. Kept verbatim so existing callers
    that omit ``target`` see bit-for-bit unchanged behaviour.

    Args:
        args: Required: sagemaker_job_name, endpoint_name.
              Optional: instance_type (default ml.m5.xlarge).

    Returns:
        dict: target, endpoint_name, endpoint_url.
    """
    job_name = args["sagemaker_job_name"]
    endpoint_name = args["endpoint_name"]
    instance_type = args.get("instance_type", "ml.m5.xlarge")

    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    desc = sm.describe_training_job(TrainingJobName=job_name)
    artifact_s3 = desc["ModelArtifacts"]["S3ModelArtifacts"]
    image_uri = desc["AlgorithmSpecification"]["TrainingImage"]

    model_name = f"{endpoint_name}-model"
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=SM_ROLE,
        PrimaryContainer={"Image": image_uri, "ModelDataUrl": artifact_s3},
    )
    sm.create_endpoint_config(
        EndpointConfigName=f"{endpoint_name}-config",
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "InstanceType": instance_type,
            "InitialInstanceCount": 1,
        }],
    )
    sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=f"{endpoint_name}-config")
    endpoint_url = f"https://runtime.sagemaker.{AWS_REGION}.amazonaws.com/endpoints/{endpoint_name}/invocations"
    return {
        "target": "sagemaker",
        "endpoint_name": endpoint_name,
        "endpoint_url": endpoint_url,
    }


def _deploy_model_bedrock(args: dict) -> dict:
    """Import a fine-tuned model artifact into Bedrock via Custom Model Import.

    Uses ``bedrock.create_model_import_job`` to pull the training job's
    S3 model artifact into a new Bedrock model. The import runs async.

    R5 follow-up: persists a ``jobs.<job_id>`` row with
    ``kind='bedrock_import'`` so the bedrock-import poller (EventBridge
    schedule, 15-min cadence — see ``lambda/bedrock_import_poller``)
    can find the in-flight import, call ``get_model_import_job``, and
    on terminal status resume the agent thread. The Bedrock service does
    NOT emit an EventBridge ``Bedrock Model Import Job State Change``
    event (verified against the Bedrock EventBridge docs, May 2026 —
    only ``Model Customization Job State Change`` and
    ``Batch Inference Job State Change`` are emitted), so a scheduled
    poller is the only viable async-resume path.

    Args:
        args: Required: sagemaker_job_name.
              Optional: bedrock_model_name (default: training job name
              with non-alphanumerics stripped; Bedrock model names must
              match ``^[a-zA-Z0-9-_.]+$``), thread_id (CURRENT_THREAD_ID,
              required for poller resume), _user_id (CURRENT_USER_ID,
              required for poller resume identity).

    Returns:
        dict: target, bedrock_model_name, import_job_arn,
              import_job_identifier, sagemaker_job_name, thread_id, job_id.
    """
    job_name = args["sagemaker_job_name"]
    thread_id = args.get("thread_id", "") or ""
    user_id = args.get("_user_id", "") or ""

    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    desc = sm.describe_training_job(TrainingJobName=job_name)
    status = desc.get("TrainingJobStatus")
    if status != "Completed":
        raise RuntimeError(
            f"Training job {job_name!r} is in status {status!r}. "
            f"Bedrock import requires Completed."
        )
    artifact_s3 = desc["ModelArtifacts"]["S3ModelArtifacts"]

    # Derive a safe default Bedrock model name from the training job
    # name: Bedrock's Custom Model Import validates ^[a-zA-Z0-9-_.]+$
    # and caps at 63 chars.
    default_name = "".join(c if c.isalnum() or c in "-_." else "-" for c in job_name)[:63]
    bedrock_model_name = args.get("bedrock_model_name") or default_name

    job_id = str(uuid.uuid4())
    # QA R5 closure: Bedrock Custom Model Import requires the model as an
    # UNPACKED HuggingFace-format S3 prefix (config.json etc. as individual
    # objects) — pointing it at model.tar.gz fails with "could not find the
    # expected file …/model.tar.gz/config.json". Unpacking a ~1 GB artifact
    # takes minutes, far beyond the Gateway tool-call budget, so this tool
    # seeds the DDB record and hands off to the background self-invoke,
    # matching the submit_* tools' pattern. The 15-min poller resumes the
    # thread when the import reaches a terminal state.
    now = int(time.time())
    if thread_id:
        ddb = _ddb_table()
        ddb.update_item(
            Key={"task_id": thread_id},
            UpdateExpression=(
                "SET jobs       = if_not_exists(jobs, :empty_map), "
                "    created_at = if_not_exists(created_at, :t), "
                "    user_id    = if_not_exists(user_id, :uid), "
                "    thread_id  = if_not_exists(thread_id, :tid)"
            ),
            ExpressionAttributeValues={
                ":empty_map": {}, ":t": now, ":uid": user_id, ":tid": thread_id,
            },
        )
        ddb.update_item(
            Key={"task_id": thread_id},
            UpdateExpression="SET jobs.#jid = :job, updated_at = :t",
            ExpressionAttributeNames={"#jid": job_id},
            ExpressionAttributeValues={":job": {
                "job_id": job_id, "kind": "bedrock_import",
                "sagemaker_job_name": job_name,
                "bedrock_model_name": bedrock_model_name,
                "model_artifact_s3_uri": artifact_s3,
                "status": "UNPACKING",
                "status_message": "Unpacking model.tar.gz to HuggingFace-format S3 prefix for Bedrock import",
                "created_at": now, "updated_at": now,
            }, ":t": now},
        )
    else:
        logger.warning(
            "[deploy_model] target=bedrock called without thread_id; "
            "poller resume will not be available for this import."
        )

    _invoke_background({
        "_bg_tool": "deploy_model_bedrock",
        "thread_id": thread_id, "job_id": job_id, "user_id": user_id,
        "artifact_s3": artifact_s3, "bedrock_model_name": bedrock_model_name,
        "sagemaker_job_name": job_name,
    })

    return {
        "target":             "bedrock",
        "bedrock_model_name": bedrock_model_name,
        "sagemaker_job_name": job_name,
        "thread_id":          thread_id,
        "job_id":             job_id,
        "status":             "SUBMITTING",
        "message": ("Unpacking the artifact to HuggingFace format and starting the "
                    "Bedrock Custom Model Import in the background (~2 min), then the "
                    "import itself runs ~10-25 min. The session is resumed automatically "
                    "on a terminal state (15-min poller cadence)."),
    }


def _deploy_model(args: dict) -> dict:
    """Deploy a trained model — dispatcher on ``target``.

    R5: ``target`` selects the deploy mechanism.
      - ``"sagemaker"`` (default, legacy): create a SageMaker real-time
        endpoint via ``_deploy_model_sagemaker``.
      - ``"bedrock"``: Bedrock Custom Model Import via
        ``_deploy_model_bedrock``.

    Callers that omit ``target`` see the legacy SageMaker path unchanged.
    """
    target = (args.get("target") or "sagemaker").lower()
    if target == "sagemaker":
        return _deploy_model_sagemaker(args)
    if target == "bedrock":
        return _deploy_model_bedrock(args)
    raise ValueError(
        f"target={target!r} is not supported. Use 'sagemaker' or 'bedrock'."
    )


def _list_hub_models(args: dict) -> dict:
    """List SageMaker Hub models available for fine-tuning.

    Discovers fine-tuneable models in a SageMaker Hub (default: the public
    ``SageMakerPublicHub``) and returns per-model metadata the agent needs
    before picking a ``model_id`` for ``submit_training_job``: name, version,
    EULA status and URL, supported recipes (if declared in hub tags), license,
    and the Hub content ARN.

    Read-only tool. Pairs with the R2 EULA hard-rule in
    ``agent/.claude/skills/planning/SKILL.md``: when ``requires_eula=True``,
    the agent must surface the EULA terms and get explicit affirmative
    acceptance before calling ``submit_training_job``.

    Args:
        args: Tool arguments.
            - ``hub_name`` (str, optional): Hub to enumerate. Defaults to
              ``"SageMakerPublicHub"``.
            - ``filter`` (str, optional): Case-insensitive substring match
              applied to ``HubContentName``. Useful for narrowing e.g. to
              ``"Llama"`` or ``"Nova"``.
            - ``_user_id`` (str): ``CURRENT_USER_ID``. Currently unused at
              call time; accepted for parity with other tools.

    Returns:
        dict with keys:
            - ``hub_name`` (str)
            - ``total`` (int) — number of models after filter.
            - ``models`` (list[dict]) — each with ``name``, ``version``,
              ``arn``, ``requires_eula`` (bool), ``eula_url`` (str, may be
              empty), ``supported_recipes`` (list[str]), ``license`` (str),
              ``description`` (str).
    """
    hub_name = args.get("hub_name") or "SageMakerPublicHub"
    filter_substr = (args.get("filter") or "").lower().strip()
    sm = boto3.client("sagemaker", region_name=AWS_REGION)

    models: list[dict[str, Any]] = []
    paginator = sm.get_paginator("list_hub_contents")
    # HubContentType=Model restricts to fine-tuneable model artifacts.
    # Non-model content (notebooks, tutorials) is filtered out at the Hub
    # level rather than post-hoc so we don't pay DescribeHubContent for them.
    for page in paginator.paginate(HubName=hub_name, HubContentType="Model"):
        for item in page.get("HubContentSummaries", []):
            name = item.get("HubContentName", "")
            if filter_substr and filter_substr not in name.lower():
                continue

            # DescribeHubContent gives us the full metadata blob including
            # tags. We swallow per-item errors so one malformed Hub entry
            # does not nuke the whole listing.
            version = item.get("HubContentVersion", "")
            try:
                detail = sm.describe_hub_content(
                    HubName=hub_name,
                    HubContentType="Model",
                    HubContentName=name,
                    HubContentVersion=version,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("describe_hub_content failed for %s@%s: %s", name, version, exc)
                continue

            # Hub content tags are used to declare EULA and recipe support.
            # We tolerate their absence — fall back to empty/False defaults.
            tags = {t.get("Key", ""): t.get("Value", "") for t in detail.get("HubContentSearchKeywords", [])}
            requires_eula = str(tags.get("requires_eula", "false")).lower() == "true"
            eula_url = tags.get("eula_url", "") or detail.get("HubContentMarkdown", "")[:0]  # no marker = empty
            recipes_raw = tags.get("supported_recipes", "")
            supported_recipes = [r.strip() for r in recipes_raw.split(",") if r.strip()]
            license_str = tags.get("license", "") or detail.get("HubContentDocument", "")[:0]

            models.append({
                "name":               name,
                "version":            version,
                "arn":                detail.get("HubContentArn", ""),
                "requires_eula":      requires_eula,
                "eula_url":           eula_url,
                "supported_recipes":  supported_recipes,
                "license":            license_str,
                "description":        item.get("HubContentDescription", ""),
            })

    return {"hub_name": hub_name, "total": len(models), "models": models}


def _submit_eval_job(args: dict) -> dict:
    """Launch an async SageMaker Processing job that runs mlflow.genai.evaluate.

    Mirrors ``_submit_training_job``: pre-creates the MLflow run, seeds a
    ``kind: "eval"`` entry on the thread row, and tags the Processing job
    with ThreadId/JobId/Kind so the EventBridge callback can close the row
    on state-change events.

    Args:
        args: Required: thread_id, eval_dataset_s3_uri, target_model,
              judge_model, scorers, task. Optional: instance_type
              (default ml.m5.large — SageMaker Processing only allows
              x86_64 instance families), run_name, _user_id.

    Returns:
        dict: thread_id, job_id, processing_job_name,
              mlflow_run_id, mlflow_run_url.
    """
    if not EVAL_IMAGE_URI:
        raise RuntimeError("EVAL_IMAGE_URI env var not set — CDK wiring incomplete")

    thread_id     = args["thread_id"]
    dataset_uri   = args["eval_dataset_s3_uri"]
    target_model  = args["target_model"]
    judge_model   = args.get("judge_model") or DEFAULT_JUDGE_MODEL
    scorers       = args["scorers"]
    # R6: optional list of Lambda ARNs that implement the custom-scorer
    # contract (see agent/.claude/skills/mlflow/SKILL.md §Custom Scorers).
    # Each arn must match ^arn:aws:lambda:[a-z0-9-]+:\d+:function:[A-Za-z0-9-_]+$.
    # The eval container invokes each Lambda per row with a
    # {inputs, outputs, expectations} payload and logs the returned
    # {name, score, reason?} to the MLflow run alongside the built-in
    # scorer metrics. Omit or pass [] to preserve legacy behaviour.
    custom_scorer_arns = args.get("custom_scorer_lambda_arns") or []
    _validate_custom_scorer_arns(custom_scorer_arns)
    task          = args["task"]
    # SageMaker Processing instance-type enum has no Graviton/ARM families,
    # so the eval image is built linux/amd64 (see
    # cdk/lib/constructs/arm-build-construct.ts) and this defaults to an
    # x86_64 instance. Picking an ARM instance here would fail
    # CreateProcessingJob validation before the container ever runs.
    instance_type = args.get("instance_type", "ml.m5.large")
    run_name      = args.get("run_name")
    user_id       = args.get("_user_id", "")

    logger.info(
        "[submit_eval_job] arg_keys=%s thread_id=%r user_id=%r target=%r judge=%r scorers=%s",
        sorted(args.keys()), thread_id, user_id, target_model, judge_model, scorers,
    )

    now = int(time.time())
    fingerprint = {
        "kind":           "eval",
        "target_model":   target_model,
        "dataset_s3_uri": dataset_uri,
        "task":           task,
    }
    existing = _find_in_flight_duplicate(
        thread_id=thread_id, fingerprint=fingerprint, now=now,
    )
    if existing:
        logger.info(
            "[submit_eval_job] returning in-flight duplicate job_id=%r status=%r",
            existing.get("job_id"), existing.get("status"),
        )
        return {
            "thread_id":           thread_id,
            "job_id":              existing["job_id"],
            "processing_job_name": existing.get("processing_job_name", ""),
            "mlflow_run_id":       existing.get("mlflow_run_id", ""),
            "mlflow_run_url":      existing.get("mlflow_run_url", ""),
            "status":              existing.get("status", "SUBMITTING"),
            "deduplicated":        True,
        }

    job_id = str(uuid.uuid4())
    job_name = f"{PROJECT_NAME}-eval-{int(time.time())}"

    # Pre-create MLflow run — container resumes via MLFLOW_RUN_ID.
    experiment_id, mlflow_run_id, mlflow_run_url = _start_mlflow_run(
        thread_id=thread_id,
        job_id=job_id,
        job_name=run_name or job_name,
        model_id=target_model,
        dataset_name=dataset_uri,
        training_type=f"eval:{task}",
        instance_type=instance_type,
        max_steps=0,
        learning_rate=0.0,
        user_id=user_id,
    )

    ddb = _ddb_table()
    # Seed thread row + jobs map defensively (same pattern as training submit).
    ddb.update_item(
        Key={"task_id": thread_id},
        UpdateExpression=(
            "SET jobs       = if_not_exists(jobs, :empty_map), "
            "    created_at = if_not_exists(created_at, :t), "
            "    user_id    = if_not_exists(user_id, :uid), "
            "    thread_id  = if_not_exists(thread_id, :tid)"
        ),
        ExpressionAttributeValues={
            ":empty_map": {},
            ":t":         now,
            ":uid":       user_id,
            ":tid":       thread_id,
        },
    )
    job_record = {
        "job_id":                    job_id,
        "kind":                      "eval",
        "target_model":              target_model,
        "judge_model":               judge_model,
        "dataset_s3_uri":            dataset_uri,
        "scorers":                   scorers,
        "custom_scorer_lambda_arns": custom_scorer_arns,
        "task":                      task,
        "instance_type":             instance_type,
        "processing_job_name":       job_name,
        "status":                    "SUBMITTING",
        "status_message":            f"Queueing eval job {job_name} for SageMaker CreateProcessingJob",
        "created_at":                now,
        "updated_at":                now,
        "mlflow_experiment_id":      experiment_id,
        "mlflow_run_id":             mlflow_run_id,
        "mlflow_run_url":            mlflow_run_url,
    }
    ddb.update_item(
        Key={"task_id": thread_id},
        UpdateExpression="SET jobs.#jid = :job, updated_at = :t",
        ExpressionAttributeNames={"#jid": job_id},
        ExpressionAttributeValues={":job": job_record, ":t": now},
    )

    _invoke_background({
        "_bg_tool":                   "submit_eval_job",
        "thread_id":                  thread_id,
        "job_id":                     job_id,
        "job_name":                   job_name,
        "target_model":               target_model,
        "judge_model":                judge_model,
        "scorers":                    scorers,
        "custom_scorer_lambda_arns":  custom_scorer_arns,
        "task":                       task,
        "instance_type":              instance_type,
        "dataset_uri":                dataset_uri,
        "mlflow_run_id":              mlflow_run_id,
    })

    return {
        "thread_id":           thread_id,
        "job_id":              job_id,
        "processing_job_name": job_name,
        "mlflow_run_id":       mlflow_run_id,
        "mlflow_run_url":      mlflow_run_url,
        "status":              "SUBMITTING",
    }


def _background_submit_eval(payload: dict) -> None:
    """Background worker: run the slow CreateProcessingJob path and report
    PENDING / FAILED back to DDB with the real error text on failure.
    """
    thread_id     = payload["thread_id"]
    job_id        = payload["job_id"]
    job_name      = payload["job_name"]
    target_model  = payload["target_model"]
    judge_model   = payload["judge_model"]
    scorers       = payload["scorers"]
    custom_scorer_arns = payload.get("custom_scorer_lambda_arns") or []
    task          = payload["task"]
    instance_type = payload["instance_type"]
    dataset_uri   = payload["dataset_uri"]
    mlflow_run_id = payload.get("mlflow_run_id", "")

    try:
        # Bypass ScriptProcessor — its .run(code=...) expects a LOCAL path that
        # gets uploaded to S3 and mounted at /opt/ml/processing/input/code.
        # Passing a container-absolute path made the SDK try to resolve
        # "/opt/ml/code/entrypoint.py" on the Lambda filesystem, where it
        # obviously doesn't exist, failing with "code wasn't found".
        # The boto3 CreateProcessingJob API lets us invoke the entrypoint
        # already baked into the image via AppSpecification.ContainerEntrypoint.
        sm = boto3.client("sagemaker", region_name=AWS_REGION)
        env_vars = {
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_RUN_ID":       mlflow_run_id,
            "TARGET_MODEL":        target_model,
            "JUDGE_MODEL":         judge_model,
            "SCORERS":             ",".join(scorers),
            # R6: comma-separated Lambda ARNs the eval container will wrap
            # as MLflow scorers per row. Empty → no custom scorers.
            "CUSTOM_SCORER_LAMBDA_ARNS": ",".join(custom_scorer_arns),
            "EVAL_DATASET_S3_URI": dataset_uri,
            "TASK":                task,
            # Processing containers don't auto-set a region; entrypoint.py
            # needs this to construct bedrock-runtime / sagemaker-runtime
            # clients without hitting NoRegionError.
            "AWS_REGION":          AWS_REGION,
        }
        # HF_TOKEN is optional here — only gated datasets require it, and
        # financebench / ultrachat are public. Mirror the training path
        # (_background_submit_training) so gated HF datasets don't 401 at
        # load_dataset() time and leave the processing job to die inside
        # the MLflow predict_fn smoke call instead of at dataset load.
        hf_token = _get_hf_token_optional()
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token
        sm.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=SM_ROLE,
            AppSpecification={
                "ImageUri": EVAL_IMAGE_URI,
                "ContainerEntrypoint": ["python3", "/opt/ml/code/entrypoint.py"],
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": instance_type,
                    "VolumeSizeInGB": 30,
                }
            },
            Environment=env_vars,
            Tags=[
                {"Key": "ThreadId",    "Value": thread_id},
                {"Key": "JobId",       "Value": job_id},
                {"Key": "ProjectName", "Value": PROJECT_NAME},
                {"Key": "Kind",        "Value": "eval"},
            ],
            StoppingCondition={"MaxRuntimeInSeconds": 86400},
        )
    except Exception as exc:
        logger.exception("[submit_eval_job:bg] CreateProcessingJob failed")
        _mark_job(
            thread_id, job_id,
            status="FAILED",
            message=f"CreateProcessingJob failed: {type(exc).__name__}: {exc}",
        )
        return

    _mark_job(
        thread_id, job_id,
        status="IN_PROGRESS",
        message=f"SageMaker accepted {job_name}; waiting for instance provisioning",
    )


def _parse_profile_export_jsonl_from_path(local_path: str) -> dict:
    """Return the aggregate row of an AI Benchmark profile_export.jsonl.

    The file is one JSON record per line; the last record with
    concurrency == 'aggregate' carries the numbers we surface.
    Returns {} if the file is empty or malformed — caller must handle.
    """
    import json  # noqa: PLC0415
    last_agg = None
    try:
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("concurrency") == "aggregate":
                    last_agg = rec
            if last_agg is None:
                # Fall back to the final row if no explicit aggregate present.
                f.seek(0)
                rows = [json.loads(x) for x in f if x.strip()]
                last_agg = rows[-1] if rows else {}
    except Exception as e:
        print(f"[callback] profile_export.jsonl parse failed: {e}")
        return {}
    wanted = ("ttft_ms_p50", "ttft_ms_p99", "inter_token_ms_p50",
              "inter_token_ms_p99", "request_latency_p50_ms",
              "request_latency_p99_ms", "throughput_requests_per_sec",
              "throughput_tokens_per_sec")
    return {k: last_agg[k] for k in wanted if k in last_agg and
            isinstance(last_agg[k], (int, float))}


def _metrics_from_aiperf_export(doc: dict) -> dict:
    """Map an AIPerf ``profile_export_aiperf.json`` document onto the flat
    metric keys this flow surfaces (QA BUG-020 follow-up).

    Real AIPerf output (verified against benchmark job
    sample-mlops-agent-rec-c301c07d-558, aiperf 0.8.0): aggregates live in
    ``profile_export_aiperf.json`` as ``{metric: {unit, avg, p50, p99, …}}``;
    ``profile_export.jsonl`` holds per-request records only (no aggregate
    row), so the previous aggregate-row parser always returned {}.

    Args:
        doc: Parsed profile_export_aiperf.json contents.

    Returns:
        Flat metrics dict; empty when the document lacks the expected keys.
    """
    def pick(metric: str, stat: str) -> Any:
        return (doc.get(metric) or {}).get(stat)

    mapping = {
        "ttft_ms_p50":                pick("time_to_first_token", "p50"),
        "ttft_ms_p99":                pick("time_to_first_token", "p99"),
        "inter_token_ms_p50":         pick("inter_token_latency", "p50"),
        "inter_token_ms_p99":         pick("inter_token_latency", "p99"),
        "request_latency_p50_ms":     pick("request_latency", "p50"),
        "request_latency_p99_ms":     pick("request_latency", "p99"),
        "throughput_requests_per_sec": pick("request_throughput", "avg"),
        "throughput_tokens_per_sec":  pick("output_token_throughput", "avg"),
        "request_count":              pick("request_count", "avg"),
        "benchmark_duration_sec":     pick("benchmark_duration", "avg"),
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _parse_profile_export_jsonl(s3_uri: str) -> dict:
    """Locate the benchmark output tarball under ``s3_uri`` and parse metrics.

    QA BUG-020 follow-up, two real-output corrections:
      1. The AI Benchmark job writes ``output.tar.gz`` under a job-named
         subprefix (``<s3_uri>/bmk-…/output/output.tar.gz``), not at the
         prefix root — list the prefix instead of assuming the key.
      2. Aggregates come from ``profile_export_aiperf.json`` (see
         ``_metrics_from_aiperf_export``); the legacy aggregate-row jsonl
         parser is kept as a fallback for older output formats.
    """
    import tarfile  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    import os  # noqa: PLC0415
    s3 = boto3.client("s3")
    prefix = s3_uri[5:].rstrip("/")
    bucket, _, key_prefix = prefix.partition("/")
    tar_key = ""
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
        for obj in resp.get("Contents", []):
            if obj["Key"].endswith("output.tar.gz"):
                tar_key = obj["Key"]
                break
    except Exception as e:
        print(f"[recommendation] list under {key_prefix} failed: {e}")
        return {}
    if not tar_key:
        print(f"[recommendation] no output.tar.gz under {key_prefix} yet")
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        local_tar = os.path.join(tmp, "output.tar.gz")
        try:
            s3.download_file(bucket, tar_key, local_tar)
        except Exception as e:
            print(f"[recommendation] could not fetch {tar_key}: {e}")
            return {}
        with tarfile.open(local_tar, "r:gz") as tf:
            tf.extractall(tmp, filter="data")
        for root, _, files in os.walk(tmp):
            if "profile_export_aiperf.json" in files:
                with open(os.path.join(root, "profile_export_aiperf.json"), encoding="utf-8") as f:
                    metrics = _metrics_from_aiperf_export(json.load(f))
                if metrics:
                    return metrics
        for root, _, files in os.walk(tmp):
            if "profile_export.jsonl" in files:
                return _parse_profile_export_jsonl_from_path(
                    os.path.join(root, "profile_export.jsonl"))
    return {}


_NOT_FOUND_MARKERS = (
    "could not find",          # sagemaker friendly phrasing
    "does not exist",
    "validationexception",     # generic
    "resourcenotfound",        # sagemaker / s3
)


def _is_not_found_error(exc: Exception) -> bool:
    """F-D (eng-review pass 3): treat 'resource not found / already deleted'
    as teardown success. Without this, idempotent re-invocations flip
    teardown_complete back to False even though the first call already
    cleaned up."""
    msg = str(exc).lower()
    if any(m in msg for m in _NOT_FOUND_MARKERS):
        return True
    # botocore ClientError exposes error code on the response envelope.
    resp = getattr(exc, "response", None) or {}
    code = (resp.get("Error") or {}).get("Code", "").lower()
    return code in {"validationexception", "resourcenotfound",
                    "resourcenotfoundexception"}


def _teardown_recommendation_resources(entry: dict) -> bool:
    """Sequential delete of every resource the submit path created.
    Each delete is independent try/except so a partial failure doesn't
    wedge the rest. Returns True iff every delete either succeeded OR
    the resource was already gone (idempotent semantics per F-D).

    F-F (eng-review pass 3): SageMaker rejects delete_endpoint while an
    InferenceComponent is still attached. Between delete_inference_component
    and delete_endpoint we wait on the inference_component_deleted waiter
    with a bounded 60 s ceiling so the second call doesn't race the first.
    """
    sm = boto3.client("sagemaker")
    all_ok = True
    ic_name = entry.get("inference_component_name")
    if ic_name:
        try:
            sm.delete_inference_component(InferenceComponentName=ic_name)
            # F-F: block briefly until the IC is actually gone. Without the
            # waiter, delete_endpoint fires while the IC is still
            # Deleting and we get "Endpoint has inference components
            # attached". 60 s ceiling is ample — IC teardown is typically
            # <20 s in practice.
            try:
                sm.get_waiter("inference_component_deleted").wait(
                    InferenceComponentName=ic_name,
                    WaiterConfig={"Delay": 10, "MaxAttempts": 6},
                )
            except Exception as wait_exc:
                # Waiter failure here is informational only — delete_endpoint
                # below will bubble up the real blocking error if the IC
                # genuinely didn't delete.
                print(f"[teardown] IC waiter non-fatal: {wait_exc}")
        except Exception as e:
            if _is_not_found_error(e):
                pass  # already gone — idempotent success
            else:
                print(f"[teardown] delete_inference_component({ic_name}) failed: {e}")
                all_ok = False
    for attr, fn_name, arg_key in (
        ("endpoint_name",        "delete_endpoint",        "EndpointName"),
        ("endpoint_config_name", "delete_endpoint_config", "EndpointConfigName"),
        ("model_name",           "delete_model",           "ModelName"),
    ):
        name = entry.get(attr)
        if not name:
            continue
        try:
            getattr(sm, fn_name)(**{arg_key: name})
        except Exception as e:
            if _is_not_found_error(e):
                continue  # already gone — idempotent success
            print(f"[teardown] {fn_name}({name}) failed: {e}")
            all_ok = False
    return all_ok


# ── R1 Task 8: submit_monitoring_job ──────────────────────────────────────
# Batch drift + data-quality + (optional) classification-quality report
# via Evidently, triggered per MCP call against a completed tabular
# training job. Reads baseline + model artifact URIs from the DDB thread
# row (stamped by the callback Lambda per R1 Task 3) so the agent never
# passes S3 URIs. Fire-and-forget: ≤5 s sync pre-flight, then async
# self-invoke to CreateProcessingJob.


def _find_source_job_entry(*, sm: Any, thread_id: str, source_job: str, desc: dict) -> dict | None:
    """Locate the DDB jobs-map entry for ``source_job``.

    Looks in the current thread's row first, then — QA BUG-004 — falls back to
    the job's *own* thread resolved via its SageMaker ``ThreadId``/``JobId``
    resource tags (the same mechanism the EventBridge callback uses). Starter
    tiles open a fresh thread, so cross-session lookups are the common case.

    Args:
        sm: boto3 SageMaker client (for the list_tags fallback).
        thread_id: The current conversation's thread id.
        source_job: SageMaker training job name to locate.
        desc: describe_training_job response for ``source_job``.

    Returns:
        The jobs-map entry dict, or None when unresolvable.
    """
    ddb = _ddb_table()
    row = ddb.get_item(Key={"task_id": thread_id}).get("Item") or {}
    for entry in (row.get("jobs") or {}).values():
        if entry.get("sagemaker_job_name") == source_job:
            return entry
    tags = {t["Key"]: t["Value"] for t in (desc.get("Tags") or [])}
    if not tags:
        try:
            arn = desc.get("TrainingJobArn", "")
            tags = {t["Key"]: t["Value"] for t in sm.list_tags(ResourceArn=arn).get("Tags", [])}
        except Exception as exc:  # noqa: BLE001 — fall through to None, callers raise loudly
            print(f"[sagemaker-skill] list_tags fallback failed for {source_job}: {exc}")
            return None
    src_thread = tags.get("ThreadId", "")
    src_job_id = tags.get("JobId", "")
    if not src_thread or not src_job_id:
        return None
    src_row = ddb.get_item(Key={"task_id": src_thread}).get("Item") or {}
    return (src_row.get("jobs") or {}).get(src_job_id)


def _list_recent_training_jobs(args: dict) -> dict:
    """List recent project training jobs across ALL sessions (QA BUG-004).

    Lets the agent resolve prompts like "the most recent Completed XGBoost
    job" when the current thread has no candidate. Backed by SageMaker
    ``list_training_jobs`` (name-prefixed to this project), enriched with
    ``training_type``/``dataset`` from each job's own DDB thread row via its
    ThreadId/JobId tags.

    Args:
        args: Tool arguments — optional ``status_equals`` (default
            "Completed"), optional ``max_results`` (default 10, cap 25).

    Returns:
        {"jobs": [{sagemaker_job_name, status, training_type, dataset_name,
        thread_id, creation_time}, …]} newest first.
    """
    status = args.get("status_equals", "Completed")
    max_results = min(int(args.get("max_results", 10) or 10), 25)
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    resp = sm.list_training_jobs(
        NameContains=f"{PROJECT_NAME}-job-",
        StatusEquals=status,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=max_results,
    )
    ddb = _ddb_table()
    jobs: list[dict] = []
    for summary in resp.get("TrainingJobSummaries", []):
        name = summary["TrainingJobName"]
        entry: dict = {}
        thread_tag = ""
        try:
            tags = {
                t["Key"]: t["Value"]
                for t in sm.list_tags(ResourceArn=summary["TrainingJobArn"]).get("Tags", [])
            }
            thread_tag = tags.get("ThreadId", "")
            job_tag = tags.get("JobId", "")
            if thread_tag and job_tag:
                row = ddb.get_item(Key={"task_id": thread_tag}).get("Item") or {}
                entry = (row.get("jobs") or {}).get(job_tag) or {}
        except Exception as exc:  # noqa: BLE001 — enrichment is best-effort
            print(f"[sagemaker-skill] enrichment failed for {name}: {exc}")
        jobs.append({
            "sagemaker_job_name": name,
            "status": summary.get("TrainingJobStatus", ""),
            "training_type": entry.get("training_type", ""),
            "dataset_name": entry.get("dataset_name", ""),
            "thread_id": thread_tag,
            "creation_time": str(summary.get("CreationTime", "")),
        })
    return {"jobs": jobs}


def _submit_monitoring_job(args: dict) -> dict:
    """Kick off a batch monitoring Processing job against a tabular
    training job's baseline.

    Mirrors ``_submit_eval_job`` in shape: pre-flight lookups → DDB seed
    → MLflow run pre-create → async self-invoke. Returns the DDB record
    shape immediately; EventBridge callback (Kind=monitoring) resumes
    the agent on terminal state.
    """
    if not MONITORING_IMAGE_URI:
        raise RuntimeError(
            "MONITORING_IMAGE_URI env var not set — CDK wiring incomplete"
        )

    thread_id = args["thread_id"]
    source_job = args["sagemaker_job_name"]
    explicit_current = args.get("current_data_s3_uri")
    use_eval_split = bool(args.get("use_training_eval_split", False))
    target_column = args.get("target_column", "")
    instance_type = args.get("instance_type", "ml.m5.large")
    run_name = args.get("run_name")
    user_id = args.get("_user_id", "")

    # Pre-flight: describe source training job. Fail loudly if it's
    # missing or not Completed — monitoring needs a finished booster.
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    try:
        desc = sm.describe_training_job(TrainingJobName=source_job)
    except Exception as exc:
        raise RuntimeError(
            f"Source training job {source_job!r} not found or inaccessible: {exc}"
        ) from exc
    if desc["TrainingJobStatus"] != "Completed":
        raise RuntimeError(
            f"Source training job {source_job!r} is {desc['TrainingJobStatus']}; "
            "monitoring requires Completed."
        )
    artifact_s3 = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
    if not artifact_s3:
        raise RuntimeError(f"Source training job {source_job!r} has no ModelArtifacts")

    # Resolve baseline + eval_split URIs from the source job's DDB entry —
    # current thread first, then the job's own thread via its SageMaker tags
    # (BUG-004: starter tiles run in a fresh thread).
    source_entry = _find_source_job_entry(
        sm=sm, thread_id=thread_id, source_job=source_job, desc=desc,
    )
    if not source_entry:
        raise RuntimeError(
            f"No DDB job record for {source_job!r} in this thread or via its "
            "ThreadId/JobId tags — was the job submitted by this project?"
        )
    ddb = _ddb_table()
    if source_entry.get("training_type") not in ("xgboost", "sklearn"):
        raise RuntimeError(
            f"training_type={source_entry.get('training_type')!r} — monitoring "
            "supports xgboost/sklearn only."
        )
    baseline_uri = source_entry.get("baseline_s3_uri")
    eval_split_uri = source_entry.get("eval_split_s3_uri")
    if not baseline_uri:
        raise RuntimeError(
            f"No baseline_s3_uri on {source_job!r} — training job may predate "
            "baseline emission. Re-submit the training job."
        )
    if use_eval_split and not eval_split_uri:
        raise RuntimeError(
            "use_training_eval_split=true but eval_split_s3_uri missing"
        )
    if explicit_current and use_eval_split:
        raise RuntimeError(
            "Pass current_data_s3_uri OR use_training_eval_split, not both"
        )
    if not explicit_current and not use_eval_split:
        raise RuntimeError(
            "Must set either current_data_s3_uri or use_training_eval_split=true"
        )
    resolved_current = explicit_current or eval_split_uri

    # head_object pre-flight on every S3 URI the container will touch.
    # Cheap (~100ms total) and catches typos / permission errors before
    # we pay the Processing-job boot cost.
    s3 = boto3.client("s3", region_name=AWS_REGION)
    for uri, label in (
        (baseline_uri, "baseline"),
        (resolved_current, "current_data"),
        (artifact_s3, "model_artifact"),
    ):
        bkt, _, key = uri[5:].partition("/")
        try:
            s3.head_object(Bucket=bkt, Key=key)
        except Exception as exc:
            raise RuntimeError(f"{label} URI not found: {uri} ({exc})") from exc

    # Idempotency: same-fingerprint retry within the existing 600 s window
    # returns the in-flight job rather than double-creating.
    now = int(time.time())
    fingerprint = {
        "kind": "monitoring",
        "source_training_job": source_job,
        "current_data_s3_uri": resolved_current,
        "target_column": target_column,
    }
    existing = _find_in_flight_duplicate(
        thread_id=thread_id, fingerprint=fingerprint, now=now,
    )
    if existing:
        return {
            "thread_id": thread_id,
            "job_id": existing["job_id"],
            "processing_job_name": existing.get("processing_job_name", ""),
            "mlflow_run_id": existing.get("mlflow_run_id", ""),
            "mlflow_run_url": existing.get("mlflow_run_url", ""),
            "baseline_s3_uri": baseline_uri,
            "current_data_s3_uri": resolved_current,
            "model_artifact_s3_uri": artifact_s3,
            "status": existing.get("status", "SUBMITTING"),
            "deduplicated": True,
        }

    job_id = str(uuid.uuid4())
    job_name = f"{PROJECT_NAME}-monitor-{int(time.time())}"
    exp_id, run_id, run_url = _start_mlflow_run(
        thread_id=thread_id, job_id=job_id,
        job_name=run_name or job_name,
        model_id=source_job, dataset_name=resolved_current,
        training_type="monitoring", instance_type=instance_type,
        max_steps=0, learning_rate=0.0, user_id=user_id,
    )

    # Seed thread row + jobs map (same pattern as _submit_eval_job).
    ddb.update_item(
        Key={"task_id": thread_id},
        UpdateExpression=(
            "SET jobs = if_not_exists(jobs, :empty_map), "
            "    created_at = if_not_exists(created_at, :t), "
            "    user_id = if_not_exists(user_id, :uid), "
            "    thread_id = if_not_exists(thread_id, :tid)"
        ),
        ExpressionAttributeValues={
            ":empty_map": {}, ":t": now, ":uid": user_id, ":tid": thread_id,
        },
    )
    job_record = {
        "job_id": job_id,
        "kind": "monitoring",
        "source_training_job": source_job,
        "baseline_s3_uri": baseline_uri,
        "current_data_s3_uri": resolved_current,
        "model_artifact_s3_uri": artifact_s3,
        "target_column": target_column,
        "instance_type": instance_type,
        "processing_job_name": job_name,
        "status": "SUBMITTING",
        "status_message": f"Queueing {job_name}",
        "created_at": now,
        "updated_at": now,
        "mlflow_experiment_id": exp_id,
        "mlflow_run_id": run_id,
        "mlflow_run_url": run_url,
    }
    ddb.update_item(
        Key={"task_id": thread_id},
        UpdateExpression="SET jobs.#jid = :job, updated_at = :t",
        ExpressionAttributeNames={"#jid": job_id},
        ExpressionAttributeValues={":job": job_record, ":t": now},
    )

    _invoke_background({
        "_bg_tool": "submit_monitoring_job",
        "thread_id": thread_id,
        "job_id": job_id,
        "job_name": job_name,
        "baseline_s3_uri": baseline_uri,
        "current_data_s3_uri": resolved_current,
        "model_artifact_s3_uri": artifact_s3,
        "target_column": target_column,
        "instance_type": instance_type,
        "mlflow_run_id": run_id,
    })

    return {
        "thread_id": thread_id,
        "job_id": job_id,
        "processing_job_name": job_name,
        "mlflow_run_id": run_id,
        "mlflow_run_url": run_url,
        "baseline_s3_uri": baseline_uri,
        "current_data_s3_uri": resolved_current,
        "model_artifact_s3_uri": artifact_s3,
        "status": "SUBMITTING",
    }


def _background_submit_monitoring(payload: dict) -> None:
    """Background worker — run CreateProcessingJob out of band and flip
    the DDB status to IN_PROGRESS (or FAILED) so the agent gets a quick
    answer from the sync handler path.
    """
    thread_id = payload["thread_id"]
    job_id = payload["job_id"]
    job_name = payload["job_name"]
    instance_type = payload["instance_type"]
    try:
        sm = boto3.client("sagemaker", region_name=AWS_REGION)
        env_vars = {
            "MLFLOW_TRACKING_URI":  MLFLOW_TRACKING_URI,
            "MLFLOW_RUN_ID":        payload.get("mlflow_run_id", ""),
            "BASELINE_S3_URI":      payload["baseline_s3_uri"],
            "CURRENT_DATA_S3_URI":  payload["current_data_s3_uri"],
            "MODEL_ARTIFACT_S3_URI": payload["model_artifact_s3_uri"],
            "TARGET_COLUMN":        payload.get("target_column", ""),
            "AWS_REGION":           AWS_REGION,
        }
        sm.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=SM_ROLE,
            AppSpecification={
                "ImageUri": MONITORING_IMAGE_URI,
                "ContainerEntrypoint": ["python3", "/opt/ml/code/entrypoint.py"],
            },
            ProcessingResources={"ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": 30,
            }},
            Environment=env_vars,
            Tags=[
                {"Key": "ThreadId", "Value": thread_id},
                {"Key": "JobId", "Value": job_id},
                {"Key": "ProjectName", "Value": PROJECT_NAME},
                {"Key": "Kind", "Value": "monitoring"},
            ],
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
        )
    except Exception as exc:
        logger.exception("[submit_monitoring_job:bg] CreateProcessingJob failed")
        _mark_job(
            thread_id, job_id, status="FAILED",
            message=f"CreateProcessingJob failed: {type(exc).__name__}: {exc}",
        )
        return
    _mark_job(
        thread_id, job_id, status="IN_PROGRESS",
        message=f"SageMaker accepted {job_name}; waiting for instance",
    )


def _submit_recommendation_job(args: dict) -> dict:
    thread_id         = args["thread_id"]
    source_job        = args["sagemaker_job_name"]
    instance_type     = args["instance_type"]
    input_tokens      = int(args.get("input_tokens", 500))
    output_tokens     = int(args.get("output_tokens", 150))
    concurrency       = list(args.get("concurrency_levels") or [1, 4, 16])
    max_latency_p99   = int(args.get("max_latency_p99_ms", 5000))
    max_invoke_pm     = args.get("max_invocations_per_minute")
    run_name          = args.get("run_name")
    serving_image_arg = args.get("serving_image_uri")
    serving_env_arg   = args.get("serving_env")
    user_id           = args.get("_user_id", "")

    # Pre-flight 1: instance family
    if not (instance_type.startswith("ml.g") or instance_type.startswith("ml.p")):
        raise RuntimeError(
            f"instance_type={instance_type!r} rejected — ml.g* or ml.p* required "
            "(TGI-based serving assumes GPU)."
        )

    # Pre-flight 2: describe source training job
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    try:
        desc = sm.describe_training_job(TrainingJobName=source_job)
    except Exception as exc:
        raise RuntimeError(f"Source training job {source_job!r} not found: {exc}") from exc
    if desc["TrainingJobStatus"] != "Completed":
        raise RuntimeError(
            f"Source training job {source_job!r} is {desc['TrainingJobStatus']}; "
            "benchmark requires Completed."
        )
    artifact_s3 = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
    training_image = desc.get("AlgorithmSpecification", {}).get("TrainingImage", "")
    if not artifact_s3:
        raise RuntimeError(f"Source training job {source_job!r} has no ModelArtifacts.")

    # Pre-flight 3: training_type must be LLM. Resolve the entry from the
    # current thread or the job's own thread via tags (BUG-004).
    ddb = _ddb_table()
    source_entry = _find_source_job_entry(
        sm=sm, thread_id=thread_id, source_job=source_job, desc=desc,
    )
    if not source_entry or source_entry.get("training_type") not in ("sft", "dpo", "grpo"):
        raise RuntimeError(
            f"training_type={source_entry.get('training_type') if source_entry else None!r} — "
            "benchmark requires sft/dpo/grpo (LLM fine-tune)."
        )

    # Pre-flight 4: artifact exists
    s3 = boto3.client("s3", region_name=AWS_REGION)
    bkt, _, key = artifact_s3[5:].partition("/")
    try:
        s3.head_object(Bucket=bkt, Key=key)
    except Exception as exc:
        raise RuntimeError(f"model artifact not found: {artifact_s3} ({exc})") from exc

    # Pre-flight 5: resolve serving image (override or auto-derive)
    if serving_image_arg:
        serving_image = serving_image_arg
        serving_env   = serving_env_arg or {}
    else:
        serving_image = _derive_serving_from_training(training_image)
        if serving_image is None:
            raise RuntimeError(
                f"Cannot auto-derive a serving container from training image "
                f"{training_image!r}. Pass serving_image_uri + serving_env "
                "(any SageMaker-compatible LLM serving image — TGI/vLLM/LMI/Triton/custom)."
            )
        serving_env = _tgi_default_env({"input_tokens": input_tokens,
                                        "output_tokens": output_tokens})

    workload_spec = {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "concurrency_levels": concurrency, "max_latency_p99_ms": max_latency_p99,
        # QA BUG-019: the AI Benchmark (AIPerf) workload spec requires the
        # model's tokenizer; stash it here so the callback can build the
        # aiperf spec without re-resolving the source thread.
        "tokenizer": source_entry.get("model_id", ""),
    }
    if max_invoke_pm is not None:
        workload_spec["max_invocations_per_minute"] = int(max_invoke_pm)

    # Idempotency dedup (same 10-min window as other submits). F-7 from
    # eng review: use the same 3-key shape as _submit_training_job, so
    # _find_in_flight_duplicate's simple dict-equality matcher works.
    # Two submits with the same training job + instance but different
    # workload_spec dedup together — acceptable for v1 demo usage; if
    # real users want per-workload dedup we'll add workload_spec_hash as
    # a stored DDB field AND include it here in one pass.
    now = int(time.time())
    fingerprint = {
        "kind": "recommendation", "source_training_job": source_job,
        "instance_type": instance_type,
    }
    existing = _find_in_flight_duplicate(thread_id=thread_id, fingerprint=fingerprint, now=now)
    if existing:
        return {
            "thread_id": thread_id, "job_id": existing["job_id"],
            "recommender_job_name": existing.get("recommender_job_name", ""),
            "endpoint_name": existing.get("endpoint_name", ""),
            "instance_type": instance_type,
            "estimated_wall_clock_minutes": existing.get("estimated_wall_clock_minutes", 0),
            "status": existing.get("status", "SUBMITTING"), "deduplicated": True,
        }

    # Seed names derived from rec_id (so callback can reverse-lookup from tags)
    rec_id = str(uuid.uuid4())
    base_name = f"{PROJECT_NAME}-rec-{rec_id[:12]}"
    endpoint_name = base_name
    model_name = base_name
    endpoint_config_name = base_name
    workload_config_name = f"{base_name}-wl"
    recommender_job_name = base_name  # == AIBenchmarkJobName

    # Seed thread row + jobs.<rec_id>
    ddb.update_item(
        Key={"task_id": thread_id},
        UpdateExpression=("SET jobs = if_not_exists(jobs, :empty_map), "
                          "created_at = if_not_exists(created_at, :t), "
                          "user_id = if_not_exists(user_id, :uid), "
                          "thread_id = if_not_exists(thread_id, :tid)"),
        ExpressionAttributeValues={":empty_map": {}, ":t": now, ":uid": user_id, ":tid": thread_id},
    )
    eta_min = _estimate_wall_clock_minutes(instance_type)
    job_record = {
        "job_id": rec_id, "kind": "recommendation",
        "source_training_job": source_job,
        "model_artifact_s3_uri": artifact_s3, "instance_type": instance_type,
        "serving_image_uri": serving_image, "serving_env": serving_env,
        "workload_spec": workload_spec,
        "endpoint_name": endpoint_name, "model_name": model_name,
        "endpoint_config_name": endpoint_config_name,
        "ai_workload_config_name": workload_config_name,
        "recommender_job_name": recommender_job_name,
        "run_name": run_name or recommender_job_name,
        "status": "SUBMITTING",
        "status_message": f"Queueing recommendation {recommender_job_name}",
        "created_at": now, "updated_at": now,
        "estimated_wall_clock_minutes": eta_min,
        "results": {}, "teardown_complete": False,
    }
    ddb.update_item(
        Key={"task_id": thread_id},
        UpdateExpression="SET jobs.#jid = :job, updated_at = :t",
        ExpressionAttributeNames={"#jid": rec_id},
        ExpressionAttributeValues={":job": job_record, ":t": now},
    )

    # Fire async self-invoke
    _invoke_background({
        "_bg_tool": "submit_recommendation_job",
        "thread_id": thread_id, "job_id": rec_id,
        "endpoint_name": endpoint_name, "model_name": model_name,
        "endpoint_config_name": endpoint_config_name,
        "artifact_s3": artifact_s3, "serving_image": serving_image,
        "serving_env": serving_env, "instance_type": instance_type,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
    })

    return {
        "thread_id": thread_id, "job_id": rec_id,
        "sagemaker_job_name": source_job,
        "recommender_job_name": recommender_job_name,
        "endpoint_name": endpoint_name, "model_name": model_name,
        "endpoint_config_name": endpoint_config_name,
        "ai_workload_config_name": workload_config_name,
        "instance_type": instance_type,
        "model_artifact_s3_uri": artifact_s3,
        "serving_image_uri": serving_image,
        "workload_spec": workload_spec,
        "estimated_wall_clock_minutes": eta_min,
        "status": "SUBMITTING",
    }


def _background_submit_recommendation(payload: dict) -> None:
    """Async Lambda (≤10 s): fires the three non-blocking control-plane
    calls to deploy the endpoint, then exits. EventBridge wakes the
    callback Lambda when the endpoint reaches InService."""
    thread_id     = payload["thread_id"]
    job_id        = payload["job_id"]
    endpoint_name = payload["endpoint_name"]
    try:
        sm = boto3.client("sagemaker", region_name=AWS_REGION)
        sm.create_model(
            ModelName=payload["model_name"],
            ExecutionRoleArn=SM_ROLE,
            PrimaryContainer={
                "Image":       payload["serving_image"],
                "Environment": payload["serving_env"],
                "ModelDataSource": {"S3DataSource": {
                    "S3Uri":          payload["artifact_s3"],
                    "S3DataType":     "S3Object",
                    "CompressionType":"Gzip",
                }},
            },
        )
        # QA BUG-020: the model is deployed directly on the variant — no
        # inference component. IC scheduling never offered the variant's GPU
        # ("not enough hardware resources" on every host size/memory tried),
        # and AWS's reference implementation of this benchmark flow targets
        # a bare endpoint with the model on the ProductionVariant. A classic
        # variant also means InService == model loaded and serving, so the
        # callback can start the benchmark with no second provisioning phase.
        # The generous timeouts are the F-2 cold-start protections (large
        # model.tar.gz download + TGI/vLLM warm-up), now on the variant.
        sm.create_endpoint_config(
            EndpointConfigName=payload["endpoint_config_name"],
            ProductionVariants=[{
                "VariantName":           "AllTraffic",
                "ModelName":             payload["model_name"],
                "InstanceType":          payload["instance_type"],
                "InitialInstanceCount":  1,
                "ModelDataDownloadTimeoutInSeconds":            1200,  # 20 min
                "ContainerStartupHealthCheckTimeoutInSeconds":   600,  # 10 min
            }],
        )
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=payload["endpoint_config_name"],
            Tags=[
                {"Key": "Kind",        "Value": "recommendation"},
                {"Key": "RecId",       "Value": job_id},
                {"Key": "ThreadId",    "Value": thread_id},
                {"Key": "ProjectName", "Value": PROJECT_NAME},
            ],
        )
    except Exception as exc:
        logger.exception("[submit_recommendation_job:bg] deploy failed")
        _mark_job(thread_id, job_id, status="FAILED",
                  message=f"Endpoint provisioning failed: {type(exc).__name__}: {exc}")
        return
    _mark_job(thread_id, job_id, status="DEPLOYING",
              message=f"Endpoint {endpoint_name} provisioning (~5-10 min)")


def _get_recommendation_results(args: dict) -> dict:
    """Check the AI Benchmark job status, parse + teardown on terminal.

    NOTE: Because AWS does not emit a terminal EventBridge event for AI
    Benchmark jobs, this tool is the ONLY place that discovers the
    benchmark completed. Each call:

      1. Looks up the DDB entry by recommender_job_name.
      2. If DDB already has status=COMPLETED or FAILED (previous call
         already parsed + tore down), returns metrics immediately.
      3. Otherwise calls describe_ai_benchmark_job. If the AWS status is
         pre-terminal, returns a BENCHMARKING message for the agent to
         retry later. If terminal, parses profile_export.jsonl, writes
         metrics + teardown_complete into DDB, tears down the temp
         endpoint / IC / model, and returns the full metrics payload.

    Safe to call repeatedly — teardown helpers are idempotent.
    """
    thread_id = args["thread_id"]
    target_name = args["recommender_job_name"]
    ddb = _ddb_table()
    row = ddb.get_item(Key={"task_id": thread_id}).get("Item") or {}
    rec_id = None
    entry = None
    for jid, e in (row.get("jobs") or {}).items():
        if e.get("recommender_job_name") == target_name:
            rec_id, entry = jid, e
            break
    if entry is None:
        raise RuntimeError(
            f"recommender_job_name={target_name!r} not found under thread {thread_id!r}"
        )

    # Fast path — we already wrote terminal results on a prior call.
    if entry.get("status") in ("COMPLETED", "FAILED"):
        return _format_recommendation_response(target_name, entry)

    # Active path — ask AWS.
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    try:
        desc = sm.describe_ai_benchmark_job(AIBenchmarkJobName=target_name)
    except Exception as exc:
        # Benchmark may not exist yet if the endpoint is still deploying.
        return {
            "recommender_job_name": target_name,
            "status": entry.get("status", "DEPLOYING"),
            "message": f"Benchmark not yet started or not found ({exc}); "
                       "try again in a minute.",
        }
    aws_status = desc.get("AIBenchmarkJobStatus", "")
    terminal = {"Completed", "Failed", "Stopped"}
    if aws_status not in terminal:
        return {
            "recommender_job_name": target_name,
            "status": "BENCHMARKING",
            "instance_type": entry.get("instance_type", ""),
            "message": f"AI benchmark status={aws_status!r}; try again later "
                       "(~10-30 min total after endpoint goes InService).",
        }

    # Terminal — parse + teardown.
    # NOTE: `_parse_profile_export_jsonl` and `_teardown_recommendation_resources`
    # MUST live in this same module (`lambda/skills/sagemaker/handler.py`).
    # Task 8 copy-pastes both helpers from `lambda/callback/handler.py` into
    # here so `_get_recommendation_results` has no cross-Lambda dependency.
    #
    # F-B (eng-review pass 3): AWS does not guarantee `output.tar.gz` is in S3
    # at the same instant `AIBenchmarkJobStatus` flips to Completed. Retry the
    # parse with bounded backoff when metrics come back empty; if still empty
    # after 3 tries, DEFER teardown so a later call can recover the metrics.
    metrics = {}
    s3_out = desc.get("OutputConfig", {}).get("S3OutputLocation", "")
    if aws_status == "Completed" and s3_out:
        for _attempt in range(3):
            metrics = _parse_profile_export_jsonl(s3_out)
            if metrics:
                break
            time.sleep(5)  # nosemgrep: arbitrary-sleep — tarball upload typically lands within a few seconds
    defer_teardown = (aws_status == "Completed" and not metrics)
    if defer_teardown:
        # Leave resources running so the next get_recommendation_results can
        # recover the metrics. Report BENCHMARKING so the agent re-checks
        # later. Do NOT write COMPLETED into DDB — status must stay pre-
        # terminal so the fast path doesn't short-circuit next time.
        return {
            "recommender_job_name": target_name,
            "status": "BENCHMARKING",
            "instance_type": entry.get("instance_type", ""),
            "message": ("Benchmark marked Completed by AWS but profile_export.tar.gz "
                        "is not yet in S3 (<~30 s propagation). Try again shortly."),
        }
    teardown_ok = _teardown_recommendation_resources(entry)
    status = aws_status.upper()
    status_msg = (
        f"Benchmark {aws_status}; " +
        (f"failure: {desc.get('FailureReason','unknown')}" if aws_status == "Failed" else
         (f"TTFT p99={metrics.get('ttft_ms_p99','?')}ms, "
          f"throughput={metrics.get('throughput_tokens_per_sec','?')} tok/s"
          if metrics else "profile_export parse failed"))
    )
    # F-C (eng-review pass 3): guard the terminal transition with a
    # ConditionExpression so two concurrent agent turns can't both write the
    # terminal row and race the teardown. Only the call that transitions the
    # row from BENCHMARKING/DEPLOYING to COMPLETED/FAILED wins; the loser
    # falls through to the fast path on the re-read.
    try:
        ddb.update_item(
            Key={"task_id": thread_id},
            UpdateExpression=("SET jobs.#jid.#s = :s, jobs.#jid.status_message = :m, "
                              "jobs.#jid.results = :r, jobs.#jid.teardown_complete = :tc, "
                              "jobs.#jid.profile_s3_uri = :p, updated_at = :t"),
            ConditionExpression="jobs.#jid.#s IN (:bench, :deploy, :submit)",
            ExpressionAttributeNames={"#jid": rec_id, "#s": "status"},
            ExpressionAttributeValues={
                # DynamoDB rejects Python floats — Decimal-ize the AIPerf
                # metrics (str round-trip keeps full precision).
                ":s": status, ":m": status_msg,
                ":r": {k: (Decimal(str(v)) if isinstance(v, float) else v)
                       for k, v in metrics.items()},
                ":tc": teardown_ok,
                ":p": s3_out,
                ":t": int(time.time()),
                ":bench": "BENCHMARKING", ":deploy": "DEPLOYING", ":submit": "SUBMITTING",
            },
        )
    except ddb.meta.client.exceptions.ConditionalCheckFailedException:
        # Another invocation beat us to the terminal write. That's fine;
        # our teardown attempt (if any) against already-deleted resources
        # is treated as success by _teardown_recommendation_resources (F-D).
        # Fall through to the re-read + format path.
        pass
    # Re-read so response carries whichever write won the condition race.
    row = ddb.get_item(Key={"task_id": thread_id}).get("Item") or {}
    entry = (row.get("jobs") or {}).get(rec_id) or {}
    return _format_recommendation_response(target_name, entry)


def _format_recommendation_response(recommender_job_name: str, entry: dict) -> dict:
    """Format a recommendation job entry into a response dict."""
    return {
        "recommender_job_name": recommender_job_name,
        "status": entry.get("status", "UNKNOWN"),
        "instance_type": entry.get("instance_type", ""),
        "source_training_job": entry.get("source_training_job", ""),
        "workload_spec": entry.get("workload_spec", {}),
        "metrics": entry.get("results", {}),
        "profile_s3_uri": entry.get("profile_s3_uri", ""),
        "teardown_complete": bool(entry.get("teardown_complete", False)),
        "summary": entry.get("status_message", ""),
    }


_DISPATCH: dict[str, Any] = {
    "submit_training_job": _submit_training_job,
    "complete_training_job": _complete_training_job,
    "deploy_model": _deploy_model,
    "list_hub_models": _list_hub_models,
    "submit_eval_job": _submit_eval_job,
    "submit_monitoring_job": _submit_monitoring_job,
    "list_recent_training_jobs": _list_recent_training_jobs,
    "submit_recommendation_job": _submit_recommendation_job,
    "get_recommendation_results": _get_recommendation_results,
}


def handler(event: dict, context: Any) -> dict:
    """Gateway MCP tool dispatcher.

    Args:
        event: AgentCore Gateway passes the tool arguments map as `event`.
        context: Lambda context; tool name is in
            `context.client_context.custom['bedrockAgentCoreToolName']`,
            formatted as `${target_name}___${tool_name}`.

    Returns:
        dict: MCP tool result with content list.
    """
    # Self-invocation path: submit_*_job fires an async lambda.invoke with
    # InvocationType=Event and a `_bg_tool` sentinel in the payload. There is no
    # Gateway client_context on those events — detect and route separately.
    if isinstance(event, dict) and event.get("_bg_tool"):
        _background_dispatcher(event)
        return {"status": "ok", "background": event["_bg_tool"]}

    # Gateway prefixes the tool with the target name; strip everything up to and including `___`
    raw_tool = context.client_context.custom.get("bedrockAgentCoreToolName", "") if getattr(context, "client_context", None) else ""
    tool_name = raw_tool.split("___", 1)[1] if "___" in raw_tool else raw_tool
    arguments = event or {}

    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}

    # QA BUG-014: the agent intermittently passes the literal env-var NAME
    # (with or without ${}) instead of its value — observed live for both
    # thread_id ("CURRENT_THREAD_ID" monitoring row) and tag values
    # ("${CURRENT_THREAD_ID}" eval CreateProcessingJob failure). Accepting a
    # placeholder silently detaches the job from its session and breaks the
    # async resume, so reject loudly with a self-correcting instruction.
    for key in ("thread_id", "_user_id"):
        val = str(arguments.get(key, "") or "")
        if "CURRENT_THREAD_ID" in val or "CURRENT_USER_ID" in val:
            return {"content": [{"type": "text", "text": (
                f"Error: {key}={val!r} is the literal environment-variable "
                "placeholder, not its value. Read the real value first (e.g. "
                "Bash: echo $CURRENT_THREAD_ID $CURRENT_USER_ID) and retry "
                "with the resolved UUID."
            )}], "isError": True}

    try:
        result = fn(arguments)
        # DDB returns numerics as decimal.Decimal; json.dumps can't serialize
        # those by default. Coerce to int/float at the boundary.
        return {"content": [{"type": "text",
                             "text": json.dumps(result, default=_json_default)}]}
    except Exception as e:
        logger.exception("[handler] tool=%s raised", tool_name)
        return {"content": [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}], "isError": True}
