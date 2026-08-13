"""HuggingFace skill Lambda — Gateway MCP target.

Tools:
  - upload_model
  - hf_snapshot_download       — stage model / small repo files into S3 + return manifest.
  - retrieve_dataset_metadata  — splits + schema (+ optional rows preview) via Datasets Server REST.
  - prepare_eval_dataset       — emit an eval-spec JSON to S3 (no dataset load here).
  - update_model_card
  - manage_tags

Design note — dataset heavy lifting lives in the SageMaker Processing
container (lambda/skills/sagemaker/eval/entrypoint.py). Lambda only handles
token fetch, small previews, and spec authoring. This keeps us inside the
Lambda 15-min timeout and 10 GB /tmp ceiling regardless of dataset size.

Token source: WORKLOAD_ACCESS_TOKEN env var (Token Vault, Phase 2) with
fallback to SSM /<project>/dev/hf-token.
"""
import json
import logging
import os
import tarfile
import tempfile
from typing import Any

# Redirect every ~/.cache / $HOME write to /tmp BEFORE importing huggingface_hub
# or datasets. Lambda's default $HOME is /home/sbx_user<N>, which is read-only;
# the first call to anything that expands `~` or `Path.home()` fails with
# `[Errno 30] Read-only file system: '/home/sbx_user1051'`. Setting HF_HOME
# alone is NOT enough — datasets, pyarrow, fsspec, and huggingface_hub's
# auth helpers all hit $HOME through different code paths. Overriding HOME
# covers all of them in one shot. /tmp is the only writable path in Lambda
# (512 MB default, up to 10 GB with configured ephemeral storage).
os.environ["HOME"] = "/tmp"  # nosec B108
os.environ["HF_HOME"] = "/tmp/hf"  # nosec B108
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf/datasets"  # nosec B108
os.environ["HF_HUB_CACHE"] = "/tmp/hf/hub"  # nosec B108
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf/hub"  # nosec B108
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf/transformers"  # nosec B108
os.environ["HF_MODULES_CACHE"] = "/tmp/hf/modules"  # nosec B108
os.environ["XDG_CACHE_HOME"] = "/tmp/hf/xdg"  # nosec B108
os.makedirs("/tmp/hf", exist_ok=True)  # nosec B108

import boto3  # noqa: E402 — HOME/HF_* env vars above must be set before these imports
import requests  # noqa: E402
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROJECT_NAME = os.environ.get("PROJECT_NAME", "sample-mlops-agent")
SESSION_BUCKET = os.environ.get("SESSION_BUCKET", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Hard ceiling on inline sample rows. retrieve_dataset_metadata is a
# schema-exploration tool, not a data pipeline — large downloads belong in
# the Processing container.
_PREVIEW_MAX_ROWS = 50

# HuggingFace Datasets Server — public REST API that returns splits, column
# schema, sample rows, and parquet URLs without downloading any parquet
# into the caller. Docs: https://huggingface.co/docs/dataset-viewer
_DATASETS_SERVER = "https://datasets-server.huggingface.co"
_DATASETS_SERVER_TIMEOUT = 15  # seconds per GET


def _has_hf_token() -> bool:
    """Return True if an HF token is reachable without raising.

    Public datasets don't need a token — load_dataset just gets 401 if we
    send an empty one. Use this to skip the token fetch on anonymous loads.
    """
    if os.environ.get("WORKLOAD_ACCESS_TOKEN"):
        return True
    try:
        boto3.client("ssm", region_name=AWS_REGION).get_parameter(
            Name=f"/{PROJECT_NAME}/dev/hf-token", WithDecryption=True
        )
        return True
    except Exception:
        return False


def _get_hf_token() -> str:
    """Return HuggingFace token: Token Vault first, SSM fallback.

    Returns:
        str: HuggingFace API token.

    Raises:
        RuntimeError: If neither source yields a non-empty token.
    """
    vault_token = os.environ.get("WORKLOAD_ACCESS_TOKEN", "")
    if vault_token:
        return vault_token
    try:
        return boto3.client("ssm", region_name=AWS_REGION).get_parameter(
            Name=f"/{PROJECT_NAME}/dev/hf-token", WithDecryption=True
        )["Parameter"]["Value"]
    except Exception as e:
        raise RuntimeError(f"Cannot retrieve HuggingFace token: {e}") from e


def _upload_model(args: dict) -> dict:
    """Upload trained model artifact from S3 to HuggingFace Hub.

    Args:
        args: Required: artifact_s3, repo_id. Optional: private (bool).

    Returns:
        dict: repo_url.
    """
    artifact_s3 = args["artifact_s3"]
    repo_id = args["repo_id"]
    private = bool(args.get("private", False))

    api = HfApi(token=_get_hf_token())
    with tempfile.TemporaryDirectory() as tmpdir:
        bucket, key = artifact_s3[5:].split("/", 1)
        tar_path = os.path.join(tmpdir, "model.tar.gz")
        boto3.client("s3", region_name=AWS_REGION).download_file(bucket, key, tar_path)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmpdir, filter="data")
        os.remove(tar_path)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(folder_path=tmpdir, repo_id=repo_id)
    return {"repo_url": f"https://huggingface.co/{repo_id}"}


def _hf_snapshot_download(args: dict) -> dict:
    """Stage a HuggingFace model or dataset snapshot into S3 and return a manifest.

    The manifest lets the agent pick the right file by path/size without a
    guess-and-retry loop.

    Args:
        args: Required: repo_id, s3_prefix.
              Optional: repo_type (model|dataset), allow_patterns (list[str]),
                         ignore_patterns (list[str]), revision.

    Returns:
        dict with:
          - s3_uri: s3://<bucket>/<s3_prefix>
          - repo_id, repo_type, revision
          - files: list of {path, size_bytes, s3_key}
    """
    repo_id = args["repo_id"]
    s3_prefix = args["s3_prefix"].rstrip("/")
    repo_type = args.get("repo_type", "model")
    allow_patterns = args.get("allow_patterns") or None
    ignore_patterns = args.get("ignore_patterns") or None
    revision = args.get("revision") or None

    s3 = boto3.client("s3", region_name=AWS_REGION)
    files: list[dict] = []

    with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:  # nosec B108
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=tmpdir,
            token=_get_hf_token(),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            revision=revision,
        )
        for root, _, fnames in os.walk(tmpdir):
            for fname in fnames:
                local = os.path.join(root, fname)
                rel = os.path.relpath(local, tmpdir)
                size = os.path.getsize(local)
                s3_key = f"{s3_prefix}/{rel}"
                s3.upload_file(local, SESSION_BUCKET, s3_key)
                files.append({"path": rel, "size_bytes": size, "s3_key": s3_key})

    return {
        "s3_uri": f"s3://{SESSION_BUCKET}/{s3_prefix}",
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision or "main",
        "files": files,
    }


def _ds_server_get(path: str, params: dict) -> dict:
    """GET <datasets-server>/<path>?<params> with optional HF auth.

    Args:
        path: endpoint path without leading slash (e.g. "splits", "info").
        params: querystring params (e.g. {"dataset": "...", "config": "..."}).

    Returns:
        Parsed JSON body.

    Raises:
        RuntimeError: on non-2xx response, surfacing the server's
            `cause_message` if present so the agent sees the real reason
            (invalid split name, dataset not found, viewer disabled, …).
    """
    headers: dict[str, str] = {}
    if _has_hf_token():
        headers["Authorization"] = f"Bearer {_get_hf_token()}"
    url = f"{_DATASETS_SERVER}/{path}"
    resp = requests.get(url, params=params, headers=headers, timeout=_DATASETS_SERVER_TIMEOUT)
    if not resp.ok:
        # Datasets Server returns JSON errors like
        # {"error": "...", "cause_exception": "...", "cause_message": "..."}
        try:
            body = resp.json()
        except ValueError:
            body = {"error": resp.text}
        reason = body.get("cause_message") or body.get("error") or resp.text
        raise RuntimeError(
            f"datasets-server {path} HTTP {resp.status_code}: {reason}"
        )
    return resp.json()


def _retrieve_dataset_metadata(args: dict) -> dict:
    """Return splits + column schema (+ optional tiny sample) for an HF dataset.

    Uses the public HuggingFace Datasets Server REST API — no parquet download,
    no datasets.load_dataset call, no Lambda disk writes. Each call is ≤2
    HTTPS GETs returning JSON bodies in the single-digit KB range.

    This replaces the previous `hf_load_dataset` which wrapped
    datasets.load_dataset + split="train[:N]"; that approach still downloaded
    the entire parquet shard into Lambda's /tmp before slicing and OOMed on
    datasets like HuggingFaceH4/ultrachat_200k (1.5 GB parquet, 512 MB /tmp).

    Agents use the returned `available_splits` + `columns` to pick the right
    `train_split` / `test_split` for submit_training_job and the right
    `input_columns` / `expectation_columns` for prepare_eval_dataset.

    Args:
        args: Required: dataset_name (HF repo ID, e.g. "PatronusAI/financebench").
              Optional:
                - split (default "train"): which split to sample rows from.
                - config (default auto): HF config / subset name; when omitted
                  we pick the first config returned by /splits.
                - columns (list[str]): restrict records preview to subset.
                - max_rows (int): 0 = metadata only (no /rows call);
                  otherwise clamped to _PREVIEW_MAX_ROWS.

    Returns:
        dict: dataset_name, split (the split we sampled from), config,
              columns (full schema), available_splits, records (≤max_rows),
              count, card_excerpt (first ~500 chars of dataset description),
              preview=True, preview_cap.
    """
    dataset_name = args["dataset_name"]
    split_arg = args.get("split", "train")
    config_arg = args.get("config") or None
    columns_filter = args.get("columns") or None
    requested_rows = int(args.get("max_rows", 0) or 0)
    row_count = min(requested_rows, _PREVIEW_MAX_ROWS) if requested_rows > 0 else 0

    # Step 1 — splits + configs. Also doubles as an existence / viewer-supported check.
    splits_body = _ds_server_get("splits", {"dataset": dataset_name})
    splits_entries = splits_body.get("splits") or []
    if not splits_entries:
        raise RuntimeError(
            f"datasets-server /splits returned no splits for {dataset_name!r}"
            " — dataset viewer may be disabled or the dataset may not exist."
        )
    available_splits = sorted({s["split"] for s in splits_entries})

    # Prefer the caller-supplied config if it exists; otherwise use the first
    # config the server reports. Public datasets usually have one ("default").
    available_configs = sorted({s["config"] for s in splits_entries})
    config = config_arg if config_arg in available_configs else available_configs[0]

    # Pick the split for sample rows. Honour the caller; if they asked for a
    # non-existent split, fall back to the first available one so the call
    # still returns useful data — but keep the requested name in the response
    # so the agent can compare.
    effective_split = split_arg if split_arg in available_splits else available_splits[0]

    # Step 2 — column schema + dataset card description.
    info_body = _ds_server_get("info", {"dataset": dataset_name, "config": config})
    dataset_info = info_body.get("dataset_info") or {}
    features = dataset_info.get("features") or {}
    full_columns = list(features.keys())
    description = (dataset_info.get("description") or "").strip()
    card_excerpt = description[:500] + ("…" if len(description) > 500 else "")

    # Step 3 — optional sample rows. Only call /rows when the caller asked for
    # records; metadata-only calls stay at 2 GETs.
    records: list[dict] = []
    if row_count > 0:
        rows_body = _ds_server_get(
            "rows",
            {
                "dataset": dataset_name,
                "config": config,
                "split": effective_split,
                "offset": 0,
                "length": row_count,
            },
        )
        for entry in rows_body.get("rows") or []:
            row = entry.get("row") or {}
            if columns_filter:
                row = {c: row.get(c) for c in columns_filter if c in row}
            records.append(row)

    return {
        "dataset_name":      dataset_name,
        "split":             effective_split,
        "requested_split":   split_arg,
        "config":            config,
        "columns":           full_columns,
        "available_splits":  available_splits,
        "available_configs": available_configs,
        "records":           records,
        "count":             len(records),
        "card_excerpt":      card_excerpt,
        "preview":           True,
        "preview_cap":       _PREVIEW_MAX_ROWS,
    }


# Transformation matrix kept in sync with
# lambda/skills/sagemaker/eval/schema_transform.py. The handler only needs to
# validate the (source, target) pair and encode the mechanism into the spec;
# the actual row transformation happens inside the eval Processing container.
_SCHEMA_TRANSFORM_MATRIX = {
    ("chat", "sft"):      "flatten_messages",
    ("sft",  "chat"):     "unflatten_to_messages",
    ("dpo",  "sft"):      "drop_rejected",
}
_SUPPORTED_SCHEMAS = {"chat", "sft", "dpo", "tabular-csv"}


def _prepare_eval_dataset(args: dict) -> dict:
    """Emit a JSON eval-spec to S3; the Processing container materialises records.

    The spec describes *how* to load + transform the dataset, not the dataset
    itself. The SageMaker Processing container (eval/entrypoint.py) reads the
    spec, calls datasets.load_dataset, applies the task-default column
    mapping, and feeds {inputs, expectations} records straight into
    mlflow.genai.evaluate. Lambda does no dataset IO.

    Schema transformation (R3): when the caller passes ``target_schema``
    that differs from ``source_schema`` (or the container's inferred
    source), the eval container will apply a record-level transformation
    via ``schema_transform.apply_transform`` before emitting records to
    MLflow. Unsupported pairs fail loudly *here* so we don't pay a
    Processing-job boot to discover the mismatch.

    Args:
        args: Required: task_type, dataset_name.
              Optional: split (default "train"), config, input_columns,
                         expectation_columns, context_columns, max_rows,
                         out_s3_prefix, source_schema, target_schema.

    Returns:
        dict: spec_s3_uri, task_type, dataset_name, split, max_rows,
              column_mapping, source_schema, target_schema,
              transform_mechanism.
    """
    task_type = args["task_type"]
    dataset_name = args["dataset_name"]
    split = args.get("split", "train")
    config = args.get("config") or None
    max_rows = int(args.get("max_rows", 0) or 0)
    out_prefix = (
        args.get("out_s3_prefix")
        or f"eval-specs/{dataset_name.replace('/', '__')}"
    ).rstrip("/")

    # R3: validate the (source_schema, target_schema) pair up-front.
    # When target_schema is omitted, container behaviour is unchanged
    # (mechanism = None, no record mutation).
    source_schema = args.get("source_schema") or None
    target_schema = args.get("target_schema") or None
    transform_mechanism: str | None = None
    if target_schema:
        if target_schema not in _SUPPORTED_SCHEMAS:
            raise ValueError(
                f"target_schema={target_schema!r} is not one of "
                f"{sorted(_SUPPORTED_SCHEMAS)}."
            )
        if source_schema and source_schema not in _SUPPORTED_SCHEMAS:
            raise ValueError(
                f"source_schema={source_schema!r} is not one of "
                f"{sorted(_SUPPORTED_SCHEMAS)}."
            )
        # If the caller omits source_schema, the container infers it from
        # dataset columns at materialisation time — we let the container
        # resolve the mechanism then. Otherwise we validate here so agents
        # get an immediate refusal on unsupported pairs.
        if source_schema:
            if source_schema == target_schema:
                transform_mechanism = "identity"
            else:
                mech = _SCHEMA_TRANSFORM_MATRIX.get((source_schema, target_schema))
                if mech is None:
                    raise ValueError(
                        f"No transformation mechanism for source_schema="
                        f"{source_schema!r} → target_schema={target_schema!r}. "
                        f"Supported pairs: "
                        f"{sorted(_SCHEMA_TRANSFORM_MATRIX.keys())} + identity."
                    )
                transform_mechanism = mech

    column_mapping = {
        "inputs":       args.get("input_columns") or None,
        "expectations": args.get("expectation_columns") or None,
        "context":      args.get("context_columns") or None,
    }

    spec = {
        "version": 2,  # bumped: v2 carries source/target schema + mechanism
        "task_type": task_type,
        "dataset_name": dataset_name,
        "split": split,
        "config": config,
        "max_rows": max_rows,
        "column_mapping": column_mapping,
        "source_schema": source_schema,
        "target_schema": target_schema,
        "transform_mechanism": transform_mechanism,
    }

    key = f"{out_prefix}/eval_spec.json"
    boto3.client("s3", region_name=AWS_REGION).put_object(
        Bucket=SESSION_BUCKET,
        Key=key,
        Body=json.dumps(spec).encode(),
        ContentType="application/json",
    )
    return {
        "spec_s3_uri":          f"s3://{SESSION_BUCKET}/{key}",
        "task_type":            task_type,
        "dataset_name":         dataset_name,
        "split":                split,
        "max_rows":             max_rows,
        "column_mapping":       column_mapping,
        "source_schema":        source_schema,
        "target_schema":        target_schema,
        "transform_mechanism":  transform_mechanism,
    }


def _update_model_card(args: dict) -> dict:
    """Update or append to the README.md on a HuggingFace repo.

    Args:
        args: Required: repo_id, content. Optional: append (bool).

    Returns:
        dict: status, repo_id.
    """
    repo_id = args["repo_id"]
    content = args["content"]
    append = bool(args.get("append", False))
    api = HfApi(token=_get_hf_token())
    if append:
        try:
            path = api.hf_hub_download(repo_id, "README.md")
            with open(path, encoding="utf-8") as f:
                current = f.read()
        except Exception:
            current = ""
        new_content = current + "\n" + content
    else:
        new_content = content
    api.upload_file(
        path_or_fileobj=new_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Update model card",
    )
    return {"status": "updated", "repo_id": repo_id}


def _manage_tags(args: dict) -> dict:
    """Add or remove tags on a HuggingFace repo.

    Args:
        args: Required: repo_id, tags (list[str]). Optional: remove (bool).

    Returns:
        dict: status, repo_id, tags.
    """
    repo_id = args["repo_id"]
    tags = args.get("tags", [])
    remove = bool(args.get("remove", False))
    api = HfApi(token=_get_hf_token())
    info = api.repo_info(repo_id=repo_id)
    existing = list(getattr(info, "tags", []) or [])
    if remove:
        new_tags = [t for t in existing if t not in tags]
    else:
        new_tags = existing + [t for t in tags if t not in existing]
    api.update_repo_settings(repo_id=repo_id, tags=new_tags)
    return {"status": "updated", "repo_id": repo_id, "tags": new_tags}


_DISPATCH: dict[str, Any] = {
    "upload_model": _upload_model,
    "hf_snapshot_download": _hf_snapshot_download,
    "retrieve_dataset_metadata": _retrieve_dataset_metadata,
    "prepare_eval_dataset": _prepare_eval_dataset,
    "update_model_card": _update_model_card,
    "manage_tags": _manage_tags,
}


def handler(event: dict, context: Any) -> dict:
    """Gateway MCP tool dispatcher for HuggingFace skills.

    Args:
        event: AgentCore Gateway passes the tool arguments map as `event`.
        context: Lambda context; tool name is in
            `context.client_context.custom['bedrockAgentCoreToolName']`,
            formatted as `${target_name}___${tool_name}`.

    Returns:
        dict: MCP content response.
    """
    raw_tool = context.client_context.custom.get("bedrockAgentCoreToolName", "") if getattr(context, "client_context", None) else ""
    tool_name = raw_tool.split("___", 1)[1] if "___" in raw_tool else raw_tool
    arguments = event or {}
    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}
    try:
        return {"content": [{"type": "text", "text": json.dumps(fn(arguments))}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
