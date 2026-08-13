import json
import os
import time

import boto3

PROJECT_NAME = os.environ.get("PROJECT_NAME", "sample-mlops-agent")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "sample-mlops-agent-metadata")
TERMINAL = {"Completed", "Failed", "Stopped"}


def _materialize_baseline_objects(*, artifact_s3: str, base_prefix: str) -> None:
    """Extract ``baseline/*`` members from a tabular training job's
    ``model.tar.gz`` and upload them as standalone S3 objects under
    ``base_prefix`` (QA BUG-003).

    Args:
        artifact_s3: s3:// URI of the job's model.tar.gz.
        base_prefix: s3:// prefix to upload baseline objects under
            (``…/output/baseline``).

    Raises:
        RuntimeError: If the tarball is oversized or contains no baseline
            members — callers must not stamp URIs in that case.
    """
    import tarfile  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    from urllib.parse import urlparse  # noqa: PLC0415

    src = urlparse(artifact_s3)
    dst = urlparse(base_prefix)
    s3 = boto3.client("s3")
    head = s3.head_object(Bucket=src.netloc, Key=src.path.lstrip("/"))
    size = head.get("ContentLength", 0)
    # Tabular (xgboost/sklearn) artifacts are KB–MB; anything huge means we
    # were called for the wrong job type — refuse rather than OOM the Lambda.
    if size > 256 * 1024 * 1024:
        raise RuntimeError(f"model.tar.gz is {size} bytes (>256 MB); refusing to extract")
    uploaded = 0
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = os.path.join(tmp, "model.tar.gz")
        s3.download_file(src.netloc, src.path.lstrip("/"), tar_path)
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf.getmembers():
                name = member.name.lstrip("./")
                # Only plain files under baseline/, and no path traversal.
                if not member.isfile() or not name.startswith("baseline/") or ".." in name:
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                key = f"{dst.path.lstrip('/')}/{name.split('/', 1)[1]}"
                s3.upload_fileobj(fobj, dst.netloc, key)
                uploaded += 1
    if uploaded == 0:
        raise RuntimeError(f"no baseline/* members found in {artifact_s3}")
    print(f"[callback] materialized {uploaded} baseline objects under {base_prefix}")


def _stamp_tabular_baseline_uris(
    *, thread_id: str, job_id: str, artifact_s3: str, training_type: str,
) -> None:
    """R1 Task 3. After a tabular (xgboost/sklearn) training job Completes,
    derive baseline + eval_split S3 URIs from the model artifact path and
    write them onto the existing jobs.<job_id> entry. LLM training types
    never emit a baseline so stamping would leave dead URIs on the thread
    row — we exit early for them.

    Runs as a second update_item after ``_process_state_change`` has
    already written the status. Cheap (single conditional update), and
    idempotent (same update with identical values on retry).
    """
    if training_type not in ("xgboost", "sklearn"):
        return
    if not artifact_s3:
        return
    # QA BUG-003: the training script writes /opt/ml/model/baseline/*.csv, and
    # SageMaker packs everything under /opt/ml/model INSIDE model.tar.gz — the
    # files are NOT uploaded as standalone siblings. Stamping sibling URIs
    # without materialising them left every monitoring run 403ing on its
    # baseline pre-flight. Extract baseline/* from the tarball and upload them
    # to the derived prefix; stamp only after the objects actually exist.
    base_prefix = artifact_s3.rsplit("/output/", 1)[0] + "/output/baseline"
    baseline_uri = f"{base_prefix}/baseline.csv"
    eval_uri = f"{base_prefix}/eval_split.csv"
    try:
        _materialize_baseline_objects(artifact_s3=artifact_s3, base_prefix=base_prefix)
    except Exception as e:
        # Fail loudly and do NOT stamp — a stamped-but-missing URI is exactly
        # the silent failure mode this fix removes.
        print(f"[callback] baseline extraction failed for {job_id}: {e}; not stamping")
        return
    ddb = boto3.resource("dynamodb").Table(JOBS_TABLE)
    try:
        ddb.update_item(
            Key={"task_id": thread_id},
            UpdateExpression=(
                "SET jobs.#jid.baseline_s3_uri   = :b, "
                "    jobs.#jid.eval_split_s3_uri = :e"
            ),
            ExpressionAttributeNames={"#jid": job_id},
            ExpressionAttributeValues={":b": baseline_uri, ":e": eval_uri},
            ConditionExpression="attribute_exists(jobs.#jid)",
        )
        print(f"[callback] stamped baseline URIs for {job_id}: {baseline_uri}")
    except ddb.meta.client.exceptions.ConditionalCheckFailedException:
        print(f"[callback] job {job_id} not present; skipping baseline URI stamp")
    except Exception as e:
        print(f"[callback] baseline URI stamp failed for {job_id}: {e}")


def _get_agentcore_endpoint() -> str:
    ssm_key = os.environ.get("AGENTCORE_ENDPOINT_SSM", f"/{PROJECT_NAME}/dev/agentcore/endpoint")
    if ssm_key:
        try:
            return boto3.client("ssm").get_parameter(Name=ssm_key)["Parameter"]["Value"]
        except Exception as e:
            print(f"[callback] SSM lookup failed: {e}")
    return os.environ.get("AGENTCORE_ENDPOINT", "")


def _invoke_agentcore(endpoint: str, session_id: str, message: str, user_id: str = "") -> None:
    """Resume an AgentCore session, carrying user identity context.

    Args:
        endpoint: Full runtime URL from SSM (https://.../runtimes/<arn>).
        session_id: The AgentCore session/thread ID.
        message: Human-readable resume message describing job outcome.
        user_id: Cognito sub from DynamoDB. Empty string omits user header.
    """
    try:
        agent_runtime_arn = endpoint.split("/runtimes/", 1)[1]
    except IndexError:
        print(f"[callback] Malformed endpoint URL, cannot extract ARN: {endpoint}")
        return

    region = os.environ.get("AWS_REGION", "us-east-1")
    payload = json.dumps({
        "session_id": session_id,
        "user_id": user_id,
        "message": message,
    }).encode()

    invoke_kwargs: dict = dict(
        agentRuntimeArn=agent_runtime_arn,
        runtimeSessionId=session_id,
        qualifier="DEFAULT",
        payload=payload,
    )
    if user_id:
        invoke_kwargs["runtimeUserId"] = user_id

    try:
        client = boto3.client("bedrock-agentcore", region_name=region)
        response = client.invoke_agent_runtime(**invoke_kwargs)
        # Drain the AgentCore streaming body — the resume is fire-and-forget;
        # we only need the request to complete.
        for _ in response.get("response", []):  # nosemgrep: pass-body-range
            pass
        print(f"[callback] resumed session {session_id} for user={user_id or '(unknown)'}")
    except Exception as e:
        print(f"[callback] AgentCore resume failed: {e}")


def _build_resume_message(job_name: str, status: str, item: dict, *, kind: str = "training") -> str:
    """Build a rich resume message with artifact path and MLflow context.

    Args:
        job_name: SageMaker job name (training or processing) OR
            Bedrock imported model name when kind='bedrock_import'.
        status: Terminal status string (Completed/Failed/Stopped) OR
            Bedrock import status (Completed/Failed/InProgress).
        item: The nested jobs.<job_id> entry from DynamoDB.
        kind: "training" | "eval" | "monitoring" | "bedrock_import" —
            drives the message wording.
    """
    artifact_path = ""
    sm = boto3.client("sagemaker")
    if status == "Completed" and kind == "training":
        try:
            desc = sm.describe_training_job(TrainingJobName=job_name)
            artifact_path = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
        except Exception as e:
            print(f"[callback] Could not get artifacts: {e}")

    mlflow_run_id = item.get("mlflow_run_id", "")
    mlflow_run_url = item.get("mlflow_run_url", "")

    nouns = {"training": "Training job", "eval": "Evaluation job",
             "monitoring": "Monitoring job",
             "bedrock_import": "Bedrock model import"}
    noun = nouns.get(kind, "Training job")
    msg = f"{noun} {job_name} is now {status}."
    if artifact_path:
        msg += f" Model artifacts: {artifact_path}."
    if mlflow_run_id:
        msg += f" MLflow run_id: {mlflow_run_id}."
    if mlflow_run_url:
        msg += f" MLflow run URL: {mlflow_run_url}."
    if kind == "eval" and status == "Completed":
        msg += (
            " Review the MLflow results and then invoke the"
            " compliance-documentation skill to generate the report."
        )
    if kind == "monitoring" and status == "Completed":
        # R1 Task 9: deliberate divergence from eval — monitoring is a
        # read, not an audit artifact. Do NOT invoke compliance-documentation.
        msg += (
            " Surface drifted_columns_share, drifted_columns_count, and accuracy"
            " (if present) from the MLflow run to the user. Do NOT invoke the"
            " compliance-documentation skill."
        )
    if kind == "bedrock_import":
        # R5: Bedrock Custom Model Import has no EventBridge event, so this
        # branch is exercised by the bedrock-import scheduled poller, not by
        # an EventBridge state-change handler. The import_job_arn lets the
        # agent retrieve the imported model with bedrock.get_model_import_job.
        import_arn = item.get("import_job_arn", "")
        if import_arn:
            msg += f" Bedrock import job ARN: {import_arn}."
        if status == "Completed":
            msg += (
                " The imported model is now invokable via Bedrock"
                " InvokeModel using the imported model name."
            )
    msg += " Please continue."
    return msg


def _tags_from_describe(describe_resp: dict, sm, arn_key: str) -> dict:
    """Extract a tag map from a describe_* response, falling back to list_tags.

    describe_training_job / describe_processing_job usually return Tags inline,
    but some SDK shapes omit them, in which case we need list_tags.
    """
    tags = {t["Key"]: t["Value"] for t in (describe_resp.get("Tags") or [])}
    if tags:
        return tags
    try:
        arn = describe_resp[arn_key]
        ltags = sm.list_tags(ResourceArn=arn).get("Tags", [])
        return {t["Key"]: t["Value"] for t in ltags}
    except Exception as e:
        print(f"[callback] list_tags fallback failed: {e}")
        return {}


def _process_state_change(
    *,
    sm,
    job_name: str,
    status: str,
    tags: dict,
    failure_reason: str,
    kind: str,
) -> dict:
    """Common tail: update DynamoDB row and resume AgentCore on terminal state.

    Returns the response dict to hand back to EventBridge.
    """
    thread_id = tags.get("ThreadId", "")
    job_id    = tags.get("JobId", "")
    if not thread_id or not job_id:
        print(f"[callback] missing ThreadId/JobId tags on {job_name} ({kind}) — skipping")
        return {"statusCode": 404}

    ddb = boto3.resource("dynamodb").Table(JOBS_TABLE)
    try:
        update_expr = (
            "SET jobs.#jid.#s           = :s, "
            "    jobs.#jid.completed_at = :t, "
            "    jobs.#jid.updated_at   = :t, "
            "    updated_at             = :t"
        )
        expr_names = {"#jid": job_id, "#s": "status"}
        expr_values: dict = {":s": status.upper(), ":t": int(time.time())}
        if failure_reason:
            update_expr += ", jobs.#jid.status_message = :m"
            expr_values[":m"] = failure_reason
        ddb.update_item(
            Key={"task_id": thread_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )
        print(f"[callback] updated {job_name} (thread={thread_id}, job={job_id}, kind={kind}) → {status.upper()}")
    except Exception as e:
        print(f"[callback] DynamoDB update failed: {e}")
        return {"statusCode": 500}

    if status in TERMINAL:
        endpoint = _get_agentcore_endpoint()
        if endpoint:
            try:
                row = ddb.get_item(Key={"task_id": thread_id}).get("Item", {}) or {}
            except Exception as e:
                print(f"[callback] get_item for resume failed: {e}")
                row = {}
            job_entry = (row.get("jobs") or {}).get(job_id, {})
            user_id = row.get("user_id", "")
            msg = _build_resume_message(job_name, status, job_entry, kind=kind)
            _invoke_agentcore(endpoint, thread_id, msg, user_id=user_id)

    return {"statusCode": 200}


def _handle_training_event(detail: dict) -> dict:
    job_name = detail.get("TrainingJobName", "")
    status   = detail.get("TrainingJobStatus", "")
    if not job_name:
        return {"statusCode": 400}
    sm = boto3.client("sagemaker")
    try:
        desc = sm.describe_training_job(TrainingJobName=job_name)
    except Exception as e:
        print(f"[callback] describe_training_job failed: {e}")
        return {"statusCode": 500}
    tags = _tags_from_describe(desc, sm, "TrainingJobArn")
    failure = desc.get("FailureReason", "") if status == "Failed" else ""
    result = _process_state_change(
        sm=sm, job_name=job_name, status=status, tags=tags,
        failure_reason=failure, kind="training",
    )

    # R1 Task 3: on Completed tabular training jobs, stamp the baseline +
    # eval_split S3 URIs into the jobs sub-record so submit_monitoring_job
    # can resolve them from DDB rather than the agent having to pass them.
    if status == "Completed" and result.get("statusCode") == 200:
        thread_id = tags.get("ThreadId", "")
        job_id = tags.get("JobId", "")
        artifact_s3 = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
        if thread_id and job_id and artifact_s3:
            try:
                ddb = boto3.resource("dynamodb").Table(JOBS_TABLE)
                row = ddb.get_item(Key={"task_id": thread_id}).get("Item", {}) or {}
                job_entry = (row.get("jobs") or {}).get(job_id, {})
                training_type = job_entry.get("training_type", "")
                _stamp_tabular_baseline_uris(
                    thread_id=thread_id, job_id=job_id,
                    artifact_s3=artifact_s3, training_type=training_type,
                )
            except Exception as e:
                print(f"[callback] baseline URI lookup failed: {e}")

    return result


def _handle_processing_event(detail: dict) -> dict:
    job_name = detail.get("ProcessingJobName", "")
    status   = detail.get("ProcessingJobStatus", "")
    if not job_name:
        return {"statusCode": 400}
    sm = boto3.client("sagemaker")
    try:
        desc = sm.describe_processing_job(ProcessingJobName=job_name)
    except Exception as e:
        print(f"[callback] describe_processing_job failed: {e}")
        return {"statusCode": 500}
    tags = _tags_from_describe(desc, sm, "ProcessingJobArn")
    failure = desc.get("FailureReason", "") if status == "Failed" else ""
    # R1 Task 9: Both eval and monitoring Processing jobs flow through this
    # handler. The submit handlers tag Kind accordingly so we can branch.
    # Unknown or missing Kind defaults to eval for backwards compat with
    # older eval jobs that predate the Kind tag.
    kind_tag = tags.get("Kind", "").lower()
    kind = "monitoring" if kind_tag == "monitoring" else "eval"
    return _process_state_change(
        sm=sm, job_name=job_name, status=status, tags=tags,
        failure_reason=failure, kind=kind,
    )


def _aiperf_workload_spec(spec: dict) -> dict:
    """Translate our user-facing workload spec into the AIPerf inline spec
    the SageMaker AI Benchmark service validates (QA BUG-019).

    The service's WorkloadSpec.Inline schema (per the SageMaker dev guide,
    "workload configuration for benchmarking") is:
    ``{"benchmark": {"type": "aiperf"}, "parameters": {...}, "tooling": {...}}``
    — the flat {input_tokens, output_tokens, concurrency_levels, …} shape we
    store on the DDB entry is rejected with pydantic validation errors.

    AIPerf runs ONE concurrency per job, so we benchmark the highest
    requested level (worst case for the p99 gate).

    Args:
        spec: The DDB entry's workload_spec map (Decimals allowed).

    Returns:
        The aiperf-shaped spec dict, json.dumps-able with ``_jsonable``.

    Raises:
        RuntimeError: If the spec carries no tokenizer — AIPerf requires the
            deployed model's tokenizer and guessing one corrupts token counts.
    """
    tokenizer = str(spec.get("tokenizer") or "")
    if not tokenizer:
        raise RuntimeError(
            "workload_spec has no tokenizer — resubmit the recommendation job "
            "(newer submit stamps the source model_id as the tokenizer)."
        )
    levels = [int(c) for c in (spec.get("concurrency_levels") or [1])]
    input_tokens = int(spec.get("input_tokens", 500))
    output_tokens = int(spec.get("output_tokens", 150))
    return {
        "benchmark": {"type": "aiperf"},
        "parameters": {
            "tokenizer": tokenizer,
            "concurrency": max(levels),
            "request_count": 30,
            "streaming": True,
            "prompt_input_tokens_mean": input_tokens,
            "prompt_input_tokens_stddev": max(1, input_tokens // 10),
            "output_tokens_mean": output_tokens,
            "output_tokens_stddev": max(1, output_tokens // 10),
        },
        "tooling": {"api_standard": "openai", "version": "0.8.0"},
    }


def _jsonable(obj: object) -> object:
    """json.dumps ``default`` for DynamoDB values (QA BUG-016).

    DynamoDB deserializes every number as decimal.Decimal; convert to int
    when integral, else float. Any other unknown type raises TypeError as
    json.dumps normally would.

    Args:
        obj: The non-serializable object json.dumps encountered.

    Returns:
        An int or float for Decimal inputs.
    """
    from decimal import Decimal  # noqa: PLC0415
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _normalize_endpoint_status(raw: str) -> str:
    """Map an endpoint status onto the API casing used throughout this module.

    EventBridge 'SageMaker Endpoint State Change' events carry UPPER_SNAKE
    statuses (``IN_SERVICE``, ``FAILED``) while the DescribeEndpoint API uses
    CamelCase (``InService``, ``Failed``). QA BUG-012: comparing the raw event
    value against API casing silently dropped every endpoint event — the
    benchmark never started and failed endpoints were never torn down.

    Args:
        raw: Status string in either casing.

    Returns:
        The API-cased status, or ``raw`` unchanged when unrecognised.
    """
    key = raw.replace("_", "").upper()
    return {
        "INSERVICE": "InService",
        "FAILED": "Failed",
        "CREATING": "Creating",
        "UPDATING": "Updating",
        "DELETING": "Deleting",
        "ROLLINGBACK": "RollingBack",
        "OUTOFSERVICE": "OutOfService",
        "SYSTEMUPDATING": "SystemUpdating",
        "UPDATEROLLBACKFAILED": "UpdateRollbackFailed",
    }.get(key, raw)


def _handle_endpoint_event(detail: dict) -> dict:
    """React to 'SageMaker Endpoint State Change'. Only InService → start
    benchmark; Failed → mark failed and tear down partials. All other
    endpoints (not tagged Kind=recommendation) are ignored."""
    import json  # noqa: PLC0415
    endpoint_name = detail.get("EndpointName", "")
    if not endpoint_name:
        return {"statusCode": 400}
    sm = boto3.client("sagemaker")
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
    except Exception as e:
        print(f"[callback] describe_endpoint failed: {e}")
        return {"statusCode": 500}
    # Prefer the live DescribeEndpoint status over the (possibly stale,
    # differently-cased) event detail; normalize both — see BUG-012.
    endpoint_status = _normalize_endpoint_status(
        desc.get("EndpointStatus") or detail.get("EndpointStatus", "")
    )
    tags = _tags_from_describe(desc, sm, "EndpointArn")
    if tags.get("Kind") != "recommendation":
        # F-5: log once per ignored endpoint so CloudWatch Insights can
        # measure noise at production scale. EventBridge can't filter on
        # resource tags, so account-wide endpoint state changes all land
        # here and we filter in code.
        print(f"[callback] ignoring endpoint event kind={tags.get('Kind','')!r} "
              f"name={endpoint_name!r}")
        return {"statusCode": 200}
    thread_id = tags.get("ThreadId", "")
    rec_id    = tags.get("RecId", "")
    if not thread_id or not rec_id:
        return {"statusCode": 404}
    ddb = boto3.resource("dynamodb").Table(JOBS_TABLE)
    row = ddb.get_item(Key={"task_id": thread_id}).get("Item") or {}
    entry = (row.get("jobs") or {}).get(rec_id) or {}
    if not entry:
        return {"statusCode": 404}
    if endpoint_status == "Failed":
        _teardown_recommendation_resources(entry)
        # Endpoints are tagged RecId (not JobId); _process_state_change keys
        # the jobs-map update on the JobId tag, so map it explicitly — without
        # this the FAILED stamp silently 404s (found by BUG-012 regression test).
        _process_state_change(sm=sm, job_name=endpoint_name, status="Failed",
                              tags={**tags, "JobId": rec_id},
                              failure_reason=desc.get("FailureReason", ""),
                              kind="recommendation")
        return {"statusCode": 200}
    if endpoint_status != "InService":
        return {"statusCode": 200}  # transient state; wait
    # --- Start the benchmark ---
    # QA BUG-020: the model is deployed on the endpoint variant (classic
    # variant, see _background_submit_recommendation), so InService already
    # means "model loaded and serving" — no inference component is created,
    # and the benchmark targets the bare endpoint exactly like AWS's
    # reference implementation of this flow.
    try:
        # QA BUG-016/BUG-019: translate to the AIPerf spec shape the service
        # validates, with Decimal-safe serialization (DDB numbers).
        sm.create_ai_workload_config(
            AIWorkloadConfigName=entry["ai_workload_config_name"],
            AIWorkloadConfigs={"WorkloadSpec": {
                "Inline": json.dumps(_aiperf_workload_spec(entry["workload_spec"]),
                                     default=_jsonable)}},
        )
        sm.create_ai_benchmark_job(
            AIBenchmarkJobName=entry["recommender_job_name"],
            BenchmarkTarget={"Endpoint": {"Identifier": endpoint_name}},
            OutputConfig={"S3OutputLocation":
                f"s3://{os.environ['SESSION_BUCKET']}/recommendation-benchmarks/"
                f"{rec_id}/"},
            AIWorkloadConfigIdentifier=entry["ai_workload_config_name"],
            RoleArn=os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"],
            Tags=[{"Key": "Kind", "Value": "recommendation"},
                  {"Key": "RecId", "Value": rec_id}],
        )
    except Exception as e:
        print(f"[callback] benchmark start failed: {e}")
        _teardown_recommendation_resources(entry)
        ddb.update_item(Key={"task_id": thread_id},
            UpdateExpression="SET jobs.#jid.#s = :s, jobs.#jid.status_message = :m, updated_at = :t",
            ExpressionAttributeNames={"#jid": rec_id, "#s": "status"},
            ExpressionAttributeValues={":s": "FAILED", ":m": str(e), ":t": int(time.time())})
        return {"statusCode": 200}
    ddb.update_item(Key={"task_id": thread_id},
        UpdateExpression="SET jobs.#jid.#s = :s, jobs.#jid.status_message = :m, updated_at = :t",
        ExpressionAttributeNames={"#jid": rec_id, "#s": "status"},
        ExpressionAttributeValues={":s": "BENCHMARKING",
                                   ":m": f"AI benchmark job {entry['recommender_job_name']} started",
                                   ":t": int(time.time())})
    # Design decision (post-eng-review): AWS does NOT emit a
    # "SageMaker AI Benchmark Job State Change" EventBridge event, so
    # we cannot auto-resume the agent thread when the benchmark finishes.
    # Agent-resume-on-completion is deliberately OUT OF SCOPE for v1.
    # Instead, the agent calls `get_recommendation_results` which itself
    # calls describe_ai_benchmark_job and, if terminal, parses + tears
    # down. User drives the "is it done yet?" cadence from the frontend.
    return {"statusCode": 200}


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


def _parse_profile_export_jsonl(s3_uri: str) -> dict:
    """Download <s3_uri>/output.tar.gz, extract profile_export.jsonl, parse."""
    import tarfile
    import tempfile
    import os  # noqa: PLC0415
    s3 = boto3.client("s3")
    # s3_uri points to the OutputConfig.S3OutputLocation *prefix*.
    # The benchmark writes output.tar.gz there.
    prefix = s3_uri[5:].rstrip("/")
    bucket, _, key_prefix = prefix.partition("/")
    tar_key = f"{key_prefix}/output.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        local_tar = os.path.join(tmp, "output.tar.gz")
        try:
            s3.download_file(bucket, tar_key, local_tar)
        except Exception as e:
            print(f"[callback] could not fetch {tar_key}: {e}")
            return {}
        with tarfile.open(local_tar, "r:gz") as tf:
            tf.extractall(tmp, filter="data")
        for root, _, files in os.walk(tmp):
            for f in files:
                if f == "profile_export.jsonl":
                    return _parse_profile_export_jsonl_from_path(os.path.join(root, f))
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


EVENT_SOURCES = {
    "SageMaker Training Job State Change":   _handle_training_event,
    "SageMaker Processing Job State Change": _handle_processing_event,
    "SageMaker Endpoint State Change":       _handle_endpoint_event,
}


def handler(event: dict, context) -> dict:
    """EventBridge callback — routes on detail-type.

    Training completion resumes the agent thread with the training artifacts;
    Processing completion does the same but tags the message as an eval
    result so the agent knows to invoke generate_compliance_report.
    """
    detail_type = event.get("detail-type", "")
    fn = EVENT_SOURCES.get(detail_type)
    if fn is None:
        print(f"[callback] unrecognised detail-type: {detail_type!r}")
        return {"statusCode": 400}
    return fn(event.get("detail", {}))
