"""Bedrock Custom Model Import poller.

Triggered by an EventBridge schedule (every 15 minutes by default). Scans
DynamoDB for in-flight ``jobs.<job_id>`` entries with
``kind='bedrock_import'`` and ``status='IN_PROGRESS'``, calls
``bedrock.get_model_import_job`` for each, and on a terminal status
updates the DDB row + resumes the agent thread via
``handler._invoke_agentcore``.

Why a poller and not an EventBridge state-change rule: AWS Bedrock does
NOT emit a ``Bedrock Model Import Job State Change`` EventBridge event
(only ``Model Customization Job State Change`` and ``Batch Inference Job
State Change``). The 15-minute cadence matches the typical 10–30 min
import wall-clock without spending too many empty-scan invocations.

This module is packaged into the same ``.build`` asset as ``handler.py``
(see ``scripts/build-callback-zip.sh``). The CDK stack wires this
``handler`` callable as the entrypoint of a separate Lambda Function and
attaches the EventBridge schedule.
"""
import os
import time

import boto3

from handler import (
    JOBS_TABLE,
    _build_resume_message,
    _get_agentcore_endpoint,
    _invoke_agentcore,
)

# Bedrock Custom Model Import status values per
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_GetModelImportJob.html
# Treat InProgress + (legacy) Queued + (defensive) Submitted as in-flight;
# everything else terminates the row.
_TERMINAL = {"Completed", "Failed", "Stopped"}
_IN_FLIGHT = {"InProgress", "Submitted", "Queued"}


def _scan_in_flight_imports() -> list[dict]:
    """Return [{thread_id, job_id, entry}, ...] for all rows whose
    jobs.<job_id> entries have kind=bedrock_import + status=IN_PROGRESS.

    Uses a paginated scan because we don't have a GSI on (kind, status).
    For the demo project's traffic this is fine; if scan cost matters,
    add a sparse GSI keyed on ``status_kind = 'IN_PROGRESS#bedrock_import'``.
    """
    ddb = boto3.resource("dynamodb").Table(JOBS_TABLE)
    matches: list[dict] = []
    scan_kwargs: dict = {}
    while True:
        resp = ddb.scan(**scan_kwargs)
        for row in resp.get("Items", []):
            thread_id = row.get("task_id") or row.get("thread_id")
            jobs = row.get("jobs") or {}
            if not thread_id or not isinstance(jobs, dict):
                continue
            for job_id, entry in jobs.items():
                if not isinstance(entry, dict):
                    continue
                if entry.get("kind") != "bedrock_import":
                    continue
                if entry.get("status") != "IN_PROGRESS":
                    continue
                matches.append({"thread_id": thread_id, "job_id": job_id, "entry": entry,
                                "user_id": row.get("user_id", "")})
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return matches


def _check_one(*, thread_id: str, job_id: str, entry: dict, user_id: str) -> None:
    """Probe one bedrock_import row. On terminal status, update DDB +
    resume the agent thread. On in-flight status, do nothing (next tick
    will re-check). Errors are logged and swallowed so one bad row does
    not block the rest of the tick.
    """
    region = os.environ.get("AWS_REGION", "us-east-1")
    ident = entry.get("import_job_identifier") or entry.get("import_job_arn", "")
    if not ident:
        print(f"[poller] {thread_id}/{job_id}: missing import_job_identifier; skipping")
        return
    try:
        bedrock = boto3.client("bedrock", region_name=region)
        desc = bedrock.get_model_import_job(jobIdentifier=ident)
    except Exception as e:
        print(f"[poller] {thread_id}/{job_id}: get_model_import_job({ident}) failed: {e}")
        return

    bedrock_status = desc.get("status", "")
    if bedrock_status in _IN_FLIGHT:
        return
    if bedrock_status not in _TERMINAL:
        # Unknown status — log and treat as in-flight; safer than
        # spuriously closing the row.
        print(f"[poller] {thread_id}/{job_id}: unrecognised status={bedrock_status!r}; skipping")
        return

    failure_msg = desc.get("failureMessage", "") or ""
    now = int(time.time())
    ddb = boto3.resource("dynamodb").Table(JOBS_TABLE)
    update_expr = (
        "SET jobs.#jid.#s           = :s, "
        "    jobs.#jid.completed_at = :t, "
        "    jobs.#jid.updated_at   = :t, "
        "    updated_at             = :t"
    )
    expr_values: dict = {":s": bedrock_status.upper(), ":t": now}
    if failure_msg:
        update_expr += ", jobs.#jid.status_message = :m"
        expr_values[":m"] = failure_msg
    try:
        ddb.update_item(
            Key={"task_id": thread_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames={"#jid": job_id, "#s": "status"},
            ExpressionAttributeValues=expr_values,
        )
        print(f"[poller] closed {thread_id}/{job_id} → {bedrock_status.upper()}")
    except Exception as e:
        print(f"[poller] DDB update failed for {thread_id}/{job_id}: {e}")
        return

    endpoint = _get_agentcore_endpoint()
    if not endpoint:
        print(f"[poller] no AgentCore endpoint; skipping resume for {thread_id}/{job_id}")
        return
    msg = _build_resume_message(
        entry.get("bedrock_model_name", ident),
        bedrock_status,
        entry,
        kind="bedrock_import",
    )
    _invoke_agentcore(endpoint, thread_id, msg, user_id=user_id)


def handler(event: dict, context) -> dict:
    """EventBridge schedule entrypoint."""
    rows = _scan_in_flight_imports()
    print(f"[poller] tick: {len(rows)} in-flight bedrock_import row(s)")
    for row in rows:
        _check_one(
            thread_id=row["thread_id"],
            job_id=row["job_id"],
            entry=row["entry"],
            user_id=row["user_id"],
        )
    return {"statusCode": 200, "checked": len(rows)}
