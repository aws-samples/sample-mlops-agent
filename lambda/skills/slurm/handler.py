"""Slurm skill Lambda — Gateway MCP target.

Exposes 4 tools to the AgentCore Gateway:
  - submit_slurm_job
  - check_slurm_job_status
  - cancel_slurm_job
  - list_slurm_jobs

In MOCK_MODE=1 (default until pcluster infra is wired), the SSH path is
replaced with an in-memory simulator so the tool chain works end-to-end
without requiring a real Slurm head node.
"""
import json
import os
import re
import shlex
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3

PROJECT_NAME = os.environ.get("PROJECT_NAME", "sample-mlops-agent")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "sample-mlops-agent-metadata")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

SLURM_HOST = os.environ.get("SLURM_HEAD_NODE_HOST", "")
SLURM_USER = os.environ.get("SLURM_SSH_USER", "ec2-user")
SLURM_SSH_SECRET_ARN = os.environ.get("SLURM_SSH_SECRET_ARN", "")
# Path to a known_hosts file pinning the head node's public key. When set, the SSH
# client verifies the head node against it and REFUSES unknown keys — closing the
# MITM window that paramiko.AutoAddPolicy (trust-on-first-use) left open.
SLURM_KNOWN_HOSTS = os.environ.get("SLURM_KNOWN_HOSTS", "")
# Default to mock mode: the main project does not yet deploy pcluster infra.
MOCK_MODE = os.environ.get("MOCK_MODE", "1") == "1"

# SLURM short state → canonical status (covers both squeue shortcodes and sacct long-form)
_STATE_MAP = {
    "PD": "PENDING",
    "R": "RUNNING",
    "CG": "COMPLETING",
    "CD": "COMPLETED",
    "F": "FAILED",
    "CA": "CANCELLED",
    "TO": "TIMEOUT",
    "PENDING": "PENDING",
    "RUNNING": "RUNNING",
    "COMPLETED": "COMPLETED",
    "FAILED": "FAILED",
    "CANCELLED": "CANCELLED",
    "CANCELLED+": "CANCELLED",
    "TIMEOUT": "TIMEOUT",
}

_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}

# Slurm job ids are integers, optionally with an array-task suffix (123_4) or a
# step suffix (123.batch). Anything else must never reach a shell command string.
_SLURM_JOB_ID_RE = re.compile(r"^\d+(?:_\d+)?(?:\.[A-Za-z0-9_+-]+)?$")


def _assert_slurm_job_id(job_id: str) -> str:
    """Validate a Slurm job id before it is interpolated into an SSH shell command.

    Args:
        job_id: The Slurm job id (DDB-sourced, originally parsed from remote output).

    Returns:
        str: The validated job id, unchanged.

    Raises:
        ValueError: If job_id is not a well-formed Slurm job id — this blocks any
            shell-metacharacter payload from reaching the head node.
    """
    if not _SLURM_JOB_ID_RE.match(job_id or ""):
        raise ValueError(f"Refusing unsafe Slurm job id: {job_id!r}")
    return job_id


def _ddb_table() -> Any:
    """Return DynamoDB Table resource bound to the shared metadata table."""
    return boto3.resource("dynamodb", region_name=AWS_REGION).Table(JOBS_TABLE)


def _load_ssh_key() -> str:
    """Fetch the Slurm SSH private key from Secrets Manager, write to /tmp, return path.

    Returns:
        str: Absolute path to the written key file (chmod 600).

    Raises:
        RuntimeError: If SLURM_SSH_SECRET_ARN is not configured.
    """
    if not SLURM_SSH_SECRET_ARN:
        raise RuntimeError("SLURM_SSH_SECRET_ARN is not set — cannot SSH without a key")
    sm = boto3.client("secretsmanager", region_name=AWS_REGION)
    secret = sm.get_secret_value(SecretId=SLURM_SSH_SECRET_ARN)["SecretString"]
    # /tmp is the only writable filesystem in Lambda; one invocation per
    # sandbox, chmod 0600 below.
    key_path = "/tmp/slurm_rsa"  # nosemgrep: hardcoded-tmp-path # nosec B108
    with open(key_path, "w", encoding="utf-8") as fh:
        fh.write(secret)
    os.chmod(key_path, 0o600)
    return key_path


def _ssh_run(cmd: str) -> tuple[int, str, str]:
    """Execute `cmd` on the Slurm head node over SSH.

    Args:
        cmd: Remote shell command to run.

    Returns:
        tuple[int, str, str]: (exit_code, stdout, stderr).

    Raises:
        RuntimeError: If SLURM_HEAD_NODE_HOST is not set.
    """
    if not SLURM_HOST:
        raise RuntimeError("SLURM_HEAD_NODE_HOST is not set — cannot reach head node")
    # Host-key verification is mandatory: without a pinned known_hosts file we cannot
    # tell the real head node from an attacker's, so refuse before doing any work.
    # (Checked before importing paramiko so a misconfig fails fast and cheaply.)
    if not SLURM_KNOWN_HOSTS:
        raise RuntimeError(
            "SLURM_KNOWN_HOSTS is not set — refusing to SSH without a pinned head-node "
            "host key (set SLURM_KNOWN_HOSTS to a known_hosts file to enable Slurm SSH)"
        )
    # Imported lazily so MOCK_MODE runs never require the paramiko layer.
    import paramiko  # type: ignore

    key_path = _load_ssh_key()
    client = paramiko.SSHClient()
    # Load the pinned key(s) and REJECT any head node whose key is not already
    # trusted. AutoAddPolicy (trust-on-first-use) would accept an attacker's key on
    # the first connection, so we require explicit provisioning of known_hosts.
    client.load_host_keys(SLURM_KNOWN_HOSTS)
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        hostname=SLURM_HOST,
        username=SLURM_USER,
        key_filename=key_path,
        timeout=30,
    )
    try:
        # cmd is assembled exclusively from shlex.quote()d fragments (see
        # callers) and the host key is pinned via SLURM_KNOWN_HOSTS.
        _, stdout, stderr = client.exec_command(cmd)  # nosec B601
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
    finally:
        client.close()
    return exit_code, out, err


def _mock_sbatch(job_name: str) -> tuple[int, str, str]:
    """Simulate sbatch without SSH — returns a deterministic-looking fake job id."""
    fake_job_id = str(100000 + int(uuid.uuid4().int % 9000))
    return 0, f"Submitted batch job {fake_job_id}", ""


def _mock_sacct(job_id: str, current_status: str) -> tuple[int, str, str]:
    """Simulate sacct: progress SUBMITTED → PENDING → RUNNING → COMPLETED on each poll."""
    progression = {
        "SUBMITTED": "PENDING",
        "PENDING": "RUNNING",
        "RUNNING": "COMPLETED",
    }
    next_state = progression.get(current_status, current_status)
    elapsed = "00:30:00" if next_state == "RUNNING" else "06:00:00"
    return 0, f"{job_id}|{next_state}|0:0|{elapsed}", ""


def _submit_slurm_job(args: dict) -> dict:
    """Submit a Slurm job on the pcluster head node and record it in DynamoDB.

    Args:
        args: Tool arguments. Required: thread_id, script_path.
              Optional: job_name, workflow_step, _user_id.

    Returns:
        dict: task_id, slurm_job_id, job_name, status.
    """
    thread_id = args["thread_id"]
    script_path = args["script_path"]
    job_name = args.get("job_name", f"{PROJECT_NAME}-job-{int(time.time())}")
    workflow_step = args.get("workflow_step", "unknown")
    user_id = args.get("_user_id", "")

    comment = f"user_id={user_id},thread_id={thread_id}"
    # shlex.quote every interpolated value: job_name / script_path / comment all
    # originate from MCP tool args (agent-influenced) and are run via SSH on the
    # head node — without quoting a value like `x.sh; rm -rf ~` would inject.
    cmd = (
        f"sbatch --job-name={shlex.quote(job_name)} "
        f"--comment={shlex.quote(comment)} {shlex.quote(script_path)}"
    )

    if MOCK_MODE:
        exit_code, out, err = _mock_sbatch(job_name)
    else:
        exit_code, out, err = _ssh_run(cmd)

    if exit_code != 0:
        raise RuntimeError(f"sbatch failed (exit {exit_code}): {err}")

    # "Submitted batch job 12345" → "12345"
    slurm_job_id = out.strip().split()[-1]
    task_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    _ddb_table().put_item(Item={
        "task_id": task_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "slurm_job_id": slurm_job_id,
        "job_name": job_name,
        "workflow_step": workflow_step,
        "script_path": script_path,
        "status": "SUBMITTED",
        "status_message": f"Submitted Slurm job {slurm_job_id}",
        "submitted_at": now_iso,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "retry_count": 0,
        "mock_mode": MOCK_MODE,
    })

    return {
        "task_id": task_id,
        "slurm_job_id": slurm_job_id,
        "job_name": job_name,
        "status": "SUBMITTED",
    }


def _check_slurm_job_status(args: dict) -> dict:
    """Check a Slurm job's current state via sacct and update DynamoDB.

    Args:
        args: Tool arguments. Required: task_id.

    Returns:
        dict: task_id, slurm_job_id, status, exit_code, elapsed, terminal.
    """
    task_id = args["task_id"]

    item = _ddb_table().get_item(Key={"task_id": task_id}).get("Item")
    if not item:
        raise ValueError(f"No task found for task_id={task_id!r}")

    slurm_job_id = item.get("slurm_job_id", "")
    if not slurm_job_id:
        raise ValueError(f"task_id={task_id!r} has no slurm_job_id — not a Slurm task")

    current_status = item.get("status", "UNKNOWN")

    if MOCK_MODE:
        exit_code, out, err = _mock_sacct(slurm_job_id, current_status)
    else:
        cmd = (
            f"sacct -j {_assert_slurm_job_id(slurm_job_id)} "
            f"--format=JobID,State,ExitCode,Elapsed "
            f"--noheader --parsable2"
        )
        exit_code, out, err = _ssh_run(cmd)

    if exit_code != 0:
        raise RuntimeError(f"sacct failed (exit {exit_code}): {err}")

    # Skip step lines like "12345.batch" — only consider the top-level job row.
    lines = [ln for ln in out.splitlines() if ln and "." not in ln.split("|")[0]]
    if not lines:
        return {
            "task_id": task_id,
            "slurm_job_id": slurm_job_id,
            "status": "UNKNOWN",
            "message": "no sacct output",
        }

    parts = lines[0].split("|")
    raw_state = parts[1] if len(parts) > 1 else "UNKNOWN"
    exit_str = parts[2] if len(parts) > 2 else "0:0"
    elapsed = parts[3] if len(parts) > 3 else ""
    status = _STATE_MAP.get(raw_state.upper(), raw_state)

    _ddb_table().update_item(
        Key={"task_id": task_id},
        UpdateExpression="SET #s = :s, exit_code = :e, elapsed = :el, updated_at = :u",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": status,
            ":e": exit_str,
            ":el": elapsed,
            ":u": int(time.time()),
        },
    )

    return {
        "task_id": task_id,
        "slurm_job_id": slurm_job_id,
        "status": status,
        "exit_code": exit_str,
        "elapsed": elapsed,
        "terminal": status in _TERMINAL,
    }


def _cancel_slurm_job(args: dict) -> dict:
    """Cancel a running Slurm job via scancel and mark DynamoDB row CANCELLED.

    Args:
        args: Tool arguments. Required: task_id.

    Returns:
        dict: task_id, slurm_job_id, status.
    """
    task_id = args["task_id"]

    item = _ddb_table().get_item(Key={"task_id": task_id}).get("Item")
    if not item:
        raise ValueError(f"No task found for task_id={task_id!r}")

    slurm_job_id = item.get("slurm_job_id", "")
    if not slurm_job_id:
        raise ValueError(f"task_id={task_id!r} has no slurm_job_id — not a Slurm task")

    if not MOCK_MODE:
        _ssh_run(f"scancel {_assert_slurm_job_id(slurm_job_id)}")

    _ddb_table().update_item(
        Key={"task_id": task_id},
        UpdateExpression="SET #s = :s, updated_at = :u",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": "CANCELLED", ":u": int(time.time())},
    )

    return {"task_id": task_id, "slurm_job_id": slurm_job_id, "status": "CANCELLED"}


def _list_slurm_jobs(args: dict) -> dict:
    """List Slurm jobs for a thread (scans the metadata table and filters).

    Args:
        args: Tool arguments. Required: thread_id.

    Returns:
        dict: jobs — list of {task_id, slurm_job_id, job_name, status}.
    """
    thread_id = args["thread_id"]
    # The metadata table's PK is task_id with no thread GSI; scan with filter is fine
    # at this scale (per-thread rows, not cross-tenant aggregation).
    resp = _ddb_table().scan(
        FilterExpression="thread_id = :tid AND attribute_exists(slurm_job_id)",
        ExpressionAttributeValues={":tid": thread_id},
    )
    jobs = [
        {
            "task_id": it.get("task_id", ""),
            "slurm_job_id": it.get("slurm_job_id", ""),
            "job_name": it.get("job_name", ""),
            "status": it.get("status", "UNKNOWN"),
        }
        for it in resp.get("Items", [])
    ]
    return {"jobs": jobs}


_DISPATCH: dict[str, Any] = {
    "submit_slurm_job": _submit_slurm_job,
    "check_slurm_job_status": _check_slurm_job_status,
    "cancel_slurm_job": _cancel_slurm_job,
    "list_slurm_jobs": _list_slurm_jobs,
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
    raw_tool = (
        context.client_context.custom.get("bedrockAgentCoreToolName", "")
        if getattr(context, "client_context", None)
        else ""
    )
    tool_name = raw_tool.split("___", 1)[1] if "___" in raw_tool else raw_tool
    arguments = event or {}

    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}

    try:
        result = fn(arguments)
        return {"content": [{"type": "text", "text": json.dumps(result)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
