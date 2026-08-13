"""HyperPod skill Lambda — Gateway MCP target.

Exposes 2 read-only tools:
  - list_nodes      : describe_cluster + list_cluster_nodes
  - check_versions  : fan out a bundled shell script via SSM AWS-RunShellScript

Design notes:
  * Rate-limit every SSM call to 3 TPS — that's the upstream-documented
    ceiling and matches ``awslabs/agent-plugins/plugins/sagemaker-ai/skills/hyperpod-ssm``.
  * No interactive commands. The bundled script is non-TTY and purely
    introspective (package queries, command version flags).
  * The version-check script is a single file baked into the image;
    handler reads it at cold-start and passes it as the
    ``AWS-RunShellScript`` ``commands`` parameter. No S3 upload round-trip.
  * Both tools are read-only. No confirmation gate applies — the
    agent/.claude/skills/hyperpod/SKILL.md docs mark them as such.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

import boto3


logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

# SSM per-account send_command ceiling per the upstream HyperPod SSM skill.
# The Lambda's RequestResponse invocation path runs one command at a time
# anyway, but a shared token bucket protects against concurrent Lambda
# invocations and keeps us well under the account limit.
_RATE_LIMIT_TPS = 3.0
_rate_lock = Lock()
_last_call_monotonic: list[float] = [0.0]  # mutable single-element list for closure binding


def _rate_limit_acquire() -> None:
    """Block up to 1.5 s to enforce a 3 TPS ceiling on SSM calls.

    The implementation is a minimal monotonic-gap enforcer rather than a
    full token bucket — we only care about the steady-state rate, not
    burst allowance. 1/3.0 s == ~333 ms minimum gap between consecutive
    calls.
    """
    min_gap = 1.0 / _RATE_LIMIT_TPS
    with _rate_lock:
        now = time.monotonic()
        gap = now - _last_call_monotonic[0]
        if gap < min_gap:
            wait = min_gap - gap
            # Cap the sleep at 1.5 s — beyond that something is wrong
            # (caller is spamming) and we'd rather fail loudly than hide.
            if wait > 1.5:
                raise RuntimeError(
                    f"Rate-limit wait {wait:.2f}s > 1.5s cap — too many SSM "
                    f"calls queued; reduce instance_ids or categories."
                )
            time.sleep(wait)
        _last_call_monotonic[0] = time.monotonic()


def _load_version_check_script() -> str:
    """Read the bundled script once at cold-start. Returns the script text.

    The Dockerfile copies it to /var/task/scripts/hyperpod_check_versions.sh.
    We read + cache in a module-level variable on first call.
    """
    global _VERSION_SCRIPT_CACHE  # noqa: PLW0603
    if _VERSION_SCRIPT_CACHE is not None:
        return _VERSION_SCRIPT_CACHE
    script_path = Path(__file__).parent / "scripts" / "hyperpod_check_versions.sh"
    if not script_path.exists():
        # Defensive: the Dockerfile copies this, but if it didn't the
        # error should point at the deploy pipeline, not the handler.
        raise RuntimeError(
            f"Bundled script missing: {script_path}. Check the "
            f"lambda/skills/hyperpod/Dockerfile COPY step."
        )
    _VERSION_SCRIPT_CACHE = script_path.read_text()
    return _VERSION_SCRIPT_CACHE


_VERSION_SCRIPT_CACHE: str | None = None


# ── list_nodes ────────────────────────────────────────────────────────────


def _list_nodes(args: dict) -> dict:
    """Enumerate HyperPod cluster nodes.

    Returns cluster_arn, cluster_id, and a flat list of node dicts with
    node_id, instance_group, instance_type, status. Paginates through
    ``list_cluster_nodes`` so clusters beyond 100 nodes are fully covered.
    """
    cluster_name = args["cluster_name"]
    sm = boto3.client("sagemaker", region_name=AWS_REGION)

    desc = sm.describe_cluster(ClusterName=cluster_name)
    cluster_arn = desc.get("ClusterArn", "")
    cluster_id = cluster_arn.split("/")[-1] if cluster_arn else ""

    nodes: list[dict[str, Any]] = []
    paginator = sm.get_paginator("list_cluster_nodes")
    for page in paginator.paginate(ClusterName=cluster_name):
        for item in page.get("ClusterNodeSummaries", []):
            nodes.append({
                "node_id":        item.get("InstanceId", ""),
                "instance_group": item.get("InstanceGroupName", ""),
                "instance_type":  item.get("InstanceType", ""),
                "status":         item.get("InstanceStatus", {}).get("Status", ""),
                "launch_time":    item.get("LaunchTime").isoformat() if item.get("LaunchTime") else "",
            })

    return {
        "cluster_name": cluster_name,
        "cluster_arn":  cluster_arn,
        "cluster_id":   cluster_id,
        "total":        len(nodes),
        "nodes":        nodes,
    }


# ── check_versions ────────────────────────────────────────────────────────


_VALID_CATEGORIES = {
    "cuda", "cudnn", "nccl", "efa", "ofi-nccl", "gdrcopy",
    "mpi", "neuron", "python", "pytorch", "runtime",
}

_PER_NODE_TIMEOUT_SEC = 60
# Upstream uses the SSM target format ``sagemaker-cluster:<CLUSTER_ID>_<GROUP>-<INSTANCE_ID>``.
# That mapping is not free (requires querying describe_cluster + instance
# group from each node); for v1 we use InstanceIds directly and rely on
# the HyperPod cluster SSM association that publishes each node as a
# regular managed instance. Upgrade path documented in the plan.


def _check_versions(args: dict) -> dict:
    """Fan out the bundled version-check script across cluster nodes.

    Behaviour:
      1. Resolve target node list — explicit ``instance_ids`` if passed,
         otherwise every InService node from ``list_cluster_nodes``.
      2. Filter categories — accept only values from _VALID_CATEGORIES.
      3. Push the script per node via SSM AWS-RunShellScript with a 60 s
         per-node timeout, respecting the 3 TPS rate limit.
      4. Poll ``get_command_invocation`` until terminal.
      5. Parse each node's stdout as JSON into the response table.

    Failures on individual nodes are collected under ``failed`` — one
    flaky node never nukes the whole audit.
    """
    cluster_name = args["cluster_name"]
    instance_ids = args.get("instance_ids") or []
    categories = args.get("categories") or []

    # Validate category filter early so we fail before touching AWS.
    bad_cats = [c for c in categories if c not in _VALID_CATEGORIES]
    if bad_cats:
        raise ValueError(
            f"Invalid categories: {bad_cats}. Supported: "
            f"{sorted(_VALID_CATEGORIES)}."
        )

    # Resolve default node list if caller omitted instance_ids.
    if not instance_ids:
        listing = _list_nodes({"cluster_name": cluster_name})
        instance_ids = [
            n["node_id"] for n in listing["nodes"]
            if n["status"] == "Running" and n["node_id"]
        ]
        if not instance_ids:
            return {
                "cluster_name": cluster_name,
                "results":      {},
                "failed":       [],
                "note":         "No InService nodes found — nothing to audit.",
            }

    script = _load_version_check_script()
    # Pass CATEGORIES env var to the script so it only prints the subset.
    # Empty == all.
    categories_env = ",".join(categories)
    command_text = f"export CATEGORIES={categories_env!r}\n{script}"

    ssm = boto3.client("ssm", region_name=AWS_REGION)
    results: dict[str, dict[str, str]] = {}
    failed: list[dict[str, str]] = []

    for node_id in instance_ids:
        _rate_limit_acquire()
        try:
            send = ssm.send_command(
                InstanceIds=[node_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": [command_text]},
                TimeoutSeconds=_PER_NODE_TIMEOUT_SEC,
            )
        except Exception as exc:  # noqa: BLE001
            failed.append({"node_id": node_id, "reason": f"send_command: {exc}"})
            continue

        command_id = send["Command"]["CommandId"]
        # Poll until terminal. SSM latency is usually 1-3 s for short
        # commands; we back off gently starting at 1 s.
        deadline = time.time() + _PER_NODE_TIMEOUT_SEC + 10
        per_node: dict[str, Any] | None = None
        while time.time() < deadline:
            _rate_limit_acquire()
            try:
                per_node = ssm.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=node_id,
                )
            except ssm.exceptions.InvocationDoesNotExist:
                time.sleep(1)  # nosemgrep: arbitrary-sleep — SSM invocation poll backoff
                continue
            if per_node["Status"] in ("Success", "Cancelled", "TimedOut", "Failed"):
                break
            time.sleep(1)  # nosemgrep: arbitrary-sleep — SSM invocation poll backoff

        if not per_node or per_node["Status"] != "Success":
            failed.append({
                "node_id": node_id,
                "reason":  f"command status={per_node['Status'] if per_node else 'TIMEOUT'}",
                "stderr":  (per_node or {}).get("StandardErrorContent", "")[:500],
            })
            continue

        stdout = per_node.get("StandardOutputContent", "").strip()
        try:
            results[node_id] = json.loads(stdout)
        except Exception as exc:  # noqa: BLE001
            failed.append({
                "node_id": node_id,
                "reason":  f"non-JSON stdout: {exc}",
                "stdout":  stdout[:500],
            })

    return {
        "cluster_name":      cluster_name,
        "categories":        categories or sorted(_VALID_CATEGORIES),
        "nodes_audited":     len(instance_ids),
        "nodes_succeeded":   len(results),
        "nodes_failed":      len(failed),
        "results":           results,
        "failed":            failed,
    }


# ── dispatcher ────────────────────────────────────────────────────────────


_DISPATCH: dict[str, Any] = {
    "list_nodes":     _list_nodes,
    "check_versions": _check_versions,
}


def handler(event: dict, context: Any) -> dict:
    """Gateway MCP tool dispatcher. Mirrors the sagemaker skill handler."""
    raw_tool = (
        context.client_context.custom.get("bedrockAgentCoreToolName", "")
        if getattr(context, "client_context", None) else ""
    )
    tool_name = raw_tool.split("___", 1)[1] if "___" in raw_tool else raw_tool
    arguments = event or {}

    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}

    try:
        result = fn(arguments)
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}
    except Exception as e:
        logger.exception("[hyperpod handler] tool=%s raised", tool_name)
        return {"content": [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}], "isError": True}
