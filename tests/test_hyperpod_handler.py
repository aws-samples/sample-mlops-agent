"""R8 — HyperPod skill handler tests.

Covers:
  - Dispatch: Gateway prefix stripping (`hyperpod-skill___list_nodes` etc.).
  - list_nodes: describe_cluster + paginated list_cluster_nodes shape.
  - check_versions: validates categories, falls back to list_nodes when
    instance_ids omitted, parses stdout JSON.
  - Rate limiter: enforces ≥ 333 ms gap between consecutive SSM calls.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch


def _ensure_stub(name: str) -> None:
    if name in sys.modules:
        return
    mod = ModuleType(name)
    mod.__getattr__ = lambda attr, _m=mod: MagicMock(name=f"{name}.{attr}")  # type: ignore[attr-defined]
    sys.modules[name] = mod


def _import_hyperpod():
    """Fresh import of the hyperpod handler."""
    lam_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "lambda", "skills", "hyperpod")
    )
    for p in list(sys.path):
        if "/lambda/skills/" in p:
            sys.path.remove(p)
    sys.path.insert(0, lam_dir)
    sys.modules.pop("handler", None)
    return importlib.import_module("handler")


def _ctx(tool_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        client_context=SimpleNamespace(custom={"bedrockAgentCoreToolName": tool_name})
    )


# ── list_nodes ────────────────────────────────────────────────────────────


def test_list_nodes_returns_cluster_metadata_and_paginated_nodes():
    h = _import_hyperpod()
    sm = MagicMock()
    sm.describe_cluster.return_value = {
        "ClusterArn": "arn:aws:sagemaker:us-east-1:1:cluster/xyz",
    }
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"ClusterNodeSummaries": [
            {"InstanceId": "i-aaa", "InstanceGroupName": "worker",
             "InstanceType": "ml.p5.48xlarge",
             "InstanceStatus": {"Status": "Running"}},
        ]},
        {"ClusterNodeSummaries": [
            {"InstanceId": "i-bbb", "InstanceGroupName": "worker",
             "InstanceType": "ml.p5.48xlarge",
             "InstanceStatus": {"Status": "Running"}},
        ]},
    ]
    sm.get_paginator.return_value = paginator

    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.return_value = sm
        out = h._list_nodes({"cluster_name": "my-cluster"})

    assert out["cluster_id"] == "xyz"
    assert out["total"] == 2
    assert [n["node_id"] for n in out["nodes"]] == ["i-aaa", "i-bbb"]


def test_handler_dispatches_list_nodes():
    h = _import_hyperpod()
    h._DISPATCH = {"list_nodes": lambda args: {"total": 0, "nodes": []}}
    resp = h.handler(
        {"cluster_name": "c", "_user_id": "u"},
        _ctx("hyperpod-skill___list_nodes"),
    )
    assert resp.get("isError") is not True
    assert "total" in resp["content"][0]["text"]


def test_handler_rejects_unknown_tool():
    h = _import_hyperpod()
    resp = h.handler({}, _ctx("hyperpod-skill___does_not_exist"))
    assert resp["isError"] is True
    assert "does_not_exist" in resp["content"][0]["text"]


# ── check_versions ────────────────────────────────────────────────────────


def test_check_versions_rejects_invalid_category():
    h = _import_hyperpod()
    import pytest  # noqa: PLC0415
    with pytest.raises(ValueError) as exc:
        h._check_versions({
            "cluster_name": "c",
            "categories": ["cuda", "not_a_thing"],
        })
    assert "not_a_thing" in str(exc.value)


def test_check_versions_empty_cluster_returns_empty_results():
    """When instance_ids omitted and list_nodes finds no nodes, return
    an empty-but-structured response rather than hitting SSM at all."""
    h = _import_hyperpod()
    with patch.object(h, "_list_nodes", return_value={"nodes": []}):
        out = h._check_versions({"cluster_name": "c"})
    assert out["results"] == {}
    assert "No InService nodes" in out["note"]


def test_check_versions_parses_stdout_json_per_node():
    h = _import_hyperpod()
    # Pretend the bundled script was loaded.
    with patch.object(h, "_load_version_check_script", return_value="#!/bin/sh\n"):
        ssm = MagicMock()
        # send_command returns a CommandId; get_command_invocation
        # returns Success with a JSON stdout.
        ssm.send_command.return_value = {"Command": {"CommandId": "c1"}}
        ssm.get_command_invocation.return_value = {
            "Status": "Success",
            "StandardOutputContent": json.dumps({
                "cuda_driver": "550.90.07", "nccl": "2.19.3",
            }),
        }
        with patch.object(h, "boto3") as mock_boto:
            mock_boto.client.return_value = ssm
            out = h._check_versions({
                "cluster_name": "c",
                "instance_ids": ["i-aaa"],
                "categories": ["cuda", "nccl"],
            })
    assert out["nodes_succeeded"] == 1
    assert out["nodes_failed"] == 0
    assert out["results"]["i-aaa"]["cuda_driver"] == "550.90.07"


def test_check_versions_collects_per_node_failures():
    """One node returning non-Success must not nuke the audit."""
    h = _import_hyperpod()
    with patch.object(h, "_load_version_check_script", return_value="#!/bin/sh\n"):
        ssm = MagicMock()
        ssm.send_command.return_value = {"Command": {"CommandId": "cc"}}
        ssm.get_command_invocation.side_effect = [
            {"Status": "Success",
             "StandardOutputContent": '{"cuda_driver":"550.0"}'},
            {"Status": "Failed",
             "StandardErrorContent": "agent offline"},
        ]
        with patch.object(h, "boto3") as mock_boto:
            mock_boto.client.return_value = ssm
            out = h._check_versions({
                "cluster_name": "c",
                "instance_ids": ["i-good", "i-bad"],
            })
    assert out["nodes_succeeded"] == 1
    assert out["nodes_failed"] == 1
    assert out["failed"][0]["node_id"] == "i-bad"
    assert "Failed" in out["failed"][0]["reason"]


# ── rate limiter ──────────────────────────────────────────────────────────


def test_rate_limit_acquire_enforces_min_gap():
    """Three consecutive calls should take ≥ 2 × 333 ms total."""
    h = _import_hyperpod()
    # Reset the module-level last-call timestamp so the first call is
    # free (no sleep).
    h._last_call_monotonic[0] = 0.0
    start = time.monotonic()
    for _ in range(3):
        h._rate_limit_acquire()
    elapsed = time.monotonic() - start
    # 1/3.0 = 0.333 s min gap; two gaps across three calls → ≥ ~0.65 s.
    # Generous lower bound to avoid flakes on CI.
    assert elapsed >= 0.60, f"expected ≥ 0.60s across 3 calls, got {elapsed:.3f}s"
