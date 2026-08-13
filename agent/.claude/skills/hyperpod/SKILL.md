# HyperPod Skill

> See `agent/.claude/skills/planning/SKILL.md` for cross-skill rules (EULA, one-decision-per-turn, routing, confirmation gate).

Read-only audit of SageMaker HyperPod clusters. Two tools: enumerate nodes,
and collect installed software versions via SSM. No cluster mutation. No
interactive commands.

## Confirmation Rule (Read-Only Skill)

Both tools are read-only — they describe cluster state and pull
version strings from nodes via AWS-RunShellScript. No confirmation is
required before invoking them.

| Tool                              | Kind      |
| --------------------------------- | --------- |
| `hyperpod-skill___list_nodes`     | read-only |
| `hyperpod-skill___check_versions` | read-only |

Do NOT request interactive shell commands from the agent — SSM
`AWS-RunShellScript` forbids TTY, and the bundled version-check script
is the only command the skill sends. If the user asks "ssh into node X
and run top", refuse and explain.

## List Nodes

Use MCP tool: `hyperpod-skill___list_nodes`

Enumerates the nodes in a HyperPod cluster, returning each node's
id, instance group, instance type, status, and launch time. Paginates
past 100 nodes automatically.

Parameters:

- `cluster_name` (string, required): HyperPod cluster name (not ARN).
- `_user_id` (string, required): Always pass `CURRENT_USER_ID`.

Returns: `{ "cluster_name": "...", "cluster_arn": "...", "cluster_id": "...", "total": N, "nodes": [...] }`

Typical use:

1. User says "audit my HyperPod cluster named foo".
2. Call `list_nodes(cluster_name="foo")` — read-only, no confirmation.
3. Surface node count, instance-group breakdown, and InService count.
4. Ask the user _one_ question (per one-decision-per-turn): full audit
   across every node, or just a sample?
5. Follow up with `check_versions` passing `instance_ids=[…]` for the
   chosen sample.

## Check Versions

Use MCP tool: `hyperpod-skill___check_versions`

Pushes a bundled shell script per node via SSM `AWS-RunShellScript`. The
script prints a JSON object with version strings for the standard HPC
stack. Rate-limited to 3 TPS (SSM account ceiling).

Parameters:

- `cluster_name` (string, required).
- `instance_ids` (list[string], optional): specific nodes to audit.
  Omit to audit every InService node. Use `list_nodes` to pick a
  sample on large clusters — `check_versions` on a 256-node cluster
  would be ~85 s wall clock minimum.
- `categories` (list[string], optional): subset of the 11 supported
  categories: `cuda`, `cudnn`, `nccl`, `efa`, `ofi-nccl`, `gdrcopy`,
  `mpi`, `neuron`, `python`, `pytorch`, `runtime`. Omit for all.
- `_user_id` (string, required).

Returns:

```json
{
  "cluster_name": "...",
  "categories":   ["cuda", "nccl", "pytorch", ...],
  "nodes_audited": 3,
  "nodes_succeeded": 3,
  "nodes_failed": 0,
  "results": {
    "i-0abcd1234": {"cuda_driver": "550.90.07", "cuda_toolkit": "12.4",
                    "nccl": "2.19.3", "pytorch": "2.4.1"},
    ...
  },
  "failed": []
}
```

### Failure handling

Per-node failures (SSM agent offline, timeout, malformed stdout) are
collected under `failed` rather than propagated. One flaky node never
nukes the whole audit. When reporting back, cite both successes and
failures so the user can see partial coverage.

### When to use

- User asks whether cluster nodes have matching CUDA / NCCL versions.
- Incident triage: "why are jobs failing?" → check toolchain drift.
- Change-management: "did the MNP rollout land on every node?"

### When NOT to use

- Real-time workload submission — use `slurm-skill` (currently mocked).
- Cluster creation / deletion — out of scope for v1.
- Running arbitrary commands — tool surface is deliberately narrow.
- EKS-backed HyperPod clusters — v1 targets Slurm-shape clusters only.
