# Slurm Skill

> See `agent/.claude/skills/planning/SKILL.md` for cross-skill rules (EULA, one-decision-per-turn, routing, confirmation gate).

## Confirmation Required Before State-Changing Actions

Before calling any tool marked **state-changing** in the table below, summarize
the resolved parameters back to the user, ask for explicit confirmation, and
only invoke the tool after an affirmative reply. Read-only tools may be invoked
without confirmation. (MOCK_MODE=1 does not relax this rule — the real-mode
flip must be transparent.)

| Tool                                   | Kind           |
| -------------------------------------- | -------------- |
| `slurm-skill___submit_slurm_job`       | state-changing |
| `slurm-skill___check_slurm_job_status` | read-only      |
| `slurm-skill___cancel_slurm_job`       | state-changing |
| `slurm-skill___list_slurm_jobs`        | read-only      |

All Slurm operations are performed via MCP tools on the `mlops-gateway` server
(target name `slurm-skill`). Never SSH to the head node or run `sbatch` /
`sacct` / `scancel` from Bash directly — always use the MCP tools below.

The skill currently runs in `MOCK_MODE=1` until the pcluster, the SSH-key
secret, and a pinned head-node host key (`SLURM_KNOWN_HOSTS`, a provisioned
known_hosts file — the handler refuses to SSH without it) are deployed; the
tool surface is identical in either mode, and the mock simulator progresses
a submitted job `SUBMITTED → PENDING → RUNNING → COMPLETED` on each
`check_slurm_job_status` call.

## Confirmation Required Before Billable Actions

`submit_slurm_job` launches work on a real HPC cluster once `MOCK_MODE=0`.
Before calling it, you MUST:

1. Summarize the resolved parameters back to the user (`script_path`,
   `job_name`, `workflow_step`).
2. Ask for explicit confirmation (e.g. "Proceed? (yes/no)").
3. Only invoke the tool after an affirmative reply.

This mirrors the SageMaker skill rule — cost-incurring actions are not
implementation details, even when the cluster is mocked.

## Submit Slurm Job

Use MCP tool: `slurm-skill___submit_slurm_job`

Queues an `sbatch` invocation on the head node, records the task in the
shared `<project>-metadata` DynamoDB table, and returns the canonical
`task_id` you will use for every follow-up call.

### Pre-flight

- `script_path` must be an **absolute path on the head node** (e.g.
  `/home/ec2-user/jobs/train.sh`). The skill does not upload scripts —
  the sbatch file is expected to already exist on the cluster. If the user
  has not provided one, ask for it; do not invent a path.
- Confirm the job is intended to run on the cluster and not on SageMaker
  (`sagemaker-skill___submit_training_job` is the right tool for SageMaker
  training).

### Parameters

- `thread_id` (string): always pass the value of the `CURRENT_THREAD_ID`
  env var (this is the AgentCore session_id and the PK of the DynamoDB
  thread row).
- `script_path` (string): absolute path of the sbatch script on the head
  node.
- `job_name` (string, optional): Slurm job name. Defaults to
  `<project>-job-<unix-timestamp>`.
- `workflow_step` (string, optional): label for multi-step pipelines
  (e.g. `ingest`, `train`, `validate`). Defaults to `unknown`.
- `_user_id` (string): always pass the value of `CURRENT_USER_ID`. Stored
  on the DynamoDB row for attribution.

Returns:
`{ "task_id": "...", "slurm_job_id": "...", "job_name": "...", "status": "SUBMITTED" }`

Keep the returned `task_id` — it is the required key for every other
tool in this skill. The `slurm_job_id` is informational (what you would
see in `squeue` / `sacct` output).

**After a successful submit**, reply with both identifiers and the
initial status, e.g.:

> Submitted Slurm job `<job_name>` (slurm_job_id `<slurm_job_id>`,
> task_id `<task_id>`). Status: SUBMITTED.

## Check Slurm Job Status

Use MCP tool: `slurm-skill___check_slurm_job_status`

Polls `sacct` on the head node for the job's current state and updates the
DynamoDB row. Returns the canonical status and whether the job has reached
a terminal state (`COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`).

Parameters:

- `task_id` (string): the `task_id` returned by `submit_slurm_job`. Do NOT
  pass the raw `slurm_job_id` — the skill keys off the DynamoDB row.

Returns:
`{ "task_id": "...", "slurm_job_id": "...", "status": "PENDING|RUNNING|COMPLETED|...", "exit_code": "0:0", "elapsed": "00:30:00", "terminal": false }`

### Polling guidance

There is no EventBridge callback for Slurm jobs today (unlike SageMaker).
To avoid burning a turn on a long sleep, prefer **one check per turn**:

- After `submit_slurm_job`, immediately call `check_slurm_job_status` once
  so the user sees the job move past `SUBMITTED`.
- If the result is not `terminal`, tell the user the job is still running
  and instruct them to ask again (e.g. "check on task `<task_id>`") when
  they want an update. Do NOT loop on `check_slurm_job_status` within a
  single turn.
- If the result is `terminal`, surface `status`, `exit_code`, and
  `elapsed` in the reply.

## Cancel Slurm Job

Use MCP tool: `slurm-skill___cancel_slurm_job`

Cancels a running job via `scancel` and marks the DynamoDB row
`CANCELLED`.

Parameters:

- `task_id` (string): the `task_id` returned by `submit_slurm_job`.

Returns: `{ "task_id": "...", "slurm_job_id": "...", "status": "CANCELLED" }`

Only invoke on explicit user request ("cancel the job", "kill task
…"). Never cancel pre-emptively on a transient status-check error.

## List Slurm Jobs

Use MCP tool: `slurm-skill___list_slurm_jobs`

Returns every Slurm task recorded on the current thread, with the last
known status from DynamoDB. Useful when the user asks "what did I submit
this session?" or before canceling a specific job.

Parameters:

- `thread_id` (string): always pass `CURRENT_THREAD_ID`.

Returns:
`{ "jobs": [{ "task_id": "...", "slurm_job_id": "...", "job_name": "...", "status": "..." }, ...] }`

Note: `status` here is the cached DynamoDB value — it only advances when
someone calls `check_slurm_job_status` on that task. If the user wants
fresh state, follow the list with targeted status checks.

## Guidelines

- **Always pass `thread_id = CURRENT_THREAD_ID` and `_user_id =
CURRENT_USER_ID`** where the tool accepts them — the DynamoDB row uses
  them for attribution and per-thread listing.
- **Never probe the filesystem or shell environment** for `thread_id` or
  `session_id`; they are injected as env vars.
- **Do not mix Slurm and SageMaker abstractions**: a Slurm `task_id` is
  not a SageMaker `job_id`, and the two skills do not share storage
  beyond the fact that both write to the `<project>-metadata` table.
- **MLflow is not wired into this skill.** Slurm submissions do not
  pre-create MLflow runs; if the user wants metrics logged, they must
  instrument the sbatch script themselves.
