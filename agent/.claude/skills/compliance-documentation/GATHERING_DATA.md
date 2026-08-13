# Gathering Data

Before you can write findings, you need the raw evaluation evidence. Two
MCP tools give it to you.

## `mlflow-skill___query_metrics`

**Call:**

```
query_metrics(run_id=<run_id from resume message>, _user_id=CURRENT_USER_ID)
```

**Response shape:**

```json
{
  "run_id": "abc123...",
  "metrics": {
    "Safety/mean": 0.94,
    "Safety/count": 50,
    "Correctness/mean": 0.71,
    "Correctness/count": 50,
    "RelevanceToQuery/mean": 0.88,
    ...
  }
}
```

**What to extract:**

- For every scorer the eval used, pull the `<Scorer>/mean` value. That's
  the aggregate pass signal per scorer.
- `<Scorer>/count` tells you how many rows the scorer ran on. If this is
  lower than the dataset size, rows were skipped — call that out.
- Ignore `<Scorer>/sum` / `<Scorer>/variance` unless you need to explain
  an outlier.

If `metrics` is empty or missing a scorer the run should have produced,
treat it as `UNKNOWN` in findings — do NOT fabricate a number.

## `mlflow-skill___analyze_trace`

**Call:**

```
analyze_trace(run_id=<same run_id>, _user_id=CURRENT_USER_ID)
```

**Response shape:**

```json
{
  "params": {
    "target_model": "bedrock:/global.amazon.nova-2-lite-v1:0",
    "judge_model": "bedrock:/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "dataset_repo_id": "PatronusAI/financebench",
    "dataset_split": "train[:50]",
    "scorers": "[\"Safety\",\"Correctness\",\"RelevanceToQuery\"]",
    "task_type": "qa"
  },
  "metrics": { ... same as query_metrics ... },
  "tags": {
    "thread_id": "tsk_...",
    "job_id": "sample-mlops-agent-eval-171...",
    "user_id": "...",
    "kind": "eval"
  },
  "status": "FINISHED"
}
```

**What to extract:**

- `params.target_model` and `params.judge_model` — name them explicitly
  in the Methodology section.
- `params.dataset_repo_id` + `params.dataset_split` — the evaluation
  corpus. Quote exactly; don't paraphrase "financebench" into "finance
  benchmark".
- `params.scorers` — JSON string; parse it and use the list to build one
  Finding per entry.
- `tags.job_id` — fallback for the `generate_compliance_report` `job_id`
  parameter if the runtime didn't inject it.
- `status` — if not `FINISHED`, the run was partial; flag in Exec Summary.

## Missing data

Any of the following are legitimate reasons to mark a finding `UNKNOWN`
rather than guess:

- Scorer listed in `params.scorers` but no `<Scorer>/mean` in `metrics`
- `metrics` dict empty (run crashed before logging)
- `status` != `FINISHED`

State the gap plainly in the finding body. Example:

> Correctness — UNKNOWN. The scorer was configured but MLflow returned
> no `Correctness/mean`; likely the judge model call failed for this
> batch. Re-run with judge_model verified before drawing conclusions.
