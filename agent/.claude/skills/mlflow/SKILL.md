# MLflow Skill

> See `agent/.claude/skills/planning/SKILL.md` for cross-skill rules (EULA, one-decision-per-turn, routing, confirmation gate).

## Confirmation Required Before State-Changing Actions

Before calling any tool marked **state-changing** in the table below, summarize
the resolved parameters back to the user, ask for explicit confirmation, and
only invoke the tool after an affirmative reply. Read-only tools may be invoked
without confirmation.

| Tool                                        | Kind                      |
| ------------------------------------------- | ------------------------- |
| `mlflow-skill___list_scorers`               | read-only                 |
| `mlflow-skill___query_metrics`              | read-only                 |
| `mlflow-skill___retrieve_traces`            | read-only                 |
| `mlflow-skill___analyze_trace`              | read-only                 |
| `mlflow-skill___generate_compliance_report` | state-changing (see note) |

> **Note on `generate_compliance_report`:** do NOT ask for a separate
> confirmation before this call. It only runs when a compliance report was
> already explicitly requested (in the user's task or the EventBridge resume
> message), so authoring + persisting the report IS the approved action. See
> the "Confirmation Rule — DO NOT re-confirm" section in
> `compliance-documentation/SKILL.md`.

All MLflow operations are performed via MCP tools on the `mlops-gateway` server
(target name `mlflow-skill`). The tracking server is a SageMaker MLflow App;
the Lambda resolves the ARN and signs SigV4 automatically — no local client,
no tokens, no CLI. Do NOT run `pip install mlflow`, do NOT invoke the `mlflow`
CLI, do NOT construct `MlflowClient()` from Bash.

## Tool Surface

- `mlflow-skill___list_scorers` — live catalog of MLflow judge scorers.
- `mlflow-skill___query_metrics` — metrics for one run.
- `mlflow-skill___retrieve_traces` — recent runs for an experiment.
- `mlflow-skill___analyze_trace` — full params/metrics/tags/status for one run.
- `mlflow-skill___generate_compliance_report` — deterministic report renderer
  (details in `agent/.claude/skills/compliance-documentation/SKILL.md`).

## List Scorers

Use MCP tool: `mlflow-skill___list_scorers`

Returns the live catalog of MLflow built-in judge scorers by introspecting
`mlflow.genai.scorers` at runtime. No precomputed alias map — the agent is
responsible for the semantic translation between user vocabulary and the
canonical scorer names.

### When you MUST call this

Before calling `sagemaker-skill___submit_eval_job` whenever the user
describes scorers in anything other than the canonical capitalised form
(e.g. "faithfulness", "answer relevance", "correctness", "safety"). The
SageMaker eval container accepts canonical names only and will raise
`ValueError` with the full valid-names list on unknown inputs — burning
a Processing instance. `list_scorers` is cheap; call it.

### Parameters

None.

### Returns

```json
{
  "scorers": [
    { "name": "Correctness",            "doc": "Judges whether the response is correct given the expected answer." },
    { "name": "RelevanceToQuery",       "doc": "Judges whether the response is relevant to the user's query." },
    { "name": "RetrievalGroundedness",  "doc": "Judges whether the response is grounded in the retrieved context." },
    { "name": "RetrievalRelevance",     "doc": "Judges whether the retrieved context is relevant to the query." },
    { "name": "Safety",                 "doc": "Judges whether the response is safe / avoids harmful content." },
    ...
  ]
}
```

### Agent-side matching protocol

1. Build a candidate list of canonical scorer names by walking the returned
   `scorers` array and reading each `doc` line.
2. For every user-supplied scorer term, pick the canonical name whose
   docstring best matches the term semantically. Do NOT assume a mapping.
   Common industry usage that you should recognise from the docstrings:
   RAG "faithfulness" talks about grounding in retrieved context — for
   **static datasets** (anything authored via `prepare_eval_dataset`,
   i.e. every starter-tile eval) map it to `Guidelines`, which the eval
   container runs as `answer_groundedness` against the dataset's context
   column. `RetrievalGroundedness` is trace-based (needs live RETRIEVER
   spans from a real retriever app) and returns no metric on static
   datasets. "Answer relevance" talks about the answer vs the user's
   query → `RelevanceToQuery`.
3. If two scorers look like plausible matches (or none does), stop and
   ask the user to disambiguate. Offer the top 2–3 candidates by name +
   docstring, let the user pick. Never guess on a billable path.
4. Include the resolved `user term → canonical name` mapping in the
   confirmation summary you show before `submit_eval_job` so the user can
   catch a wrong match before it runs.

## Custom Scorers (R6)

`submit_eval_job` accepts a `custom_scorer_lambda_arns: string[]` argument
for deterministic scorers that MLflow's built-in catalog doesn't cover —
math correctness, code execution, regex match, JSON-schema validation,
etc. Each ARN is a standard AWS Lambda ARN; the eval Processing
container invokes each one per evaluation row with a JSON payload and
logs the returned score to the MLflow run alongside the built-in scorer
metrics.

### Contract

**Input** — the Lambda receives an event shaped exactly as:

```json
{"inputs": {...}, "outputs": {...}, "expectations": {...}}
```

where each sub-object carries column values for a single row. `inputs`
maps columns declared as the eval's inputs (e.g. `question`); `outputs`
is whatever the target model returned (single string); `expectations` is
the ground-truth set (e.g. `answer`).

**Output** — the Lambda MUST return (as its JSON response body):

```json
{ "score": 1.0, "reason": "optional free text" }
```

- `score` must be numeric (int or float).
- The MLflow metric key is ALWAYS the Lambda's function name, derived
  from the ARN (any `:alias`/`:version` suffix stripped). A `name` field
  in the response body is ignored — name the Lambda what you want the
  metric called.
- `reason` is logged to the eval Processing job's CloudWatch logs; it is
  NOT surfaced on the per-row result_df. May be omitted.

### Constraints

- **Per-row 5 s timeout.** The eval container configures a 5 s boto3
  read timeout. Slower scorers must be re-implemented as LLM-judges.
- **Errors surface as null metrics.** Non-200, timeout, malformed
  response, non-numeric score → the row's scorer value is `null` and
  the evaluation continues. Check CloudWatch logs for the eval
  Processing job to see rejection reasons.
- **Naming.** Scorer Lambdas must be named `<anything>-scorer-<suffix>`
  to match the IAM allowlist on the SageMaker execution role
  (`science-agent-stack.ts` sid `CustomScorerInvoke`).

### Reference scorer

A deployed, copyable example ships with the gateway stack: the
`<project>-math-scorer-ref` Lambda (sympy-based symbolic-equivalence
check, source at
`lambda/skills/sagemaker/eval/reference_scorers/math_correctness/`).
When the user asks for math-correctness scoring, pass that Lambda's ARN
in `custom_scorer_lambda_arns` — do not reimplement it.

### When to use

- Deterministic ground-truth matching (exact-match, regex, JSON-schema).
- Execution-based scoring (compile-and-run code, assert sympy equality).
- Domain-specific metrics no judge can reason about reliably (unit-test
  pass rate, API contract conformance).

Do NOT use for subjective quality (use MLflow built-in `Correctness`,
`RelevanceToQuery`, `Safety`) or for anything requiring a call-out to an
LLM (wrap those as MLflow built-ins with a bespoke `Guidelines`
scorer instead).

## Query Metrics

Use MCP tool: `mlflow-skill___query_metrics`

Returns the aggregated metrics dict for a single MLflow run. For eval runs,
the per-scorer means land as keys like `Correctness/mean`,
`RetrievalGroundedness/mean`. Null = null — if `query_metrics` doesn't return
a metric, don't invent a value. The `compliance-documentation` skill's "Rules"
section is the authoritative guide.

Parameters:

- `run_id` (string): MLflow run ID.
- `_user_id` (string): always pass `CURRENT_USER_ID`.

Returns: `{ "run_id": "...", "metrics": { "Correctness/mean": 0.83, ... } }`

## Retrieve Traces

Use MCP tool: `mlflow-skill___retrieve_traces`

Lists recent MLflow runs in an experiment with status and a metrics summary.
Experiments in this project are named `<project>/<thread_id>` — one
experiment per AgentCore session. Pass the experiment name the agent's
environment injected, not a guess.

Parameters:

- `experiment_name` (string): MLflow experiment name, usually
  `sample-mlops-agent/<CURRENT_THREAD_ID>`.
- `max_results` (int, optional, default 10).
- `_user_id` (string): always pass `CURRENT_USER_ID`.

Returns:
`{ "runs": [{ "run_id": "...", "status": "...", "metrics": {...} }, ...] }`

Use this when the user asks "what ran in this session?" or when you need to
find a `run_id` from context (e.g. "show me metrics for the last eval").

## Analyze Trace

Use MCP tool: `mlflow-skill___analyze_trace`

Returns the full `params / metrics / tags / status` payload for one run.
Useful when building a compliance report — `tags` carries `thread_id`,
`job_id`, `user_id`, and `training_type` / `task`.

Parameters:

- `run_id` (string): MLflow run ID.
- `_user_id` (string): always pass `CURRENT_USER_ID`.

Returns:
`{ "run_id": "...", "params": {...}, "metrics": {...}, "tags": {...}, "status": "..." }`

## Generate Compliance Report

Use MCP tool: `mlflow-skill___generate_compliance_report`

Renders a compliance report for a completed eval run. **The agent authors
the narrative** (title, sections, findings) under the
`compliance-documentation` skill; this tool is the deterministic
render/persist step. It writes `report.md` + `report.docx` to S3 and stamps
the thread row's eval job entry with `compliance_report_s3_uri`,
`compliance_report_format`, and `compliance_report_at` so the frontend
download button can surface the artifact.

Parameters:

- `thread_id` (string): always `CURRENT_THREAD_ID`.
- `job_id` (string): eval `job_id` returned by `submit_eval_job`.
- `run_id` (string): MLflow run ID the report describes.
- `sections` (list[{heading, body}]): ordered narrative sections.
- `findings` (list[{name, status, body}], optional): PASS/FAIL/PARTIAL/UNKNOWN
  — the .docx colors them green/red/grey.
- `metrics` (dict, optional): metric name → value; rendered as a
  two-column Metrics Summary table.
- `title` (string, optional; default "LLM Evaluation Compliance Report").
- `appendix_failing_rows` (list[dict], optional): override the auto
  worst-10 appendix. Omit in the normal case — the Lambda pulls
  `eval_results.parquet` from the MLflow run and picks the 10 rows with
  the lowest mean scorer score.
- `formats` (list[string], optional): subset of `["md","docx"]`; default both.
- `out_s3_prefix` (string, optional).

Returns:
`{ "artifact_s3_uri": "s3://.../report.docx", "artifact_format": "docx", "artifacts": {"md": "s3://...", "docx": "s3://..."}, "appendix_rows": 10, ... }`

Surface `artifact_s3_uri` to the user — the frontend renders a download
button off the same value.
