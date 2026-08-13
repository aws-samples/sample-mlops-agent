# Compliance Documentation Skill

> See `agent/.claude/skills/planning/SKILL.md` for cross-skill rules (EULA, one-decision-per-turn, routing, confirmation gate).

## Confirmation Rule — DO NOT re-confirm before persisting

This skill has no MCP tools of its own; it ends by invoking
`mlflow-skill___generate_compliance_report`, which writes a compliance-report
artifact to S3 and logs it to MLflow.

**Do NOT ask for confirmation before persisting the report.** This skill only
runs when the report was already explicitly requested — either the user asked
for it in their task ("…then create a compliance report") or the EventBridge
resume message instructs it. In both cases, authoring _and saving_ the report
IS the requested action, not a separate surprise side effect. Asking "reply yes
to write it to S3" after the user already said "create a compliance report"
double-prompts them for something they already approved.

So: gather the data, decide pass/fail, author the narrative, and immediately
call `mlflow-skill___generate_compliance_report` to persist it — all in one
turn, no interstitial yes/no. After it writes, report the S3 location and the
one-line verdict. (This is a deliberate exception to the general state-changing
confirmation gate in `planning/SKILL.md`: the report's creation was the
user's explicit instruction, so the gate is already satisfied.)

Invoked after an LLM evaluation run completes. Your job is to read the
MLflow metrics + per-row eval results, author a short narrative compliance
report, and persist it by calling `mlflow-skill___generate_compliance_report`.

The tool handles rendering, S3 upload, and stamping the DynamoDB thread
row. You handle the judgment: what passed, what failed, what to recommend.

## When to Invoke

This skill fires when the EventBridge callback resumes the thread with a
message like:

> Evaluation job sample-mlops-agent-eval-1713... is now Completed. MLflow
> run URL: https://.../runs/abc123. Review the MLflow results and then
> invoke the compliance-documentation skill to generate the report.
> Please continue.

Do NOT invoke proactively on other turns — the user isn't waiting for it.

## Workflow

### 1. Gather the data

Use:

- `mlflow-skill___query_metrics` with the run_id from the resume message
  → aggregate scorer means (`Safety/mean`, `Correctness/mean`, etc.)
- `mlflow-skill___analyze_trace` with the same run_id → params (model,
  judge, dataset), tags (thread_id, job_id, user_id)

See `GATHERING_DATA.md` for the exact shape of these responses and which
fields to extract.

### 2. Decide pass/fail

Apply the thresholds in `WRITING_FINDINGS.md`. Each scorer the run used
becomes one Finding with `status: PASS|FAIL|PARTIAL|UNKNOWN`.

### 3. Author the narrative

Write `sections` (Executive Summary, Methodology, Results, Recommendations)
following `REPORT_STRUCTURE.md`. Keep it tight — compliance readers scan,
they don't read. One paragraph per section is usually right.

### 4. Persist

Call `mlflow-skill___generate_compliance_report` with:

- `thread_id`: `CURRENT_THREAD_ID`
- `job_id`: the eval `job_id` from the resume-message context (the
  runtime injects it; if absent, look it up from the thread row via the
  MLflow run tag `job_id`)
- `run_id`: the MLflow run_id you analysed
- `title`: e.g. `"Nova 2 Lite on PatronusAI/financebench — Compliance Report"`
- `sections`: `[{heading, body}, ...]` (from step 3)
- `findings`: `[{name, status, body}, ...]` (from step 2)
- `metrics`: optional; pass the scorer-mean dict from `query_metrics` to
  get the Metrics Summary table in the .docx
- Omit `appendix_failing_rows` — the Lambda auto-attaches the worst 10
  rows from `eval_results.parquet`

### 5. Reply to the user

Surface the returned `artifact_s3_uri` (docx) so the frontend download
button wires up. One-sentence summary of the report outcome — e.g.:

> Compliance report generated: 4 of 5 findings PASS (Safety FAIL). Download: s3://.../report.docx

## Subfiles

- `GATHERING_DATA.md` — exact shapes of the MLflow responses you'll read.
- `WRITING_FINDINGS.md` — scorer thresholds and status mapping.
- `REPORT_STRUCTURE.md` — section template and tone guidance.

## Rules

- **Never invent metrics.** Only report scores `query_metrics` actually
  returned. Missing = missing; say so rather than guessing.
- **Never skip `generate_compliance_report`.** The markdown you write in
  chat is ephemeral; the tool is the audit trail.
- **Don't ask the user to pick a title or sections.** This skill is
  invoked because the user already asked for an eval; confirmations were
  taken earlier. Just produce the report.
