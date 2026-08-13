# SageMaker Skill

> See `agent/.claude/skills/planning/SKILL.md` for cross-skill rules (EULA, one-decision-per-turn, routing, confirmation gate).

All SageMaker operations are performed via MCP tools on the `mlops-gateway` server.
Never run Python scripts directly — always use the MCP tools below.

## Confirmation Required Before State-Changing Actions

Some MCP tools create AWS infrastructure, write to persistent storage, or produce
externally visible side effects. Before calling any tool marked **state-changing**
in the table below, you MUST:

1. **Run all read-only discovery FIRST, without confirmation.** Read-only tools
   (`retrieve_dataset_metadata`, `list_scorers`, `list_hub_models`, env-var
   reads via Bash, etc.) require NO confirmation — invoke them silently to
   resolve every parameter. Do this BEFORE presenting any summary.
2. Present the confirmation summary **exactly once**, built from the resolved
   parameters (model, dataset, instance type, training type, scorers, etc.).
3. Ask for explicit confirmation (e.g. "Proceed? (yes/no)") and wait.
4. Only invoke the state-changing tool after the user replies affirmatively.

**Confirm exactly once.** Do NOT ask for confirmation before you have run the
read-only discovery, and do NOT ask again after it. A common bug is to present a
plan, get a "yes", THEN run the EDA, THEN present a second "revised" plan and ask
"yes" again — that double-confirms the user for no reason. Gather everything you
need (read-only) up front, then gate on a single yes/no. The only reason to ask a
second time is if the discovery surfaced a genuinely different parameter than the
first summary claimed (e.g. the requested split does not exist).

This overrides the general "do not ask for confirmation on implementation details"
rule in `CLAUDE.md`. Cost-incurring or externally visible actions are not
implementation details. Read-only tools may be invoked without confirmation.

| Tool                                           | Kind           |
| ---------------------------------------------- | -------------- |
| `sagemaker-skill___submit_training_job`        | state-changing |
| `sagemaker-skill___complete_training_job`      | state-changing |
| `sagemaker-skill___deploy_model`               | state-changing |
| `sagemaker-skill___submit_eval_job`            | state-changing |
| `sagemaker-skill___submit_recommendation_job`  | state-changing |
| `sagemaker-skill___get_recommendation_results` | read-only      |
| `sagemaker-skill___list_hub_models`            | read-only      |
| `sagemaker-skill___submit_monitoring_job`      | state-changing |

## Discover Hub Models

When the user asks "what models are available?" or requests a fine-tune without
naming a specific `model_id`, call `list_hub_models` first. It's a read-only
tool — no confirmation required. It returns a list of models in a SageMaker
Hub (default: `SageMakerPublicHub`) with per-model EULA status, supported
recipes, and license.

Typical flow:

1. User says "I want to fine-tune a Llama model."
2. Call `list_hub_models(filter="Llama")`.
3. Present the top 5–10 results back to the user (name, license, EULA
   required yes/no). Per the one-decision-per-turn rule in
   `agent/.claude/skills/planning/SKILL.md`, ask the user to pick **one**
   before moving on. Do NOT also ask for instance type / steps / dataset in
   the same turn.
4. If the chosen model has `requires_eula=true`, apply the EULA hard-rule
   from `planning/SKILL.md` — surface the EULA terms, require explicit
   affirmative acceptance naming the model, and only then proceed.
5. Proceed to the normal pre-flight EDA for `submit_training_job`.

### Parameters

- `hub_name` (string, optional): Hub to enumerate. Default `SageMakerPublicHub`.
- `filter` (string, optional): Case-insensitive substring match against the
  model name. E.g. `"Llama"`, `"Nova"`, `"Qwen"`.
- `_user_id` (string, required): Always pass `CURRENT_USER_ID`.

### Returns

```json
{
  "hub_name": "SageMakerPublicHub",
  "total": 12,
  "models": [
    {
      "name": "meta-llama/Llama-3.1-8B-Instruct",
      "version": "1.0.0",
      "arn": "arn:aws:sagemaker:us-east-1:...:hub-content/...",
      "requires_eula": true,
      "eula_url": "https://llama.meta.com/llama3/license/",
      "supported_recipes": ["sft", "dpo"],
      "license": "Llama-3 Community License",
      "description": "..."
    }
  ]
}
```

Models without Hub tags declaring recipes/EULA return empty defaults
(`supported_recipes: []`, `requires_eula: false`, `eula_url: ""`). When in
doubt, ask the user to confirm licence compatibility with their use case.

## Submit Training Job

### Pre-flight: Dataset Analysis is REQUIRED

Before calling `submit_training_job`, you MUST perform exploratory data
analysis (EDA) on `dataset_name` and then **explicitly confirm the resolved
parameters back to the user** alongside the other training parameters. Do
this for every submission and every retry, even if you "remember" the
dataset from a previous turn. Silent split-name or schema failures waste
an instance-hour.

The EDA procedure differs by `training_type`. Pick the right path — do
NOT skip ahead to `submit_training_job` on the assumption that the
container will figure it out.

#### LLM training types — `grpo`, `sft`, `dpo`

These load data via HuggingFace `datasets` and expect text/conversation
columns.

1. **HuggingFace datasets** (`dataset_name` is a repo ID like
   `HuggingFaceH4/ultrachat_200k`): call
   `huggingface-skill___retrieve_dataset_metadata` to discover the actual
   split names and column list. The response's `available_splits` field
   lists every valid split — use it to fill `train_split` / `test_split`.
   No guess-and-retry required; a single metadata-only call (`max_rows=0`)
   is enough.
2. **S3 datasets** (`dataset_name` starts with `s3://`): inspect the
   prefix layout (e.g. `train/` vs `train_sft/` subdirectories, file
   format, header columns of a representative shard) so you can
   explicitly set `train_split` / `test_split` and verify the schema
   matches the chosen `training_type`.

#### Tabular training types — `xgboost`, `sklearn`

The container loads these via pandas (CSV) or `sklearn.datasets`; it
does NOT use the `train`/`test` split mechanism at all (ignore
`train_split`/`test_split` params — they're not read). Supported forms
of `dataset_name` for tabular training:

- `iris`, `breast_cancer`, `wine`, `digits` — bundled sklearn datasets
  loaded in-process, no network I/O. Default `target_column` = `"target"`.
- `s3://<bucket>/<key>.csv` — a single CSV object.
- `s3://<bucket>/<prefix>/` — directory-style prefix; every `*.csv`
  object beneath it is concatenated.
- `<hf-org>/<hf-repo>` — only if the dataset is already tabular-shaped
  (`.to_pandas()` on the train split must succeed). If the user asked
  for a random HF repo ID for a tabular job (e.g. `scikit-learn/iris`),
  translate it to the equivalent bundled name (`iris`) — it's faster
  and more reliable than pulling through HF.

Additional rules for tabular:

- `model_id` is a **slug for the SageMaker job name only** — the script
  doesn't load a model from it. Pass `"xgboost"` / `"sklearn"` verbatim.
- **`target_column` is mandatory** for S3-CSV and HuggingFace inputs.
  For bundled sklearn datasets it defaults to `"target"`. Resolve and
  confirm this column name before submit — the script raises immediately
  if it's missing or not a column in the loaded DataFrame.
- EDA for S3 inputs: `head` the first few rows of one CSV shard to
  confirm the column list includes `target_column` and that feature
  columns are numeric (XGBoost doesn't handle raw string features).
- EDA for bundled sklearn: state the dataset's known shape (e.g.
  "`iris`: 150 rows, 4 features, 3-class target") in the confirmation
  summary instead of calling the metadata tool.
- Hyperparameters: `max_depth`, `n_estimators` (mapped to `num_round`),
  and `learning_rate` (mapped to xgboost's `eta`) are the three the
  agent should reason about. The script picks a sensible `objective`
  based on class count if the user didn't specify one.

#### Confirmation summary

Fold the findings into the confirmation summary required by the
"Confirmation Required Before Billable Actions" section above — the user
should see:

- For LLM: the resolved `train_split`, `test_split`, and column list.
- For tabular: the resolved `dataset_name` form, `target_column`, row
  count, feature count, and class count.

…before saying "yes".

### Parameters

- `thread_id` (string): always pass the value of the `CURRENT_THREAD_ID` env var
  (this is the AgentCore session_id and the PK of the DynamoDB thread row)
- `model_id` (string): HuggingFace model ID (LLM types), or `"xgboost"` / `"sklearn"`
  slug for tabular types — the slug only shows up in the SageMaker job name.
- `dataset_name` (string): dataset ID, S3 URI, or bundled sklearn name — see the
  per-training-type guidance above for which forms each path accepts.
- `training_type` (string): `grpo` | `sft` | `dpo` | `xgboost` | `sklearn` (default: grpo)
- `instance_type` (string): SageMaker instance, e.g. `ml.g5.2xlarge`
- `max_steps` (int): training steps (default: 500). LLM types only; xgboost/sklearn
  ignore this in favour of `n_estimators`/`num_round`.
- `max_samples` (int, LLM only): cap the training dataset to the first N rows
  (default: 0 = no cap). Use for quick demo runs — e.g. the fine-tune starter
  tile passes 2000. For `grpo`, the eval set is additionally capped to
  `max_samples // 10`.
- `train_split` (string, LLM only): split name for the training set, resolved from
  the mandatory pre-flight EDA. Do NOT rely on the `"train"` default; always pass
  the split you observed (e.g. `train_sft` for `HuggingFaceH4/ultrachat_200k`).
- `test_split` (string, LLM only): split name for the eval set, resolved from the
  mandatory pre-flight EDA. Do NOT rely on the `"test"` default; always pass the
  split you observed (e.g. `test_sft` for `HuggingFaceH4/ultrachat_200k`).
- `target_column` (string, tabular only): name of the label column. Required for
  S3-CSV and HuggingFace inputs; defaults to `"target"` for bundled sklearn
  datasets. Resolve via EDA and confirm before submit.
- `max_depth` (int, tabular only): XGBoost tree depth (default 6).
- `n_estimators` (int, tabular only): number of boosting rounds (default 100).
- `learning_rate` (float): training step size. Maps to `eta` for xgboost.
- `_user_id` (string): always pass the value of `CURRENT_USER_ID` env var

Returns: `{ "thread_id": "...", "job_id": "...", "sagemaker_job_name": "...", "mlflow_run_id": "...", "mlflow_run_url": "..." }`

Each submission:

- Creates a fresh `job_id` under the thread row's `jobs` map.
- Pre-creates an MLflow run in RUNNING state (the training container resumes
  it via `MLFLOW_RUN_ID`; HF's `MLflowCallback` will log metrics into that
  run instead of creating a new one).

**After a successful submit you MUST surface `mlflow_run_url` to the user** —
this is the live MLflow dashboard link for the run. Example reply format:

> Submitted SageMaker job `<sagemaker_job_name>`.
> Follow training metrics in MLflow: `<mlflow_run_url>`

Retries within a thread are additional submissions on the same `thread_id`;
prior entries are preserved in the `jobs` map.

**Do NOT** probe the filesystem or shell environment for `session_id` /
`thread_id` — it is already injected as the `CURRENT_THREAD_ID` env var.

## Complete Training Job

Use MCP tool: `sagemaker-skill___complete_training_job`

Parameters:

- `sagemaker_job_name` (string): job name from submit response
- `_user_id` (string): always pass CURRENT_USER_ID

Returns: `{ "status": "Completed|Failed|...", "artifact_s3": "s3://..." }`

## Deploy Model

Use MCP tool: `sagemaker-skill___deploy_model`

Two deploy targets via the `target` parameter:

- `target="sagemaker"` (default): create a SageMaker real-time endpoint.
  Use for low-latency real-time inference where you want fine-grained
  instance-type control.
- `target="bedrock"`: import the training job's artifact into Bedrock via
  Custom Model Import. Use for managed serverless inference and when you
  want the model to be available via the Bedrock Converse API.

### Parameters (common)

- `sagemaker_job_name` (string, required): Completed training job name. The
  artifact is read from `ModelArtifacts.S3ModelArtifacts` on the training
  job.
- `target` (string, optional): `sagemaker` (default) or `bedrock`.
- `_user_id` (string, required): always pass `CURRENT_USER_ID`.

### Parameters (sagemaker target)

- `endpoint_name` (string): Desired endpoint name.
- `instance_type` (string, default `ml.m5.xlarge`).

Returns: `{ "target": "sagemaker", "endpoint_name": "...", "endpoint_url": "..." }`

### Parameters (bedrock target)

- `thread_id` (string, required): always pass `CURRENT_THREAD_ID`. The
  import poller uses it to resume this session when the import reaches a
  terminal state — without it there is no async resume and you must poll
  manually.
- `bedrock_model_name` (string, optional): Name for the imported Bedrock
  model. Must match `^[a-zA-Z0-9-_.]+$` and be ≤ 63 chars. Default: the
  SageMaker training job name with non-alphanumerics replaced by `-`.

Returns: `{ "target": "bedrock", "bedrock_model_name": "...", "import_job_arn": "...", "import_job_identifier": "...", "sagemaker_job_name": "...", "thread_id": "...", "job_id": "..." }`

### Preconditions + caveats

- The source training job must be in `Completed` status. Bedrock import
  fails loudly (MCP error) when it isn't.
- Bedrock Custom Model Import works best with LoRA fine-tunes; full
  fine-tunes may exceed Bedrock's size ceiling. There is no automated
  LoRA-only precondition — warn the user before importing a full
  fine-tune of a large model.
- Bedrock import is async and resumes like a training job: a scheduled
  poller (15-min cadence) tracks the import job and sends a continuation
  message to this thread on terminal status. Do NOT busy-poll after
  submitting — end the turn and wait for the resume, exactly as after
  `submit_training_job`. The resume only happens when `thread_id` was
  passed.

## Submit Eval Job

Use MCP tool: `sagemaker-skill___submit_eval_job`

Kicks off an async SageMaker Processing job that runs `mlflow.genai.evaluate`
against a target model with a Bedrock judge. Returns immediately; the
EventBridge callback closes the thread row when the Processing job finishes.

Parameters:

- `thread_id` (string): always pass `CURRENT_THREAD_ID`
- `eval_dataset_s3_uri` (string): JSON eval-spec S3 URI returned by
  `huggingface-skill___prepare_eval_dataset` (`spec_s3_uri`). The
  Processing container loads the dataset internally — do not pre-materialise.
- `target_model` (string): model under evaluation, format
  `bedrock:/<model_id>` or `sagemaker:/<endpoint_name>`. Pass the exact
  Bedrock model ID the user asked for — do NOT rewrite it to a different
  inference-profile prefix (e.g. if the user says
  `global.amazon.nova-2-lite-v1:0`, submit it verbatim; do not swap in
  `us.amazon.nova-lite-v1:0`).
- `judge_model` (string, optional): LLM-as-judge, format
  `bedrock:/<model_id>`. Defaults to
  `bedrock:/us.anthropic.claude-sonnet-4-5-20250929-v1:0` — only pass
  this if you need a different judge.
- `scorers` (list[string]): canonical MLflow scorer class names ONLY. The
  agent MUST call `mlflow-skill___list_scorers` first to discover the live
  catalog, then reason about which canonical name matches each user-supplied
  term by reading the returned docstrings. Pass the resolved canonical names
  verbatim — the container rejects plain-English inputs (`faithfulness`,
  `answer_relevance`, …) and raises `ValueError` with the full valid-names
  list. If a user term is ambiguous against the catalog, prompt the user to
  disambiguate before submitting. See `agent/.claude/skills/mlflow/SKILL.md`
  → "List Scorers" for the matching protocol.
- `task` (string): `question_answering` | `rag` | `summarization` |
  `text_generation` | `classification`
- `custom_scorer_lambda_arns` (list[string], optional): Lambda ARNs
  implementing the custom-scorer contract — see
  `agent/.claude/skills/mlflow/SKILL.md` → "Custom Scorers (R6)". Omit for
  MLflow built-ins only.
- `instance_type` (string, default `ml.m5.large` — SageMaker Processing only allows x86_64 instance families)
- `run_name` (string, optional): friendly MLflow run name
- `_user_id` (string): always pass `CURRENT_USER_ID`

Returns:
`{ "thread_id": "...", "job_id": "...", "processing_job_name": "...", "mlflow_run_id": "...", "mlflow_run_url": "..." }`

**After a successful submit you MUST surface `mlflow_run_url` to the user** —
same pattern as `submit_training_job`. When the job completes, EventBridge
fires a callback that resumes the agent with an "Evaluation job … is now
Completed" message instructing you to invoke the
`compliance-documentation` skill.

## List Recent Training Jobs

Use MCP tool: `sagemaker-skill___list_recent_training_jobs`

Lists this project's recent SageMaker training jobs across **all sessions**,
newest first, enriched with `training_type` and `dataset_name`. Use it when
the user says "the most recent Completed XGBoost job" (or SFT fine-tune)
and the current session contains no candidate — pick the newest entry with
the matching `training_type` and pass its `sagemaker_job_name` to
`submit_monitoring_job` or `submit_recommendation_job`. Do NOT ask the user
to paste a job name before trying this tool.

Parameters: `status_equals` (default `Completed`), `max_results` (default 10),
`_user_id` (always pass `CURRENT_USER_ID`).

## Submit Monitoring Job

Use MCP tool: `sagemaker-skill___submit_monitoring_job`

Kicks off an async SageMaker Processing job that runs **Evidently** drift

- data-quality checks (and classification-quality metrics when labels
  are available) against a completed tabular training job's baseline.
  Returns immediately; the EventBridge callback updates the thread row
  when the Processing job finishes.

### When to use

- After a tabular training job (`training_type=xgboost` or `sklearn`)
  has Completed, to compare a new "current" dataset against the
  training-time baseline the job auto-emitted.
- Do NOT use for LLM training types — they do not emit a baseline and
  the monitoring container is tabular-only.

### Parameters

- `thread_id` (string, required): always pass `CURRENT_THREAD_ID`.
- `sagemaker_job_name` (string, required): the Completed training job
  whose baseline + model artifact drive this run. May come from a
  **different session** — the handler resolves the job's own thread via
  its SageMaker `ThreadId`/`JobId` tags (use
  `list_recent_training_jobs` to find candidates). Handler resolves
  `baseline_s3_uri` and `model_artifact_s3_uri` from that thread row —
  do NOT pass S3 URIs.
- `current_data_s3_uri` (string, optional): `s3://bucket/key.csv` or
  `s3://bucket/prefix/`. Mutually exclusive with `use_training_eval_split`.
- `use_training_eval_split` (bool, optional, default `false`): when
  `true`, handler reads `jobs.<source>.eval_split_s3_uri` from the
  thread row and uses that as the current frame.
- `target_column` (string, optional): label column name in the current
  CSV. Enables ClassificationPreset when the column is present.
- `instance_type` (string, default `ml.m5.large`, x86_64 only).
- `run_name` (string, optional): MLflow run display name.
- `_user_id` (string, required): always pass `CURRENT_USER_ID`.

Caller must set **exactly one** of `current_data_s3_uri` OR
`use_training_eval_split=true`. The handler refuses submission
otherwise.

### Returns

```json
{
  "thread_id": "…",
  "job_id": "…",
  "processing_job_name": "…",
  "mlflow_run_id": "…",
  "mlflow_run_url": "…",
  "baseline_s3_uri": "s3://…/baseline/baseline.csv",
  "current_data_s3_uri": "s3://…/baseline/eval_split.csv",
  "model_artifact_s3_uri": "s3://…/output/model.tar.gz",
  "status": "SUBMITTING"
}
```

**After a successful submit you MUST surface `mlflow_run_url` to the
user** — same pattern as submit_training_job / submit_eval_job. The
monitoring report HTML and JSON are logged as artifacts on that run;
`drifted_columns_share` and `drifted_columns_count` scalars appear
under the Metrics tab.

### Follow-up

When the callback fires "Monitoring job X is now Completed", surface
the scalar metrics (`drifted_columns_share`, `drifted_columns_count`,
`accuracy` if labelled) and link `mlflow_run_url`. Do NOT invoke
`compliance-documentation` — monitoring is a regression-free read of
the data, not an audit artifact. The callback resume message says so
explicitly.

### Expected signal on the iris starter tile

Iris train/eval splits are drawn from the same random shuffle, so
Evidently will report **near-zero drift**. That is the correct signal
for a working model and an unchanged data distribution. Surface it
plainly; do not imply the monitoring is broken.

## Submit Recommendation Job

Use MCP tool: `sagemaker-skill___submit_recommendation_job`

Kicks off an async SageMaker AI Benchmark job against a trained LLM on a
single user-specified GPU instance. The submit-side handler deploys a
temporary endpoint; when the endpoint reaches InService an EventBridge
callback creates the workload config + benchmark job against the bare endpoint
(the model is deployed on the endpoint variant — no inference component).
Teardown (endpoint/config/model) happens inside `get_recommendation_results`
the first time AWS reports the benchmark terminal — see that tool's docs
below. Wall-clock ~15-40 min; typical spend ~$2-6 for g5/g6 families.

### Confirmation required before calling

Per the "Confirmation Required Before Billable Actions" rule at the top of
this file, you MUST summarize instance_type + default workload spec + the
~15-40 min wall-clock + the ~$2-6 cost range back to the user and ask for
explicit confirmation before invoking this tool.

### When to use

- After an LLM fine-tune (training_type in `sft`, `dpo`, `grpo`) has
  Completed, to decide which instance type to deploy on.
- Do NOT use for tabular training types — the TGI-based serving container
  won't load xgboost/sklearn artifacts. The handler rejects these.
- Multi-instance comparison = N parallel calls, one per candidate.

### Parameters

- `thread_id` (string, required): `CURRENT_THREAD_ID`.
- `sagemaker_job_name` (string, required): Completed LLM fine-tune.
- `instance_type` (string, required): single GPU instance (ml.g* or ml.p*).
  CPU rejected.
- `input_tokens` (int, optional, default 500): synthetic input length.
- `output_tokens` (int, optional, default 150): synthetic output length.
- `concurrency_levels` (list[int], optional, default [1,4,16]). The AIPerf
  benchmark runs ONE concurrency per job — only the HIGHEST requested level
  is exercised (worst case for the p99 gate). Tell the user this when they
  request multiple levels.
- `max_latency_p99_ms` (int, optional, default 5000).
- `max_invocations_per_minute` (int, optional): traffic ceiling.
- `run_name` (string, optional).
- `serving_image_uri` (string, optional): override auto-derived TGI image.
  Any SageMaker-compatible LLM serving image (TGI/vLLM/LMI/Triton/custom).
- `serving_env` (dict, optional): env vars for the serving container;
  caller owns this when serving_image_uri is overridden.
- `_user_id` (string, required): `CURRENT_USER_ID`.

### Returns

```json
{
  "recommender_job_name": "sample-mlops-agent-rec-…",
  "endpoint_name": "…",
  "instance_type": "ml.g6.xlarge",
  "estimated_wall_clock_minutes": 20,
  "status": "SUBMITTING"
}
```

**After submit you MUST surface** `recommender_job_name` and
`estimated_wall_clock_minutes` to the user. Example:

> Submitted benchmark `rec-xxx` on ml.g6.xlarge. Expect ~20 min. Call
> `get_recommendation_results` when you want me to re-check status.

Tell the user to ping you when they want a status re-check. Do NOT
loop-poll `get_recommendation_results` on your own — the user drives the
cadence. One call per user message is the right cadence.

## Get Recommendation Results

Use MCP tool: `sagemaker-skill___get_recommendation_results`

Actively checks the AI benchmark job status and, on terminal, parses the
AIPerf aggregates (`profile_export_aiperf.json` from the job's
`output.tar.gz`) + tears down the temporary endpoint/config/model.
Safe to call repeatedly — DDB short-circuits the second call onward.

### Parameters

- `thread_id` (string, required): `CURRENT_THREAD_ID`.
- `recommender_job_name` (string, required): from submit response.
- `_user_id` (string, required): `CURRENT_USER_ID`.

### Returns (Completed)

```json
{
  "status": "COMPLETED", "instance_type": "ml.g6.xlarge",
  "metrics": { "ttft_ms_p99": 812, "inter_token_ms_p99": 58,
               "request_latency_p99_ms": 3900,
               "throughput_tokens_per_sec": 1710, ... },
  "workload_spec": {…}, "profile_s3_uri": "s3://…",
  "teardown_complete": true, "summary": "…"
}
```

### Returns (pre-terminal)

```json
{ "status": "DEPLOYING" | "BENCHMARKING",
  "message": "AI benchmark status='InProgress'; try again later …" }
```

No `metrics` until status reaches COMPLETED or FAILED.

### How to handle pre-terminal responses

If the response is `DEPLOYING` or `BENCHMARKING`, surface the `message`
string to the user verbatim and stop. The user will ping you again to
re-check; each user message triggers at most ONE `get_recommendation_results`
call — never call it a second time in the same turn.

### Follow-up

On `COMPLETED`: surface the metrics table. Typical next step: user picks
an instance and you call `deploy_model`. Do NOT auto-deploy — always let
the user confirm which result they want to ship to production.
On `FAILED`: surface `summary` (includes FailureReason). Teardown still
ran, so no cleanup action is needed from the user.
