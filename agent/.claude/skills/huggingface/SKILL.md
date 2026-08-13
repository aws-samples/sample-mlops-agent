# HuggingFace Skill

All HuggingFace operations are performed via MCP tools on the `mlops-gateway` server.

> See `agent/.claude/skills/planning/SKILL.md` for cross-skill rules (EULA, one-decision-per-turn, routing, confirmation gate).

## Confirmation Required Before State-Changing Actions

Before calling any tool marked **state-changing** in the table below, summarize
the resolved parameters back to the user, ask for explicit confirmation, and
only invoke the tool after an affirmative reply. Read-only tools may be invoked
without confirmation.

| Tool                                            | Kind           |
| ----------------------------------------------- | -------------- |
| `huggingface-skill___upload_model`              | state-changing |
| `huggingface-skill___hf_snapshot_download`      | read-only      |
| `huggingface-skill___retrieve_dataset_metadata` | read-only      |
| `huggingface-skill___prepare_eval_dataset`      | state-changing |
| `huggingface-skill___update_model_card`         | state-changing |
| `huggingface-skill___manage_tags`               | state-changing |

## Upload Model

Use MCP tool: `huggingface-skill___upload_model`

Parameters:

- `artifact_s3` (string): S3 URI of model artifact (tar.gz)
- `repo_id` (string): HuggingFace repo, e.g. `username/model-name`
- `private` (bool): create private repo (default: false)
- `_user_id` (string): always pass CURRENT_USER_ID

Returns: `{ "repo_url": "https://huggingface.co/..." }`

## Snapshot Download (model weights / small repo files)

Use MCP tool: `huggingface-skill___hf_snapshot_download`

Parameters:

- `repo_id` (string): HuggingFace repo ID
- `s3_prefix` (string): S3 key prefix destination
- `repo_type` (string): `model` | `dataset` (default: model)
- `allow_patterns` (list[string], optional): glob includes, e.g. `["*.safetensors"]`
- `ignore_patterns` (list[string], optional): glob excludes
- `revision` (string, optional): branch/tag/commit

Returns: `{ "s3_uri": "s3://...", "files": [{path, size_bytes, s3_key}, ...] }`

Use this for model weights or small artefacts. **Do NOT use it to stage huge
eval datasets** — use `prepare_eval_dataset` instead; the Processing container
materialises the dataset at eval time.

## Retrieve Dataset Metadata (dataset EDA entry point)

Use MCP tool: `huggingface-skill___retrieve_dataset_metadata`

Calls the HuggingFace Datasets Server REST API to return splits, column
schema, and (optionally) a small sample of rows — entirely from JSON
endpoints, with no parquet download and no Lambda disk IO. This is the
canonical dataset EDA tool — you MUST call it before:

- authoring an eval spec with `prepare_eval_dataset`, and
- submitting a training job via `sagemaker-skill___submit_training_job` with
  a HuggingFace `dataset_name` (see that skill's "Pre-flight: Dataset
  Analysis is REQUIRED" section).

What to surface from the response:

- **`available_splits`** — the full list of valid split names for the
  dataset. Use this to set `train_split` / `test_split` on
  `submit_training_job` (e.g. `ultrachat_200k` → `train_sft`, `test_sft`),
  or to pick the right `split` for `prepare_eval_dataset`. No more
  guess-and-retry on split names.
- **`columns`** — the full column schema (always returned, regardless of
  the `columns` filter). Confirm the dataset is shaped for the chosen
  `training_type` (SFT vs DPO vs GRPO vs XGBoost) or `task_type`
  (QA vs RAG vs summarization …) before committing to a billable job.
- **`card_excerpt`** — first ~500 chars of the dataset-card description,
  when the dataset publishes one. Useful for a one-line "what is this
  dataset?" explanation in the confirmation summary you show the user.
- **`available_configs`** — dataset configs / subsets; most datasets have
  one called `default`, but benchmarks like MMLU ship many.

Parameters:

- `dataset_name` (string): HuggingFace dataset repo ID
- `split` (string, default `train`): the split to sample rows from. If it
  does not exist, the tool transparently falls back to the first available
  split and reports both `requested_split` and `split` in the response.
- `config` (string, optional): defaults to the first config the server
  reports (usually `default`).
- `columns` (list[string], optional): restrict returned records to a
  subset; the full schema is always reported in `columns`.
- `max_rows` (int, optional, default 0): 0 = metadata-only (2 HTTP calls).
  > 0 = additionally fetch sample rows; server hard-caps to 50.

Returns: `{ dataset_name, split, requested_split, config, columns, available_splits, available_configs, records, count, card_excerpt, preview: true, preview_cap }`

## Prepare Eval Dataset (spec, not data)

Use MCP tool: `huggingface-skill___prepare_eval_dataset`

Authors a small JSON eval-spec describing how to load and reshape the
dataset. Writes the spec to S3 and returns `spec_s3_uri`. Pass that URI as
`eval_dataset_s3_uri` to `sagemaker-skill___submit_eval_job` — the
Processing container reads the spec, loads the dataset, applies the
task-default column mapping, and streams records into
`mlflow.genai.evaluate`. No dataset IO happens in Lambda.

Parameters:

- `task_type` (string): `question_answering` | `rag` | `summarization` |
  `text_generation` | `classification`
- `dataset_name` (string): HuggingFace dataset repo ID
- `split` (string, default `train`)
- `config` (string, optional)
- `input_columns` / `expectation_columns` / `context_columns`
  (list[string], optional): override the task defaults when the dataset
  uses custom column names. Use the preview above to pick these.
  For `task_type="rag"`, the Processing container auto-flattens list or
  list-of-dict context columns (e.g. the `evidence` field on
  `PatronusAI/financebench`) — no need to pre-join passages.
- `max_rows` (int, default 0 = all rows)
- `out_s3_prefix` (string, optional): S3 prefix for the spec JSON
- `source_schema` (string, optional): `chat` | `sft` | `dpo` | `tabular-csv`.
  If omitted, the eval container infers from the dataset's column names at
  materialisation time.
- `target_schema` (string, optional): same enum. When present and different
  from source, the eval container applies a row-level transformation
  before evaluation. Supported pairs: `chat→sft` (flatten messages into
  a single text column), `sft→chat` (split on role markers back into
  messages), `dpo→sft` (drop rejected, use chosen as target). Unsupported
  pairs — `chat→dpo` (can't infer pairings), `tabular-csv→anything` — are
  refused by the handler before any S3 write.

Returns: `{ "spec_s3_uri": "...", "source_schema": ..., "target_schema": ..., "transform_mechanism": ... }`

### When to use target_schema

Set `target_schema` when the dataset shape does not match the downstream
consumer. Typical case: an HF dataset in OpenAI chat shape
(`messages: [{role, content}, ...]`) needs to feed an eval expecting SFT
single-text rows — pass `target_schema="sft"` and the container's
`schema_transform` module collapses messages into role-prefixed text.

Do NOT set `target_schema` when source and target already agree — leave
both fields unset for the legacy code path.

## Update Model Card

Use MCP tool: `huggingface-skill___update_model_card`

Parameters:

- `repo_id` (string): HuggingFace repo ID
- `content` (string): README.md content (markdown)
- `append` (bool): append to existing README (default: false)
- `_user_id` (string): always pass CURRENT_USER_ID

Returns: `{ "status": "updated", "repo_id": "..." }`

## Manage Tags

Use MCP tool: `huggingface-skill___manage_tags`

Parameters:

- `repo_id` (string): HuggingFace repo ID
- `tags` (list[string]): tags to add or remove
- `remove` (bool): if true, remove the tags; else add them (default: false)
- `_user_id` (string): always pass CURRENT_USER_ID

Returns: `{ "status": "updated", "tags": [...] }`
