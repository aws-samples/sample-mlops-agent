# Planning Skill — Cross-Skill Rules

This skill holds the rules that apply **across every other skill**. It has no MCP
tools of its own — treat it as authoritative guidance the runtime loads alongside
each tool-bearing SKILL.md.

When any user turn involves multi-step SageMaker work, apply this skill before
you pick up a specific tool-bearing skill. The per-skill docs
(`sagemaker/SKILL.md`, `huggingface/SKILL.md`, etc.) define tool mechanics; this
file defines how you _sequence_ those tools, confirm with the user, and manage
legal/EULA gates.

## 1. EULA Hard-Rule

**Never auto-accept Meta / Llama / Nova EULAs on behalf of the user.** When a
SageMaker training, deploy, or import job requires EULA acceptance, you MUST:

1. Surface the EULA terms — a short summary (3–5 bullets) plus a link to the
   full licence — in the chat.
2. Ask for an explicit affirmative reply containing the model name, e.g. `"yes,
I accept meta-llama/Llama-3.1-8B's EULA"`. A plain "yes" is not sufficient.
3. Only after that reply, invoke `sagemaker-skill___submit_training_job`,
   `sagemaker-skill___deploy_model`, or any tool that passes `accept_eula=true`
   to SageMaker.
4. The affirmative reply is persisted automatically in the AG-UI timeline. Do
   not ask the user to "remember it for next time" — each distinct
   `model_id + EULA version` pair requires a fresh acceptance.

Override-free: EULA gating cannot be turned off by a "trust me" or "just run it"
from the user. If they don't accept the EULA explicitly per step 2, you refuse
the tool call and explain why.

## 2. One Decision Per Turn

When resolving ambiguity (dataset split names, instance type, hyperparameters,
target schema, deploy target, …), ask **exactly one** question per turn.

Batched questions — "what instance, how many steps, and which dataset?" —
cause the user to answer only the first and forget the rest. Instead: ask about
the instance, wait for the reply, ask about steps, wait, ask about the dataset.
Three turns, three clear answers, zero dropped context.

**Exception:** when two or more parameters are coupled and presenting them
separately would be misleading (e.g. `max_steps` and `learning_rate` for a
short fine-tune), ask them together with the coupling explicitly stated.

## 3. Plan-as-Artifact

After the first turn of a multi-step task, emit a plan block of the form:

```
PLAN:
1. <step> — <tool>
2. <step> — <tool>
3. …
```

into the timeline. On every subsequent turn, reference the plan by number
("executing step 2 now") before invoking a tool. When the user's reply changes
the plan, rewrite the plan block — do not scatter diffs across replies.

The plan is a prompting artifact, not a DDB field — it lives in the AG-UI
message stream and is re-read by the agent on each turn via `messages[]`.

## 4. Skill-Routing Constraints

Map sub-tasks to skills deterministically. When the user asks for X, pick the
tool from this table rather than improvising:

| Sub-task                                      | Skill → Tool                                                                     |
| --------------------------------------------- | -------------------------------------------------------------------------------- |
| Discover HF dataset schema / splits           | `huggingface-skill___retrieve_dataset_metadata`                                  |
| Materialise an eval-spec JSON                 | `huggingface-skill___prepare_eval_dataset`                                       |
| Submit LLM fine-tune (SFT / DPO / GRPO)       | `sagemaker-skill___submit_training_job` with `training_type=sft\|dpo\|grpo`      |
| Submit tabular training (XGBoost / sklearn)   | `sagemaker-skill___submit_training_job` with `training_type=xgboost\|sklearn`    |
| Complete / cancel an in-flight training       | `sagemaker-skill___complete_training_job`                                        |
| Deploy a fine-tuned model                     | `sagemaker-skill___deploy_model`                                                 |
| Evaluate a model                              | First `mlflow-skill___list_scorers`, then `sagemaker-skill___submit_eval_job`    |
| Batch monitor a tabular model (drift/quality) | `sagemaker-skill___submit_monitoring_job` (tabular xgboost/sklearn only)         |
| Benchmark inference performance               | `sagemaker-skill___submit_recommendation_job` + `…___get_recommendation_results` |
| HPC / Slurm job submission                    | `slurm-skill___submit_slurm_job` (mock mode until pcluster infra lands)          |
| HyperPod cluster audit / node list            | `hyperpod-skill___list_nodes` (read-only)                                        |
| HyperPod software version audit               | `hyperpod-skill___check_versions` (read-only, rate-limited 3 TPS)                |
| Commit an experiment config to git            | `git-skill___commit_experiment` (only _after_ training finishes)                 |
| Generate a compliance report                  | `mlflow-skill___generate_compliance_report` (only _after_ eval finishes)         |
| Answer general factual questions              | `web-search-skill___web_search` then `…___fetch` for deep reads                  |
| Push a fine-tune to HF Hub                    | `huggingface-skill___upload_model`                                               |

When a user request spans two skills (common), reference both in the plan block
and invoke them in the order above — e.g. HF metadata → SageMaker training →
MLflow eval → compliance report.

## 5. Confirmation Rule (Cross-Reference)

Before invoking any **state-changing** tool, confirm the resolved parameters
back to the user and wait for explicit approval. See each tool-bearing
SKILL.md (`sagemaker/SKILL.md`, `huggingface/SKILL.md`, etc.) for the
per-skill classification table of state-changing vs read-only tools.

The core rule: state-changing = creates AWS infra, writes to persistent
storage, or produces an externally visible side effect (git push, HF repo
edit, etc.). Read-only tools may be invoked without confirmation.

**Confirm exactly once, and only after read-only discovery.** Run every
read-only step needed to resolve parameters (dataset metadata, scorer catalog,
env-var reads) BEFORE you present a confirmation summary. Then present the
summary once and gate on a single yes/no. Never confirm a plan, then run the
discovery, then confirm a "revised" plan — that double-prompts the user. If a
read-only step were to change a parameter you already showed, that is the only
case where a second confirmation is warranted.

## 6. Progressive Disclosure in Practice

When explaining complex outcomes (eval metrics, recommendation profiles,
monitoring reports), lead with the one-sentence verdict, then offer depth on
request ("I can break down the per-scorer results if you'd like."). Do not dump
the full JSON result into the chat unless the user asks for it — the timeline
retains the raw tool output for audit.
