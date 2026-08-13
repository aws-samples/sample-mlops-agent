# Report Structure

Every report has the same four sections, in this order. Pass them to
`generate_compliance_report` as `sections=[{heading, body}, ...]`.

## Tone

Compliance readers are auditors, not ML engineers. They want:

- **Claims backed by numbers.** Every assertion tied to a metric or
  a row count. "Safety is strong" is worthless; "Safety/mean = 0.96
  across 50 rows" is defensible.
- **Plain language.** No "LLM-as-a-judge" jargon in the Executive
  Summary. You can use it in Methodology.
- **Terse.** One paragraph per section is the target. If a section
  runs > 6 sentences, split it or cut it.

## Section 1 — Executive Summary

What a director reads in 30 seconds. Structure:

1. One sentence naming the model, dataset, and overall outcome.
   > "Nova 2 Lite was evaluated against PatronusAI/financebench (50
   > rows) and passed 4 of 5 scorers."
2. One sentence calling out any FAIL or Safety concern.
   > "Safety scored 0.82, below the 0.95 threshold — do not promote
   > to production until remediated."
3. One sentence on the recommendation.
   > "Re-evaluate after an adversarial-safety SFT pass."

Skip step 2 if everything passed. Skip step 3 if there's no action.

## Section 2 — Methodology

The audit trail. What was run, on what, by what judge. Include:

- Target model (from `params.target_model`)
- Judge model (from `params.judge_model`)
- Dataset repo id + split + row count (from `params` + metric counts)
- Scorer list (from `params.scorers`)
- Date of run (use today's date; the MLflow `start_time` is also fine
  if you can read it from `analyze_trace`)

Example:

> Evaluation was executed via MLflow GenAI's `evaluate()` harness on
> 2026-04-21. Target model: `bedrock:/global.amazon.nova-2-lite-v1:0`. Judge
> model for LLM-as-a-judge scorers: `bedrock:/us.anthropic.claude-sonnet-4-5-20250929-v1:0`.
> Evaluation corpus: PatronusAI/financebench, split `train[:50]`.
> Scorers applied: Safety, Correctness, RelevanceToQuery.

## Section 3 — Results

Narrative summary of the scorer outcomes. **Do not restate every
Finding body** — those render separately as PASS/FAIL tags. This
section is the connective tissue:

- Which scorers passed, as a group
- Which scorer(s) failed and the shape of the failure (scattered
  outliers vs systemic)
- Cross-scorer patterns if any (e.g. "failures cluster in rows
  discussing derivatives, suggesting a domain gap")

Keep it 3–5 sentences. The Metrics Summary table (auto-rendered from
the `metrics` param) carries the numbers.

## Section 4 — Recommendations

Action items. Each is a single bullet-equivalent sentence. Write as
plain prose paragraphs, not a bulleted list — the .docx renders
headings and paragraphs cleanly but list formatting is inconsistent.

Prioritize:

1. Blockers to production promotion (any Safety FAIL, any Correctness
   FAIL below 0.5)
2. Data gaps to close before re-eval (missing scorers, dataset too
   small, domain mismatch)
3. Process recommendations (e.g. "add adversarial subset to nightly
   eval")

If nothing failed, this section is "No remediation required.
Recommend re-evaluating on the next model checkpoint to detect
regression." One sentence, done.

## The "sections" payload

```python
sections = [
    {"heading": "Executive Summary", "body": "..."},
    {"heading": "Methodology",       "body": "..."},
    {"heading": "Results",           "body": "..."},
    {"heading": "Recommendations",   "body": "..."},
]
```

The `generate_compliance_report` Lambda renders each as an H2 in
markdown and Heading 1 in .docx. Do not nest further headings inside
the body — just paragraphs.
