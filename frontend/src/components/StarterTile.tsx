import Box from "@cloudscape-design/components/box";
import SpaceBetween from "@cloudscape-design/components/space-between";

export interface StarterTileProps {
  title: string;
  description: string;
  prompt: string;
  onSelect: (prompt: string) => void;
}

export function StarterTile({
  title,
  description,
  prompt,
  onSelect,
}: StarterTileProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(prompt)}
      style={{
        background: "var(--color-background-container-content, #ffffff)",
        border: "1px solid var(--color-border-divider-default, #d1d5db)",
        borderRadius: "8px",
        cursor: "pointer",
        padding: "16px",
        textAlign: "left",
        width: "100%",
        transition: "border-color 0.15s",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLButtonElement).style.borderColor = "#0972d3";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLButtonElement).style.borderColor =
          "var(--color-border-divider-default, #d1d5db)";
      }}
      onFocus={(e) => {
        (e.currentTarget as HTMLButtonElement).style.borderColor = "#0972d3";
      }}
      onBlur={(e) => {
        (e.currentTarget as HTMLButtonElement).style.borderColor =
          "var(--color-border-divider-default, #d1d5db)";
      }}
    >
      <SpaceBetween size="xxs">
        <Box variant="strong" fontSize="body-m">
          {title}
        </Box>
        <Box color="text-body-secondary" fontSize="body-s">
          {description}
        </Box>
      </SpaceBetween>
    </button>
  );
}

export const STARTER_TILES: Omit<StarterTileProps, "onSelect">[] = [
  {
    title: "Quick fine-tune",
    description:
      "Qwen 2.5 0.5B (Apache-2.0) on UltraChat 200k (MIT) · ml.g5.2xlarge · 100 steps · 2k-sample subset",
    prompt:
      "Fine-tune Qwen/Qwen2.5-0.5B-Instruct on HuggingFaceH4/ultrachat_200k from Hugging Face using ml.g5.2xlarge, 100 steps, learning rate 1e-5, capped to 2000 training samples so the demo finishes quickly.",
  },
  {
    title: "LLM evaluation with MLflow",
    description:
      "Evaluate global.amazon.nova-2-lite-v1:0 on 20-row slice of PatronusAI/financebench (CC-BY-NC-4.0) and generate a compliance report",
    prompt:
      "Evaluate global.amazon.nova-2-lite-v1:0 on the PatronusAI/financebench dataset from Hugging Face. Use only the first 20 rows so the demo stays under ~5 minutes and judge-model costs stay low. Log metrics (faithfulness, answer relevance, correctness) and traces to the MLflow tracking server. Once the evaluation is complete, create a compliance report.",
  },
  {
    title: "Submit Slurm job",
    description: "Launch a distributed training job on the Slurm HPC cluster",
    prompt:
      "Submit a multi-node training job to the Slurm cluster. Use the slurm-skill to queue the job, monitor its status, and report back when it completes.",
  },
  {
    title: "XGBoost training",
    description:
      "Train XGBoost on scikit-learn's iris dataset (150 rows, 3 classes) · ml.m5.xlarge · 100 rounds",
    prompt:
      "Train an XGBoost classifier on the scikit-learn iris dataset (training_type=xgboost, dataset_name=iris, target_column=target) on ml.m5.xlarge with 100 boosting rounds, learning_rate 0.1, and max_depth 4. Log per-round eval-mlogloss to MLflow and surface the run URL.",
  },
  {
    title: "Benchmark fine-tuned LLM",
    description:
      "Measure TTFT / P99 / throughput on ml.g6.xlarge · async · ~20 min · ~$2-6",
    prompt:
      "Benchmark the most recent Completed SFT fine-tune on ml.g6.xlarge " +
      "using the default workload spec (input_tokens=500, output_tokens=150, " +
      "concurrency_levels=[1,4,16], max_latency_p99_ms=5000). Summarize the " +
      "candidate + cost range + expected wall-clock to me and wait for my " +
      "confirmation before calling submit_recommendation_job. After submit, " +
      "call get_recommendation_results; if status is DEPLOYING or BENCHMARKING " +
      "tell me it's still running and stop — I'll ping you again to re-check. " +
      "Only when status is COMPLETED present the latency / throughput table. " +
      "Do NOT auto-deploy — wait for me to pick the winning instance.",
  },
  {
    title: "Monitor XGBoost predictions",
    description:
      "Evidently drift + quality report against an XGBoost iris training baseline · ml.m5.large · ~5 min",
    prompt:
      "Run a monitoring job on the most recent Completed XGBoost iris training " +
      "job. Pass use_training_eval_split=true so the monitoring container " +
      "compares the training job's baseline against its own 80/20 eval split " +
      "(legitimate unseen data, no hardcoded fixtures). Include " +
      "target_column=target so classification metrics are reported. Log the " +
      "Evidently HTML and JSON report to MLflow and surface the run URL. " +
      "Expected: near-zero drift (train and eval are the same random shuffle " +
      "of iris) — that is the correct signal for an unchanged distribution.",
  },
];
