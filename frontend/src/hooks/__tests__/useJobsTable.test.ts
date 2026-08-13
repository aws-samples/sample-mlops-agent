// QA-02 regression: task cards rendered "New Task / —" for threads whose
// newest (or only) job entry carries no model_id/dataset_name — the
// bedrock_import entry stacked on an SFT thread, and the single-entry
// monitoring / recommendation threads. getCardDisplay merges display
// fields across all entries so the card keeps its metadata; getPrimaryJob
// still returns the newest entry (status/links must track current
// activity). Fixtures mirror real DynamoDB rows from the 2026-07-26 run.
import { describe, expect, it } from "vitest";
import {
  getCardDisplay,
  getPrimaryJob,
  type JobRecord,
} from "../useJobsTable";

const sftThreadWithBedrockImport: JobRecord = {
  task_id: "thread-sft",
  jobs: {
    "job-training": {
      job_id: "job-training",
      kind: "training",
      training_type: "sft",
      model_id: "Qwen/Qwen2.5-0.5B-Instruct",
      dataset_name: "HuggingFaceH4/ultrachat_200k",
      instance_type: "ml.g5.2xlarge",
      status: "COMPLETED",
      created_at: 1785038064,
    },
    "job-import": {
      job_id: "job-import",
      kind: "bedrock_import",
      bedrock_model_name: "sample-mlops-agent-job-qwen2505bins-1785038064",
      status: "COMPLETED",
      created_at: 1785039039,
    },
  },
};

const monitoringOnlyThread: JobRecord = {
  task_id: "thread-monitor",
  jobs: {
    "job-monitor": {
      job_id: "job-monitor",
      kind: "monitoring",
      source_training_job: "sample-mlops-agent-job-xgboost-1785037891",
      instance_type: "ml.m5.large",
      status: "COMPLETED",
      created_at: 1785039440,
    },
  },
};

const recommendationOnlyThread: JobRecord = {
  task_id: "thread-benchmark",
  jobs: {
    "job-rec": {
      job_id: "job-rec",
      kind: "recommendation",
      source_training_job: "sample-mlops-agent-job-qwen2505bins-1785038064",
      instance_type: "ml.g6.xlarge",
      status: "COMPLETED",
      created_at: 1785039549,
    },
  },
};

const evalThread: JobRecord = {
  task_id: "thread-eval",
  jobs: {
    "job-eval": {
      job_id: "job-eval",
      kind: "eval",
      target_model: "bedrock:/global.amazon.nova-2-lite-v1:0",
      dataset_s3_uri: "s3://bucket/eval-specs/financebench/eval_spec.json",
      instance_type: "ml.m5.large",
      status: "COMPLETED",
      created_at: 1785038248,
    },
  },
};

describe("getPrimaryJob", () => {
  it("returns the newest entry so status tracks current activity", () => {
    expect(getPrimaryJob(sftThreadWithBedrockImport)?.job_id).toBe(
      "job-import",
    );
  });

  it("returns null for threads without submissions", () => {
    expect(getPrimaryJob({ task_id: "empty" })).toBeNull();
  });
});

describe("getCardDisplay", () => {
  it("keeps the training entry's metadata when a bedrock_import entry is newer", () => {
    expect(getCardDisplay(sftThreadWithBedrockImport)).toEqual({
      modelId: "Qwen/Qwen2.5-0.5B-Instruct",
      dataset: "HuggingFaceH4/ultrachat_200k",
      instanceType: "ml.g5.2xlarge",
    });
  });

  it("labels a monitoring-only thread and points at its source job", () => {
    expect(getCardDisplay(monitoringOnlyThread)).toEqual({
      modelId: "Monitoring",
      dataset: "sample-mlops-agent-job-xgboost-1785037891",
      instanceType: "ml.m5.large",
    });
  });

  it("labels a recommendation-only thread as Benchmark with its source job", () => {
    expect(getCardDisplay(recommendationOnlyThread)).toEqual({
      modelId: "Benchmark",
      dataset: "sample-mlops-agent-job-qwen2505bins-1785038064",
      instanceType: "ml.g6.xlarge",
    });
  });

  it("leaves eval threads unchanged (target model + eval-spec dataset)", () => {
    expect(getCardDisplay(evalThread)).toEqual({
      modelId: "bedrock:/global.amazon.nova-2-lite-v1:0",
      dataset: "s3://bucket/eval-specs/financebench/eval_spec.json",
      instanceType: "ml.m5.large",
    });
  });

  it("returns empty fields for threads without submissions", () => {
    expect(getCardDisplay({ task_id: "empty" })).toEqual({
      modelId: "",
      dataset: "",
      instanceType: "",
    });
  });
});
