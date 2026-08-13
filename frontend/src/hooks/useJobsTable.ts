import { useCallback, useEffect, useState } from "react";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  ScanCommand,
  UpdateCommand,
} from "@aws-sdk/lib-dynamodb";
import { getAWSCredentials } from "../lib/credentials";

const REGION = import.meta.env.VITE_AWS_REGION ?? "us-east-1";
const TABLE_NAME =
  import.meta.env.VITE_JOBS_TABLE_NAME ?? "sample-mlops-agent-metadata";
const POLL_MS = 10_000;

// One DynamoDB row per chat thread. PK attribute is named `task_id`
// for backwards compatibility with the existing table, but its value is
// the thread_id. Each job (training or eval) is stored under jobs[job_id].
//
// `kind` discriminates the job row. Training jobs carry model_id /
// dataset_name / training_type; eval jobs carry target_model /
// judge_model / scorers / task and later get stamped with
// compliance_report_* once the compliance-documentation skill runs.
// monitoring / recommendation / bedrock_import entries carry no
// model_id/dataset_name of their own — only instance_type plus a
// source_training_job (or bedrock_model_name) pointer.
export interface JobEntry {
  job_id: string;
  kind?:
    "training" | "eval" | "monitoring" | "recommendation" | "bedrock_import";
  sagemaker_job_name?: string;
  processing_job_name?: string;
  // Training fields
  model_id?: string;
  dataset_name?: string;
  instance_type?: string;
  training_type?: string;
  endpoint_name?: string;
  // Monitoring / recommendation / bedrock_import fields
  source_training_job?: string;
  bedrock_model_name?: string;
  model_name?: string;
  // Eval fields (written by submit_eval_job)
  target_model?: string;
  judge_model?: string;
  dataset_s3_uri?: string;
  scorers?: string[];
  task?: string;
  // Compliance report stamps (written by generate_compliance_report)
  compliance_report_s3_uri?: string;
  compliance_report_format?: "md" | "docx";
  compliance_report_at?: number;
  // Shared
  status?: string;
  status_message?: string;
  mlflow_run_id?: string;
  mlflow_run_url?: string;
  experiment_id?: string;
  created_at?: number;
  updated_at?: number;
  completed_at?: number;
}

export interface JobRecord {
  task_id: string; // == thread_id
  thread_id?: string;
  user_id?: string;
  messages?: unknown;
  // Persisted AG-UI timeline (JSON string). Presence indicates the thread has
  // chat activity — used to render an in-flight task card before submit writes
  // a jobs entry (see jobRecordToAgentJob in App.tsx).
  timeline?: string;
  jobs?: Record<string, JobEntry>;
  created_at?: number;
  updated_at?: number;
}

// Returns the most recently created job entry for a thread, or null if
// the thread has no submissions yet.
export function getPrimaryJob(r: JobRecord): JobEntry | null {
  const entries = Object.values(r.jobs ?? {});
  if (entries.length === 0) return null;
  return [...entries].sort(
    (a, b) => (b.created_at ?? 0) - (a.created_at ?? 0),
  )[0];
}

// Human card-title fallback for job kinds that never carry a
// model_id/target_model of their own.
const KIND_TITLE: Record<string, string> = {
  training: "Training",
  eval: "Evaluation",
  monitoring: "Monitoring",
  recommendation: "Benchmark",
  bedrock_import: "Bedrock import",
};

export interface CardDisplay {
  modelId: string;
  dataset: string;
  instanceType: string;
}

// Card display fields merged across ALL of the thread's job entries,
// newest first. getPrimaryJob still decides status/links/timestamps
// (newest entry = current activity), but a newer metadata-poor entry —
// e.g. the bedrock_import job stacked on top of the training job it
// deploys — must not blank out the title/dataset an older entry carries
// (QA-02). Threads whose only entry is monitoring/recommendation fall
// back to a kind label + their source_training_job pointer.
export function getCardDisplay(r: JobRecord): CardDisplay {
  const entries = Object.values(r.jobs ?? {}).sort(
    (a, b) => (b.created_at ?? 0) - (a.created_at ?? 0),
  );
  // First non-empty value of `pick` across entries, newest first.
  const first = (
    pick: (e: JobEntry) => string | undefined,
  ): string | undefined => {
    for (const e of entries) {
      const v = pick(e);
      if (v) return v;
    }
    return undefined;
  };
  const primaryKind = entries[0]?.kind;
  return {
    modelId:
      first((e) => e.model_id ?? e.target_model) ??
      (primaryKind ? (KIND_TITLE[primaryKind] ?? "") : ""),
    dataset:
      first((e) => e.dataset_name ?? e.dataset_s3_uri) ??
      first((e) => e.source_training_job) ??
      "",
    instanceType: first((e) => e.instance_type) ?? "",
  };
}

async function getDocClient() {
  const credentials = await getAWSCredentials();
  if (!credentials) throw new Error("Not authenticated");
  return DynamoDBDocumentClient.from(
    new DynamoDBClient({ region: REGION, credentials }),
  );
}

async function fetchJobs(): Promise<JobRecord[]> {
  const client = await getDocClient();
  const result = await client.send(new ScanCommand({ TableName: TABLE_NAME }));
  return (result.Items ?? []).filter(
    (item) => item["task_id"] && !item["DELETE_FLAG"],
  ) as JobRecord[];
}

export function useJobsTable(enabled = true) {
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setJobs(await fetchJobs());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    void refresh();
    const id = setInterval(() => {
      void refresh();
    }, POLL_MS);
    return () => clearInterval(id);
  }, [enabled, refresh]);

  const deleteJob = useCallback(async (taskId: string) => {
    const client = await getDocClient();
    await client.send(
      new UpdateCommand({
        TableName: TABLE_NAME,
        Key: { task_id: taskId },
        UpdateExpression: "SET DELETE_FLAG = :t",
        ExpressionAttributeValues: { ":t": true },
      }),
    );
    setJobs((prev) => prev.filter((j) => j.task_id !== taskId));
  }, []);

  return { jobs, loading, error, refresh, deleteJob };
}
