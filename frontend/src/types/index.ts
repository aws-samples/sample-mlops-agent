export type JobStatus =
  | "SUBMITTING"
  | "PENDING"
  | "IN_PROGRESS"
  | "COMPLETED"
  | "FAILED"
  | "STOPPED";

export type JobPhase = "chatting" | "executing";

export type TimelineEntry = MessageEntry | ToolEntry;

export interface MessageEntry {
  kind: "message";
  id: string;
  role: "user" | "assistant";
  content: string;
}

export interface ToolEntry {
  kind: "tool";
  id: string;
  step: number;
  name: string;
  status: string;
  args: string;
  result?: string;
  truncated?: boolean;
  isError?: boolean;
}

export interface TrainingConfig {
  modelId: string;
  dataS3: string;
  instanceType: string;
  hyperparams: Record<string, string>;
}

// Eval-job-specific fields mirrored onto the AgentJob from the
// DynamoDB jobs.<job_id> entry. Kept as a sub-object so the training
// path (config: TrainingConfig) and the eval path (evalConfig) render
// without either one having to carry nullable fields of the other.
export interface EvalConfig {
  targetModel: string;
  judgeModel: string;
  datasetS3Uri: string;
  scorers: string[];
  task: string;
  instanceType: string;
}

// Compliance report artefact stamped onto the eval job row once the
// compliance-documentation skill completes. The frontend uses
// `s3Uri` to render a Download button in TaskDetailPanel.
export interface ComplianceReport {
  s3Uri: string;
  format: "md" | "docx";
  at: number;
}

export interface AgentJob {
  threadId: string;
  runId: string;
  kind: "training" | "eval";
  config: TrainingConfig;
  evalConfig?: EvalConfig;
  complianceReport?: ComplianceReport;
  status: JobStatus;
  phase: JobPhase;
  timeline: TimelineEntry[];
  mlflowRunUrl?: string;
  /** SageMaker job name — training job name when kind="training",
   *  Processing job name when kind="eval". Used to linkify a row in
   *  TaskDetailPanel that takes the user straight to the job's SageMaker
   *  console page. */
  sagemakerJobName?: string;
  metrics?: Record<string, number>;
  errorMessage?: string;
  statusMessage?: string;
  createdAt?: number;
  updatedAt?: number;
}
