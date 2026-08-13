import { useState, useEffect, useRef } from "react";
import Badge from "@cloudscape-design/components/badge";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import SpaceBetween from "@cloudscape-design/components/space-between";
import { createPresignedMlflowUrl } from "../lib/mlflow";
import type { AgentJob, JobStatus, TimelineEntry } from "../types";

const STATUS_COLOR: Record<JobStatus, "blue" | "grey" | "green" | "red"> = {
  SUBMITTING: "blue",
  PENDING: "grey",
  IN_PROGRESS: "blue",
  COMPLETED: "green",
  FAILED: "red",
  STOPPED: "grey",
};
const STATUS_LABEL: Record<JobStatus, string> = {
  SUBMITTING: "Submitting",
  PENDING: "Pending",
  IN_PROGRESS: "In Progress",
  COMPLETED: "Completed",
  FAILED: "Failed",
  STOPPED: "Stopped",
};

interface Props {
  job: AgentJob;
  section: "header" | "body";
  onClick?: (job: AgentJob) => void;
  onOpen?: (job: AgentJob) => void;
  onDelete?: (job: AgentJob) => void;
  mlflowAppArn?: string;
}

function LogPanel({ messages }: { messages: string[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [messages]);
  return (
    <div ref={ref} className="agent-log-box">
      {messages.join("") || (
        <span style={{ color: "#6b7280", fontStyle: "italic" }}>
          Waiting for agent…
        </span>
      )}
    </div>
  );
}

// Cloudscape Badge colors don't include purple; use a styled pill so the
// "Evaluation" label is unmistakably distinct from the blue/grey status
// badge at a glance.
function EvaluationBadge() {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: "10px",
        background: "#6f42c1",
        color: "#ffffff",
        fontSize: "11px",
        fontWeight: 600,
        lineHeight: "16px",
      }}
    >
      Evaluation
    </span>
  );
}

function MlflowButton({ appArn, runUrl }: { appArn: string; runUrl: string }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const handleOpen = async () => {
    setLoading(true);
    setError(false);
    try {
      const url = await createPresignedMlflowUrl(appArn);
      const fragment = runUrl.includes("#/")
        ? "#/" + runUrl.split("#/")[1]
        : "";
      const a = document.createElement("a");
      a.href = url + fragment;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.click();
    } catch (e) {
      console.error("MLflow presigned URL failed:", e);
      setError(true);
    } finally {
      setLoading(false);
    }
  };
  return (
    <Button
      variant="normal"
      loading={loading}
      onClick={() => void handleOpen()}
    >
      {error ? "Error — retry" : "View in MLflow"}
    </Button>
  );
}

export function AgentCardContent({
  job,
  section,
  onClick,
  onOpen,
  onDelete,
  mlflowAppArn,
}: Props) {
  const logLines = job.timeline
    .filter(
      (e): e is Extract<TimelineEntry, { kind: "message" }> =>
        e.kind === "message" && e.role === "assistant",
    )
    .map((e) => e.content);
  const [showDelete, setShowDelete] = useState(false);

  if (section === "header") {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: "8px",
        }}
        onMouseEnter={() => setShowDelete(true)}
        onMouseLeave={() => setShowDelete(false)}
        onFocusCapture={() => setShowDelete(true)}
        onBlurCapture={() => setShowDelete(false)}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            onClick={onClick ? () => onClick(job) : undefined}
            tabIndex={onClick ? 0 : undefined}
            onKeyDown={
              onClick
                ? (e: React.KeyboardEvent) => {
                    if (e.key === "Enter" || e.key === " ") onClick(job);
                  }
                : undefined
            }
            data-testid="card-header-click"
            style={onClick ? { cursor: "pointer" } : undefined}
          >
            <Box variant="h3" fontWeight="bold">
              {job.kind === "eval"
                ? job.evalConfig?.targetModel || "New Evaluation"
                : job.config.modelId || "New Task"}
            </Box>
          </div>
          <div
            style={{
              marginTop: "4px",
              display: "flex",
              gap: "6px",
              flexWrap: "wrap",
            }}
          >
            {job.kind === "eval" && <EvaluationBadge />}
            <Badge color={STATUS_COLOR[job.status]}>
              {STATUS_LABEL[job.status]}
            </Badge>
          </div>
        </div>
        {onDelete && (
          <div
            style={{
              opacity: showDelete ? 1 : 0,
              pointerEvents: showDelete ? "auto" : "none",
              transition: "opacity 0.15s",
              flexShrink: 0,
            }}
          >
            <Button
              variant="icon"
              iconName="remove"
              ariaLabel="Delete job"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(job);
              }}
            />
          </div>
        )}
      </div>
    );
  }

  const createdLabel = job.createdAt
    ? new Date(job.createdAt * 1000).toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : null;

  const summary =
    job.kind === "eval" ? (
      <JobSummaryEval job={job} />
    ) : (
      <JobSummaryTraining job={job} />
    );

  return (
    <SpaceBetween size="s">
      {summary}
      {createdLabel && (
        <Box color="text-body-secondary" fontSize="body-s">
          {createdLabel}
        </Box>
      )}
      {(() => {
        const toolNames = [
          ...new Set(
            job.timeline.flatMap((e) => (e.kind === "tool" ? [e.name] : [])),
          ),
        ];
        if (toolNames.length === 0) return null;
        return (
          <SpaceBetween size="xxs" direction="horizontal">
            {toolNames.map((tool) => (
              <Badge key={tool} color="grey">
                {tool}
              </Badge>
            ))}
          </SpaceBetween>
        );
      })()}
      {job.errorMessage && (
        <Box color="text-status-error" fontSize="body-s">
          {job.errorMessage}
        </Box>
      )}
      {logLines.length > 0 && <LogPanel messages={logLines} />}
      <SpaceBetween size="xs" direction="horizontal">
        {onOpen && (
          <Button variant="normal" onClick={() => onOpen(job)}>
            Open
          </Button>
        )}
        {job.mlflowRunUrl && mlflowAppArn && (
          <MlflowButton appArn={mlflowAppArn} runUrl={job.mlflowRunUrl} />
        )}
      </SpaceBetween>
    </SpaceBetween>
  );
}

function JobSummaryTraining({ job }: { job: AgentJob }) {
  return (
    <>
      <Box color="text-body-secondary" fontSize="body-s">
        {job.config.dataS3 || "—"}
      </Box>
      {job.config.instanceType && (
        <Box>
          <Badge color="blue">{job.config.instanceType}</Badge>
        </Box>
      )}
    </>
  );
}

// Eval-card body: dataset, judge model, and the scorer list as grey
// chips. Keeps parity with the training card's information density but
// exposes the fields a compliance reader scans for first.
function JobSummaryEval({ job }: { job: AgentJob }) {
  const ec = job.evalConfig;
  const datasetLabel = ec?.datasetS3Uri || job.config.dataS3 || "—";
  return (
    <>
      <Box color="text-body-secondary" fontSize="body-s">
        Dataset: {datasetLabel}
      </Box>
      {ec?.judgeModel && (
        <Box color="text-body-secondary" fontSize="body-s">
          Judge: {ec.judgeModel}
        </Box>
      )}
      {ec?.scorers && ec.scorers.length > 0 && (
        <SpaceBetween size="xxs" direction="horizontal">
          {ec.scorers.map((s) => (
            <Badge key={s} color="grey">
              {s}
            </Badge>
          ))}
        </SpaceBetween>
      )}
      {ec?.instanceType && (
        <Box>
          <Badge color="blue">{ec.instanceType}</Badge>
        </Box>
      )}
      {job.complianceReport && (
        <Box color="text-status-success" fontSize="body-s">
          Compliance report ready
        </Box>
      )}
    </>
  );
}
