import { useEffect, useRef, useState } from "react";
import Badge from "@cloudscape-design/components/badge";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import Link from "@cloudscape-design/components/link";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Spinner from "@cloudscape-design/components/spinner";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import type { StatusIndicatorProps } from "@cloudscape-design/components/status-indicator";
import { createPresignedMlflowUrl } from "../lib/mlflow";
import { createPresignedS3GetUrl } from "../lib/s3";
import { MarkdownMessage } from "./MarkdownMessage";
import { useChat } from "../hooks/useChat";
import { useChatHistory } from "../hooks/useChatHistory";
import type { AgentJob, JobStatus, TimelineEntry } from "../types";
import { ToolCallCard } from "./ToolCallCard";

// Region used to construct SageMaker console deep-links. Matches the
// convention used in hooks/useJobsTable.ts, hooks/useChatHistory.ts, lib/s3.ts.
const REGION = import.meta.env.VITE_AWS_REGION ?? "us-east-1";

/** Build a deep-link to the SageMaker console for a training or processing
 *  job. Training jobs live under `#/jobs/<name>`; Processing jobs under
 *  `#/processing-jobs/<name>`. */
function sagemakerConsoleUrl(jobName: string, kind: AgentJob["kind"]): string {
  const resource = kind === "eval" ? "processing-jobs" : "jobs";
  return `https://console.aws.amazon.com/sagemaker/home?region=${REGION}#/${resource}/${encodeURIComponent(jobName)}`;
}

function ComplianceDownloadButton({
  s3Uri,
  format,
}: {
  s3Uri: string;
  format: "md" | "docx";
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const handleDownload = async () => {
    setLoading(true);
    setError(false);
    try {
      const url = await createPresignedS3GetUrl(s3Uri);
      const a = document.createElement("a");
      a.href = url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.click();
    } catch (e) {
      console.error("Compliance report presigned URL failed:", e);
      setError(true);
    } finally {
      setLoading(false);
    }
  };
  return (
    <Button
      variant="primary"
      loading={loading}
      onClick={() => void handleDownload()}
    >
      {error ? "Error — retry" : `Download compliance report (.${format})`}
    </Button>
  );
}

const STATUS_TYPE: Record<JobStatus, StatusIndicatorProps["type"]> = {
  SUBMITTING: "loading",
  PENDING: "pending",
  IN_PROGRESS: "in-progress",
  COMPLETED: "success",
  FAILED: "error",
  STOPPED: "stopped",
};

interface Props {
  job: AgentJob;
  agentCoreEndpoint: string;
  mlflowAppArn?: string;
  initialPrompt?: string;
}

export function TaskDetailPanel({
  job,
  agentCoreEndpoint,
  mlflowAppArn,
  initialPrompt,
}: Props) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const hasSentInitial = useRef(false);
  const [mlflowLoading, setMlflowLoading] = useState(false);
  const [mlflowError, setMlflowError] = useState(false);

  const handleOpenMlflow = async () => {
    if (!mlflowAppArn || !job.mlflowRunUrl) return;
    setMlflowLoading(true);
    setMlflowError(false);
    try {
      const url = await createPresignedMlflowUrl(mlflowAppArn);
      const fragment = job.mlflowRunUrl.includes("#/")
        ? "#/" + job.mlflowRunUrl.split("#/")[1]
        : "";
      const a = document.createElement("a");
      a.href = url + fragment;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.click();
    } catch {
      setMlflowError(true);
    } finally {
      setMlflowLoading(false);
    }
  };

  const formatTs = (ts?: number) =>
    ts
      ? new Date(ts * 1000).toLocaleString(undefined, {
          year: "numeric",
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        })
      : null;

  const createdLabel = formatTs(job.createdAt);
  const updatedLabel = formatTs(job.updatedAt);

  const {
    timeline: liveTimeline,
    streaming,
    error,
    send,
    stop,
  } = useChat({
    agentCoreEndpoint,
    threadId: job.threadId,
  });

  const { timeline: storedTimeline, loading: historyLoading } = useChatHistory(
    job.threadId,
    true,
  );

  useEffect(() => {
    if (initialPrompt && !hasSentInitial.current) {
      hasSentInitial.current = true;
      void send(initialPrompt);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // storedTimeline is the source of truth for historical turns.
  // job.timeline holds in-memory entries from the current live session.
  // liveTimeline holds new entries from this page view.
  // De-duplicate by id so an entry never appears twice.
  const seen = new Set<string>();
  const timeline: TimelineEntry[] = [
    ...storedTimeline,
    ...job.timeline,
    ...liveTimeline,
  ].filter((e) => {
    if (seen.has(e.id)) return false;
    seen.add(e.id);
    return true;
  });

  useEffect(() => {
    if (scrollRef.current)
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [timeline.length]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || streaming) return;
    setInput("");
    await send(text);
  };

  return (
    <SpaceBetween size="l">
      {/* Metadata */}
      <ColumnLayout columns={2} borders="horizontal">
        {job.kind === "eval" ? (
          <>
            <Box>
              <Box variant="awsui-key-label">Target model</Box>
              <Box>{job.evalConfig?.targetModel || "—"}</Box>
            </Box>
            <Box>
              <Box variant="awsui-key-label">Status</Box>
              <StatusIndicator type={STATUS_TYPE[job.status]}>
                {job.status}
              </StatusIndicator>
            </Box>
            <Box>
              <Box variant="awsui-key-label">Judge model</Box>
              <Box>{job.evalConfig?.judgeModel || "—"}</Box>
            </Box>
            <Box>
              <Box variant="awsui-key-label">Instance</Box>
              {job.evalConfig?.instanceType ? (
                <Badge color="blue">{job.evalConfig.instanceType}</Badge>
              ) : (
                <Box>—</Box>
              )}
            </Box>
            <Box>
              <Box variant="awsui-key-label">Dataset</Box>
              <Box>{job.evalConfig?.datasetS3Uri || "—"}</Box>
            </Box>
            {job.evalConfig?.task && (
              <Box>
                <Box variant="awsui-key-label">Task</Box>
                <Badge color="grey">{job.evalConfig.task}</Badge>
              </Box>
            )}
            {job.evalConfig?.scorers && job.evalConfig.scorers.length > 0 && (
              <Box>
                <Box variant="awsui-key-label">Scorers</Box>
                <SpaceBetween size="xxs" direction="horizontal">
                  {job.evalConfig.scorers.map((s) => (
                    <Badge key={s} color="grey">
                      {s}
                    </Badge>
                  ))}
                </SpaceBetween>
              </Box>
            )}
            {job.sagemakerJobName && (
              <Box>
                <Box variant="awsui-key-label">Processing job</Box>
                <Link
                  href={sagemakerConsoleUrl(job.sagemakerJobName, "eval")}
                  external
                  externalIconAriaLabel="Opens in the SageMaker console"
                >
                  {job.sagemakerJobName}
                </Link>
              </Box>
            )}
          </>
        ) : (
          <>
            <Box>
              <Box variant="awsui-key-label">Model</Box>
              <Box>{job.config.modelId || "—"}</Box>
            </Box>
            <Box>
              <Box variant="awsui-key-label">Status</Box>
              <StatusIndicator type={STATUS_TYPE[job.status]}>
                {job.status}
              </StatusIndicator>
            </Box>
            <Box>
              <Box variant="awsui-key-label">Dataset</Box>
              <Box>{job.config.dataS3 || "—"}</Box>
            </Box>
            <Box>
              <Box variant="awsui-key-label">Instance</Box>
              {job.config.instanceType ? (
                <Badge color="blue">{job.config.instanceType}</Badge>
              ) : (
                <Box>—</Box>
              )}
            </Box>
            {Object.keys(job.config.hyperparams).length > 0 && (
              <Box>
                <Box variant="awsui-key-label">Hyperparameters</Box>
                <SpaceBetween size="xxs" direction="horizontal">
                  {Object.entries(job.config.hyperparams).map(([k, v]) => (
                    <Badge key={k} color="grey">
                      {k}: {v}
                    </Badge>
                  ))}
                </SpaceBetween>
              </Box>
            )}
            {job.sagemakerJobName && (
              <Box>
                <Box variant="awsui-key-label">Training job</Box>
                <Link
                  href={sagemakerConsoleUrl(job.sagemakerJobName, "training")}
                  external
                  externalIconAriaLabel="Opens in the SageMaker console"
                >
                  {job.sagemakerJobName}
                </Link>
              </Box>
            )}
          </>
        )}
        {createdLabel && (
          <Box>
            <Box variant="awsui-key-label">Created</Box>
            <Box>{createdLabel}</Box>
          </Box>
        )}
        {updatedLabel && (
          <Box>
            <Box variant="awsui-key-label">Updated</Box>
            <Box>{updatedLabel}</Box>
          </Box>
        )}
        {job.statusMessage && (
          <Box>
            <Box variant="awsui-key-label">
              {job.status === "FAILED" ? "Error" : "Message"}
            </Box>
            <Box color="text-body-secondary" fontSize="body-s">
              {job.statusMessage}
            </Box>
          </Box>
        )}
      </ColumnLayout>

      <SpaceBetween size="xs" direction="horizontal">
        {job.mlflowRunUrl && mlflowAppArn && (
          <Button
            variant="normal"
            loading={mlflowLoading}
            onClick={() => void handleOpenMlflow()}
          >
            {mlflowError ? "Error — retry" : "View in MLflow"}
          </Button>
        )}
        {job.complianceReport && (
          <ComplianceDownloadButton
            s3Uri={job.complianceReport.s3Uri}
            format={job.complianceReport.format}
          />
        )}
      </SpaceBetween>

      {/* Timeline */}
      {historyLoading && timeline.length === 0 && (
        <div
          style={{
            padding: "12px 0",
            color: "var(--color-text-body-secondary)",
          }}
        >
          <Spinner size="normal" /> Loading conversation history…
        </div>
      )}
      {timeline.length > 0 && (
        <div
          ref={scrollRef}
          style={{
            maxHeight: "50vh",
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: "12px",
          }}
        >
          {timeline.map((e) => {
            if (e.kind === "tool") return <ToolCallCard key={e.id} entry={e} />;
            return (
              <div
                key={e.id}
                style={{
                  display: "flex",
                  justifyContent: e.role === "user" ? "flex-end" : "flex-start",
                }}
              >
                {e.role === "user" ? (
                  <div
                    style={{
                      background:
                        "var(--color-background-notification-blue, #0972d3)",
                      color: "#ffffff",
                      borderRadius: "8px",
                      padding: "10px 14px",
                      maxWidth: "80%",
                      fontSize: "14px",
                      lineHeight: "22px",
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                    }}
                  >
                    {e.content}
                  </div>
                ) : (
                  <div style={{ maxWidth: "80%" }}>
                    <MarkdownMessage content={e.content} />
                  </div>
                )}
              </div>
            );
          })}
          {streaming && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                padding: "6px 0",
                color: "var(--color-text-body-secondary, #687078)",
              }}
            >
              <Spinner size="normal" />
              <span style={{ fontSize: "12px" }}>Thinking…</span>
            </div>
          )}
        </div>
      )}

      {error && <Box color="text-status-error">{error}</Box>}

      {/* Follow-up input */}
      <div style={{ display: "flex", gap: "8px", alignItems: "flex-end" }}>
        <div style={{ flex: 1 }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void handleSend();
              }
            }}
            placeholder="Ask a follow-up…"
            disabled={streaming}
            rows={2}
            style={{
              width: "100%",
              padding: "8px 12px",
              border: "1px solid var(--color-border-control-default, #adb5bd)",
              borderRadius: "4px",
              fontFamily: "inherit",
              fontSize: "14px",
              resize: "none",
              boxSizing: "border-box",
            }}
          />
        </div>
        {streaming ? (
          <Button variant="normal" onClick={stop}>
            Stop
          </Button>
        ) : (
          <Button
            variant="primary"
            onClick={() => void handleSend()}
            disabled={!input.trim()}
          >
            Send
          </Button>
        )}
      </div>
    </SpaceBetween>
  );
}
