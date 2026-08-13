import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import Alert from "@cloudscape-design/components/alert";
import AppLayout from "@cloudscape-design/components/app-layout";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import Cards from "@cloudscape-design/components/cards";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Flashbar from "@cloudscape-design/components/flashbar";
import Header from "@cloudscape-design/components/header";
import Modal from "@cloudscape-design/components/modal";
import SideNavigation from "@cloudscape-design/components/side-navigation";
import Spinner from "@cloudscape-design/components/spinner";
import SpaceBetween from "@cloudscape-design/components/space-between";
import TopNavigation from "@cloudscape-design/components/top-navigation";
import { AgentCardContent } from "./components/AgentCard";
import { ArchitecturePage } from "./components/ArchitecturePage";
import { NewTaskPanel } from "./components/NewTaskPanel";
import { SkillDetail } from "./components/SkillDetail";
import { SkillsPage } from "./components/SkillsPage";
import { EvalsPage } from "./components/EvalsPage";
import { TaskDetailPanel } from "./components/TaskDetailPanel";
import {
  useJobsTable,
  getPrimaryJob,
  getCardDisplay,
  type JobEntry,
  type JobRecord,
} from "./hooks/useJobsTable";
import { useHashRoute } from "./lib/router";
import {
  type AuthConfig,
  getStoredToken,
  getUserDisplayName,
  handleCallback,
  initiateLogin,
  loadConfig,
  parseJwt,
  signOut,
} from "./lib/auth";
import type { AgentJob } from "./types";
import "./App.css";
import "./index.css";

function jobsReducer(
  state: AgentJob[],
  action: React.SetStateAction<AgentJob[]>,
): AgentJob[] {
  return typeof action === "function" ? action(state) : action;
}

type AuthState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "authenticated"; config: AuthConfig; token: string };

// Pull the eval-specific fields off a JobEntry. Returns undefined for
// training rows so the card/detail panels don't conditionally render an
// empty eval section.
function evalConfigFromEntry(entry: JobEntry): AgentJob["evalConfig"] {
  if (entry.kind !== "eval") return undefined;
  return {
    targetModel: entry.target_model ?? "",
    judgeModel: entry.judge_model ?? "",
    datasetS3Uri: entry.dataset_s3_uri ?? "",
    scorers: entry.scorers ?? [],
    task: entry.task ?? "",
    instanceType: entry.instance_type ?? "",
  };
}

function complianceReportFromEntry(
  entry: JobEntry,
): AgentJob["complianceReport"] {
  if (!entry.compliance_report_s3_uri) return undefined;
  return {
    s3Uri: entry.compliance_report_s3_uri,
    format: entry.compliance_report_format ?? "docx",
    at: entry.compliance_report_at ?? entry.updated_at ?? 0,
  };
}

// Surface the thread's primary (most-recent) job on the tile. Threads that
// have no submissions yet (chat-only turns, pre-submit clarifications)
// return null and are filtered out of the tile list.
function jobRecordToAgentJob(r: JobRecord): AgentJob | null {
  const entry = getPrimaryJob(r);
  if (!entry) {
    // No submitted job yet, but the thread has chat activity → it's a task
    // being prepared (the agent is running EDA / awaiting confirmation before
    // submit_eval_job or submit_training_job writes the jobs entry). Render a
    // "chatting" placeholder card so the in-flight task stays visible in the
    // Tasks view during that window instead of vanishing (a running task must
    // never disappear from the list). A truly empty row — no timeline — is a
    // stale/abandoned thread and is still filtered out.
    if (!r.timeline) return null;
    return {
      threadId: r.task_id,
      runId: r.task_id,
      kind: "training",
      config: { modelId: "", dataS3: "", instanceType: "", hyperparams: {} },
      status: "SUBMITTING",
      phase: "chatting",
      timeline: [],
      createdAt: r.created_at,
      updatedAt: r.updated_at,
    };
  }
  const statusMap: Record<string, AgentJob["status"]> = {
    PENDING: "PENDING",
    IN_PROGRESS: "IN_PROGRESS",
    COMPLETED: "COMPLETED",
    FAILED: "FAILED",
    STOPPED: "STOPPED",
  };
  const kind: AgentJob["kind"] = entry.kind === "eval" ? "eval" : "training";
  // Title/dataset/instance merge across ALL job entries (QA-02): the
  // primary (newest) entry drives status/links, but e.g. a bedrock_import
  // entry carries no display metadata and must not blank the card.
  const display = getCardDisplay(r);
  return {
    threadId: r.task_id,
    runId: entry.job_id,
    kind,
    config: {
      modelId: display.modelId,
      dataS3: display.dataset,
      instanceType: display.instanceType,
      hyperparams: {},
    },
    evalConfig: evalConfigFromEntry(entry),
    complianceReport: complianceReportFromEntry(entry),
    status: statusMap[entry.status ?? ""] ?? "PENDING",
    phase: "executing",
    timeline: [],
    mlflowRunUrl: entry.mlflow_run_url,
    sagemakerJobName: entry.sagemaker_job_name ?? entry.processing_job_name,
    errorMessage: entry.status === "FAILED" ? entry.status_message : undefined,
    statusMessage: entry.status_message,
    createdAt: entry.created_at ?? r.created_at,
    updatedAt: entry.updated_at ?? r.updated_at,
  };
}

export default function App() {
  const [auth, setAuth] = useState<AuthState>({ status: "loading" });
  const [jobs, dispatch] = useReducer(jobsReducer, []);
  const [navOpen, setNavOpen] = useState(true);
  const route = useHashRoute();
  const [newTaskOpen, setNewTaskOpen] = useState(false);
  const [selectedJob, setSelectedJob] = useState<AgentJob | null>(null);
  const [openedJob, setOpenedJob] = useState<AgentJob | null>(null);
  const [pendingDeleteJob, setPendingDeleteJob] = useState<AgentJob | null>(
    null,
  );
  const [deleteError, setDeleteError] = useState<string | null>(null);
  // Fresh threadId per NewTaskPanel open
  const newTaskThreadId = useRef<string>(crypto.randomUUID());

  useEffect(() => {
    async function init() {
      try {
        const config = await loadConfig();
        const devJwt = import.meta.env.VITE_JWT as string | undefined;
        if (devJwt) {
          setAuth({ status: "authenticated", config, token: devJwt });
          return;
        }
        const code = new URLSearchParams(window.location.search).get("code");
        if (code) {
          const token = await handleCallback(config, code);
          setAuth({ status: "authenticated", config, token });
          return;
        }
        const stored = getStoredToken();
        if (stored) {
          setAuth({ status: "authenticated", config, token: stored });
          return;
        }
        await initiateLogin(config);
      } catch (err) {
        setAuth({ status: "error", message: String(err) });
      }
    }
    void init();
  }, []);

  const agentCoreEndpoint =
    auth.status === "authenticated" ? auth.config.agentCoreEndpoint : "";

  const { jobs: dbJobs, deleteJob } = useJobsTable(
    auth.status === "authenticated",
  );
  const mlflowAppArn =
    auth.status === "authenticated" ? auth.config.mlflowAppArn : "";

  // One DynamoDB row = one thread = one tile. Rows without submitted jobs
  // (chat-only, pre-submit clarifications) return null and are filtered out.
  const liveThreadIds = new Set(jobs.map((j) => j.threadId));
  const historicalJobs = dbJobs
    .filter((r) => !liveThreadIds.has(r.task_id))
    .map(jobRecordToAgentJob)
    .filter((j): j is AgentJob => j !== null);
  const displayJobs = [...jobs, ...historicalJobs];

  // Stable ref so callbacks always see the latest jobs without stale closures
  const displayJobsRef = useRef(displayJobs);
  useEffect(() => {
    displayJobsRef.current = displayJobs;
  });

  // Stable ref so handleRunFinished can check DynamoDB state without stale closures
  const dbJobsRef = useRef(dbJobs);
  useEffect(() => {
    dbJobsRef.current = dbJobs;
  });

  useEffect(() => {
    if (dbJobs.length === 0) return;
    dispatch((prev) =>
      prev.map((job) => {
        const record = dbJobs.find((r) => r.task_id === job.threadId);
        if (!record) return job;
        // Match the job by runId (= job_id) first; fall back to the primary
        // job in case the in-memory job predates the submit that created it.
        const entry = (record.jobs ?? {})[job.runId] ?? getPrimaryJob(record);
        if (!entry) return job;
        const statusMap: Record<string, AgentJob["status"]> = {
          PENDING: "PENDING",
          IN_PROGRESS: "IN_PROGRESS",
          COMPLETED: "COMPLETED",
          FAILED: "FAILED",
          STOPPED: "STOPPED",
        };
        const kind: AgentJob["kind"] =
          entry.kind === "eval" ? "eval" : (job.kind ?? "training");
        return {
          ...job,
          runId: entry.job_id,
          kind,
          config: {
            modelId: entry.model_id ?? entry.target_model ?? job.config.modelId,
            dataS3:
              entry.dataset_name ?? entry.dataset_s3_uri ?? job.config.dataS3,
            instanceType: entry.instance_type ?? job.config.instanceType,
            hyperparams: job.config.hyperparams,
          },
          evalConfig: evalConfigFromEntry(entry) ?? job.evalConfig,
          complianceReport:
            complianceReportFromEntry(entry) ?? job.complianceReport,
          status: statusMap[entry.status ?? ""] ?? job.status,
          mlflowRunUrl: entry.mlflow_run_url ?? job.mlflowRunUrl,
          sagemakerJobName:
            entry.sagemaker_job_name ??
            entry.processing_job_name ??
            job.sagemakerJobName,
          errorMessage:
            entry.status === "FAILED"
              ? (entry.status_message ?? job.errorMessage)
              : undefined,
          statusMessage: entry.status_message ?? job.statusMessage,
          createdAt: entry.created_at ?? record.created_at ?? job.createdAt,
          updatedAt: entry.updated_at ?? record.updated_at ?? job.updatedAt,
        };
      }),
    );
  }, [dbJobs]);

  const handleNewTaskOpen = useCallback(() => {
    newTaskThreadId.current = crypto.randomUUID();
    setNewTaskOpen(true);
  }, []);

  // Only navigate when submit has written an entry under jobs.<job_id> for
  // this thread. A chat-only row (thread with empty jobs map) returns null
  // from jobRecordToAgentJob — stay in NewTaskPanel in that case.
  const handleRunFinished = useCallback((threadId: string) => {
    const dbRecord = dbJobsRef.current.find((r) => r.task_id === threadId);
    if (!dbRecord) return;
    const live = displayJobsRef.current.find((j) => j.threadId === threadId);
    const fromRecord = jobRecordToAgentJob(dbRecord);
    const target = live ?? fromRecord;
    if (!target) return;
    setOpenedJob(target);
    setNewTaskOpen(false);
    newTaskThreadId.current = crypto.randomUUID();
  }, []);

  // No-op: task cards are created only when submit writes a jobs entry.
  // Eagerly adding a card here caused premature navigation to TaskDetailPanel
  // before the agent had a chance to ask clarifying questions.
  const handleSubmit = useCallback((_text: string) => {}, []);

  // When NewTaskPanel is open and the thread row now has at least one
  // submitted job, navigate to its detail view automatically.
  useEffect(() => {
    if (!newTaskOpen) return;
    const threadId = newTaskThreadId.current;
    const dbRecord = dbJobs.find((r) => r.task_id === threadId);
    if (!dbRecord) return;
    const live = displayJobsRef.current.find((j) => j.threadId === threadId);
    const fromRecord = jobRecordToAgentJob(dbRecord);
    const target = live ?? fromRecord;
    if (!target) return;
    setOpenedJob(target);
    setNewTaskOpen(false);
    newTaskThreadId.current = crypto.randomUUID();
  }, [dbJobs, newTaskOpen]);

  const handleOpenJob = useCallback((job: AgentJob) => {
    setOpenedJob(job);
  }, []);

  const handleDeleteJob = useCallback((job: AgentJob) => {
    setPendingDeleteJob(job);
  }, []);

  const handleConfirmDelete = useCallback(async () => {
    if (!pendingDeleteJob) return;
    const job = pendingDeleteJob;
    setPendingDeleteJob(null);
    dispatch((prev) => prev.filter((j) => j.threadId !== job.threadId));
    if (selectedJob?.threadId === job.threadId) setSelectedJob(null);
    if (openedJob?.threadId === job.threadId) setOpenedJob(null);
    try {
      // Soft-delete the whole thread row (PK == threadId).
      await deleteJob(job.threadId);
    } catch {
      setDeleteError(
        `Failed to delete "${job.config.modelId || job.runId}". The job may reappear shortly.`,
      );
    }
  }, [pendingDeleteJob, deleteJob, selectedJob, openedJob]);

  if (auth.status === "loading") {
    return (
      <Box textAlign="center" padding={{ top: "xxxl" }}>
        <SpaceBetween size="m" direction="vertical" alignItems="center">
          <Spinner size="large" />
          <Box color="text-body-secondary">Authenticating…</Box>
        </SpaceBetween>
      </Box>
    );
  }

  if (auth.status === "error") {
    return (
      <Box padding="xl">
        <Alert
          type="error"
          header="Authentication failed"
          action={
            <Button onClick={() => window.location.reload()}>Try again</Button>
          }
        >
          {auth.message}
        </Alert>
      </Box>
    );
  }

  const claims = parseJwt(auth.token);

  // Keep openedJob in sync with the latest data from displayJobs
  const resolvedOpenedJob = openedJob
    ? (displayJobs.find((j) => j.threadId === openedJob.threadId) ?? openedJob)
    : null;

  return (
    <>
      {pendingDeleteJob && (
        <Modal
          visible
          onDismiss={() => setPendingDeleteJob(null)}
          header="Delete job?"
          footer={
            <Box float="right">
              <SpaceBetween direction="horizontal" size="xs">
                <Button
                  variant="link"
                  onClick={() => setPendingDeleteJob(null)}
                >
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  onClick={() => void handleConfirmDelete()}
                >
                  Delete
                </Button>
              </SpaceBetween>
            </Box>
          }
        >
          Are you sure you want to delete{" "}
          <strong>
            {pendingDeleteJob.config.modelId || pendingDeleteJob.runId}
          </strong>
          ? This cannot be undone.
        </Modal>
      )}
      <div id="app-top-nav">
        <TopNavigation
          identity={{
            href: "/",
            title: "Sample MLOps Agent",
            logo: { src: "/favicon.svg", alt: "Sample MLOps Agent" },
          }}
          utilities={[
            {
              type: "menu-dropdown",
              text: getUserDisplayName(claims),
              description: claims.email,
              iconName: "user-profile",
              items: [{ id: "signout", text: "Sign out" }],
              onItemClick: ({ detail }) => {
                if (detail.id === "signout") {
                  signOut(auth.config);
                }
              },
            },
          ]}
          i18nStrings={{
            overflowMenuTriggerText: "More",
            overflowMenuTitleText: "All",
          }}
        />
      </div>

      <AppLayout
        notifications={
          deleteError ? (
            <Flashbar
              items={[
                {
                  type: "error",
                  content: deleteError,
                  dismissible: true,
                  onDismiss: () => setDeleteError(null),
                  id: "delete-error",
                },
              ]}
            />
          ) : undefined
        }
        navigationOpen={navOpen}
        onNavigationChange={({ detail }) => setNavOpen(detail.open)}
        navigation={
          <SideNavigation
            header={{ text: "Sample MLOps Agent", href: "#/" }}
            activeHref={
              route.name === "skills" || route.name === "skill-detail"
                ? "#/skills"
                : route.name === "evals"
                  ? "#/evals"
                  : route.name === "architecture"
                    ? "#/architecture"
                    : "#/"
            }
            items={[
              { type: "link", text: "Tasks", href: "#/" },
              { type: "link", text: "Skills", href: "#/skills" },
              { type: "link", text: "Evals", href: "#/evals" },
              { type: "link", text: "Architecture", href: "#/architecture" },
            ]}
          />
        }
        toolsWidth={420}
        toolsHide={
          route.name !== "tasks" || !selectedJob || newTaskOpen || !!openedJob
        }
        toolsOpen={
          route.name === "tasks" && !!selectedJob && !newTaskOpen && !openedJob
        }
        onToolsChange={({ detail }) => {
          if (!detail.open) setSelectedJob(null);
        }}
        tools={
          selectedJob ? (
            <TaskDetailPanel
              job={selectedJob}
              agentCoreEndpoint={agentCoreEndpoint}
              mlflowAppArn={mlflowAppArn}
            />
          ) : null
        }
        content={
          route.name === "skills" ? (
            <SkillsPage
              sessionBucket={
                auth.status === "authenticated"
                  ? (auth.config.sessionBucket ?? "")
                  : ""
              }
              skillsPrefix={
                auth.status === "authenticated"
                  ? (auth.config.skillsPrefix ?? "skills/")
                  : "skills/"
              }
            />
          ) : route.name === "skill-detail" ? (
            <SkillDetail
              skillId={route.skillId}
              sessionBucket={
                auth.status === "authenticated"
                  ? (auth.config.sessionBucket ?? "")
                  : ""
              }
              skillsPrefix={
                auth.status === "authenticated"
                  ? (auth.config.skillsPrefix ?? "skills/")
                  : "skills/"
              }
            />
          ) : route.name === "evals" ? (
            <EvalsPage
              cfg={{
                agentRuntimeName:
                  auth.status === "authenticated"
                    ? (auth.config.agentRuntimeName ?? "sample_mlops_agent")
                    : "sample_mlops_agent",
                onlineEvalConfigName:
                  auth.status === "authenticated"
                    ? (auth.config.onlineEvalConfigName ?? "")
                    : "",
                evalResultsLogGroupPrefix:
                  auth.status === "authenticated"
                    ? (auth.config.evalResultsLogGroupPrefix ?? "")
                    : "",
                runtimeLogGroup:
                  auth.status === "authenticated"
                    ? (auth.config.runtimeLogGroup ?? "")
                    : "",
                runtimeLogGroupArn:
                  auth.status === "authenticated"
                    ? (auth.config.runtimeLogGroupArn ?? "")
                    : "",
              }}
              sessionBucket={
                auth.status === "authenticated"
                  ? (auth.config.sessionBucket ?? "")
                  : ""
              }
              skillsPrefix={
                auth.status === "authenticated"
                  ? (auth.config.skillsPrefix ?? "skills/")
                  : "skills/"
              }
            />
          ) : route.name === "architecture" ? (
            <ArchitecturePage />
          ) : newTaskOpen ? (
            <NewTaskPanel
              agentCoreEndpoint={agentCoreEndpoint}
              threadId={newTaskThreadId.current}
              onSubmit={handleSubmit}
              onRunFinished={handleRunFinished}
              onClose={() => setNewTaskOpen(false)}
            />
          ) : resolvedOpenedJob ? (
            <ContentLayout
              header={
                <Header
                  variant="h1"
                  description={
                    resolvedOpenedJob.config.modelId || resolvedOpenedJob.runId
                  }
                  actions={
                    <Button
                      iconName="arrow-left"
                      onClick={() => setOpenedJob(null)}
                    >
                      Back to Tasks
                    </Button>
                  }
                >
                  {resolvedOpenedJob.config.modelId || resolvedOpenedJob.runId}
                </Header>
              }
            >
              <TaskDetailPanel
                job={resolvedOpenedJob}
                agentCoreEndpoint={agentCoreEndpoint}
                mlflowAppArn={mlflowAppArn}
                initialPrompt={
                  resolvedOpenedJob.status === "COMPLETED"
                    ? `Summarize this training session for job "${resolvedOpenedJob.threadId}": what model was trained, on what dataset, with what configuration, what were the outcomes, and any MLflow metrics if available.`
                    : undefined
                }
              />
            </ContentLayout>
          ) : (
            <ContentLayout
              header={
                <Header
                  variant="h1"
                  description="Submit tasks and track experiments in MLflow."
                  actions={
                    <Button
                      variant="primary"
                      iconName="add-plus"
                      onClick={handleNewTaskOpen}
                    >
                      New Task
                    </Button>
                  }
                >
                  Tasks
                </Header>
              }
            >
              <Cards
                cardDefinition={{
                  header: (job) => (
                    <AgentCardContent
                      job={job}
                      section="header"
                      onClick={setSelectedJob}
                      onDelete={handleDeleteJob}
                    />
                  ),
                  sections: [
                    {
                      id: "body",
                      content: (job) => (
                        <AgentCardContent
                          job={job}
                          section="body"
                          onOpen={handleOpenJob}
                          mlflowAppArn={mlflowAppArn}
                        />
                      ),
                    },
                  ],
                }}
                items={displayJobs}
                trackBy="threadId"
                cardsPerRow={[
                  { cards: 1 },
                  { minWidth: 640, cards: 2 },
                  { minWidth: 1024, cards: 3 },
                  { minWidth: 1400, cards: 4 },
                ]}
                empty={
                  <Box
                    textAlign="center"
                    color="inherit"
                    padding={{ top: "xxxl", bottom: "l" }}
                  >
                    <SpaceBetween size="xs">
                      <Box variant="strong" color="inherit">
                        No tasks
                      </Box>
                      <Box color="text-body-secondary">
                        Click &ldquo;New Task&rdquo; to start your first task.
                      </Box>
                    </SpaceBetween>
                  </Box>
                }
              />
            </ContentLayout>
          )
        }
      />
    </>
  );
}
