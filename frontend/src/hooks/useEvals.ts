// Evals data functions + React hooks. Metrics come from CloudWatch
// GetMetricData over the Bedrock-AgentCore/Evaluations namespace (metric name =
// evaluator name, dimensioned by service.name = <runtime>.DEFAULT — NOT
// "bedrock-agentcore", per recon's post-mortem). Batch eval + recommendation
// use the data-plane bedrock-agentcore client.
import { useCallback, useEffect, useState } from "react";
import { GetMetricDataCommand } from "@aws-sdk/client-cloudwatch";
import {
  StartBatchEvaluationCommand,
  GetBatchEvaluationCommand,
  StartRecommendationCommand,
  GetRecommendationCommand,
  type RecommendationConfig,
} from "@aws-sdk/client-bedrock-agentcore";
import { cloudWatch, bedrockAgentCore } from "../lib/evals";
import { loadConfig, type AuthConfig } from "../lib/auth";

export const DEFAULT_EVALUATORS = [
  "Builtin.GoalSuccessRate",
  "Builtin.Helpfulness",
  "Builtin.Correctness",
];

export interface EvalConfig {
  agentRuntimeName: string;
  onlineEvalConfigName: string;
  evalResultsLogGroupPrefix: string;
  // Real runtime app log group (/runtimes/<runtimeId>-DEFAULT) — spans + LLO
  // events live here; the batch data source must read from it.
  runtimeLogGroup: string;
  // Its ARN — the tool-description recommendation API requires logGroupArns
  // with >=1 entry.
  runtimeLogGroupArn: string;
}

export interface SummaryEntry {
  avg: number | null;
  count: number;
}

function serviceName(runtime: string): string {
  return `${runtime}.DEFAULT`;
}

// One GetMetricData query per evaluator: an average series + a sample-count
// series, both SEARCH()ed over the evaluations namespace filtered to the
// runtime's service. Returns { evaluatorName: {avg, count} }.
export async function fetchSummary(
  evaluators: string[],
  days: number,
  runtime = "sample_mlops_agent",
): Promise<Record<string, SummaryEntry>> {
  const cw = await cloudWatch();
  const svc = serviceName(runtime);
  const now = Date.now();
  // Pin the service.name dimension to THIS runtime's value (recon post-mortem:
  // the real value is "<runtime>.DEFAULT", not "bedrock-agentcore"); without
  // the value the SEARCH would aggregate across every service.
  const queries = evaluators.flatMap((ev, i) => [
    {
      Id: `avg_${i}`,
      Expression: `SEARCH('{Bedrock-AgentCore/Evaluations,"service.name"} MetricName="${ev}" "service.name"="${svc}"', 'Average', 86400)`,
      ReturnData: true,
    },
    {
      Id: `cnt_${i}`,
      Expression: `SEARCH('{Bedrock-AgentCore/Evaluations,"service.name"} MetricName="${ev}" "service.name"="${svc}"', 'SampleCount', 86400)`,
      ReturnData: true,
    },
  ]);
  const res = await cw.send(
    new GetMetricDataCommand({
      MetricDataQueries: queries,
      StartTime: new Date(now - days * 86400_000),
      EndTime: new Date(now),
    }),
  );
  const byId: Record<string, { values: number[] }> = {};
  for (const r of res.MetricDataResults ?? []) {
    if (r.Id) byId[r.Id] = { values: (r.Values ?? []) as number[] };
  }
  const out: Record<string, SummaryEntry> = {};
  evaluators.forEach((ev, i) => {
    const avgVals = byId[`avg_${i}`]?.values ?? [];
    const cntVals = byId[`cnt_${i}`]?.values ?? [];
    const count = cntVals.reduce((a, b) => a + b, 0);
    const avg =
      avgVals.length > 0
        ? avgVals.reduce((a, b) => a + b, 0) / avgVals.length
        : null;
    out[ev] = { avg, count };
  });
  return out;
}

export async function startBatchEval(
  cfg: EvalConfig,
  evaluators: string[],
): Promise<{ batchEvaluationId?: string; batchEvaluationArn?: string }> {
  const bac = await bedrockAgentCore();
  const res = await bac.send(
    new StartBatchEvaluationCommand({
      // Name must match [a-zA-Z][a-zA-Z0-9_]{0,47} — no hyphens.
      batchEvaluationName: `ui_batch_${Date.now()}`,
      evaluators: evaluators.map((evaluatorId) => ({ evaluatorId })),
      dataSourceConfig: {
        cloudWatchLogs: {
          serviceNames: [serviceName(cfg.agentRuntimeName)],
          logGroupNames: [
            "aws/spans",
            // Real app log group holding spans + LLO events (from config.json).
            cfg.runtimeLogGroup,
          ],
        },
      },
    }),
  );
  return {
    batchEvaluationId: res.batchEvaluationId,
    batchEvaluationArn: res.batchEvaluationArn,
  };
}

export async function getBatchEval(id: string): Promise<{
  status?: string;
  evaluationResults?: unknown;
  errorDetails?: string[];
  // The full ARN — StartRecommendation's batchEvaluation trace source requires
  // the ARN, NOT the id/name (which fails with "not a valid ARN").
  batchEvaluationArn?: string;
}> {
  const bac = await bedrockAgentCore();
  const res = await bac.send(
    new GetBatchEvaluationCommand({ batchEvaluationId: id }),
  );
  return {
    status: res.status,
    evaluationResults: res.evaluationResults,
    errorDetails: res.errorDetails,
    batchEvaluationArn: res.batchEvaluationArn,
  };
}

export type RecommendationType =
  "SYSTEM_PROMPT_RECOMMENDATION" | "TOOL_DESCRIPTION_RECOMMENDATION";

export async function startRecommendation(
  cfg: EvalConfig,
  type: RecommendationType,
  opts: {
    currentPrompt?: string;
    batchEvaluationArn?: string;
    tools?: { toolName: string; toolDescription: { text: string } }[];
  },
): Promise<{ recommendationId?: string }> {
  const bac = await bedrockAgentCore();
  // Build the tagged-union member explicitly per branch — a shared ternary
  // widens both keys together and TS then can't narrow to a single member.
  let recommendationConfig: RecommendationConfig;
  if (type === "SYSTEM_PROMPT_RECOMMENDATION") {
    // The API rejects an empty systemPrompt.text ("Member must have length
    // greater than or equal to 1"). The optimizer refines an EXISTING prompt,
    // so the caller must set one first (Skills tab → System prompt).
    const prompt = opts.currentPrompt?.trim();
    if (!prompt)
      throw new Error(
        "No system prompt is set — add one in the Skills tab before optimizing.",
      );
    recommendationConfig = {
      systemPromptRecommendationConfig: {
        systemPrompt: { text: prompt },
        agentTraces: {
          batchEvaluation: {
            batchEvaluationArn: opts.batchEvaluationArn ?? "",
          },
        },
        evaluationConfig: {
          evaluators: [
            {
              evaluatorArn:
                "arn:aws:bedrock-agentcore:::evaluator/Builtin.GoalSuccessRate",
            },
          ],
        },
      },
    };
  } else {
    // The API rejects an empty tools list ("At least one tool is required"),
    // so the caller MUST supply the tools to optimize — there is no
    // infer-from-traces mode. The panel passes the build-time gatewayTools
    // manifest (generated from lambda/skills/*/tool_spec.json).
    if (!opts.tools?.length)
      throw new Error(
        "tool-description recommendation requires a non-empty tools list",
      );
    recommendationConfig = {
      toolDescriptionRecommendationConfig: {
        toolDescription: { toolDescriptionText: { tools: opts.tools } },
        agentTraces: {
          cloudwatchLogs: {
            // Must be non-empty (API requires >=1 ARN); use the real runtime
            // log group where spans + LLO events land.
            logGroupArns: [cfg.runtimeLogGroupArn],
            serviceNames: [serviceName(cfg.agentRuntimeName)],
            startTime: new Date(Date.now() - 7 * 86400_000),
            endTime: new Date(),
          },
        },
      },
    };
  }
  const res = await bac.send(
    new StartRecommendationCommand({
      name: `ui-rec-${Date.now()}`,
      type,
      recommendationConfig,
    }),
  );
  return { recommendationId: res.recommendationId };
}

export async function getRecommendation(id: string): Promise<{
  status?: string;
  recommendationResult?: unknown;
  // Failure details live NESTED under the per-type result member (not at the
  // response top level), e.g. "No sessions were identified from input agent
  // traces." Surface it so the UI can show the real reason, not just "failed".
  errorMessage?: string;
}> {
  const bac = await bedrockAgentCore();
  const res = await bac.send(
    new GetRecommendationCommand({ recommendationId: id }),
  );
  const result = res.recommendationResult as
    | {
        systemPromptRecommendationResult?: { errorMessage?: string };
        toolDescriptionRecommendationResult?: { errorMessage?: string };
      }
    | undefined;
  const errorMessage =
    result?.systemPromptRecommendationResult?.errorMessage ??
    result?.toolDescriptionRecommendationResult?.errorMessage;
  return {
    status: res.status,
    recommendationResult: res.recommendationResult,
    errorMessage,
  };
}

// ── React hooks over the functions above ─────────────────────────────────────

function evalConfigFrom(c: AuthConfig): EvalConfig {
  return {
    agentRuntimeName: c.agentRuntimeName ?? "sample_mlops_agent",
    onlineEvalConfigName: c.onlineEvalConfigName ?? "",
    evalResultsLogGroupPrefix: c.evalResultsLogGroupPrefix ?? "",
    runtimeLogGroup: c.runtimeLogGroup ?? "",
    runtimeLogGroupArn: c.runtimeLogGroupArn ?? "",
  };
}

export function useSummary(days = 7) {
  const [data, setData] = useState<Record<string, SummaryEntry>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const cfg = evalConfigFrom(await loadConfig());
      setData(
        await fetchSummary(DEFAULT_EVALUATORS, days, cfg.agentRuntimeName),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [days]);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  return { data, loading, error, refresh };
}

export function useBatchEval() {
  const [status, setStatus] = useState<string | null>(null);
  const [results, setResults] = useState<unknown>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    setResults(null);
    try {
      const cfg = evalConfigFrom(await loadConfig());
      const { batchEvaluationId } = await startBatchEval(
        cfg,
        DEFAULT_EVALUATORS,
      );
      if (!batchEvaluationId) throw new Error("No batch id returned");
      const deadline = Date.now() + 5 * 60_000;
      for (;;) {
        setStatus("IN_PROGRESS");
        await new Promise((r) => setTimeout(r, 5000));
        const got = await getBatchEval(batchEvaluationId);
        if (got.status && !["IN_PROGRESS", "STARTING"].includes(got.status)) {
          setStatus(got.status);
          setResults(got.evaluationResults ?? null);
          break;
        }
        if (Date.now() > deadline)
          throw new Error("Batch evaluation timed out");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, []);

  return { status, results, running, error, run };
}
