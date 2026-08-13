import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SegmentedControl from "@cloudscape-design/components/segmented-control";
import SpaceBetween from "@cloudscape-design/components/space-between";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import { useState } from "react";
import {
  startBatchEval,
  getBatchEval,
  startRecommendation,
  getRecommendation,
  DEFAULT_EVALUATORS,
  type EvalConfig,
  type RecommendationType,
} from "../../hooks/useEvals";
import { gatewayTools } from "../../generated/gatewayTools";

interface Props {
  cfg: EvalConfig;
  currentSystemPrompt: string;
  onApplyPrompt: (prompt: string) => void;
}

// AgentCore optimization. System-prompt path needs a batch eval as its trace
// source (multi-step, client-polled so no server timeout). Tool-description
// path goes straight to StartRecommendation.
export function RecommendationsPanel({
  cfg,
  currentSystemPrompt,
  onApplyPrompt,
}: Props) {
  const [type, setType] = useState<RecommendationType>(
    "SYSTEM_PROMPT_RECOMMENDATION",
  );
  const [phase, setPhase] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendedPrompt, setRecommendedPrompt] = useState<string | null>(
    null,
  );

  async function poll<T>(
    fn: () => Promise<T>,
    done: (r: T) => boolean,
    ceilMs: number,
  ): Promise<T> {
    const deadline = Date.now() + ceilMs;
    for (;;) {
      await new Promise((r) => setTimeout(r, 5000));
      const r = await fn();
      if (done(r)) return r;
      if (Date.now() > deadline) throw new Error("Timed out");
    }
  }

  async function generate() {
    setRunning(true);
    setError(null);
    setRecommendedPrompt(null);
    try {
      let batchArn: string | undefined;
      if (type === "SYSTEM_PROMPT_RECOMMENDATION") {
        // Guard BEFORE the batch eval — an empty prompt is rejected by
        // StartRecommendation anyway, and the batch run takes minutes. Fail
        // fast and point the user at where to set the prompt.
        if (!currentSystemPrompt.trim())
          throw new Error(
            "No system prompt is set — add one in the Skills tab before optimizing.",
          );
        setPhase("Running batch evaluation (trace source)…");
        const started = await startBatchEval(cfg, DEFAULT_EVALUATORS);
        if (!started.batchEvaluationId) throw new Error("No batch id");
        const done = await poll(
          () => getBatchEval(started.batchEvaluationId!),
          (r) => !!r.status && !["IN_PROGRESS", "STARTING"].includes(r.status),
          5 * 60_000,
        );
        // StartRecommendation needs the batch's ARN, not its id/name. Prefer the
        // ARN from the terminal Get; fall back to the Start response.
        batchArn = done.batchEvaluationArn ?? started.batchEvaluationArn;
        if (!batchArn)
          throw new Error("Batch evaluation returned no ARN for the optimizer");
      }
      setPhase("Generating recommendation…");
      const { recommendationId } = await startRecommendation(cfg, type, {
        currentPrompt: currentSystemPrompt,
        batchEvaluationArn: batchArn,
        // Tool-description optimization requires the current tool set (the API
        // rejects an empty list). Supply the gateway tools baked in at build
        // time from the skill tool specs — the SPA has no live gateway access.
        tools:
          type === "TOOL_DESCRIPTION_RECOMMENDATION"
            ? gatewayTools.map((t) => ({
                toolName: t.toolName,
                toolDescription: { text: t.description },
              }))
            : undefined,
      });
      if (!recommendationId) throw new Error("No recommendation id");
      const rec = await poll(
        () => getRecommendation(recommendationId),
        (r) => !!r.status && ["COMPLETED", "FAILED"].includes(r.status),
        10 * 60_000,
      );
      if (rec.status === "FAILED")
        throw new Error(
          rec.errorMessage
            ? `Recommendation failed: ${rec.errorMessage}`
            : "Recommendation failed",
        );
      const result = rec.recommendationResult as
        | {
            systemPromptRecommendationResult?: {
              recommendedSystemPrompt?: string;
            };
          }
        | undefined;
      const prompt =
        result?.systemPromptRecommendationResult?.recommendedSystemPrompt;
      if (prompt) setRecommendedPrompt(prompt);
      setPhase("Done");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <Container
      header={
        <Header
          variant="h2"
          description="AgentCore prompt / tool-description optimization from traces"
        >
          Optimization
        </Header>
      }
    >
      <SpaceBetween size="s">
        <SegmentedControl
          selectedId={type}
          onChange={({ detail }) =>
            setType(detail.selectedId as RecommendationType)
          }
          options={[
            { id: "SYSTEM_PROMPT_RECOMMENDATION", text: "System Prompt" },
            {
              id: "TOOL_DESCRIPTION_RECOMMENDATION",
              text: "Tool Descriptions",
            },
          ]}
        />
        <Button variant="primary" loading={running} onClick={generate}>
          Generate recommendation
        </Button>
        {error && <Alert type="error">{error}</Alert>}
        {phase && !error && (
          <StatusIndicator type={running ? "in-progress" : "success"}>
            {phase}
          </StatusIndicator>
        )}
        {recommendedPrompt && (
          <SpaceBetween size="xs">
            <Box variant="code" fontSize="body-s">
              <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                {recommendedPrompt}
              </pre>
            </Box>
            <Button onClick={() => onApplyPrompt(recommendedPrompt)}>
              Apply to system prompt →
            </Button>
          </SpaceBetween>
        )}
      </SpaceBetween>
    </Container>
  );
}
