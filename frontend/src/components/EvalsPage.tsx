import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import { useEffect, useState } from "react";
import { useSummary, useBatchEval, type EvalConfig } from "../hooks/useEvals";
import { loadSystemPrompt, saveSystemPrompt } from "../hooks/useSkillsStore";
import { SummaryPanel } from "./evals/SummaryPanel";
import { BatchPanel } from "./evals/BatchPanel";
import { RecommendationsPanel } from "./evals/RecommendationsPanel";

interface Props {
  cfg: EvalConfig;
  sessionBucket: string;
  skillsPrefix: string;
}

export function EvalsPage({ cfg, sessionBucket, skillsPrefix }: Props) {
  const summary = useSummary(7);
  const batch = useBatchEval();
  // The optimizer needs the current system prompt as its baseline to refine.
  // Load it from the same S3 object the Skills-tab editor writes; an empty
  // string surfaces a clear "set it first" error in the panel rather than a
  // raw API validation failure.
  const [currentSystemPrompt, setCurrentSystemPrompt] = useState("");
  useEffect(() => {
    void loadSystemPrompt(sessionBucket, skillsPrefix)
      .then(setCurrentSystemPrompt)
      .catch(() => setCurrentSystemPrompt(""));
  }, [sessionBucket, skillsPrefix]);

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="AgentCore evaluations — online metrics, on-demand batch scoring, and optimization."
        >
          Evals
        </Header>
      }
    >
      <SpaceBetween size="l">
        <SummaryPanel
          loading={summary.loading}
          error={summary.error}
          data={summary.data}
        />
        <BatchPanel
          status={batch.status}
          results={batch.results}
          running={batch.running}
          error={batch.error}
          run={batch.run}
        />
        <RecommendationsPanel
          cfg={cfg}
          currentSystemPrompt={currentSystemPrompt}
          onApplyPrompt={async (prompt) => {
            // Apply lands in the S3 skills store's system-prompt.md object so
            // the agent picks it up on its next turn (Task 2 hydration).
            await saveSystemPrompt(sessionBucket, skillsPrefix, prompt);
            setCurrentSystemPrompt(prompt);
          }}
        />
      </SpaceBetween>
    </ContentLayout>
  );
}
