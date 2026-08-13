import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import StatusIndicator from "@cloudscape-design/components/status-indicator";

interface Props {
  status: string | null;
  results: unknown;
  running: boolean;
  error: string | null;
  run: () => void;
}

// Triggers an AgentCore batch evaluation over recent sessions and polls to
// completion (the hook does the 5s poll). Results are the raw EvaluationJobResults.
export function BatchPanel({ status, results, running, error, run }: Props) {
  return (
    <Container
      header={
        <Header
          variant="h2"
          description="Score recent sessions on-demand against all evaluators"
          actions={
            <Button variant="primary" loading={running} onClick={run}>
              Run batch evaluation
            </Button>
          }
        >
          Batch evaluation
        </Header>
      }
    >
      <SpaceBetween size="s">
        {error && <Alert type="error">{error}</Alert>}
        {status && (
          <StatusIndicator
            type={
              status === "COMPLETED"
                ? "success"
                : status === "FAILED"
                  ? "error"
                  : "in-progress"
            }
          >
            {status}
          </StatusIndicator>
        )}
        {results != null && (
          <Box variant="code" fontSize="body-s">
            <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
              {JSON.stringify(results, null, 2).slice(0, 4000)}
            </pre>
          </Box>
        )}
        {!status && !running && (
          <Box color="text-body-secondary">
            Runs AgentCore StartBatchEvaluation over the agent's recent
            sessions.
          </Box>
        )}
      </SpaceBetween>
    </Container>
  );
}
