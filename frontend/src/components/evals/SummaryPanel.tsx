import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import Spinner from "@cloudscape-design/components/spinner";
import type { SummaryEntry } from "../../hooks/useEvals";

interface Props {
  loading: boolean;
  error: string | null;
  data: Record<string, SummaryEntry>;
}

// 7-day rolling metric tiles per evaluator. Online eval metrics land in the
// Bedrock-AgentCore/Evaluations namespace once Transaction Search has indexed
// live sessions; until then every tile shows a zero count → empty state.
// An expired Cognito session surfaces as a NotAuthorized / "Not authenticated"
// error while getAWSCredentials redirects to the Hosted UI. Show a calm
// "signing you back in" message rather than a raw AWS exception.
function isSessionExpiredError(error: string | null): boolean {
  return (
    !!error &&
    /Not authenticated|NotAuthorized|Token expired|Invalid login token/i.test(
      error,
    )
  );
}

export function SummaryPanel({ loading, error, data }: Props) {
  const entries = Object.entries(data);
  const anyScored = entries.some(([, v]) => v.count > 0);
  const sessionExpired = isSessionExpiredError(error);

  return (
    <Container
      header={
        <Header
          variant="h2"
          description="7-day rolling averages from live sessions"
        >
          Online eval metrics
        </Header>
      }
    >
      {error &&
        (sessionExpired ? (
          <Alert type="info" header="Session expired">
            Your session expired — signing you back in…
          </Alert>
        ) : (
          <Alert type="error">{error}</Alert>
        ))}
      {loading ? (
        <Box textAlign="center" padding="l">
          <Spinner size="large" />
        </Box>
      ) : !anyScored ? (
        <Alert type="info" header="No sessions scored yet">
          No online-eval datapoints yet. Ensure CloudWatch Transaction Search is
          enabled and drive a few agent sessions — metrics appear within a few
          minutes of session completion.
        </Alert>
      ) : (
        <ColumnLayout columns={Math.min(entries.length, 4)} variant="text-grid">
          {entries.map(([name, v]) => (
            <div key={name}>
              <Box variant="awsui-key-label">{name}</Box>
              <Box variant="awsui-value-large">
                {v.avg === null ? "—" : v.avg.toFixed(2)}
              </Box>
              <Box color="text-body-secondary" fontSize="body-s">
                {v.count} session{v.count === 1 ? "" : "s"}
              </Box>
            </div>
          ))}
        </ColumnLayout>
      )}
    </Container>
  );
}
