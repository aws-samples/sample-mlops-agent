// AWS SDK client factories for the Evals tab, built with Cognito Identity Pool
// credentials (same pattern as lib/s3 + useJobsTable). Batch eval and
// recommendation live on the DATA-PLANE bedrock-agentcore client (verified via
// `aws bedrock-agentcore help`), NOT the -control client.
import { CloudWatchClient } from "@aws-sdk/client-cloudwatch";
import {
  CloudWatchLogsClient,
  StartQueryCommand,
  GetQueryResultsCommand,
} from "@aws-sdk/client-cloudwatch-logs";
import { BedrockAgentCoreClient } from "@aws-sdk/client-bedrock-agentcore";
import { getAWSCredentials } from "./credentials";

const REGION = import.meta.env.VITE_AWS_REGION ?? "us-east-1";

async function creds() {
  const c = await getAWSCredentials();
  if (!c) throw new Error("Not authenticated");
  return c;
}

export async function cloudWatch(): Promise<CloudWatchClient> {
  return new CloudWatchClient({ region: REGION, credentials: await creds() });
}

export async function bedrockAgentCore(): Promise<BedrockAgentCoreClient> {
  return new BedrockAgentCoreClient({
    region: REGION,
    credentials: await creds(),
  });
}

// Run a CloudWatch Logs Insights query and poll to completion. Returns the
// result rows as arrays of {field,value}. Used to read per-session eval
// results from the service-generated results log group.
export async function runLogsInsights(
  logGroupNames: string[],
  queryString: string,
  startMs: number,
  endMs: number,
): Promise<Record<string, string>[]> {
  const client = new CloudWatchLogsClient({
    region: REGION,
    credentials: await creds(),
  });
  const { queryId } = await client.send(
    new StartQueryCommand({
      logGroupNames,
      startTime: Math.floor(startMs / 1000),
      endTime: Math.floor(endMs / 1000),
      queryString,
      limit: 1000,
    }),
  );
  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const res = await client.send(new GetQueryResultsCommand({ queryId }));
    if (res.status === "Complete") {
      return (res.results ?? []).map((row) => {
        const obj: Record<string, string> = {};
        for (const f of row) if (f.field) obj[f.field] = f.value ?? "";
        return obj;
      });
    }
    if (res.status === "Failed" || res.status === "Cancelled") {
      throw new Error(`Logs Insights query ${res.status}`);
    }
  }
  throw new Error("Logs Insights query timed out");
}
