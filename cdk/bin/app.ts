#!/usr/bin/env node
import "source-map-support/register";
import * as dotenv from "dotenv";
dotenv.config();
import * as cdk from "aws-cdk-lib";
import { AgentCoreStack } from "../lib/stacks/backend/agentcore-stack";
import { ApiStack } from "../lib/stacks/backend/api-stack";
import { EvalsStack } from "../lib/stacks/backend/evals-stack";
import { GatewayStack } from "../lib/stacks/backend/gateway-stack";
import { MlflowStack } from "../lib/stacks/backend/mlflow-stack";
import { ScienceAgentStack } from "../lib/stacks/backend/science-agent-stack";
import { ScienceAgentUiStack } from "../lib/stacks/frontend/science-agent-ui-stack";

const app = new cdk.App();
const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION ?? "us-east-1",
};
const contextProjectName = app.node.tryGetContext("projectName") as string;
if (!contextProjectName) {
  console.warn(
    'WARNING: projectName not set in CDK context. Using fallback "sample-mlops-agent". Set it in cdk.json context.',
  );
}
const projectName = contextProjectName ?? "sample-mlops-agent";
const environment = (app.node.tryGetContext("environment") as string) ?? "dev";

// ── Tier 1: core infrastructure + Docker builds ─────────────────────────────
const agentStack = new ScienceAgentStack(app, `${projectName}-agent`, {
  env,
  projectName,
});

// ── Tier 2: MLflow + AgentCore Runtime (can deploy in parallel) ──────────────
const mlflowStack = new MlflowStack(app, `${projectName}-mlflow`, {
  env,
  artifactBucket: agentStack.sessionBucket,
  projectName,
  environment,
});
mlflowStack.addDependency(agentStack);

const agentCoreStack = new AgentCoreStack(app, `${projectName}-agentcore`, {
  env,
  agentRepo: agentStack.agentRepo,
  agentImageTag: agentStack.agentImageTag,
  agentBuildCompletion: agentStack.agentBuildCompletion,
  agentCoreRole: agentStack.agentCoreRole,
  metadataTable: agentStack.metadataTable,
  sessionBucket: agentStack.sessionBucket,
  mlflowSsmParamPath: `/${projectName}/${environment}/mlflow/tracking-server-arn`,
  projectName,
  sageMakerExecutionRoleArn: agentStack.sageMakerExecutionRole.roleArn,
});
agentCoreStack.addDependency(agentStack);

// ── Tier 2b: AgentCore Evaluations (evaluator + online config + txn search) ──
const runtimeName = projectName.replace(/-/g, "_");
const evalsStack = new EvalsStack(app, `${projectName}-evals`, {
  env,
  projectName,
  runtimeName,
  // Real app log group is /runtimes/<runtimeId>-DEFAULT (not /runtimes/<name>);
  // sourced from the agentcore stack so spans + LLO events are actually read.
  runtimeLogGroupName: agentCoreStack.runtimeLogGroupName,
});
evalsStack.addDependency(agentCoreStack);

// ── Tier 3: Cognito + Identity Pool ───────────────────────────────────────────
const apiStack = new ApiStack(app, `${projectName}-api`, {
  env,
  agentRuntimeArn: agentCoreStack.agentRuntimeArn,
  metadataTableArn: agentStack.metadataTable.tableArn,
  mlflowAppArn: mlflowStack.trackingServerArn,
  projectName,
  sessionBucketArn: agentStack.sessionBucket.bucketArn,
});
apiStack.addDependency(agentCoreStack);

// ── Tier 3b: Gateway (MCP + Cedar ABAC) ───────────────────────────────────────
const gatewayStack = new GatewayStack(app, `${projectName}-gateway`, {
  projectName,
  userPoolArn: apiStack.userPoolArn,
  userPoolId: apiStack.userPoolId,
  machineClientId: apiStack.machineClientId,
  cognitoTokenEndpoint: apiStack.cognitoTokenEndpoint,
  machineClientSecretArn: apiStack.machineClientSecretArn,
  region: env.region!,
  account: env.account!,
  sageMakerExecutionRoleArn: agentStack.sageMakerExecutionRole.roleArn,
  mlflowTrackingUri: mlflowStack.trackingServerArn,
  bedrockImportRoleArn: agentStack.bedrockImportRole.roleArn,
  env,
});
gatewayStack.addDependency(apiStack);
gatewayStack.addDependency(mlflowStack);

// ── Tier 4: Frontend ──────────────────────────────────────────────────────────
new ScienceAgentUiStack(app, `${projectName}-ui`, {
  env,
  userPoolId: apiStack.userPoolId,
  userPoolClientId: apiStack.userPoolClientId,
  userPoolDomainPrefix: apiStack.userPoolDomainPrefix,
  cognitoRegion: env.region ?? "us-east-1",
  projectName,
}).addDependency(apiStack);
