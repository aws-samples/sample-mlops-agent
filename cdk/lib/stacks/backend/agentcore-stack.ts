import * as cdk from "aws-cdk-lib";
import * as agentcore from "@aws-cdk/aws-bedrock-agentcore-alpha";
import * as cr from "aws-cdk-lib/custom-resources";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as ssm from "aws-cdk-lib/aws-ssm";
import * as path from "path";
import { Construct } from "constructs";

export interface AgentCoreStackProps extends cdk.StackProps {
  /** ECR repository that holds the agent image */
  agentRepo: ecr.IRepository;
  /** Image tag built by ArmBuildConstruct — used to pin the Runtime to a specific image */
  agentImageTag: string;
  /** Custom resource that completes only when the agent Docker build succeeds */
  agentBuildCompletion: cdk.CustomResource;
  /** Execution role created in ScienceAgentStack */
  agentCoreRole: iam.IRole;
  /** DynamoDB metadata table */
  metadataTable: dynamodb.ITable;
  /** S3 bucket for session/artifact data */
  sessionBucket: s3.IBucket;
  /** SSM parameter path storing the MLflow App ARN (avoids CFN cross-stack export dependency) */
  mlflowSsmParamPath: string;
  /** Project name — used to name resources and OTEL service attributes */
  projectName: string;
  /** SageMaker execution role ARN — injected into training skill at runtime */
  sageMakerExecutionRoleArn: string;
}

export class AgentCoreStack extends cdk.Stack {
  /** Full HTTPS endpoint for the Runtime — append /invocations to invoke */
  public readonly agentCoreEndpoint: string;
  /** Runtime ARN — used to scope IAM grants for callers */
  public readonly agentRuntimeArn: string;
  /** Application log group that receives OTEL spans + LLO gen-ai events —
   *  /aws/bedrock-agentcore/runtimes/<runtimeId>-DEFAULT. The eval config +
   *  batch data source MUST read from this exact name. */
  public readonly runtimeLogGroupName: string;

  constructor(scope: Construct, id: string, props: AgentCoreStackProps) {
    super(scope, id, props);

    // AgentCore writes container stdout/stderr to this log group.
    // APPLICATION_LOGS delivery (configured below) ships logs here natively —
    // no OTLP-to-CloudWatch-Logs exporter or pre-created log stream required.
    const agentLogGroup = new logs.LogGroup(this, "AgentCoreLogGroup", {
      logGroupName: `/aws/bedrock-agentcore/runtimes/${props.projectName.replace(/-/g, "_")}`,
      retention: logs.RetentionDays.ONE_MONTH,
      // DESTROY so `cdk destroy` removes the log group and a fresh deploy doesn't
      // collide with a retained one (fixed-name log group can't be recreated
      // while the old one survives). Logs are observability data, not a system
      // of record.
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const artifact = agentcore.AgentRuntimeArtifact.fromEcrRepository(
      props.agentRepo,
      props.agentImageTag,
    );

    // Declare SSM params that the Runtime env vars and PatchWorkloadIdentity both reference.
    // Declared here (before Runtime) so CDK tokens are available at construction time,
    // avoiding a second valueForStringParameter() lookup that would fail on first deploy.
    const hfOAuthCallbackUrlParam = new ssm.StringParameter(
      this,
      "HFOAuthCallbackUrlParam",
      {
        parameterName: `/${props.projectName}/dev/hf-oauth-callback-url`,
        stringValue: process.env.HF_OAUTH_CALLBACK_URL ?? "PLACEHOLDER",
      },
    );

    // Gateway params — owned by GatewayStack (which deploys after AgentCore).
    // Declare them as PLACEHOLDER here so they exist before GatewayStack's
    // PatchGatewaySsm custom resource overwrites them with real values. The
    // agent reads them from SSM at container start (agent/main.py::
    // _get_gateway_config), NOT as baked Runtime env vars — so a single
    // `cdk deploy --all` wires everything up without a second agentcore redeploy.
    new ssm.StringParameter(this, "GatewayMcpUrlParam", {
      parameterName: `/${props.projectName}/dev/gateway/mcp-url`,
      stringValue: process.env.GATEWAY_MCP_URL ?? "PLACEHOLDER",
    });
    new ssm.StringParameter(this, "CognitoTokenEndpointParam", {
      parameterName: `/${props.projectName}/dev/gateway/cognito-token-endpoint`,
      stringValue: process.env.COGNITO_TOKEN_ENDPOINT ?? "PLACEHOLDER",
    });
    new ssm.StringParameter(this, "GatewayM2MClientIdParam", {
      parameterName: `/${props.projectName}/dev/gateway/m2m-client-id`,
      stringValue: process.env.GATEWAY_M2M_CLIENT_ID ?? "PLACEHOLDER",
    });

    const runtime = new agentcore.Runtime(this, "ScienceAgentRuntime", {
      runtimeName: props.projectName.replace(/-/g, "_"),
      agentRuntimeArtifact: artifact,
      executionRole: props.agentCoreRole,
      description:
        "Science Training Agent — async SageMaker + MLflow orchestration",
      authorizerConfiguration:
        agentcore.RuntimeAuthorizerConfiguration.usingIAM(),
      lifecycleConfiguration: {
        idleRuntimeSessionTimeout: cdk.Duration.minutes(30),
        maxLifetime: cdk.Duration.hours(8),
      },
      environmentVariables: {
        JOBS_TABLE: props.metadataTable.tableName,
        SESSION_BUCKET: props.sessionBucket.bucketName,
        // Live skills + system-prompt store prefix; agent hydrates from
        // s3://<sessionBucket>/skills/ at turn start (see agent/skills_loader.py).
        // The runtime role already has sessionBucket read/write (grantReadWrite below).
        SKILLS_S3_PREFIX: "skills/",
        SAGEMAKER_EXECUTION_ROLE_ARN: props.sageMakerExecutionRoleArn,
        MLFLOW_TRACKING_URI: ssm.StringParameter.valueForStringParameter(
          this,
          props.mlflowSsmParamPath,
        ),
        AWS_REGION: this.region,
        CLAUDE_CODE_USE_BEDROCK: "1",
        ANTHROPIC_MODEL: "global.anthropic.claude-sonnet-4-6",
        // Minimal OTEL config — aws-opentelemetry-distro auto-configures distro,
        // configurator, resource attributes, exporters, and endpoints. We only
        // override the sampler (AgentCore propagates sampled=false) and set the
        // X-Ray-compatible trace ID + propagators.
        AGENT_OBSERVABILITY_ENABLED: "true",
        OTEL_PROPAGATORS: "xray,baggage",
        OTEL_PYTHON_ID_GENERATOR: "xray",
        OTEL_EXPORTER_OTLP_PROTOCOL: "http/protobuf",
        OTEL_TRACES_SAMPLER: "always_on",
        // Export LLO (large-language-object) gen-ai event records — the
        // input/output message content — to CloudWatch Logs via the ADOT
        // LLOHandler. Without this, only spans reach aws/spans and the builtin
        // eval judges (GoalSuccessRate/Helpfulness/Correctness) fail every
        // session with LogEventMissingException ("session span data is
        // incomplete"), because they read message content from these log events.
        OTEL_LOGS_EXPORTER: "otlp",
        OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED: "true",
        // Phase 3 Gateway MCP config (mcp-url, cognito-token-endpoint,
        // m2m-client-id) is NOT injected as env vars. GatewayStack deploys AFTER
        // this stack and patches those SSM params, so a value baked here at synth
        // time would be a stale PLACEHOLDER. The agent resolves them from SSM at
        // container start (see agent/main.py::_get_gateway_config), which keeps
        // `cdk deploy --all` single-pass. The M2M client secret is likewise read
        // from Secrets Manager at runtime.
      },
    });

    // Grant the execution role read/write on the data stores
    props.metadataTable.grantReadWriteData(props.agentCoreRole);
    props.sessionBucket.grantReadWrite(props.agentCoreRole);

    // Also grant ECR pull on the specific agent repository
    // (the Runtime CDK construct adds wildcard ECR pull automatically,
    //  but an explicit repo grant is belt-and-suspenders)
    props.agentRepo.grantPull(props.agentCoreRole);

    // Allow the agent runtime to read project SSM params: HF_TOKEN for gated
    // dataset access, and the Gateway MCP config (mcp-url, cognito-token-endpoint,
    // m2m-client-id) which the agent resolves at runtime via ssm:GetParameters.
    props.agentCoreRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        sid: "ReadProjectSSMParams",
        actions: ["ssm:GetParameter", "ssm:GetParameters"],
        resources: [
          `arn:aws:ssm:${this.region}:${this.account}:parameter/${props.projectName}/*`,
        ],
      }),
    );

    // Allow the agent runtime to fetch the M2M client secret for Gateway auth
    props.agentCoreRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        sid: "ReadGatewayM2MSecret",
        actions: ["secretsmanager:GetSecretValue"],
        resources: [
          `arn:aws:secretsmanager:${this.region}:${this.account}:secret:/${props.projectName}/dev/gateway-m2m-client-secret*`,
        ],
      }),
    );

    // Ensure the Runtime is only created after the Docker image build succeeds
    runtime.node.addDependency(props.agentBuildCompletion);

    this.agentRuntimeArn = runtime.agentRuntimeArn;
    // Real app log group = /aws/bedrock-agentcore/runtimes/<runtimeId>-DEFAULT
    // (runtime id is segment 1 of the ARN). This is where spans + LLO gen-ai
    // events land — NOT the /runtimes/<runtimeName> group. Publish it to SSM so
    // the evals stack + UI batch data source point at the correct group.
    const runtimeId = cdk.Fn.select(
      1,
      cdk.Fn.split("/", runtime.agentRuntimeArn),
    );
    this.runtimeLogGroupName = `/aws/bedrock-agentcore/runtimes/${runtimeId}-DEFAULT`;
    new ssm.StringParameter(this, "RuntimeLogGroupNameParam", {
      parameterName: `/${props.projectName}/dev/agentcore/runtime-log-group`,
      stringValue: this.runtimeLogGroupName,
    });
    // Also publish the log-group ARN — the tool-description optimization API
    // (StartRecommendation) requires cloudwatchLogs.logGroupArns with >=1 entry.
    new ssm.StringParameter(this, "RuntimeLogGroupArnParam", {
      parameterName: `/${props.projectName}/dev/agentcore/runtime-log-group-arn`,
      stringValue: `arn:aws:logs:${this.region}:${this.account}:log-group:${this.runtimeLogGroupName}:*`,
    });

    // ── PatchWorkloadIdentity Custom Resource ─────────────────────────────────
    // Registers the HuggingFace OAuth callback URL with the workload identity
    // after the Runtime (and its auto-created workload identity) exists.
    const patchWorkloadFn = new lambda.Function(
      this,
      "PatchWorkloadIdentityFn",
      {
        runtime: lambda.Runtime.PYTHON_3_12,
        handler: "index.handler",
        code: lambda.Code.fromAsset(
          path.join(__dirname, "patch-workload-identity-handler"),
        ),
        timeout: cdk.Duration.minutes(5),
        description:
          "Registers HF OAuth callback URL with AgentCore workload identity",
      },
    );

    patchWorkloadFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "PatchWorkloadIdentity",
        actions: [
          "bedrock-agentcore-control:GetWorkloadIdentity",
          "bedrock-agentcore-control:UpdateWorkloadIdentity",
        ],
        resources: [
          `arn:aws:bedrock-agentcore:${this.region}:${this.account}:workload-identity-directory/default/*`,
        ],
      }),
    );
    // Allow the handler to look up the workload identity name from the Runtime
    // (workloadIdentityDetails.workloadIdentityArn) without relying on a pre-populated SSM param.
    patchWorkloadFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "ReadAgentRuntimeForWorkloadIdentity",
        actions: ["bedrock-agentcore-control:GetAgentRuntime"],
        resources: [this.agentRuntimeArn],
      }),
    );

    const patchWorkloadProvider = new cr.Provider(
      this,
      "PatchWorkloadIdentityProvider",
      {
        onEventHandler: patchWorkloadFn,
      },
    );

    new cdk.CustomResource(this, "PatchWorkloadIdentity", {
      serviceToken: patchWorkloadProvider.serviceToken,
      properties: {
        // The handler resolves WorkloadIdentityName by calling GetAgentRuntime on this ID —
        // no SSM placeholder dance needed.
        RuntimeId: cdk.Fn.select(1, cdk.Fn.split("/", this.agentRuntimeArn)),
        HFOAuthCallbackUrl: hfOAuthCallbackUrlParam.stringValue,
        Region: this.region,
        // Force re-run on every Runtime update so callback URL stays registered.
        AgentImageTag: props.agentImageTag,
      },
    });

    // ── Native log delivery (APPLICATION_LOGS + USAGE_LOGS) ──────────────────
    // Configures AgentCore Runtime → CloudWatch Logs delivery so that:
    //   1. The AgentCore console shows "Log delivery: N"
    //   2. APPLICATION_LOGS session data flows to the log group used by online eval
    //   3. USAGE_LOGS token/CPU/memory metrics populate the GenAI Observability
    //      "Resource consumption" dashboard section
    const runtimeName = props.projectName.replace(/-/g, "_");
    const baseName = props.projectName;

    // APPLICATION_LOGS → reuse the existing agentLogGroup (also receives OTEL stdout).
    // The delivery.logs service principal needs explicit write permission on the group.
    agentLogGroup.addToResourcePolicy(
      new iam.PolicyStatement({
        sid: "AllowApplicationLogsDelivery",
        principals: [new iam.ServicePrincipal("delivery.logs.amazonaws.com")],
        actions: ["logs:CreateLogStream", "logs:PutLogEvents"],
        resources: [`${agentLogGroup.logGroupArn}:log-stream:*`],
        conditions: {
          StringEquals: { "aws:SourceAccount": this.account },
          ArnLike: {
            "aws:SourceArn": `arn:aws:logs:${this.region}:${this.account}:*`,
          },
        },
      }),
    );

    const appLogsSrc = new logs.CfnDeliverySource(this, "AppLogsSrc", {
      name: `${baseName}-application-src`,
      resourceArn: this.agentRuntimeArn,
      logType: "APPLICATION_LOGS",
    });
    const appLogsDest = new logs.CfnDeliveryDestination(this, "AppLogsDest", {
      name: `${baseName}-application-dest`,
      destinationResourceArn: agentLogGroup.logGroupArn,
    });
    new logs.CfnDelivery(this, "AppLogsDelivery", {
      deliverySourceName: appLogsSrc.ref,
      deliveryDestinationArn: appLogsDest.attrArn,
    });

    // USAGE_LOGS → separate log group for token/CPU/memory metrics
    // (populates "Resource consumption" section in GenAI Observability dashboard)
    const usageLogGroup = new logs.LogGroup(this, "UsageLogGroup", {
      logGroupName: `/aws/bedrock-agentcore/runtimes/${runtimeName}-usage`,
      retention: logs.RetentionDays.ONE_MONTH,
      // DESTROY: see AgentCoreLogGroup — fixed-name log group must be removed on
      // destroy so a fresh deploy doesn't collide with a retained one.
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    usageLogGroup.addToResourcePolicy(
      new iam.PolicyStatement({
        sid: "AllowUsageLogsDelivery",
        principals: [new iam.ServicePrincipal("delivery.logs.amazonaws.com")],
        actions: ["logs:CreateLogStream", "logs:PutLogEvents"],
        resources: [`${usageLogGroup.logGroupArn}:log-stream:*`],
        conditions: {
          StringEquals: { "aws:SourceAccount": this.account },
          ArnLike: {
            "aws:SourceArn": `arn:aws:logs:${this.region}:${this.account}:*`,
          },
        },
      }),
    );

    const usageLogsSrc = new logs.CfnDeliverySource(this, "UsageLogsSrc", {
      name: `${baseName}-usage-src`,
      resourceArn: this.agentRuntimeArn,
      logType: "USAGE_LOGS",
    });
    const usageLogsDest = new logs.CfnDeliveryDestination(
      this,
      "UsageLogsDest",
      {
        name: `${baseName}-usage-dest`,
        destinationResourceArn: usageLogGroup.logGroupArn,
      },
    );
    new logs.CfnDelivery(this, "UsageLogsDelivery", {
      deliverySourceName: usageLogsSrc.ref,
      deliveryDestinationArn: usageLogsDest.attrArn,
    });

    // The HTTP invocation endpoint: append /invocations for POST calls
    this.agentCoreEndpoint = cdk.Fn.join("", [
      "https://bedrock-agentcore.",
      this.region,
      ".amazonaws.com/runtimes/",
      runtime.agentRuntimeArn,
    ]);

    // Write agentCoreEndpoint to SSM so ScienceAgentUiStack can read it without
    // a CFN cross-stack export dependency (avoids CFN export lock on updates).
    new ssm.StringParameter(this, "AgentCoreEndpointParam", {
      parameterName: `/${props.projectName}/dev/agentcore/endpoint`,
      stringValue: this.agentCoreEndpoint,
    });

    // Seed HF token from environment variable
    new ssm.StringParameter(this, "HfToken", {
      parameterName: `/${props.projectName}/dev/hf-token`,
      stringValue: process.env.HF_API_TOKEN ?? "PLACEHOLDER",
    });

    // GitHub token and experiment repo for git skill
    new ssm.StringParameter(this, "GithubToken", {
      parameterName: `/${props.projectName}/dev/github-token`,
      stringValue: process.env.GITHUB_TOKEN ?? "PLACEHOLDER",
    });
    new ssm.StringParameter(this, "GitExperimentRepo", {
      parameterName: `/${props.projectName}/dev/git-experiment-repo`,
      stringValue: process.env.GIT_EXPERIMENT_REPO ?? "PLACEHOLDER",
    });

    new cdk.CfnOutput(this, "AgentCoreEndpoint", {
      value: this.agentCoreEndpoint,
      description:
        "AgentCore Runtime invocation base URL (append /invocations)",
    });
    new cdk.CfnOutput(this, "AgentRuntimeArn", {
      value: this.agentRuntimeArn,
    });
  }
}
