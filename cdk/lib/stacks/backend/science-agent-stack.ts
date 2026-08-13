import * as cdk from "aws-cdk-lib";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambdaFn from "aws-cdk-lib/aws-lambda";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as sqs from "aws-cdk-lib/aws-sqs";
import { Construct } from "constructs";
import * as path from "path";
import { ArmBuildConstruct } from "../../constructs/arm-build-construct";

export interface ScienceAgentStackProps extends cdk.StackProps {
  projectName: string;
}

export class ScienceAgentStack extends cdk.Stack {
  public readonly metadataTable: dynamodb.ITable;
  public readonly sessionBucket: s3.Bucket;
  public readonly agentCoreRole: iam.Role;
  public readonly sageMakerExecutionRole: iam.Role;
  public readonly bedrockImportRole: iam.Role;
  public readonly agentRepo: ecr.Repository;
  public readonly agentImageUri: string;
  /** Exported so AgentCoreStack can create the Runtime artifact and dependency */
  public readonly agentImageTag: string;
  public readonly agentBuildCompletion: cdk.CustomResource;

  constructor(scope: Construct, id: string, props: ScienceAgentStackProps) {
    super(scope, id, props);

    this.sessionBucket = new s3.Bucket(this, "SessionBucket", {
      bucketName: `${props.projectName}-sessions-${this.account}-${this.region}`,
      // DESTROY + autoDeleteObjects so `cdk destroy` fully removes the bucket
      // (including all object versions) and a fresh deploy doesn't collide with
      // a retained bucket. This is session/artifact scratch state, not a system
      // of record.
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      // CORS so the SPA (served from CloudFront) can call S3 directly with
      // Cognito Identity Pool creds — the Skills editor lists/reads/writes
      // skills/*, and the Evals tab writes skills/system-prompt.md. Without
      // this the browser's preflight/same-origin policy blocks the request
      // ("Failed to fetch"). The CloudFront domain is created downstream in the
      // UI stack, so we allow https origins broadly here (BlockPublicAccess +
      // enforceSSL + the Identity Pool authz still gate actual access; CORS
      // only governs which page origins the browser will expose responses to).
      cors: [
        {
          allowedMethods: [
            s3.HttpMethods.GET,
            s3.HttpMethods.PUT,
            s3.HttpMethods.POST,
            s3.HttpMethods.DELETE,
            s3.HttpMethods.HEAD,
          ],
          allowedOrigins: ["https://*.cloudfront.net"],
          allowedHeaders: ["*"],
          exposedHeaders: ["ETag", "x-amz-request-id"],
          maxAge: 3000,
        },
      ],
    });

    // CDK-managed metadata table: PK task_id (= thread_id), on-demand billing,
    // no GSIs. Created by the stack so a from-scratch `cdk deploy --all` is
    // self-contained. DESTROY removal policy keeps destroy/redeploy one-pass —
    // this is a demo/session-state table, not a system of record.
    this.metadataTable = new dynamodb.Table(this, "JobsTable", {
      tableName: `${props.projectName}-metadata`,
      partitionKey: {
        name: "task_id",
        type: dynamodb.AttributeType.STRING,
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    this.agentCoreRole = new iam.Role(this, "AgentCoreRole", {
      // bedrock-agentcore.amazonaws.com is the correct trust principal for AgentCore Runtime
      assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
    });
    // Allow the agent to invoke Claude models via Bedrock
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "BedrockModelInvocation",
        actions: [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
        ],
        resources: [
          `arn:aws:bedrock:${this.region}::foundation-model/anthropic.*`,
          `arn:aws:bedrock:::foundation-model/anthropic.*`,
          `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/global.anthropic.*`,
          `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/us.anthropic.*`,
        ],
      }),
    );
    // Least-privilege SageMaker: only the actions the agent actually calls
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "SageMakerTrainingLeastPrivilege",
        actions: [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:ListTrainingJobs",
        ],
        resources: [
          `arn:aws:sagemaker:${this.region}:${this.account}:training-job/${props.projectName}-job-*`,
        ],
      }),
    );
    // MLflow App access — two levels of permission are required:
    // 1. sagemaker:CallMlflowAppApi  — control-plane gate; authorises invoking the App endpoint.
    //    Without this, all data-plane calls return HTTP 403 regardless of sagemaker-mlflow grants.
    // 2. sagemaker-mlflow:*          — data-plane operations (GetExperimentByName, CreateRun, etc.)
    // 3. sagemaker:DescribeMlflowApp — resolves the ARN → HTTPS TrackingServerUrl (sagemaker-mlflow plugin)
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "MLflowAppAccess",
        actions: [
          "sagemaker:CallMlflowAppApi",
          "sagemaker:DescribeMlflowApp",
          "sagemaker-mlflow:*",
        ],
        resources: [
          `arn:aws:sagemaker:${this.region}:${this.account}:mlflow-app/*`,
        ],
      }),
    );
    // SageMaker execution role — used by training jobs to access S3/ECR
    this.sageMakerExecutionRole = new iam.Role(this, "SageMakerExecutionRole", {
      roleName: `${props.projectName}-SageMakerExecutionRole`,
      assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess"),
      ],
    });
    this.sessionBucket.grantReadWrite(this.sageMakerExecutionRole);

    // R5 closure: dedicated execution role for Bedrock Custom Model Import.
    // The import service assumes this role to read the training artifact —
    // passing the SageMaker execution role fails ("Provided IAM role could
    // not be assumed"): its trust policy only allows sagemaker.amazonaws.com.
    this.bedrockImportRole = new iam.Role(this, "BedrockImportRole", {
      roleName: `${props.projectName}-BedrockImportRole`,
      assumedBy: new iam.ServicePrincipal("bedrock.amazonaws.com", {
        conditions: {
          StringEquals: { "aws:SourceAccount": this.account },
        },
      }),
    });
    this.sessionBucket.grantRead(this.bedrockImportRole, "training-output/*");
    this.sageMakerExecutionRole.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          "ecr:GetAuthorizationToken",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchCheckLayerAvailability",
        ],
        resources: ["*"],
      }),
    );

    // R6: Eval Processing container invokes custom scorer Lambdas per row
    // if the caller passed `custom_scorer_lambda_arns` on submit_eval_job.
    // Scope by name pattern so a stray ARN can't invoke an unrelated Lambda.
    // `*-scorer-*` is the naming convention documented in the mlflow SKILL.md
    // §Custom Scorers; tighten with an explicit allowlist in production.
    this.sageMakerExecutionRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "CustomScorerInvoke",
        actions: ["lambda:InvokeFunction"],
        resources: [
          `arn:aws:lambda:${this.region}:${this.account}:function:*-scorer-*`,
        ],
      }),
    );

    // Eval Processing container needs to call the target model (Bedrock or a
    // SageMaker endpoint) and the judge model (always Bedrock). Without these
    // the container's _bedrock_predict / _sagemaker_predict calls 403.
    this.sageMakerExecutionRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "BedrockEvalInvoke",
        actions: ["bedrock:InvokeModel", "bedrock:Converse"],
        resources: [
          // Same-region foundation model ARNs (us-east-1 models called directly).
          `arn:aws:bedrock:${this.region}::foundation-model/*`,
          // Regional inference profiles (us.*, eu.*, apac.*) — ARN carries region.
          `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/*`,
          // Cross-region (`global.*`) inference profiles route to foundation
          // models in any region; Bedrock evaluates IAM against the underlying
          // foundation-model ARN in the target region, which may be "*" when
          // the profile is truly global. Authorize invoke against every region
          // so global.amazon.nova-2-lite-v1:0 → amazon.nova-2-lite-v1:0 works.
          `arn:aws:bedrock:*::foundation-model/*`,
          // And the cross-region profile ARN itself (profile record is in
          // us-east-1 for a global.* profile, but the pattern allows any).
          `arn:aws:bedrock:*:${this.account}:inference-profile/*`,
        ],
      }),
    );
    this.sageMakerExecutionRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "SageMakerEndpointInvoke",
        actions: ["sagemaker:InvokeEndpoint"],
        resources: [
          `arn:aws:sagemaker:${this.region}:${this.account}:endpoint/*`,
        ],
      }),
    );

    // Allow passing the SageMaker execution role to training jobs
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "PassSageMakerExecutionRole",
        actions: ["iam:PassRole"],
        resources: [this.sageMakerExecutionRole.roleArn],
        conditions: {
          StringEquals: { "iam:PassedToService": "sagemaker.amazonaws.com" },
        },
      }),
    );
    this.sessionBucket.grantReadWrite(this.agentCoreRole);
    this.metadataTable.grantReadWriteData(this.agentCoreRole);

    // CloudWatch Logs — runtime container must be able to emit logs and OTEL telemetry
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "AgentCoreCloudWatchLogs",
        actions: [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ],
        resources: [
          `arn:aws:logs:${this.region}:${this.account}:log-group:/aws/bedrock-agentcore/runtimes/*`,
          `arn:aws:logs:${this.region}:${this.account}:log-group:/aws/bedrock-agentcore/runtimes/*:*`,
        ],
      }),
    );

    // ECR — wildcard pull so AgentCore can always fetch the container image
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "AgentCoreEcrPull",
        actions: [
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchCheckLayerAvailability",
        ],
        resources: ["*"],
      }),
    );
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "AgentCoreEcrAuth",
        actions: ["ecr:GetAuthorizationToken"],
        resources: ["*"],
      }),
    );

    // AgentCore Workload Identity — required for memory and identity features
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "AgentCoreWorkloadIdentity",
        actions: [
          "bedrock-agentcore:GetWorkloadAccessToken",
          "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
        ],
        resources: [
          `arn:aws:bedrock-agentcore:${this.region}:${this.account}:workload-identity-directory/default/*`,
        ],
      }),
    );

    // Token Vault — per-user OAuth token storage (Phase 2: HuggingFace per-user tokens)
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "AgentCoreTokenVault",
        actions: [
          "bedrock-agentcore:GetWorkloadAccessTokenForUserId",
          "bedrock-agentcore:GetResourceOauth2Token",
          "secretsmanager:GetSecretValue",
        ],
        resources: [
          `arn:aws:bedrock-agentcore:${this.region}:${this.account}:workload-identity-directory/default/*`,
          `arn:aws:bedrock-agentcore:${this.region}:${this.account}:token-vault/default/*`,
          `arn:aws:secretsmanager:${this.region}:${this.account}:secret:bedrock-agentcore-identity!default/oauth2/HuggingFaceProvider*`,
        ],
      }),
    );

    // X-Ray OTLP — required to export traces via https://xray.<region>.amazonaws.com/v1/traces
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "AgentCoreXRay",
        actions: [
          "xray:PutSpans",
          "xray:PutSpansForIndexing",
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
        ],
        resources: ["*"],
      }),
    );

    // EC2 VPC networking — required if AgentCore Runtime is attached to a VPC
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "AgentCoreVpcNetworking",
        actions: [
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface",
          "ec2:AssignPrivateIpAddresses",
          "ec2:UnassignPrivateIpAddresses",
        ],
        resources: ["*"],
      }),
    );

    // SSM Parameter Store — agent reads MLflow tracking URI and agentcore endpoint at runtime
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "AgentCoreSsmRead",
        actions: ["ssm:GetParameter", "ssm:GetParameters"],
        resources: [
          `arn:aws:ssm:${this.region}:${this.account}:parameter/${props.projectName}/*`,
        ],
      }),
    );

    // AgentCore Gateway — invoke any configured AgentCore gateway (e.g. future Neptune gateway)
    this.agentCoreRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "AgentCoreInvokeGateway",
        actions: ["bedrock-agentcore:InvokeGateway"],
        resources: [
          `arn:aws:bedrock-agentcore:${this.region}:${this.account}:gateway/*`,
        ],
      }),
    );

    // ARM64 Docker builds — source is zipped to S3 assets during cdk deploy.
    // A custom resource with async polling blocks CloudFormation until each
    // build succeeds before dependent stacks (ApiStack) are allowed to proceed.
    const agentBuild = new ArmBuildConstruct(this, "AgentBuild", {
      sourcePath: path.join(__dirname, "../../../../agent"),
      namePrefix: props.projectName,
    });
    this.agentRepo = agentBuild.repository;
    this.agentImageUri = agentBuild.imageUri;
    this.agentImageTag = agentBuild.imageTag;
    this.agentBuildCompletion = agentBuild.buildCompletion;

    // SQS DLQ for Callback Lambda retry
    const dlq = new sqs.Queue(this, "CallbackDLQ", {
      retentionPeriod: cdk.Duration.days(14),
      encryption: sqs.QueueEncryption.SQS_MANAGED,
      enforceSSL: true,
    });

    // Callback Lambda + R5 bedrock-import poller — both packaged as ARM64
    // container images built by CodeBuild via ArmBuildConstruct, matching the
    // packaging convention used for every skill Lambda + AgentCore Runtime in
    // this project. A single image (lambda/callback/Dockerfile) ships both
    // entrypoints (handler.py + poller.py); each Function picks its entrypoint
    // via `cmd` override on DockerImageCode.fromEcr.
    //
    // Pinned boto3 1.43.2 is installed inside the image so the SageMaker AI
    // Benchmark Job APIs (create_ai_benchmark_job, create_ai_workload_config,
    // describe_ai_benchmark_job) are available — the Lambda runtime's built-in
    // boto3 is ~1.34.x and lacks them.
    const callbackBuild = new ArmBuildConstruct(this, "CallbackBuild", {
      sourcePath: path.join(__dirname, "../../../../lambda/callback"),
      namePrefix: `${props.projectName}-callback`,
    });
    const callbackFn = new lambdaFn.DockerImageFunction(
      this,
      "CallbackLambda",
      {
        architecture: lambdaFn.Architecture.ARM_64,
        code: lambdaFn.DockerImageCode.fromEcr(callbackBuild.repository, {
          tagOrDigest: callbackBuild.imageTag,
          cmd: ["handler.handler"],
        }),
        environment: {
          JOBS_TABLE: this.metadataTable.tableName,
          SESSION_BUCKET: this.sessionBucket.bucketName,
          SAGEMAKER_EXECUTION_ROLE_ARN: this.sageMakerExecutionRole.roleArn,
          AGENTCORE_ENDPOINT_SSM: `/${props.projectName}/dev/agentcore/endpoint`,
        },
        deadLetterQueue: dlq,
        retryAttempts: 2,
        timeout: cdk.Duration.seconds(60),
      },
    );
    // Block Lambda creation until the CodeBuild image push completes.
    callbackFn.node.addDependency(callbackBuild.buildCompletion);
    this.metadataTable.grantReadWriteData(callbackFn);
    // BUG-003: the callback extracts baseline/* from tabular model.tar.gz and
    // re-uploads them as standalone objects for submit_monitoring_job.
    this.sessionBucket.grantReadWrite(callbackFn, "training-output/*");
    // Grant Lambda permission to read AgentCore endpoint + HF token from SSM
    callbackFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "ReadAgentCoreEndpointFromSSM",
        actions: ["ssm:GetParameter"],
        resources: [
          `arn:aws:ssm:${this.region}:${this.account}:parameter/${props.projectName}/*`,
        ],
      }),
    );
    // Describe + ListTags replace the GSI — the callback reads ThreadId/JobId
    // tags off the training job to locate the thread row. Processing-job ARNs
    // also need Describe so the eval callback branch can look up the row.
    callbackFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "DescribeSageMakerJobs",
        actions: [
          "sagemaker:DescribeTrainingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:ListTags",
        ],
        resources: [
          `arn:aws:sagemaker:${this.region}:${this.account}:training-job/${props.projectName}-job-*`,
          `arn:aws:sagemaker:${this.region}:${this.account}:processing-job/${props.projectName}-eval-*`,
          // QA BUG-018: R1 monitoring jobs were never added here, so the
          // callback's DescribeProcessingJob was AccessDenied for every
          // monitoring completion — the resume silently never happened.
          `arn:aws:sagemaker:${this.region}:${this.account}:processing-job/${props.projectName}-monitor-*`,
        ],
      }),
    );
    // Grant Lambda permission to invoke the AgentCore runtime (resume agent session after job completes)
    callbackFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "InvokeAgentCoreRuntime",
        actions: [
          "bedrock-agentcore:InvokeAgentRuntime",
          "bedrock-agentcore:InvokeAgentRuntimeForUser", // NEW: user context header
        ],
        resources: [
          `arn:aws:bedrock-agentcore:${this.region}:${this.account}:runtime/${props.projectName.replace(/-/g, "_")}*`,
        ],
      }),
    );
    // Recommendation flow — start AI Benchmark Job from endpoint state change event
    callbackFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "CallbackRecommendationStartBenchmark",
        actions: [
          "sagemaker:DescribeEndpoint",
          "sagemaker:CreateInferenceComponent",
          "sagemaker:CreateAIWorkloadConfig",
          "sagemaker:CreateAIBenchmarkJob",
          "sagemaker:AddTags", // F-6: Create* with Tags=[…] needs AddTags
          // Failure-path teardown if create_ai_benchmark_job itself fails:
          "sagemaker:DeleteEndpoint",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:DeleteInferenceComponent",
          "sagemaker:DeleteModel",
          "sagemaker:DeleteAIWorkloadConfig",
          "sagemaker:ListTags", // _tags_from_describe fallback
        ],
        resources: ["*"], // SageMaker resource ARNs are name-scoped; we rely on
        //  Kind=recommendation tag filtering in the handler.
      }),
    );
    callbackFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "CallbackPassSageMakerRoleToBenchmark",
        actions: ["iam:PassRole"],
        resources: [this.sageMakerExecutionRole.roleArn],
        conditions: {
          StringEquals: { "iam:PassedToService": "sagemaker.amazonaws.com" },
        },
      }),
    );

    // EventBridge rule — SageMaker job state changes. Covers training
    // (training-job state change), evaluation (processing-job state change),
    // and recommendation flow (endpoint state change). The callback Lambda
    // branches internally on detail-type.
    new events.Rule(this, "SageMakerJobRule", {
      eventPattern: {
        source: ["aws.sagemaker"],
        detailType: [
          "SageMaker Training Job State Change",
          "SageMaker Processing Job State Change",
          "SageMaker Endpoint State Change", // NEW — wakes _handle_endpoint_event
        ],
      },
      targets: [new targets.LambdaFunction(callbackFn)],
    });

    // R5: Bedrock Custom Model Import poller. Bedrock does NOT emit a
    // "Bedrock Model Import Job State Change" EventBridge event (only Model
    // Customization + Batch Inference jobs do), so we run a 15-min schedule
    // that scans DDB for in-flight bedrock_import rows and probes each via
    // bedrock.get_model_import_job. On terminal status the poller updates
    // DDB + resumes the agent thread via _invoke_agentcore.
    //
    // Shares the same ECR image as callbackFn (callbackBuild.repository) so
    // poller.py can `from handler import` _build_resume_message /
    // _invoke_agentcore / _get_agentcore_endpoint / JOBS_TABLE without
    // duplicating code. Different entrypoint via `cmd` override.
    const bedrockImportPollerFn = new lambdaFn.DockerImageFunction(
      this,
      "BedrockImportPollerLambda",
      {
        architecture: lambdaFn.Architecture.ARM_64,
        code: lambdaFn.DockerImageCode.fromEcr(callbackBuild.repository, {
          tagOrDigest: callbackBuild.imageTag,
          cmd: ["poller.handler"],
        }),
        environment: {
          JOBS_TABLE: this.metadataTable.tableName,
          AGENTCORE_ENDPOINT_SSM: `/${props.projectName}/dev/agentcore/endpoint`,
        },
        timeout: cdk.Duration.minutes(2),
      },
    );
    bedrockImportPollerFn.node.addDependency(callbackBuild.buildCompletion);
    this.metadataTable.grantReadWriteData(bedrockImportPollerFn);
    bedrockImportPollerFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "PollerReadAgentCoreEndpointFromSSM",
        actions: ["ssm:GetParameter"],
        resources: [
          `arn:aws:ssm:${this.region}:${this.account}:parameter/${props.projectName}/*`,
        ],
      }),
    );
    bedrockImportPollerFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "PollerGetBedrockModelImportJob",
        actions: ["bedrock:GetModelImportJob", "bedrock:ListModelImportJobs"],
        // Bedrock Model Import job ARNs are name-scoped at create time and
        // not always known up-front; the poller's blast radius is limited to
        // describe/list calls.
        resources: ["*"],
      }),
    );
    bedrockImportPollerFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "PollerInvokeAgentCoreRuntime",
        actions: [
          "bedrock-agentcore:InvokeAgentRuntime",
          "bedrock-agentcore:InvokeAgentRuntimeForUser",
        ],
        resources: [
          `arn:aws:bedrock-agentcore:${this.region}:${this.account}:runtime/${props.projectName.replace(/-/g, "_")}*`,
        ],
      }),
    );
    new events.Rule(this, "BedrockImportPollerSchedule", {
      schedule: events.Schedule.rate(cdk.Duration.minutes(15)),
      targets: [new targets.LambdaFunction(bedrockImportPollerFn)],
    });
  }
}
