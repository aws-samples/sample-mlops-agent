import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as ssm from "aws-cdk-lib/aws-ssm";
import * as bedrockagentcore from "aws-cdk-lib/aws-bedrockagentcore";
import * as cr from "aws-cdk-lib/custom-resources";
import * as path from "path";
import * as fs from "fs";
import { Construct } from "constructs";
import { ArmBuildConstruct } from "../../constructs/arm-build-construct";

export interface GatewayStackProps extends cdk.StackProps {
  projectName: string;
  /** Cognito User Pool ARN — Gateway uses this for JWT validation */
  userPoolArn: string;
  /** Cognito User Pool ID */
  userPoolId: string;
  /** Cognito M2M client ID — machine-to-machine authentication */
  machineClientId: string;
  /** Cognito token endpoint URL — for M2M client credentials grant */
  cognitoTokenEndpoint: string;
  /** ARN of the Cognito M2M client secret (created in ApiStack) */
  machineClientSecretArn: string;
  /** AWS region */
  region: string;
  /** AWS account ID */
  account: string;
  /** SageMaker execution role ARN — injected as env var into skill Lambdas */
  sageMakerExecutionRoleArn: string;
  /** MLflow tracking URI (SageMaker MLflow app ARN) — injected into MLflow skill Lambda */
  mlflowTrackingUri: string;
  bedrockImportRoleArn: string;
}

export class GatewayStack extends cdk.Stack {
  /** SSM parameter name storing the Gateway MCP URL for the agent */
  public readonly gatewayMcpUrlParam: string;

  constructor(scope: Construct, id: string, props: GatewayStackProps) {
    super(scope, id, props);

    // Skill Lambdas are packaged as ARM64 container images built by CodeBuild
    // (no local Docker required). Each skill's source dir contains a Dockerfile
    // based on public.ecr.aws/lambda/python:3.12 that installs its requirements.txt.
    const skillLambdaDefaults = {
      timeout: cdk.Duration.seconds(300),
      memorySize: 1024,
      architecture: lambda.Architecture.ARM_64,
      environment: {
        PROJECT_NAME: props.projectName,
        JOBS_TABLE: `${props.projectName}-metadata`,
        SESSION_BUCKET: `${props.projectName}-sessions-${props.account}-${props.region}`,
        // Passed as CDK props (cross-stack references) — safe for clean deploys.
        // valueForStringParameter() would require the SSM params to exist at changeset time.
        SAGEMAKER_EXECUTION_ROLE_ARN: props.sageMakerExecutionRoleArn,
        BEDROCK_IMPORT_ROLE_ARN: props.bedrockImportRoleArn,
        MLFLOW_TRACKING_URI: props.mlflowTrackingUri,
      },
    };

    const buildSkillFn = (
      id: string,
      skillDir: string,
      description: string,
      extraEnv: Record<string, string> = {},
    ): lambda.DockerImageFunction => {
      const build = new ArmBuildConstruct(this, `${id}Build`, {
        sourcePath: path.join(__dirname, "../../../../lambda/skills", skillDir),
        namePrefix: `${props.projectName}-${skillDir}-skill`,
      });
      const fn = new lambda.DockerImageFunction(this, id, {
        ...skillLambdaDefaults,
        code: lambda.DockerImageCode.fromEcr(build.repository, {
          tagOrDigest: build.imageTag,
        }),
        environment: {
          ...skillLambdaDefaults.environment,
          ...extraEnv,
        },
        description,
      });
      // Block Lambda creation until the CodeBuild image push completes.
      fn.node.addDependency(build.buildCompletion);
      return fn;
    };

    const sagemakerSkillFn = buildSkillFn(
      "SageMakerSkillFn",
      "sagemaker",
      "SageMaker skill — Gateway MCP target",
    );
    const huggingfaceSkillFn = buildSkillFn(
      "HuggingFaceSkillFn",
      "huggingface",
      "HuggingFace skill — Gateway MCP target",
    );
    // Defaults (1 GB memory, 512 MB /tmp) are fine — this Lambda only does
    // token fetch, ≤50-row dataset previews via split="train[:N]", and spec
    // authoring. Heavy dataset IO lives in the SageMaker Processing
    // container (lambda/skills/sagemaker/eval/entrypoint.py).
    const gitSkillFn = buildSkillFn(
      "GitSkillFn",
      "git",
      "Git skill — Gateway MCP target",
    );
    const mlflowSkillFn = buildSkillFn(
      "MLflowSkillFn",
      "mlflow",
      "MLflow skill — Gateway MCP target",
    );
    // Slurm skill runs in MOCK_MODE until the pcluster + SSH secret are deployed.
    // When that infra exists, override MOCK_MODE=0 and inject SLURM_HEAD_NODE_HOST,
    // SLURM_SSH_SECRET_ARN, and SLURM_KNOWN_HOSTS (path to a known_hosts file
    // pinning the head node's public key — the handler refuses to SSH without it)
    // via the Lambda environment.
    // R6 closure: reference custom-scorer Lambda (sympy math_correctness).
    // Deployed so the submit_eval_job custom_scorer_lambda_arns path is
    // verifiable end-to-end; callers copy this pattern for their own scorers.
    // Explicit functionName so it matches the "*-scorer-*" invoke grants on
    // the sagemaker skill Lambda and the eval Processing task role.
    const mathScorerBuild = new ArmBuildConstruct(this, "MathScorerRefBuild", {
      sourcePath: path.join(
        __dirname,
        "../../../../lambda/skills/sagemaker/eval/reference_scorers/math_correctness",
      ),
      namePrefix: `${props.projectName}-math-scorer-ref`,
    });
    const mathScorerFn = new lambda.DockerImageFunction(this, "MathScorerRefFn", {
      functionName: `${props.projectName}-math-scorer-ref`,
      code: lambda.DockerImageCode.fromEcr(mathScorerBuild.repository, {
        tagOrDigest: mathScorerBuild.imageTag,
      }),
      memorySize: 512,
      timeout: cdk.Duration.seconds(10),
      architecture: lambda.Architecture.ARM_64,
      description: "R6 reference custom scorer (math_correctness) for submit_eval_job",
    });
    mathScorerFn.node.addDependency(mathScorerBuild.buildCompletion);

    const slurmSkillFn = buildSkillFn(
      "SlurmSkillFn",
      "slurm",
      "Slurm skill — Gateway MCP target",
      { MOCK_MODE: "1" },
    );
    // Web-search skill — Nova Web Grounding (primary) + DuckDuckGo fallback.
    // Needs a longer timeout than the default since Nova Grounding's
    // systemTool loop can issue multiple internal searches per query (the
    // upstream MCP sets read_timeout=3600 on the Bedrock client).
    const webSearchSkillFn = buildSkillFn(
      "WebSearchSkillFn",
      "web_search",
      "Web search skill — Gateway MCP target (Nova Grounding + DDG fallback)",
    );
    webSearchSkillFn.addEnvironment(
      "NOVA_GROUNDING_MODEL_ID",
      "us.amazon.nova-2-lite-v1:0",
    );

    // R8: HyperPod skill — read-only cluster enumeration + SSM-driven
    // version audit on worker nodes. Sibling to the 6 existing skill
    // Lambdas; new 7th Gateway target registered below.
    const hyperpodSkillFn = buildSkillFn(
      "HyperPodSkillFn",
      "hyperpod",
      "HyperPod skill — Gateway MCP target (list_nodes + check_versions)",
    );
    hyperpodSkillFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "HyperPodClusterRead",
        actions: ["sagemaker:DescribeCluster", "sagemaker:ListClusterNodes"],
        // Cluster ARNs are account-scoped; * here means any cluster the
        // caller's Cognito identity can discover via list_cluster_nodes.
        resources: ["*"],
      }),
    );
    hyperpodSkillFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "HyperPodSsmSendAndRead",
        actions: [
          "ssm:SendCommand",
          "ssm:GetCommandInvocation",
          "ssm:DescribeInstanceInformation",
        ],
        // SendCommand must allow the AWS-RunShellScript document and the
        // target instance ARNs. We scope document by name and leave
        // instances at * — a tighter resource scope requires CDK-time
        // knowledge of cluster ARNs which we don't have.
        resources: ["*"],
      }),
    );

    // Evaluation container image — runs mlflow.genai.evaluate() inside a
    // SageMaker Processing job. Built via CodeBuild (same pattern as skill
    // Lambdas) so devs don't need local Docker. The SageMaker skill Lambda
    // gets the image URI as an env var and passes it to CreateProcessingJob.
    const evalImageBuild = new ArmBuildConstruct(this, "EvalImageBuild", {
      sourcePath: path.join(
        __dirname,
        "../../../../lambda/skills/sagemaker/eval",
      ),
      namePrefix: `${props.projectName}-eval-image`,
      // SageMaker Processing job instance-type enum has no Graviton/ARM
      // families, so the eval image must be x86_64.
      targetArchitecture: "x86_64",
    });
    const evalImageUri = `${evalImageBuild.repository.repositoryUri}:${evalImageBuild.imageTag}`;
    sagemakerSkillFn.addEnvironment("EVAL_IMAGE_URI", evalImageUri);
    sagemakerSkillFn.node.addDependency(evalImageBuild.buildCompletion);
    new cdk.CfnOutput(this, "EvalImageUri", { value: evalImageUri });

    // R1 Task 10: monitoring container image. Same shape as eval — built
    // via CodeBuild, passed to the SageMaker skill Lambda as an env var
    // that submit_monitoring_job reads when it calls CreateProcessingJob.
    const monitoringImageBuild = new ArmBuildConstruct(
      this,
      "MonitoringImageBuild",
      {
        sourcePath: path.join(
          __dirname,
          "../../../../lambda/skills/sagemaker/monitoring",
        ),
        namePrefix: `${props.projectName}-monitoring-image`,
        // Same x86_64 reason as eval — SageMaker Processing instance-type
        // enum has no Graviton families.
        targetArchitecture: "x86_64",
      },
    );
    const monitoringImageUri = `${monitoringImageBuild.repository.repositoryUri}:${monitoringImageBuild.imageTag}`;
    sagemakerSkillFn.addEnvironment("MONITORING_IMAGE_URI", monitoringImageUri);
    sagemakerSkillFn.node.addDependency(monitoringImageBuild.buildCompletion);
    new cdk.CfnOutput(this, "MonitoringImageUri", {
      value: monitoringImageUri,
    });

    // Grant skill Lambdas SSM read for tokens
    for (const fn of [
      sagemakerSkillFn,
      huggingfaceSkillFn,
      gitSkillFn,
      mlflowSkillFn,
      slurmSkillFn,
      webSearchSkillFn,
    ]) {
      fn.addToRolePolicy(
        new iam.PolicyStatement({
          actions: ["ssm:GetParameter"],
          resources: [
            `arn:aws:ssm:${props.region}:${props.account}:parameter/${props.projectName}/*`,
          ],
        }),
      );
    }

    // Web-search skill needs bedrock:InvokeModel on the Nova Grounding
    // inference profile (us.amazon.nova-2-lite-v1:0) plus the underlying
    // foundation-model ARN in every region (cross-region profiles resolve
    // to any region at runtime). Mirrors the SageMaker execution role's
    // BedrockEvalInvoke grant shape.
    webSearchSkillFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "NovaGroundingInvoke",
        actions: ["bedrock:InvokeModel", "bedrock:Converse"],
        resources: [
          `arn:aws:bedrock:*::foundation-model/*`,
          `arn:aws:bedrock:*:${props.account}:inference-profile/*`,
        ],
      }),
    );

    // Grant SageMaker Lambda full SageMaker + S3 + DynamoDB access.
    // resources: ["*"] is required because SageMaker training job and endpoint ARNs
    // include runtime-generated timestamps/UUIDs that are not known at CDK synthesis time.
    // GetItem is needed for the idempotency scan in submit_{training,eval}_job
    // that returns the in-flight duplicate instead of double-creating a job.
    sagemakerSkillFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          // BUG-004: list_recent_training_jobs discovery tool + the
          // ThreadId/JobId tag fallback in _find_source_job_entry.
          "sagemaker:ListTrainingJobs",
          "sagemaker:ListTags",
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:CreateModel",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:AddTags",
          // Hub model discovery (read-only) — powers list_hub_models.
          // Needed for the public SageMakerPublicHub enumeration used by
          // the agent to surface Nova/Llama/Qwen models + EULA status
          // before submit_training_job.
          "sagemaker:ListHubs",
          "sagemaker:ListHubContents",
          "sagemaker:ListHubContentVersions",
          "sagemaker:DescribeHubContent",
          // R5: Bedrock Custom Model Import as a deploy_model target.
          // CreateModelImportJob kicks off the async import from the
          // training job's S3 model artifact; Get/List let the agent
          // (and future callback Lambda) track progress.
          "bedrock:CreateModelImportJob",
          "bedrock:GetModelImportJob",
          "bedrock:ListModelImportJobs",
          // R5 closure (live verify 2026-07-25): the import job is tagged
          // ThreadId/JobId for callback resolution — CreateModelImportJob
          // with tags requires TagResource.
          "bedrock:TagResource",
          // R6: Custom scorer Lambdas. The eval Processing container (not
          // this Lambda) is the hot-path invoker — it assumes the
          // SageMaker execution role, which inherits InvokeFunction via
          // the sagemakerRoleProcessingInvoke grant below. The grant
          // here covers only the handler's pre-flight validation path.
          // Scope to function:*-scorer-* is a demo guardrail; production
          // should use an explicit allowlist of scorer ARNs.
          "lambda:GetFunction",
          "s3:PutObject",
          "s3:GetObject",
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "iam:PassRole",
        ],
        resources: ["*"],
      }),
    );

    // The submit_* tools fire a self-invoke (InvocationType=Event) so the
    // slow SageMaker SDK path runs out-of-band and the MCP tool-call returns
    // well within its round-trip budget. grantInvoke(self) creates a circular
    // dependency (role→function→role), so scope the allow-invoke with an ARN
    // pattern instead. The pattern uses ${projectName}-* rather than the
    // stackName because CFN caps auto-generated Lambda names at 64 chars and
    // truncates the stack-name portion (e.g. "…-gateway-" → "…-gatewa-"),
    // which would cause an exact-stackName prefix to silently fail to match.
    sagemakerSkillFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "SelfInvokeForAsyncSubmit",
        actions: ["lambda:InvokeFunction"],
        resources: [
          `arn:aws:lambda:${props.region}:${props.account}:function:${props.projectName}-*SageMakerSkillFn*`,
        ],
      }),
    );

    // MLflow App access — mirrors the AgentCore runtime role grant
    // (science-agent-stack.ts `MLflowAppAccess`). The SageMaker skill pre-creates
    // an MLflow run before each training/eval submission so the container can
    // resume it via MLFLOW_RUN_ID. The MLflow skill needs the same access for
    // query_metrics / analyze_trace / generate_compliance_report (which reads
    // eval_results.parquet from the run artifacts). Without these grants, all
    // MLflow App data-plane calls return HTTP 403.
    const mlflowAppPolicy = new iam.PolicyStatement({
      sid: "MLflowAppAccess",
      actions: [
        "sagemaker:CallMlflowAppApi",
        "sagemaker:DescribeMlflowApp",
        "sagemaker-mlflow:*",
      ],
      resources: [
        `arn:aws:sagemaker:${props.region}:${props.account}:mlflow-app/*`,
      ],
    });
    sagemakerSkillFn.addToRolePolicy(mlflowAppPolicy);
    mlflowSkillFn.addToRolePolicy(mlflowAppPolicy);

    // Recommendation flow — describe and teardown AI Benchmark Job + endpoint
    sagemakerSkillFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "SagemakerSkillRecommendationDescribeAndTeardown",
        actions: [
          "sagemaker:DescribeAIBenchmarkJob",
          "sagemaker:StopAIBenchmarkJob",
          "sagemaker:DeleteEndpoint",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:DeleteInferenceComponent",
          "sagemaker:DeleteModel",
          "sagemaker:DeleteAIWorkloadConfig",
        ],
        resources: ["*"], // resource ARNs are name-scoped; handler filters by DDB lookup.
      }),
    );
    sagemakerSkillFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "SagemakerSkillGetBenchmarkOutput",
        actions: ["s3:GetObject"],
        resources: [
          `arn:aws:s3:::${props.projectName}-sessions-${props.account}-${props.region}/recommendation-benchmarks/*`,
        ],
      }),
    );

    // Session-bucket S3 access — the SageMaker/HuggingFace/MLflow skills all
    // stage datasets, upload training scripts, or load eval data from the shared
    // sessions bucket. Scoped to that one bucket to avoid granting * on S3.
    const sessionBucketArn = `arn:aws:s3:::${props.projectName}-sessions-${props.account}-${props.region}`;
    const sessionBucketPolicy = new iam.PolicyStatement({
      sid: "SessionBucketRW",
      actions: [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
      ],
      resources: [sessionBucketArn, `${sessionBucketArn}/*`],
    });
    huggingfaceSkillFn.addToRolePolicy(sessionBucketPolicy);
    mlflowSkillFn.addToRolePolicy(sessionBucketPolicy);
    // QA BUG-020 follow-up: get_recommendation_results lists the benchmark
    // output prefix to locate output.tar.gz (the AI Benchmark job writes it
    // under a job-named subprefix). Without s3:ListBucket the list call is
    // AccessDenied and missing-key HeadObject probes surface as 403.
    sagemakerSkillFn.addToRolePolicy(sessionBucketPolicy);

    // generate_compliance_report stamps the eval job row with the S3 URI of
    // the rendered report (md + docx). Scope to the thread-row metadata table.
    mlflowSkillFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: "MetadataTableUpdate",
        actions: ["dynamodb:UpdateItem", "dynamodb:GetItem"],
        resources: [
          `arn:aws:dynamodb:${props.region}:${props.account}:table/${props.projectName}-metadata`,
        ],
      }),
    );

    // Slurm skill needs read/write on the metadata table (records task rows)
    // and read on the SSH key secret once real pcluster infra is wired.
    // The metadata table is custom-named and managed outside CDK, so scope
    // by table name pattern rather than importing a Table ref here.
    slurmSkillFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:UpdateItem",
          "dynamodb:Scan",
        ],
        resources: [
          `arn:aws:dynamodb:${props.region}:${props.account}:table/${props.projectName}-metadata`,
        ],
      }),
    );
    slurmSkillFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["secretsmanager:GetSecretValue"],
        resources: [
          `arn:aws:secretsmanager:${props.region}:${props.account}:secret:${props.projectName}/slurm-ssh-key-*`,
        ],
      }),
    );

    // ── Interceptor Lambda ───────────────────────────────────────────────────
    const interceptorFn = new lambda.Function(this, "InterceptorFn", {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: "handler.handler",
      code: lambda.Code.fromAsset(
        path.join(__dirname, "../../../../lambda/interceptor"),
      ),
      timeout: cdk.Duration.seconds(10),
      description:
        "Gateway interceptor — injects _user_id for Cedar evaluation",
    });

    // ── Gateway IAM Role ─────────────────────────────────────────────────────
    const gatewayRole = new iam.Role(this, "GatewayRole", {
      assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
      description:
        "Execution role for AgentCore Gateway - invokes skill Lambdas",
    });

    // Grant gatewayRole invoke on the 7 skill Lambdas (R8 adds hyperpod).
    // Gateway's CreateGatewayTarget validates this grant up-front — missing
    // a function here produces a CREATE_FAILED on the target with message
    // "Gateway execution role lacks permission to invoke Lambda function …".
    for (const fn of [
      sagemakerSkillFn,
      huggingfaceSkillFn,
      gitSkillFn,
      mlflowSkillFn,
      slurmSkillFn,
      webSearchSkillFn,
      hyperpodSkillFn,
    ]) {
      fn.grantInvoke(gatewayRole);
    }
    // Gateway invokes the interceptor on every REQUEST so it can inject
    // _user_id from the JWT before the tool schema is enforced.
    interceptorFn.grantInvoke(gatewayRole);

    // ── Tool schemas — loaded at synth time ──────────────────────────────────
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const toolSchemas: Record<string, any> = {
      sagemaker: JSON.parse(
        fs.readFileSync(
          path.join(
            __dirname,
            "../../../../lambda/skills/sagemaker/tool_spec.json",
          ),
          "utf8",
        ),
      ),
      huggingface: JSON.parse(
        fs.readFileSync(
          path.join(
            __dirname,
            "../../../../lambda/skills/huggingface/tool_spec.json",
          ),
          "utf8",
        ),
      ),
      git: JSON.parse(
        fs.readFileSync(
          path.join(__dirname, "../../../../lambda/skills/git/tool_spec.json"),
          "utf8",
        ),
      ),
      mlflow: JSON.parse(
        fs.readFileSync(
          path.join(
            __dirname,
            "../../../../lambda/skills/mlflow/tool_spec.json",
          ),
          "utf8",
        ),
      ),
      slurm: JSON.parse(
        fs.readFileSync(
          path.join(
            __dirname,
            "../../../../lambda/skills/slurm/tool_spec.json",
          ),
          "utf8",
        ),
      ),
      web_search: JSON.parse(
        fs.readFileSync(
          path.join(
            __dirname,
            "../../../../lambda/skills/web_search/tool_spec.json",
          ),
          "utf8",
        ),
      ),
      // R8: HyperPod skill tool spec (list_nodes, check_versions).
      hyperpod: JSON.parse(
        fs.readFileSync(
          path.join(
            __dirname,
            "../../../../lambda/skills/hyperpod/tool_spec.json",
          ),
          "utf8",
        ),
      ),
    };

    // ── CfnGateway ──────────────────────────────────────────────────────────
    const cognitoDiscoveryUrl = `https://cognito-idp.${props.region}.amazonaws.com/${props.userPoolId}/.well-known/openid-configuration`;

    const cfnGateway = new bedrockagentcore.CfnGateway(
      this,
      "AgentCoreGateway",
      {
        name: `${props.projectName}-gateway`,
        roleArn: gatewayRole.roleArn,
        protocolType: "MCP",
        protocolConfiguration: {
          mcp: { supportedVersions: ["2025-03-26"] },
        },
        authorizerType: "CUSTOM_JWT",
        authorizerConfiguration: {
          customJwtAuthorizer: {
            allowedClients: [props.machineClientId],
            discoveryUrl: cognitoDiscoveryUrl,
          },
        },
        // Interceptor fires on every tool REQUEST — rewrites arguments._user_id
        // into params._injected_user_id so Cedar evaluation sees the JWT sub
        // before the per-tool inputSchema strips unknown keys.
        interceptorConfigurations: [
          {
            interceptor: { lambda: { arn: interceptorFn.functionArn } },
            interceptionPoints: ["REQUEST"],
          },
        ],
        description: "MLOps Agent Gateway — MCP over Cognito JWT",
      },
    );
    cfnGateway.node.addDependency(gatewayRole);
    cfnGateway.node.addDependency(interceptorFn);

    // ── CfnGatewayTargets ────────────────────────────────────────────────────
    const targetDefs: Array<{
      id: string;
      name: string;
      fn: lambda.Function;
      schema: string;
    }> = [
      {
        id: "SageMakerTarget",
        name: "sagemaker-skill",
        fn: sagemakerSkillFn,
        schema: "sagemaker",
      },
      {
        id: "HuggingFaceTarget",
        name: "huggingface-skill",
        fn: huggingfaceSkillFn,
        schema: "huggingface",
      },
      { id: "GitTarget", name: "git-skill", fn: gitSkillFn, schema: "git" },
      {
        id: "MLflowTarget",
        name: "mlflow-skill",
        fn: mlflowSkillFn,
        schema: "mlflow",
      },
      {
        id: "SlurmTarget",
        name: "slurm-skill",
        fn: slurmSkillFn,
        schema: "slurm",
      },
      {
        id: "WebSearchTarget",
        name: "web-search-skill",
        fn: webSearchSkillFn,
        schema: "web_search",
      },
      // R8: HyperPod cluster audit target.
      {
        id: "HyperPodTarget",
        name: "hyperpod-skill",
        fn: hyperpodSkillFn,
        schema: "hyperpod",
      },
    ];

    for (const def of targetDefs) {
      const target = new bedrockagentcore.CfnGatewayTarget(this, def.id, {
        gatewayIdentifier: cfnGateway.attrGatewayIdentifier,
        name: def.name,
        description: `${def.name} Lambda target`,
        targetConfiguration: {
          mcp: {
            lambda: {
              lambdaArn: def.fn.functionArn,
              toolSchema: { inlinePayload: toolSchemas[def.schema] },
            },
          },
        },
        credentialProviderConfigurations: [
          { credentialProviderType: "GATEWAY_IAM_ROLE" },
        ],
      });
      target.addDependency(cfnGateway);
    }

    // ── Patch SSM params Custom Resource ─────────────────────────────────────
    const patchSsmFn = new lambda.Function(this, "PatchGatewaySsmFn", {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: "index.handler",
      code: lambda.Code.fromAsset(
        path.join(__dirname, "patch-gateway-ssm-handler"),
      ),
      timeout: cdk.Duration.minutes(2),
      description:
        "Overwrites 3 Gateway SSM PLACEHOLDERs after CfnGateway deploy",
    });

    patchSsmFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["ssm:GetParameter", "ssm:PutParameter"],
        resources: [
          `arn:aws:ssm:${props.region}:${props.account}:parameter/${props.projectName}/dev/gateway/*`,
        ],
      }),
    );

    const patchProvider = new cr.Provider(this, "PatchGatewaySsmProvider", {
      onEventHandler: patchSsmFn,
    });

    new cdk.CustomResource(this, "PatchGatewaySsm", {
      serviceToken: patchProvider.serviceToken,
      properties: {
        GatewayMcpUrl: cfnGateway.attrGatewayUrl,
        CognitoTokenEndpoint: props.cognitoTokenEndpoint,
        MachineClientId: props.machineClientId,
        GatewayMcpUrlParam: `/${props.projectName}/dev/gateway/mcp-url`,
        CognitoTokenEndpointParam: `/${props.projectName}/dev/gateway/cognito-token-endpoint`,
        MachineClientIdParam: `/${props.projectName}/dev/gateway/m2m-client-id`,
      },
    }).node.addDependency(cfnGateway);

    // ── Store M2M client secret ARN in SSM ────────────────────────────────────
    new ssm.StringParameter(this, "GatewayM2MClientSecretArnParam", {
      parameterName: `/${props.projectName}/dev/gateway/m2m-client-secret-arn`,
      stringValue: props.machineClientSecretArn,
      description:
        "ARN of the Cognito M2M client secret for agent→Gateway auth",
    });

    // ── Store gateway MCP URL parameter name for public access ──────────────────
    this.gatewayMcpUrlParam = `/${props.projectName}/dev/gateway/mcp-url`;

    // ── CloudFormation Outputs ───────────────────────────────────────────────
    new cdk.CfnOutput(this, "GatewayId", {
      value: cfnGateway.attrGatewayIdentifier,
    });
    new cdk.CfnOutput(this, "GatewayUrl", {
      value: cfnGateway.attrGatewayUrl,
    });
    new cdk.CfnOutput(this, "GatewayArn", {
      value: cfnGateway.attrGatewayArn,
    });
  }
}
