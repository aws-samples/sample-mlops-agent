import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import * as ssm from "aws-cdk-lib/aws-ssm";
import * as path from "path";
import { Construct } from "constructs";

export interface EvalsStackProps extends cdk.StackProps {
  projectName: string;
  /** AgentCore runtime name, e.g. sample_mlops_agent (underscored). */
  runtimeName: string;
  /** Runtime application log group name that receives OTEL gen-ai event records. */
  runtimeLogGroupName: string;
}

/**
 * AgentCore Evaluations infra: the custom confirmation-gate evaluator Lambda,
 * its evaluator registration, an online-evaluation config over the agent's live
 * sessions, and CloudWatch Transaction Search enablement (prereq for reading
 * `aws/spans`). Both AWS::BedrockAgentCore::Evaluator and
 * ::OnlineEvaluationConfig CFN types were verified present (describe-type) so we
 * use CfnResource directly (Path A) — no custom-resource fallback needed.
 */
export class EvalsStack extends cdk.Stack {
  public readonly evaluatorArn: string;
  public readonly onlineEvalConfigName: string;
  public readonly evalResultsLogGroupPrefix: string;
  public readonly evalExecRoleArn: string;

  constructor(scope: Construct, id: string, props: EvalsStackProps) {
    super(scope, id, props);

    const serviceName = `${props.runtimeName}.DEFAULT`; // real OTEL service.name
    const spansLogGroup = "aws/spans";

    // ── Custom confirmation-gate evaluator Lambda ────────────────────────────
    const evaluatorFn = new lambda.Function(this, "ConfirmationGateEvaluator", {
      functionName: `${props.projectName}-eval-confirmation-gate`,
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64,
      handler: "handler.handle",
      code: lambda.Code.fromAsset(
        path.join(__dirname, "../../../../lambda/eval_confirmation_gate"),
      ),
      timeout: cdk.Duration.seconds(60),
      memorySize: 256,
      logRetention: logs.RetentionDays.ONE_MONTH,
    });

    // ── Evaluator registration (code-based, SESSION level → Lambda) ──────────
    const evaluator = new cdk.CfnResource(
      this,
      "ConfirmationGateEvaluatorReg",
      {
        type: "AWS::BedrockAgentCore::Evaluator",
        properties: {
          EvaluatorName: `${props.runtimeName}_confirmation_gate`,
          Description:
            "Scores whether the agent confirmed before billable tool calls.",
          Level: "SESSION",
          EvaluatorConfig: {
            CodeBased: {
              LambdaConfig: {
                LambdaArn: evaluatorFn.functionArn,
                LambdaTimeoutInSeconds: 60,
              },
            },
          },
        },
      },
    );
    this.evaluatorArn = evaluator.getAtt("EvaluatorArn").toString();
    const evaluatorId = evaluator.getAtt("EvaluatorId").toString();

    // ── eval_exec role assumed by the AgentCore evaluation service ───────────
    const evalExec = new iam.Role(this, "EvalExecRole", {
      assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
      description:
        "Assumed by AgentCore Evaluations to run judges + custom evaluator.",
    });
    // The CreateOnlineEvaluationConfig API validates that this role has
    // log-group-SCOPED read access to every configured log group (a `*`
    // resource is rejected). Name aws/spans + the runtime log group explicitly.
    const spansArn = `arn:aws:logs:${this.region}:${this.account}:log-group:${spansLogGroup}:*`;
    const runtimeLgArn = `arn:aws:logs:${this.region}:${this.account}:log-group:${props.runtimeLogGroupName}:*`;
    evalExec.addToPolicy(
      new iam.PolicyStatement({
        sid: "ReadConfiguredLogGroups",
        actions: [
          "logs:StartQuery",
          "logs:StopQuery",
          "logs:GetQueryResults",
          "logs:GetLogEvents",
          "logs:FilterLogEvents",
        ],
        resources: [spansArn, runtimeLgArn],
      }),
    );
    evalExec.addToPolicy(
      new iam.PolicyStatement({
        sid: "DescribeLogGroups",
        // DescribeLogGroups genuinely cannot be resource-scoped.
        actions: ["logs:DescribeLogGroups"],
        resources: ["*"],
      }),
    );
    // The eval service writes per-session results into a service-generated log
    // group under /aws/bedrock-agentcore/evaluations/results/* and validates it
    // can create + write that group at config-create time.
    evalExec.addToPolicy(
      new iam.PolicyStatement({
        sid: "WriteEvalResultsLogGroup",
        actions: [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ],
        resources: [
          `arn:aws:logs:${this.region}:${this.account}:log-group:/aws/bedrock-agentcore/evaluations/*`,
          `arn:aws:logs:${this.region}:${this.account}:log-group:/aws/bedrock-agentcore/evaluations/*:*`,
        ],
      }),
    );
    evalExec.addToPolicy(
      new iam.PolicyStatement({
        sid: "InvokeBuiltinJudges",
        actions: ["bedrock:InvokeModel"],
        resources: ["*"], // builtin judge model ids are managed by the service
      }),
    );
    evalExec.addToPolicy(
      new iam.PolicyStatement({
        sid: "InvokeCustomEvaluator",
        // The eval service validates BOTH GetFunction (to resolve the evaluator)
        // and InvokeFunction at config-create time.
        actions: ["lambda:InvokeFunction", "lambda:GetFunction"],
        resources: [evaluatorFn.functionArn],
      }),
    );
    evalExec.addToPolicy(
      new iam.PolicyStatement({
        sid: "PublishEvalMetrics",
        actions: ["cloudwatch:PutMetricData"],
        resources: ["*"],
        conditions: {
          StringEquals: {
            "cloudwatch:namespace": "Bedrock-AgentCore/Evaluations",
          },
        },
      }),
    );
    evaluatorFn.grantInvoke(
      new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
    );
    this.evalExecRoleArn = evalExec.roleArn;

    // ── Online evaluation config over the live runtime sessions ──────────────
    const configName = `${props.runtimeName}_online_eval`;
    const onlineCfg = new cdk.CfnResource(this, "OnlineEvalConfig", {
      type: "AWS::BedrockAgentCore::OnlineEvaluationConfig",
      properties: {
        OnlineEvaluationConfigName: configName,
        Description: "Continuous scoring of live MLOps-agent sessions.",
        // Enable scoring on create — the resource otherwise lands DISABLED and
        // no sessions get evaluated (metrics never appear in the Evals tab).
        ExecutionStatus: "ENABLED",
        EvaluationExecutionRoleArn: evalExec.roleArn,
        DataSourceConfig: {
          CloudWatchLogs: {
            // ServiceNames maxItems=1 — one runtime, one service.
            LogGroupNames: [spansLogGroup, props.runtimeLogGroupName],
            ServiceNames: [serviceName],
          },
        },
        Rule: { SamplingConfig: { SamplingPercentage: 100 } },
        Evaluators: [
          { EvaluatorId: "Builtin.GoalSuccessRate" },
          { EvaluatorId: "Builtin.Helpfulness" },
          { EvaluatorId: "Builtin.Correctness" },
          { EvaluatorId: evaluatorId },
        ],
      },
    });
    onlineCfg.node.addDependency(evaluator);
    // CRITICAL ordering: the config references evalExec.roleArn, which makes CFN
    // depend on the Role but NOT on its inline DefaultPolicy. Without this, the
    // config is created before the log-read policy attaches and the API's
    // permission validation fails ("does not have permissions to access the
    // specified log groups"). Depend on the role's default policy explicitly.
    if (evalExec.node.tryFindChild("DefaultPolicy")) {
      onlineCfg.node.addDependency(evalExec.node.findChild("DefaultPolicy"));
    }
    this.onlineEvalConfigName = configName;
    // Service-generated results log group (discovered by prefix in the UI).
    this.evalResultsLogGroupPrefix = `/aws/bedrock-agentcore/evaluations/results/${configName}`;

    // ── Transaction Search (indexes aws/spans; prereq for online eval) ───────
    // NOT managed here. Transaction Search is an ACCOUNT-WIDE X-Ray setting and
    // is already enabled by the agentcore runtime's AGENT_OBSERVABILITY_ENABLED
    // (verified at deploy: updateTraceSegmentDestination returns "destination is
    // already set to CloudWatchLogs"). Managing it from this stack added a
    // failure mode (the API errors when it's already set) with no benefit, so we
    // treat it as a satisfied account-level prerequisite. If it were ever off,
    // enable once via: aws xray update-trace-segment-destination --destination
    // CloudWatchLogs. The empty-state in the Evals tab's SummaryPanel already
    // tells operators to check this if metrics never appear.

    // ── SSM params so api-stack + ui-stack import without a CFN export lock ──
    // (same pattern the codebase uses for the identity-pool id / mlflow arn).
    const p = `/${props.projectName}/dev`;
    new ssm.StringParameter(this, "EvalExecRoleArnParam", {
      parameterName: `${p}/evals/exec-role-arn`,
      stringValue: this.evalExecRoleArn,
    });
    new ssm.StringParameter(this, "OnlineEvalConfigNameParam", {
      parameterName: `${p}/evals/online-config-name`,
      stringValue: configName,
    });
    new ssm.StringParameter(this, "EvalResultsPrefixParam", {
      parameterName: `${p}/evals/results-log-group-prefix`,
      stringValue: this.evalResultsLogGroupPrefix,
    });

    // ── Outputs (consumed by api-stack IAM + ui-stack config.json) ───────────
    new cdk.CfnOutput(this, "EvaluatorArn", { value: this.evaluatorArn });
    new cdk.CfnOutput(this, "EvaluatorId", { value: evaluatorId });
    new cdk.CfnOutput(this, "OnlineEvalConfigName", { value: configName });
    new cdk.CfnOutput(this, "EvalResultsLogGroupPrefix", {
      value: this.evalResultsLogGroupPrefix,
    });
    new cdk.CfnOutput(this, "EvalExecRoleArn", { value: this.evalExecRoleArn });
  }
}
