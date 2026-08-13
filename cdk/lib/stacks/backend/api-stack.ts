import * as cdk from "aws-cdk-lib";
import * as cognito from "aws-cdk-lib/aws-cognito";
import * as iam from "aws-cdk-lib/aws-iam";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as ssm from "aws-cdk-lib/aws-ssm";
import { Construct } from "constructs";

export interface ApiStackProps extends cdk.StackProps {
  agentRuntimeArn: string;
  metadataTableArn: string;
  mlflowAppArn: string;
  projectName: string;
  sessionBucketArn: string;
}

export class ApiStack extends cdk.Stack {
  public readonly userPoolId: string;
  public readonly userPoolArn: string;
  public readonly userPoolClientId: string;
  public readonly userPoolDomainPrefix: string;
  public readonly identityPoolId: string;
  public readonly machineClientId: string;
  public readonly cognitoTokenEndpoint: string;
  public readonly machineClientSecretArn: string;

  constructor(scope: Construct, id: string, props: ApiStackProps) {
    super(scope, id, props);

    // Cognito domain prefix: AWS limit is 63 chars, lowercase alphanumeric + hyphens
    // Use truncated projectName (40 chars) + hyphen + account ID (12 chars) = max 53 chars
    const safeDomainPrefix = props.projectName.substring(0, 40);
    const userPool = new cognito.UserPool(this, "UserPool", {
      selfSignUpEnabled: false,
      signInAliases: { email: true },
      autoVerify: { email: true },
    });
    const hostedUiDomain = userPool.addDomain("HostedUiDomain", {
      cognitoDomain: {
        domainPrefix: cdk.Fn.sub(`${safeDomainPrefix}-\${AWS::AccountId}`),
      },
    });
    this.userPoolDomainPrefix = hostedUiDomain.domainName;
    this.userPoolId = userPool.userPoolId;
    this.userPoolArn = userPool.userPoolArn;

    // ── M2M: Resource Server + Machine Client ─────────────────────────────────
    const resourceServer = new cognito.UserPoolResourceServer(
      this,
      "GatewayResourceServer",
      {
        userPool,
        identifier: `${props.projectName}-gateway`,
        userPoolResourceServerName: `${props.projectName}-gateway`,
        scopes: [
          new cognito.ResourceServerScope({
            scopeName: "invoke",
            scopeDescription: "Invoke Gateway tools",
          }),
        ],
      },
    );

    const machineClient = userPool.addClient("GatewayMachineClient", {
      userPoolClientName: `${props.projectName}-gateway-machine`,
      generateSecret: true,
      authFlows: {},
      oAuth: {
        flows: { clientCredentials: true },
        scopes: [
          cognito.OAuthScope.resourceServer(
            resourceServer,
            new cognito.ResourceServerScope({
              scopeName: "invoke",
              scopeDescription: "Invoke Gateway tools",
            }),
          ),
        ],
      },
    });
    machineClient.node.addDependency(resourceServer);

    // Store machine client secret in Secrets Manager so the agent can fetch it at runtime
    const machineClientSecret = new secretsmanager.Secret(
      this,
      "GatewayMachineClientSecret",
      {
        secretName: `/${props.projectName}/dev/gateway-m2m-client-secret`,
        secretStringValue: cdk.SecretValue.unsafePlainText(
          machineClient.userPoolClientSecret.unsafeUnwrap(),
        ),
        description:
          "Cognito machine client secret for AgentCore Gateway M2M auth",
      },
    );

    this.machineClientId = machineClient.userPoolClientId;
    // Cognito token endpoint: https://{domain}.auth.{region}.amazoncognito.com/oauth2/token
    this.cognitoTokenEndpoint = `https://${hostedUiDomain.domainName}.auth.${this.region}.amazoncognito.com/oauth2/token`;
    this.machineClientSecretArn = machineClientSecret.secretArn;

    const userPoolClient = userPool.addClient("AppClient", {
      authFlows: { userSrp: true },
      oAuth: {
        flows: { authorizationCodeGrant: true },
        scopes: [cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL],
        // callbackUrls placeholder — overwritten at deploy time by ScienceAgentUiStack
        // via AwsCustomResource (UpdateUserPoolClient) once the CloudFront URL is known.
        callbackUrls: ["http://localhost:5173"],
        logoutUrls: ["http://localhost:5173"],
      },
      preventUserExistenceErrors: true,
    });
    this.userPoolClientId = userPoolClient.userPoolClientId;

    // ── Brand the Cognito Hosted UI (login/logout) to match the app ──────────
    // The classic Hosted UI only accepts a restricted CSS allowlist; these
    // hex values mirror frontend/src/lib/brandTheme.ts (BRAND) so the login
    // page and the Cloudscape app read as one product. Keep them in sync.
    const hostedUiBranding = new cognito.CfnUserPoolUICustomizationAttachment(
      this,
      "HostedUiBranding",
      {
        userPoolId: userPool.userPoolId,
        clientId: userPoolClient.userPoolClientId,
        css: [
          ".banner-customizable {",
          "  background: linear-gradient(135deg, #08201e 0%, #0b7268 100%);",
          "  padding: 28px 0;",
          "}",
          ".submitButton-customizable {",
          "  background-color: #0b7268;",
          "  font-weight: 600;",
          "  border-radius: 8px;",
          "}",
          ".submitButton-customizable:hover {",
          "  background-color: #095f57;",
          "}",
          ".inputField-customizable {",
          "  border-radius: 8px;",
          "  border: 1px solid #b9c6c3;",
          "}",
          ".inputField-customizable:focus {",
          "  border-color: #0b7268;",
          "  box-shadow: 0 0 0 2px rgba(11,114,104,0.25);",
          "}",
          ".label-customizable { color: #08201e; font-weight: 600; }",
          ".textDescription-customizable { color: #4a5a5a; }",
          ".redirect-customizable { color: #c8781e; }",
          ".background-customizable { background-color: #f4f7f6; }",
        ].join("\n"),
      },
    );

    // Branding requires an existing Hosted UI domain. Declaring the dependency
    // makes CloudFormation create the domain first AND — critically for a clean
    // `cdk destroy` — delete the branding BEFORE the domain. Without this, CFN
    // may delete the domain first, then the branding delete fails with "there
    // has to be an existing domain associated with this user pool", leaving the
    // stack in DELETE_FAILED and requiring a manual --retain-resources delete.
    hostedUiBranding.node.addDependency(hostedUiDomain);

    // Logical id suffixed "V2" to force CloudFormation to create a fresh
    // Identity Pool: the original pool was deleted out-of-band in the console,
    // but CFN's stored template still contained it, so an identical-template
    // deploy was a no-op. Renaming the logical id makes CFN create the new
    // pool and issue an (idempotent, already-satisfied) delete of the old one.
    // The pool id is consumed by the UI stack via SSM (not a CFN export), so
    // the new id propagates on the UI redeploy — no cross-stack export breaks.
    const identityPool = new cognito.CfnIdentityPool(this, "IdentityPoolV2", {
      allowUnauthenticatedIdentities: false,
      cognitoIdentityProviders: [
        {
          clientId: userPoolClient.userPoolClientId,
          providerName: userPool.userPoolProviderName,
        },
      ],
    });
    this.identityPoolId = identityPool.ref;

    const authenticatedRole = new iam.Role(this, "IdentityPoolAuthRole", {
      assumedBy: new iam.FederatedPrincipal(
        "cognito-identity.amazonaws.com",
        {
          StringEquals: {
            "cognito-identity.amazonaws.com:aud": identityPool.ref,
          },
          "ForAnyValue:StringLike": {
            "cognito-identity.amazonaws.com:amr": "authenticated",
          },
        },
        "sts:AssumeRoleWithWebIdentity",
      ),
    });
    authenticatedRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["bedrock-agentcore:InvokeAgentRuntime"],
        // AgentCore accepts two URL forms:
        //   /runtimes/{runtimeArn}/invocations               → IAM resource: runtimeArn
        //   /runtimes/{runtimeArn}/runtime-endpoint/*/invocations → IAM resource: runtimeArn/runtime-endpoint/*
        // Allow both so either URL shape succeeds.
        resources: [
          props.agentRuntimeArn,
          `${props.agentRuntimeArn}/runtime-endpoint/*`,
        ],
      }),
    );
    const tableGsi = `${props.metadataTableArn}/index/*`;
    authenticatedRole.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          "dynamodb:Query",
          "dynamodb:GetItem",
          "dynamodb:Scan",
          "dynamodb:UpdateItem",
        ],
        resources: [props.metadataTableArn, tableGsi],
      }),
    );
    authenticatedRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "MlflowPresignedUrl",
        actions: ["sagemaker:CreatePresignedMlflowAppUrl"],
        resources: [props.mlflowAppArn],
      }),
    );
    authenticatedRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "ComplianceReportDownload",
        actions: ["s3:GetObject"],
        resources: [`${props.sessionBucketArn}/compliance-reports/*`],
      }),
    );

    // ── Skills store CRUD (Skills tab editor, browser-direct) ────────────────
    authenticatedRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "SkillsStoreReadWrite",
        actions: ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
        resources: [`${props.sessionBucketArn}/skills/*`],
      }),
    );
    authenticatedRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "SkillsStoreList",
        actions: ["s3:ListBucket"],
        resources: [props.sessionBucketArn],
        conditions: { StringLike: { "s3:prefix": ["skills/*"] } },
      }),
    );

    // ── Evals tab: metrics + logs reads (browser-direct) ─────────────────────
    authenticatedRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "EvalMetricsRead",
        // GetMetricData / DescribeLogGroups / StartQuery are not resource-scopable.
        actions: [
          "cloudwatch:GetMetricData",
          "logs:StartQuery",
          "logs:StopQuery",
          "logs:GetQueryResults",
          "logs:DescribeLogGroups",
        ],
        resources: ["*"],
      }),
    );

    // ── Evals tab: trigger batch eval + optimization (browser-direct) ────────
    // M4 resolved: start-batch-evaluation / start-recommendation take NO
    // execution-role arg (the role is bound at online-eval-config creation in
    // the evals stack), so NO iam:PassRole is required here. Scoped to this
    // account's agentcore evaluation/recommendation resources.
    authenticatedRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "EvalTriggers",
        actions: [
          "bedrock-agentcore:StartBatchEvaluation",
          "bedrock-agentcore:GetBatchEvaluation",
          "bedrock-agentcore:ListBatchEvaluations",
          "bedrock-agentcore:StartRecommendation",
          "bedrock-agentcore:GetRecommendation",
          "bedrock-agentcore:ListRecommendations",
        ],
        resources: [
          `arn:aws:bedrock-agentcore:${this.region}:${this.account}:*`,
        ],
      }),
    );

    // ── Evals tab: create the batch/recommendation results log group ─────────
    // When the browser triggers StartBatchEvaluation / StartRecommendation,
    // AgentCore uses the CALLER's forwarded (FAS) credentials — i.e. this
    // authenticated role — to create the service-generated results log group.
    // Without CreateLogGroup the UI fails with "FAS credentials do not have
    // permission to create CloudWatch log groups". Scope to the evaluations
    // results/batch-evaluations prefixes.
    authenticatedRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "EvalResultsLogGroupWrite",
        actions: [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:PutRetentionPolicy",
        ],
        resources: [
          `arn:aws:logs:${this.region}:${this.account}:log-group:/aws/bedrock-agentcore/evaluations/*`,
          `arn:aws:logs:${this.region}:${this.account}:log-group:/aws/bedrock-agentcore/evaluations/*:*`,
        ],
      }),
    );

    // Renamed alongside IdentityPoolV2 — the attachment references the pool's
    // ref and must be recreated against the new pool.
    new cognito.CfnIdentityPoolRoleAttachment(this, "IdentityPoolRolesV2", {
      identityPoolId: identityPool.ref,
      roles: { authenticated: authenticatedRole.roleArn },
    });

    // Write identityPoolId to SSM so ScienceAgentUiStack can read it without
    // a CFN cross-stack export dependency (avoids CFN export lock on updates).
    new ssm.StringParameter(this, "IdentityPoolIdParam", {
      parameterName: `/${props.projectName}/dev/cognito/identity-pool-id`,
      stringValue: this.identityPoolId,
    });

    new cdk.CfnOutput(this, "UserPoolId", { value: this.userPoolId });
    new cdk.CfnOutput(this, "UserPoolClientId", {
      value: this.userPoolClientId,
    });
    new cdk.CfnOutput(this, "UserPoolDomainPrefix", {
      value: this.userPoolDomainPrefix,
    });
    new cdk.CfnOutput(this, "IdentityPoolId", { value: this.identityPoolId });
    new cdk.CfnOutput(this, "MachineClientId", { value: this.machineClientId });
  }
}
