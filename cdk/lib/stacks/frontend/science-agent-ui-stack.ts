import * as cdk from "aws-cdk-lib";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";
import * as origins from "aws-cdk-lib/aws-cloudfront-origins";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import { LogGroup } from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as s3assets from "aws-cdk-lib/aws-s3-assets";
import * as ssm from "aws-cdk-lib/aws-ssm";
import * as wafv2 from "aws-cdk-lib/aws-wafv2";
import * as cr from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";
import { execFileSync } from "child_process";
import * as path from "path";
import { CloudFrontCognitoIntegration } from "../../constructs/cloudfront-cognito-integration";

export interface UiStackProps extends cdk.StackProps {
  userPoolId: string;
  userPoolClientId: string;
  userPoolDomainPrefix: string;
  cognitoRegion: string;
  projectName: string;
}

export class ScienceAgentUiStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: UiStackProps) {
    super(scope, id, props);

    const siteBucket = new s3.Bucket(this, "SiteBucket", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    const oac = new cloudfront.S3OriginAccessControl(this, "OAC");

    // WAF WebACL — scope must be CLOUDFRONT, which requires us-east-1.
    // This stack defaults to us-east-1 (CDK_DEFAULT_REGION ?? 'us-east-1').
    // AWSManagedRulesCommonRuleSet blocks OWASP Top 10 patterns (SQLi, XSS, etc.).
    const webAcl = new wafv2.CfnWebACL(this, "CloudFrontWebACL", {
      scope: "CLOUDFRONT",
      defaultAction: { allow: {} },
      visibilityConfig: {
        cloudWatchMetricsEnabled: true,
        metricName: "CloudFrontWebACL",
        sampledRequestsEnabled: true,
      },
      rules: [
        {
          name: "AWSManagedRulesCommonRuleSet",
          priority: 1,
          overrideAction: { none: {} },
          statement: {
            managedRuleGroupStatement: {
              vendorName: "AWS",
              name: "AWSManagedRulesCommonRuleSet",
            },
          },
          visibilityConfig: {
            cloudWatchMetricsEnabled: true,
            metricName: "AWSManagedRulesCommonRuleSet",
            sampledRequestsEnabled: true,
          },
        },
      ],
    });

    const distribution = new cloudfront.Distribution(this, "Distribution", {
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(siteBucket, {
          originAccessControl: oac,
        }),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
      },
      defaultRootObject: "index.html",
      errorResponses: [
        {
          httpStatus: 404,
          responseHttpStatus: 200,
          responsePagePath: "/index.html",
        },
      ],
      // Enforce TLS 1.2+ (removes support for TLS 1.0/1.1)
      minimumProtocolVersion: cloudfront.SecurityPolicyProtocol.TLS_V1_2_2021,
      // Attach WAF WebACL to block OWASP Top 10 attack patterns
      webAclId: webAcl.attrArn,
    });

    // Read identityPoolId and agentCoreEndpoint from SSM to avoid CFN cross-stack
    // export dependencies that would lock stacks from updating independently.
    const identityPoolId = ssm.StringParameter.valueForStringParameter(
      this,
      `/${props.projectName}/dev/cognito/identity-pool-id`,
    );
    const agentCoreEndpoint = ssm.StringParameter.valueForStringParameter(
      this,
      `/${props.projectName}/dev/agentcore/endpoint`,
    );
    const mlflowAppArn = ssm.StringParameter.valueForStringParameter(
      this,
      `/${props.projectName}/dev/mlflow/tracking-server-arn`,
    );
    // Skills store + Evals tab identifiers. The session bucket follows the
    // project's fixed naming convention; the eval identifiers come from SSM
    // params published by the evals stack (no CFN export lock).
    const sessionBucket = `${props.projectName}-sessions-${this.account}-${this.region}`;
    const onlineEvalConfigName = ssm.StringParameter.valueForStringParameter(
      this,
      `/${props.projectName}/dev/evals/online-config-name`,
    );
    const evalResultsLogGroupPrefix =
      ssm.StringParameter.valueForStringParameter(
        this,
        `/${props.projectName}/dev/evals/results-log-group-prefix`,
      );
    const agentRuntimeName = props.projectName.replace(/-/g, "_");
    // Real runtime app log group (/runtimes/<runtimeId>-DEFAULT) that holds
    // spans + LLO events — the batch-eval data source must read from THIS, not
    // /runtimes/<name>. Published to SSM by the agentcore stack.
    const runtimeLogGroup = ssm.StringParameter.valueForStringParameter(
      this,
      `/${props.projectName}/dev/agentcore/runtime-log-group`,
    );
    const runtimeLogGroupArn = ssm.StringParameter.valueForStringParameter(
      this,
      `/${props.projectName}/dev/agentcore/runtime-log-group-arn`,
    );

    const cloudfrontUrl = `https://${distribution.distributionDomainName}`;
    const cognitoDomain = `https://${props.userPoolDomainPrefix}.auth.${props.cognitoRegion}.amazoncognito.com`;

    // Stage agent skill docs + architecture diagram into frontend/.agent-skills/
    // so they are included in the FrontendAsset zip (CodeBuild only sees the
    // zipped frontend tree; it cannot reach back into agent/.claude/skills/).
    const frontendDir = path.join(__dirname, "../../../../frontend");
    execFileSync(
      "node",
      [path.join(frontendDir, "scripts/copy-agent-skills.mjs")],
      {
        stdio: "inherit",
      },
    );

    // Upload frontend source as CDK asset — zips and stores in bootstrap S3 bucket
    const frontendAsset = new s3assets.Asset(this, "FrontendAsset", {
      path: frontendDir,
      exclude: [
        "node_modules",
        "node_modules/**",
        "dist",
        "dist/**",
        ".git",
        ".git/**",
        ".env.local",
        ".env.*.local",
      ],
    });

    // CodeBuild: npm ci + vite build + write config.json + deploy to S3 + invalidate CloudFront
    const buildProject = new codebuild.Project(this, "FrontendBuild", {
      projectName: `${this.stackName}-frontend-build`,
      source: codebuild.Source.s3({
        bucket: frontendAsset.bucket,
        path: frontendAsset.s3ObjectKey,
      }),
      artifacts: codebuild.Artifacts.s3({
        bucket: siteBucket,
        includeBuildId: false,
        packageZip: false,
        name: "/",
        encryption: false,
      }),
      environment: {
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
        computeType: codebuild.ComputeType.SMALL,
      },
      environmentVariables: {
        VITE_IDENTITY_POOL_ID: { value: identityPoolId },
        VITE_AGENTCORE_ENDPOINT: { value: agentCoreEndpoint },
        VITE_JOBS_TABLE_NAME: { value: `${props.projectName}-metadata` },
        COGNITO_CLIENT_ID: { value: props.userPoolClientId },
        COGNITO_DOMAIN: { value: cognitoDomain },
        MLFLOW_APP_ARN: { value: mlflowAppArn },
        SESSION_BUCKET: { value: sessionBucket },
        SKILLS_PREFIX: { value: "skills/" },
        ONLINE_EVAL_CONFIG_NAME: { value: onlineEvalConfigName },
        EVAL_RESULTS_LOG_GROUP_PREFIX: { value: evalResultsLogGroupPrefix },
        AGENT_RUNTIME_NAME: { value: agentRuntimeName },
        RUNTIME_LOG_GROUP: { value: runtimeLogGroup },
        RUNTIME_LOG_GROUP_ARN: { value: runtimeLogGroupArn },
        DISTRIBUTION_ID: { value: distribution.distributionId },
        ASSET_HASH: { value: frontendAsset.assetHash },
      },
      buildSpec: codebuild.BuildSpec.fromObject({
        version: "0.2",
        phases: {
          install: {
            "runtime-versions": { nodejs: "20" },
            commands: ["rm -f .env.local .env.*.local", "npm ci"],
          },
          build: {
            commands: ["npm run build"],
          },
          post_build: {
            commands: [
              'printf \'{"agentCoreEndpoint":"%s","identityPoolId":"%s","cognitoClientId":"%s","cognitoDomain":"%s","mlflowAppArn":"%s","sessionBucket":"%s","skillsPrefix":"%s","onlineEvalConfigName":"%s","evalResultsLogGroupPrefix":"%s","agentRuntimeName":"%s","runtimeLogGroup":"%s","runtimeLogGroupArn":"%s"}\' "$VITE_AGENTCORE_ENDPOINT" "$VITE_IDENTITY_POOL_ID" "$COGNITO_CLIENT_ID" "$COGNITO_DOMAIN" "$MLFLOW_APP_ARN" "$SESSION_BUCKET" "$SKILLS_PREFIX" "$ONLINE_EVAL_CONFIG_NAME" "$EVAL_RESULTS_LOG_GROUP_PREFIX" "$AGENT_RUNTIME_NAME" "$RUNTIME_LOG_GROUP" "$RUNTIME_LOG_GROUP_ARN" > dist/config.json',
              'aws cloudfront create-invalidation --distribution-id $DISTRIBUTION_ID --paths "/*"',
            ],
          },
        },
        artifacts: {
          files: ["**/*"],
          "base-directory": "dist",
        },
      }),
      timeout: cdk.Duration.minutes(15),
      logging: {
        cloudWatch: {
          logGroup: new LogGroup(this, "FrontendBuildLogs", {
            retention: logs.RetentionDays.ONE_DAY,
            removalPolicy: cdk.RemovalPolicy.DESTROY,
          }),
        },
      },
    });

    siteBucket.grantReadWrite(buildProject);
    frontendAsset.grantRead(buildProject);
    buildProject.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["cloudfront:CreateInvalidation"],
        resources: [
          `arn:aws:cloudfront::${this.account}:distribution/${distribution.distributionId}`,
        ],
      }),
    );

    // onEvent: start the CodeBuild build
    const onEventFn = new lambda.Function(this, "OnEventFn", {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: "index.handler",
      timeout: cdk.Duration.minutes(1),
      logGroup: new LogGroup(this, "OnEventFnLogs", {
        retention: logs.RetentionDays.ONE_DAY,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      }),
      code: lambda.Code.fromInline(`
import boto3

codebuild_client = boto3.client('codebuild')

def handler(event, context):
    print(f"Event: {event}")
    if event['RequestType'] == 'Delete':
        return {'PhysicalResourceId': event.get('PhysicalResourceId', 'frontend-build')}
    project_name = event['ResourceProperties']['ProjectName']
    response = codebuild_client.start_build(projectName=project_name)
    build_id = response['build']['id']
    print(f"Build started: {build_id}")
    return {
        'PhysicalResourceId': f"{project_name}-build",
        'Data': {'BuildId': build_id},
    }
`),
    });

    // isComplete: poll until build succeeds or fails
    const isCompleteFn = new lambda.Function(this, "IsCompleteFn", {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: "index.handler",
      timeout: cdk.Duration.minutes(1),
      logGroup: new LogGroup(this, "IsCompleteFnLogs", {
        retention: logs.RetentionDays.ONE_DAY,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      }),
      code: lambda.Code.fromInline(`
import boto3

codebuild_client = boto3.client('codebuild')

def handler(event, context):
    print(f"Event: {event}")
    if event['RequestType'] == 'Delete':
        return {'IsComplete': True}
    build_id = event.get('Data', {}).get('BuildId')
    if not build_id:
        raise Exception("BuildId missing from event data")
    builds = codebuild_client.batch_get_builds(ids=[build_id]).get('builds', [])
    if not builds:
        raise Exception(f"Build {build_id} not found")
    status = builds[0]['buildStatus']
    print(f"Build status: {status}")
    if status == 'SUCCEEDED':
        return {'IsComplete': True, 'Data': {'BuildId': build_id, 'Status': status}}
    elif status in ('FAILED', 'FAULT', 'STOPPED', 'TIMED_OUT'):
        logs_link = builds[0].get('logs', {}).get('deepLink', '')
        raise Exception(f"Frontend build failed ({status}). Logs: {logs_link}")
    return {'IsComplete': False}
`),
    });

    onEventFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["codebuild:StartBuild"],
        resources: [buildProject.projectArn],
      }),
    );
    isCompleteFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["codebuild:BatchGetBuilds"],
        resources: [buildProject.projectArn],
      }),
    );

    const buildProvider = new cr.Provider(this, "BuildProvider", {
      onEventHandler: onEventFn,
      isCompleteHandler: isCompleteFn,
      queryInterval: cdk.Duration.seconds(30),
      totalTimeout: cdk.Duration.minutes(15),
      logGroup: new LogGroup(this, "BuildProviderLogs", {
        retention: logs.RetentionDays.ONE_DAY,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      }),
    });

    const buildTrigger = new cdk.CustomResource(this, "BuildTrigger", {
      serviceToken: buildProvider.serviceToken,
      properties: {
        ProjectName: buildProject.projectName,
        AssetHash: frontendAsset.assetHash,
      },
    });
    buildTrigger.node.addDependency(distribution);

    // Wire CloudFront URL into Cognito callbackUrls automatically.
    new CloudFrontCognitoIntegration(this, "CloudFrontCognitoIntegration", {
      distribution,
      userPoolId: props.userPoolId,
      userPoolClientId: props.userPoolClientId,
      region: this.region,
    });

    new cdk.CfnOutput(this, "CloudFrontUrl", {
      value: cloudfrontUrl,
    });
  }
}
