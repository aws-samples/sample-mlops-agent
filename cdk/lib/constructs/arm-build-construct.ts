import * as cdk from "aws-cdk-lib";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3assets from "aws-cdk-lib/aws-s3-assets";
import * as cr from "aws-cdk-lib/custom-resources";
import { LogGroup } from "aws-cdk-lib/aws-logs";
import * as crypto from "crypto";
import * as fs from "fs";
import * as path from "path";
import { Construct } from "constructs";

export interface ArmBuildConstructProps {
  /** Absolute path to the directory containing the Dockerfile and source code */
  readonly sourcePath: string;
  /** ECR repository name (optional — CDK auto-generates if omitted) */
  readonly repositoryName?: string;
  /** Human-readable name prefix for CodeBuild project and related resources */
  readonly namePrefix: string;
  /** Dockerfile filename (default: Dockerfile) */
  readonly dockerfileName?: string;
  /** Build timeout in minutes (default: 30) */
  readonly buildTimeoutMinutes?: number;
  /**
   * Target architecture for the built image (default: `arm64`).
   *
   * Use `x86_64` for images pulled by services that don't support Graviton —
   * notably SageMaker Processing jobs (eval image), whose instance-type enum
   * has no ARM families.
   */
  readonly targetArchitecture?: "arm64" | "x86_64";
}

/**
 * Builds a Docker image via CodeBuild and pushes it to ECR.
 *
 * Default architecture is `arm64`. Pass `targetArchitecture: "x86_64"` for
 * images consumed by services without Graviton support (e.g. SageMaker
 * Processing). The class name is retained for historical reasons; it predates
 * the architecture prop.
 *
 * Source code is zipped and uploaded to S3 as a CDK asset during cdk deploy —
 * no GitHub or CodeCommit required. A hash of the source files is used as the
 * image tag so the build is skipped if the image already exists in ECR.
 *
 * A custom resource with async polling ensures CloudFormation waits for the
 * build to complete before advancing to dependent stacks.
 */
export class ArmBuildConstruct extends Construct {
  /** ECR repository where the image is pushed */
  public readonly repository: ecr.Repository;
  /** Image tag derived from source hash */
  public readonly imageTag: string;
  /** Full ECR image URI (repository:tag) */
  public readonly imageUri: string;
  /** Custom resource that completes only when the CodeBuild job succeeds */
  public readonly buildCompletion: cdk.CustomResource;

  constructor(scope: Construct, id: string, props: ArmBuildConstructProps) {
    super(scope, id);

    const stack = cdk.Stack.of(this);
    const absoluteSourcePath = path.resolve(props.sourcePath);

    this.imageTag = this.calculateSourceHash(absoluteSourcePath);

    this.repository = new ecr.Repository(this, "Repository", {
      ...(props.repositoryName ? { repositoryName: props.repositoryName } : {}),
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [{ maxImageCount: 5, description: "Keep last 5 images" }],
    });

    this.imageUri = `${this.repository.repositoryUri}:${this.imageTag}`;

    const targetArch = props.targetArchitecture ?? "arm64";
    // Pick a same-arch CodeBuild host so `docker build` produces a native
    // image for targetArch without needing qemu/--platform. The built image's
    // architecture matches the CodeBuild image's architecture.
    const buildImage =
      targetArch === "arm64"
        ? codebuild.LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0
        : codebuild.LinuxBuildImage.AMAZON_LINUX_2_5;

    // CDK zips the source directory and uploads it to the bootstrap S3 assets bucket
    const sourceAsset = new s3assets.Asset(this, "SourceAsset", {
      path: absoluteSourcePath,
    });

    const buildProject = new codebuild.Project(this, "BuildProject", {
      projectName: `${props.namePrefix}-arm-build`,
      description: `${targetArch} Docker build for ${props.namePrefix}`,
      source: codebuild.Source.s3({
        bucket: sourceAsset.bucket,
        path: sourceAsset.s3ObjectKey,
      }),
      environment: {
        buildImage,
        computeType: codebuild.ComputeType.SMALL,
        privileged: true,
      },
      timeout: cdk.Duration.minutes(props.buildTimeoutMinutes ?? 30),
      environmentVariables: {
        ECR_REPO_URI: { value: this.repository.repositoryUri },
        IMAGE_TAG: { value: this.imageTag },
        AWS_ACCOUNT_ID: { value: stack.account },
        AWS_REGION: { value: stack.region },
        DOCKERFILE_NAME: { value: props.dockerfileName ?? "Dockerfile" },
      },
      buildSpec: codebuild.BuildSpec.fromObject({
        version: "0.2",
        phases: {
          pre_build: {
            commands: [
              "aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws",
              "aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com",
              "REPO_NAME=$(echo $ECR_REPO_URI | cut -d'/' -f2) && if aws ecr describe-images --repository-name $REPO_NAME --image-ids imageTag=$IMAGE_TAG --region $AWS_REGION 2>/dev/null; then echo \"Image exists, skipping build\"; export SKIP_BUILD=true; else export SKIP_BUILD=false; fi",
            ],
          },
          build: {
            commands: [
              'if [ "$SKIP_BUILD" = "false" ]; then docker build -f $DOCKERFILE_NAME -t $ECR_REPO_URI:$IMAGE_TAG -t $ECR_REPO_URI:latest .; fi',
            ],
          },
          post_build: {
            commands: [
              'if [ "$SKIP_BUILD" = "false" ]; then docker push $ECR_REPO_URI:$IMAGE_TAG; docker push $ECR_REPO_URI:latest; fi',
              'echo "Image URI: $ECR_REPO_URI:$IMAGE_TAG"',
            ],
          },
        },
      }),
    });

    this.repository.grantPullPush(buildProject.role!);
    sourceAsset.grantRead(buildProject.role!);

    buildProject.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["ecr:DescribeImages", "ecr:BatchGetImage"],
        resources: [this.repository.repositoryArn],
      }),
    );
    buildProject.addToRolePolicy(
      new iam.PolicyStatement({
        actions: [
          "ecr-public:GetAuthorizationToken",
          "sts:GetServiceBearerToken",
        ],
        resources: ["*"],
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
    request_type = event['RequestType']
    props = event['ResourceProperties']
    project_name = props['ProjectName']
    image_tag = props['ImageTag']

    if request_type == 'Delete':
        return {'PhysicalResourceId': event.get('PhysicalResourceId', f"{project_name}-{image_tag}")}

    response = codebuild_client.start_build(projectName=project_name)
    build_id = response['build']['id']
    print(f"Build started: {build_id}")
    return {
        'PhysicalResourceId': f"{project_name}-{image_tag}",
        'Data': {'BuildId': build_id},
    }
`),
    });

    // isComplete: poll until succeeded or failed
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
        raise Exception(f"Build failed ({status}). Logs: {logs_link}")

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

    const provider = new cr.Provider(this, "Provider", {
      onEventHandler: onEventFn,
      isCompleteHandler: isCompleteFn,
      queryInterval: cdk.Duration.seconds(30),
      totalTimeout: cdk.Duration.minutes(props.buildTimeoutMinutes ?? 30),
      logGroup: new LogGroup(this, "ProviderLogs", {
        retention: logs.RetentionDays.ONE_DAY,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      }),
    });

    this.buildCompletion = new cdk.CustomResource(this, "TriggerAndWaitBuild", {
      serviceToken: provider.serviceToken,
      properties: {
        ProjectName: buildProject.projectName,
        ImageTag: this.imageTag,
        SourceHash: this.imageTag,
      },
    });
    this.buildCompletion.node.addDependency(buildProject);

    new cdk.CfnOutput(this, "ImageUri", { value: this.imageUri });
  }

  private calculateSourceHash(sourcePath: string): string {
    const hash = crypto.createHash("sha256");

    const addToHash = (p: string) => {
      if (fs.statSync(p).isDirectory()) {
        for (const entry of fs.readdirSync(p)) {
          if (["node_modules", "__pycache__", ".git", ".venv"].includes(entry))
            continue;
          addToHash(path.join(p, entry));
        }
      } else {
        hash.update(fs.readFileSync(p));
      }
    };

    addToHash(sourcePath);
    return hash.digest("hex").substring(0, 12);
  }
}
