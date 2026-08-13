import * as cdk from "aws-cdk-lib";
import * as cr from "aws-cdk-lib/custom-resources";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as ssm from "aws-cdk-lib/aws-ssm";
import { Construct } from "constructs";

export interface MlflowStackProps extends cdk.StackProps {
  artifactBucket: s3.IBucket;
  projectName: string;
  environment: string;
}

export class MlflowStack extends cdk.Stack {
  public readonly trackingServerArn: string;

  constructor(scope: Construct, id: string, props: MlflowStackProps) {
    super(scope, id, props);

    const appName = `${props.projectName}-${props.environment}`;

    const mlflowRole = new iam.Role(this, "MlflowRole", {
      assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
    });

    mlflowRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "MlflowArtifactAccess",
        actions: [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation",
        ],
        resources: [
          props.artifactBucket.bucketArn,
          `${props.artifactBucket.bucketArn}/mlflow-artifacts/*`,
        ],
      }),
    );

    mlflowRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "MlflowAppAccess",
        actions: ["sagemaker-mlflow:*"],
        resources: ["*"],
      }),
    );

    mlflowRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "ModelRegistryAccess",
        actions: [
          "sagemaker:AddTags",
          "sagemaker:CreateModelPackageGroup",
          "sagemaker:CreateModelPackage",
          "sagemaker:DescribeModelPackageGroup",
          "sagemaker:UpdateModelPackage",
        ],
        resources: ["*"],
      }),
    );

    // ARN is deterministic: constructed at synth time from concrete region/account values
    const appArn = `arn:aws:sagemaker:${this.region}:${this.account}:mlflow-app/${appName}`;

    const mlflowApp = new cr.AwsCustomResource(this, "MlflowApp", {
      installLatestAwsSdk: true, // required: CreateMlflowApp is not in the runtime-bundled SDK
      onCreate: {
        service: "SageMaker",
        action: "createMlflowApp",
        parameters: {
          Name: appName,
          ArtifactStoreUri: `s3://${props.artifactBucket.bucketName}/mlflow-artifacts`,
          RoleArn: mlflowRole.roleArn,
          AutomaticModelRegistration: true,
        },
        physicalResourceId: cr.PhysicalResourceId.fromResponse("Arn"),
      },
      onUpdate: {
        service: "SageMaker",
        action: "describeMlflowApp",
        // describeMlflowApp takes Arn, not Name
        parameters: { Arn: appArn },
        physicalResourceId: cr.PhysicalResourceId.fromResponse("Arn"),
      },
      onDelete: {
        service: "SageMaker",
        action: "deleteMlflowApp",
        // deleteMlflowApp takes Arn, not Name
        parameters: { Arn: appArn },
        // Tolerate "not found" errors so cdk destroy succeeds even if the app
        // was already deleted, is mid-async-deletion, or was never created.
        // The SageMaker MLflow API surfaces this as code `ResourceNotFound`
        // (no "Exception" suffix) with message "MLflow app entity does not
        // exist" — match both the short and long-form codes plus the message.
        ignoreErrorCodesMatching:
          "ResourceNotFound.*|ValidationException|.*does not exist.*",
      },
      policy: cr.AwsCustomResourcePolicy.fromStatements([
        new iam.PolicyStatement({
          // Explicit actions — fromSdkCalls does not cover iam:PassRole
          actions: [
            "sagemaker:CreateMlflowApp",
            "sagemaker:DeleteMlflowApp",
            "sagemaker:DescribeMlflowApp",
          ],
          resources: ["*"],
        }),
        new iam.PolicyStatement({
          // Required: CreateMlflowApp passes mlflowRole to SageMaker
          actions: ["iam:PassRole"],
          resources: [mlflowRole.roleArn],
          conditions: {
            StringEquals: { "iam:PassedToService": "sagemaker.amazonaws.com" },
          },
        }),
      ]),
    });

    this.trackingServerArn = mlflowApp.getResponseField("Arn");

    new ssm.StringParameter(this, "TrackingServerArnParam", {
      parameterName: `/${props.projectName}/${props.environment}/mlflow/tracking-server-arn`,
      stringValue: this.trackingServerArn,
    });

    new cdk.CfnOutput(this, "TrackingServerArn", {
      value: this.trackingServerArn,
    });
  }
}
