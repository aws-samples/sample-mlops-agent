/**
 * CloudFrontCognitoIntegration
 *
 * Custom resource that runs after CloudFront is created and wires the real
 * distribution URL into two places that need it at deploy time:
 *   1. Cognito UserPoolClient  – callbackUrls / logoutUrls
 *   2. API Gateway HTTP API    – CORS allowedOrigins
 *
 * Eliminates the manual two-pass deploy and the --context allowedOrigins flag.
 */

import * as cdk from 'aws-cdk-lib';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as cr from 'aws-cdk-lib/custom-resources';
import { Construct } from 'constructs';

export interface CloudFrontCognitoIntegrationProps {
  /** CloudFront distribution whose domain name is used as the callback / CORS origin */
  distribution: cloudfront.Distribution;
  /** Cognito User Pool ID */
  userPoolId: string;
  /** Cognito UserPoolClient ID to patch */
  userPoolClientId: string;
  /** API Gateway HTTP API ID to patch CORS on (optional — only used if API Gateway exists) */
  apiId?: string;
  /** AWS region (for ARN construction) */
  region: string;
}

export class CloudFrontCognitoIntegration extends Construct {
  constructor(scope: Construct, id: string, props: CloudFrontCognitoIntegrationProps) {
    super(scope, id);

    const { distribution, userPoolId, userPoolClientId, apiId, region } = props;

    const handlerFn = new lambda.Function(this, 'IntegrationFn', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      timeout: cdk.Duration.minutes(5),
      code: lambda.Code.fromInline(`
import boto3
import cfnresponse
import logging
import traceback

def handler(event, context):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f"Event: {event}")

    try:
        if event['RequestType'] == 'Delete':
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
            return

        props = event['ResourceProperties']
        cloudfront_url  = props['CloudFrontUrl']
        user_pool_id    = props['UserPoolId']
        client_id       = props['ClientId']
        api_id          = props.get('ApiId')

        # ── 1. Cognito UserPoolClient: set callbackUrls / logoutUrls ──────────
        cognito = boto3.client('cognito-idp')
        cognito.update_user_pool_client(
            UserPoolId=user_pool_id,
            ClientId=client_id,
            AllowedOAuthFlows=['code'],
            AllowedOAuthFlowsUserPoolClient=True,
            AllowedOAuthScopes=['openid', 'email'],
            CallbackURLs=[cloudfront_url],
            LogoutURLs=[cloudfront_url],
            SupportedIdentityProviders=['COGNITO'],
            ExplicitAuthFlows=['ALLOW_USER_SRP_AUTH', 'ALLOW_REFRESH_TOKEN_AUTH'],
            PreventUserExistenceErrors='ENABLED',
        )
        logger.info(f"Cognito client {client_id}: callbackUrls -> {cloudfront_url}")

        # ── 2. API Gateway HTTP API: update CORS allowedOrigins (if apiId provided) ───
        if api_id:
            apigw = boto3.client('apigatewayv2')
            apigw.update_api(
                ApiId=api_id,
                CorsConfiguration={
                    'AllowOrigins': [cloudfront_url],
                    'AllowMethods': ['POST', 'GET'],
                    'AllowHeaders': ['Authorization', 'Content-Type'],
                    'MaxAge': 3600,
                },
            )
            logger.info(f"API Gateway {api_id}: CORS allowedOrigins -> {cloudfront_url}")
        else:
            logger.info("No API Gateway configured; skipping CORS setup")

        cfnresponse.send(event, context, cfnresponse.SUCCESS, {
            'CloudFrontUrl': cloudfront_url,
        })

    except Exception as exc:
        logger.error(f"Error: {exc}")
        logger.error(traceback.format_exc())
        cfnresponse.send(event, context, cfnresponse.FAILED, {'Error': str(exc)})
`),
    });

    handlerFn.addToRolePolicy(new iam.PolicyStatement({
      actions: ['cognito-idp:UpdateUserPoolClient'],
      resources: [
        cdk.Stack.of(this).formatArn({
          service: 'cognito-idp',
          resource: 'userpool',
          resourceName: userPoolId,
        }),
      ],
    }));

    // Only add API Gateway permissions if apiId is provided
    if (apiId) {
      handlerFn.addToRolePolicy(new iam.PolicyStatement({
        actions: ['apigateway:PATCH'],
        // API Gateway v2 resource ARN pattern
        resources: [`arn:aws:apigateway:${region}::/apis/${apiId}`],
      }));
    }

    const provider = new cr.Provider(this, 'Provider', {
      onEventHandler: handlerFn,
    });

    // Build properties object conditionally
    const customResourceProps: Record<string, string> = {
      CloudFrontUrl: `https://${distribution.distributionDomainName}`,
      UserPoolId: userPoolId,
      ClientId: userPoolClientId,
    };
    if (apiId) {
      customResourceProps.ApiId = apiId;
    }

    // Include the CloudFront URL in the trigger so the resource re-runs
    // whenever the distribution changes (e.g. domain rotation).
    const customResource = new cdk.CustomResource(this, 'Resource', {
      serviceToken: provider.serviceToken,
      properties: customResourceProps,
    });

    customResource.node.addDependency(distribution);

    const descSuffix = apiId ? 'API Gateway CORS' : 'Cognito callbacks';
    new cdk.CfnOutput(this, 'IntegrationStatus', {
      value: `Cognito + ${descSuffix} wired to https://${distribution.distributionDomainName}`,
    });
  }
}
