import {
  SageMakerClient,
  CreatePresignedMlflowAppUrlCommand,
} from '@aws-sdk/client-sagemaker';
import { getAWSCredentials } from './credentials';

const REGION = import.meta.env.VITE_AWS_REGION ?? 'us-east-1';

/**
 * Generate a one-time-use presigned URL for the SageMaker MLflow App.
 * URL is valid for 5 minutes. Browser-accessible — no SigV4 required on the client.
 */
export async function createPresignedMlflowUrl(appArn: string): Promise<string> {
  const credentials = await getAWSCredentials();
  if (!credentials) throw new Error('Not authenticated');
  const client = new SageMakerClient({ region: REGION, credentials });
  const { AuthorizedUrl } = await client.send(
    new CreatePresignedMlflowAppUrlCommand({ Arn: appArn, ExpiresInSeconds: 300 })
  );
  if (!AuthorizedUrl) throw new Error('No URL returned from CreatePresignedMlflowAppUrl');
  return AuthorizedUrl;
}
