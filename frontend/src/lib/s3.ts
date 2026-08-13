import { GetObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { getAWSCredentials } from "./credentials";

const REGION = import.meta.env.VITE_AWS_REGION ?? "us-east-1";

// Parse "s3://bucket/key/with/slashes" → { bucket, key }. Throws on any
// other scheme so a misconfigured row surfaces as an error, not a
// silently-broken Download button.
function parseS3Uri(uri: string): { bucket: string; key: string } {
  if (!uri.startsWith("s3://")) {
    throw new Error(`Not an s3:// URI: ${uri}`);
  }
  const rest = uri.slice("s3://".length);
  const slash = rest.indexOf("/");
  if (slash <= 0) {
    throw new Error(`Malformed s3:// URI (missing key): ${uri}`);
  }
  return { bucket: rest.slice(0, slash), key: rest.slice(slash + 1) };
}

/**
 * Create a short-lived presigned GET URL for an s3:// artefact using the
 * caller's Cognito credentials. Used by the compliance-report Download
 * button in TaskDetailPanel — the browser opens the URL directly, so no
 * credentials leak into the href we hand to the user.
 *
 * expiresInSeconds defaults to 300 (5 min) — long enough to click and
 * download, short enough that a copy-pasted URL goes cold quickly.
 */
export async function createPresignedS3GetUrl(
  s3Uri: string,
  expiresInSeconds: number = 300,
): Promise<string> {
  const credentials = await getAWSCredentials();
  if (!credentials) throw new Error("Not authenticated");
  const { bucket, key } = parseS3Uri(s3Uri);
  const client = new S3Client({ region: REGION, credentials });
  return getSignedUrl(
    client,
    new GetObjectCommand({ Bucket: bucket, Key: key }),
    { expiresIn: expiresInSeconds },
  );
}
