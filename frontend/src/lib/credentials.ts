import { getStoredToken, initiateLogin, loadConfig } from "./auth";

const TOKEN_KEY = "sample_mlops_agent_id_token";
const IDENTITY_POOL_ID = import.meta.env.VITE_IDENTITY_POOL_ID as string;
const REGION = IDENTITY_POOL_ID?.split(":")[0] ?? "us-east-1";

export interface AWSCredentials {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken: string;
}

// Cache credentials until 5 minutes before expiry
let cached: { credentials: AWSCredentials; expiresAt: number } | null = null;

// Guard so concurrent callers (multiple data hooks) don't each fire a redirect.
let reauthInFlight = false;

// The Cognito Identity ID token has expired (or was rejected). Clear local
// session state and bounce through the Cognito Hosted UI to get a fresh token.
// If the user still has a valid Cognito session cookie this is seamless.
async function triggerReauth(): Promise<void> {
  if (reauthInFlight) return;
  reauthInFlight = true;
  cached = null;
  sessionStorage.removeItem(TOKEN_KEY);
  try {
    await initiateLogin(await loadConfig());
  } catch {
    // If we can't kick off login (e.g. config fetch failed), leave the guard
    // set so we don't loop; the caller gets null and surfaces "Not authenticated".
  }
}

function isAuthExpiryError(e: unknown): boolean {
  return (
    e instanceof Error &&
    /NotAuthorized|Token expired|Invalid login token/i.test(e.message)
  );
}

async function cognitoPost(target: string, body: object): Promise<unknown> {
  const res = await fetch(`https://cognito-identity.${REGION}.amazonaws.com`, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-amz-json-1.1",
      "X-Amz-Target": `AWSCognitoIdentityService.${target}`,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok)
    throw new Error(`CognitoIdentity.${target} failed: ${await res.text()}`);
  return res.json();
}

export async function getAWSCredentials(): Promise<AWSCredentials | null> {
  // getStoredToken() returns null (and clears storage) once the token is within
  // 60s of expiry, so we never hand an expired token to Cognito. Distinguish an
  // EXPIRED session (a raw token was present) from simply being logged out /
  // dev mode (no token at all) — only the former should force a re-login.
  const hadToken = sessionStorage.getItem(TOKEN_KEY) !== null;
  const idToken = getStoredToken();
  if (!idToken) {
    cached = null;
    if (hadToken) void triggerReauth();
    return null;
  }

  // Return cached credentials if still valid
  if (cached && Date.now() < cached.expiresAt) return cached.credentials;

  // Extract the user pool provider from the token's iss claim:
  // "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_xxx" → strip https://
  let loginKey: string;
  try {
    const payload = JSON.parse(
      atob(idToken.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")),
    ) as { iss: string };
    loginKey = payload.iss.replace("https://", "");
  } catch {
    return null;
  }

  const logins = { [loginKey]: idToken };

  try {
    const { IdentityId } = (await cognitoPost("GetId", {
      IdentityPoolId: IDENTITY_POOL_ID,
      Logins: logins,
    })) as { IdentityId: string };

    const { Credentials } = (await cognitoPost("GetCredentialsForIdentity", {
      IdentityId,
      Logins: logins,
    })) as {
      Credentials: {
        AccessKeyId: string;
        SecretKey: string;
        SessionToken: string;
        Expiration: number;
      };
    };

    const credentials: AWSCredentials = {
      accessKeyId: Credentials.AccessKeyId,
      secretAccessKey: Credentials.SecretKey,
      sessionToken: Credentials.SessionToken,
    };

    // Cache until 5 minutes before expiry
    cached = {
      credentials,
      expiresAt: Credentials.Expiration * 1000 - 5 * 60 * 1000,
    };

    return credentials;
  } catch (e) {
    // The token cleared the local 60s-skew check but Cognito still rejected it
    // (clock skew, revoked, or just-expired). Re-authenticate instead of
    // surfacing the raw "Invalid login token" error to every panel.
    if (isAuthExpiryError(e)) {
      void triggerReauth();
      return null;
    }
    throw e;
  }
}

/**
 * Extract the Cognito `sub` claim from the stored ID token.
 * Returns empty string if no token is present or the token cannot be decoded.
 */
export function getCognitoSub(): string {
  const idToken = sessionStorage.getItem(TOKEN_KEY);
  if (!idToken) return "";
  try {
    const payload = JSON.parse(
      atob(idToken.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")),
    ) as { sub?: string };
    return payload.sub ?? "";
  } catch {
    return "";
  }
}
