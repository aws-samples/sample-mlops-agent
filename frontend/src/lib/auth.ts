export type JwtClaims = {
  email?: string;
  name?: string;
  "cognito:username"?: string;
  given_name?: string;
  family_name?: string;
};

export function parseJwt(token: string): JwtClaims {
  try {
    const payload = token.split(".")[1];
    const decoded = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(decoded) as JwtClaims;
  } catch {
    return {};
  }
}

export function getUserDisplayName(claims: JwtClaims): string {
  if (claims.name) return claims.name;
  if (claims.given_name && claims.family_name)
    return `${claims.given_name} ${claims.family_name}`;
  if (claims.email) return claims.email;
  return claims["cognito:username"] ?? "Account";
}

export type AuthConfig = {
  agentCoreEndpoint: string;
  cognitoClientId: string;
  cognitoDomain: string; // https://sample-mlops-agent-<account>.auth.<region>.amazoncognito.com
  mlflowAppArn: string;
  // Skills store + Evals tab identifiers (optional — undefined-safe for
  // back-compat with configs published before these features).
  sessionBucket?: string;
  skillsPrefix?: string;
  onlineEvalConfigName?: string;
  evalResultsLogGroupPrefix?: string;
  agentRuntimeName?: string;
  runtimeLogGroup?: string;
  runtimeLogGroupArn?: string;
};

const TOKEN_KEY = "sample_mlops_agent_id_token";
const TOKEN_EXP_KEY = "sample_mlops_agent_token_exp";
const PKCE_VERIFIER_KEY = "sample_mlops_agent_pkce_verifier";

export async function loadConfig(): Promise<AuthConfig> {
  const res = await fetch("/config.json");
  if (!res.ok) throw new Error(`Failed to load config.json: ${res.status}`);
  return res.json() as Promise<AuthConfig>;
}

export function getStoredToken(): string | null {
  const token = sessionStorage.getItem(TOKEN_KEY);
  const exp = sessionStorage.getItem(TOKEN_EXP_KEY);
  if (!token || !exp) return null;
  // Treat as expired 60 s before actual expiry
  if (Date.now() / 1000 > parseInt(exp, 10) - 60) {
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(TOKEN_EXP_KEY);
    return null;
  }
  return token;
}

// ── PKCE helpers ────────────────────────────────────────────────────────────

function base64UrlEncode(buffer: ArrayBuffer): string {
  return btoa(String.fromCharCode(...new Uint8Array(buffer)))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=/g, "");
}

async function generatePKCE(): Promise<{
  verifier: string;
  challenge: string;
}> {
  const bytes = new Uint8Array(96);
  crypto.getRandomValues(bytes);
  const verifier = base64UrlEncode(bytes.buffer).slice(0, 128);
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(verifier),
  );
  const challenge = base64UrlEncode(digest);
  return { verifier, challenge };
}

// ── Public auth flow ─────────────────────────────────────────────────────────

export async function initiateLogin(config: AuthConfig): Promise<void> {
  const { verifier, challenge } = await generatePKCE();
  sessionStorage.setItem(PKCE_VERIFIER_KEY, verifier);

  const redirectUri = window.location.origin;
  const params = new URLSearchParams({
    response_type: "code",
    client_id: config.cognitoClientId,
    redirect_uri: redirectUri,
    scope: "openid email",
    code_challenge: challenge,
    code_challenge_method: "S256",
  });
  window.location.assign(`${config.cognitoDomain}/oauth2/authorize?${params}`);
}

export async function handleCallback(
  config: AuthConfig,
  code: string,
): Promise<string> {
  const verifier = sessionStorage.getItem(PKCE_VERIFIER_KEY);
  if (!verifier)
    throw new Error("PKCE verifier missing — cannot exchange code");

  const redirectUri = window.location.origin;
  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: config.cognitoClientId,
    code,
    redirect_uri: redirectUri,
    code_verifier: verifier,
  });

  const res = await fetch(`${config.cognitoDomain}/oauth2/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Token exchange failed: ${text}`);
  }

  const data = (await res.json()) as { id_token: string; expires_in: number };
  const exp = Math.floor(Date.now() / 1000) + data.expires_in;
  sessionStorage.setItem(TOKEN_KEY, data.id_token);
  sessionStorage.setItem(TOKEN_EXP_KEY, String(exp));
  sessionStorage.removeItem(PKCE_VERIFIER_KEY);

  // Remove the code from the URL without a page reload
  window.history.replaceState({}, document.title, window.location.pathname);

  return data.id_token;
}

export function signOut(config: AuthConfig): void {
  sessionStorage.clear();
  const redirectUri = encodeURIComponent(window.location.origin);
  window.location.assign(
    `${config.cognitoDomain}/logout?client_id=${config.cognitoClientId}&logout_uri=${redirectUri}`,
  );
}
