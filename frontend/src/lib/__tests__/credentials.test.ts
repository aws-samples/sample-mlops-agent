import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// auth.ts is mocked so we control token validity + observe the re-login trigger.
const getStoredToken = vi.fn();
const initiateLogin = vi.fn(async (_config: unknown) => {});
const loadConfig = vi.fn(async () => ({}) as never);

vi.mock("../auth", () => ({
  getStoredToken: () => getStoredToken(),
  initiateLogin: (c: unknown) => initiateLogin(c),
  loadConfig: () => loadConfig(),
}));

const TOKEN_KEY = "sample_mlops_agent_id_token";

async function importFresh() {
  vi.resetModules();
  return import("../credentials");
}

beforeEach(() => {
  getStoredToken.mockReset();
  initiateLogin.mockReset();
  initiateLogin.mockResolvedValue(undefined);
  loadConfig.mockReset();
  loadConfig.mockResolvedValue({} as never);
  sessionStorage.clear();
  vi.restoreAllMocks();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("getAWSCredentials", () => {
  it("returns null WITHOUT re-login when there is no token (logged out / dev)", async () => {
    getStoredToken.mockReturnValue(null); // no valid token
    // No raw token in storage either.
    const { getAWSCredentials } = await importFresh();
    const creds = await getAWSCredentials();
    expect(creds).toBeNull();
    expect(initiateLogin).not.toHaveBeenCalled();
  });

  it("triggers re-login when a token exists but has expired", async () => {
    // A raw token is present, but getStoredToken returns null (expired + cleared).
    sessionStorage.setItem(TOKEN_KEY, "expired.jwt.token");
    getStoredToken.mockReturnValue(null);
    const { getAWSCredentials } = await importFresh();
    const creds = await getAWSCredentials();
    expect(creds).toBeNull();
    expect(initiateLogin).toHaveBeenCalledTimes(1);
  });

  it("re-authenticates (not throws) when Cognito rejects the token as expired", async () => {
    // Valid-looking, unexpired token locally; iss decodes fine.
    const payload = btoa(
      JSON.stringify({
        iss: "https://cognito-idp.us-east-1.amazonaws.com/pool",
      }),
    );
    const token = `h.${payload}.s`;
    sessionStorage.setItem(TOKEN_KEY, token);
    getStoredToken.mockReturnValue(token);

    // Cognito responds NotAuthorized (token revoked / clock skew).
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        text: async () =>
          '{"__type":"NotAuthorizedException","message":"Invalid login token. Token expired"}',
      })) as never,
    );

    const { getAWSCredentials } = await importFresh();
    const creds = await getAWSCredentials();
    expect(creds).toBeNull();
    expect(initiateLogin).toHaveBeenCalledTimes(1);
  });
});
