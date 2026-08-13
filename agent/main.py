import json
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import boto3

# Runtime (Dockerfile `COPY . .` into /app) imports this as a top-level module;
# the test suite imports the package as `agent.skills_loader`. Support both.
try:
    from agent import skills_loader
except ImportError:  # pragma: no cover - container/runtime path
    import skills_loader  # type: ignore[no-redef]

from opentelemetry import trace as otel_trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from bedrock_agentcore.runtime import BedrockAgentCoreApp, RequestContext
from claude_agent_sdk import (
    AssistantMessage,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ProcessError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Allow nested claude_agent_sdk subprocess to find its own Claude binary
os.environ.pop("CLAUDECODE", None)

# ── Configuration ──────────────────────────────────────────────────────────────
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
SESSION_BUCKET = os.environ.get("SESSION_BUCKET", "")
WORKSPACE_BASE = os.environ.get("WORKSPACE_BASE", "/tmp/workspaces")  # nosec B108
CLAUDE_HOME = Path.home() / ".claude"

ALLOWED_TOOLS = [
    "Read", "Write", "Edit", "Bash", "Glob", "Grep",
    "TodoRead", "TodoWrite", "WebFetch", "WebSearch",
    # Gateway MCP tools — wildcard matches every target/function the Gateway
    # exposes (sagemaker-skill, huggingface-skill, git-skill, mlflow-skill, …).
    # Keeps ALLOWED_TOOLS from drifting when Gateway targets are added.
    "mcp__mlops-gateway__*",
]

TOOL_STATUS_MAP = {
    "Bash":  "Running command",
    "Read":  "Reading file",
    "Write": "Writing file",
}

PROJECT_NAME = os.environ.get("PROJECT_NAME", "sample-mlops-agent")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "sample-mlops-agent-metadata")

_M2M_TOKEN_CACHE: dict = {}

# Gateway MCP configuration is resolved from SSM at runtime (NOT baked into the
# Runtime's env vars). GatewayStack writes these params via its PatchGatewaySsm
# custom resource *after* AgentCoreStack deploys, so reading them at container
# start — rather than at CDK synth time — lets a single `cdk deploy --all` wire
# everything up without a second agentcore redeploy. Values are stable for the
# container lifetime, so the first successful lookup is cached.
_GATEWAY_CONFIG_CACHE: dict = {}


def _get_gateway_config() -> dict:
    """Resolve Gateway MCP config from SSM, cached for the container lifetime.

    Reads three parameters written by GatewayStack's PatchGatewaySsm custom
    resource:
      /{PROJECT_NAME}/dev/gateway/mcp-url
      /{PROJECT_NAME}/dev/gateway/cognito-token-endpoint
      /{PROJECT_NAME}/dev/gateway/m2m-client-id

    Returns:
        dict: {"mcp_url": str, "token_endpoint": str, "m2m_client_id": str}.
              Values are empty strings if the params are still PLACEHOLDER
              (Gateway not yet deployed) or the lookup fails — callers treat an
              empty mcp_url as "Gateway not configured" and skip MCP setup.
    """
    if _GATEWAY_CONFIG_CACHE.get("resolved"):
        return _GATEWAY_CONFIG_CACHE

    result = {"mcp_url": "", "token_endpoint": "", "m2m_client_id": ""}
    param_map = {
        "mcp_url": f"/{PROJECT_NAME}/dev/gateway/mcp-url",
        "token_endpoint": f"/{PROJECT_NAME}/dev/gateway/cognito-token-endpoint",
        "m2m_client_id": f"/{PROJECT_NAME}/dev/gateway/m2m-client-id",
    }
    try:
        import boto3 as _boto3
        ssm_client = _boto3.client("ssm", region_name=AWS_REGION)
        resp = ssm_client.get_parameters(Names=list(param_map.values()))
        by_name = {p["Name"]: p["Value"] for p in resp.get("Parameters", [])}
        for key, name in param_map.items():
            value = by_name.get(name, "")
            # Treat the deploy-time PLACEHOLDER sentinel as "not configured".
            result[key] = "" if value == "PLACEHOLDER" else value
    except Exception as e:
        # Gateway params may not exist yet (first deploy before GatewayStack).
        # Log and return empties — the caller skips Gateway MCP setup.
        logger.warning("[Gateway] Failed to resolve gateway config from SSM: %s", e)
        return result

    # Only cache once the MCP URL is present — otherwise a cold start that races
    # ahead of the PatchGatewaySsm resource would pin empty values permanently.
    if result["mcp_url"]:
        _GATEWAY_CONFIG_CACHE.update(result)
        _GATEWAY_CONFIG_CACHE["resolved"] = True
    return result

_CONTEXT_KEYS: dict[str, list[str]] = {
    "Bash":  ["command"],
    "Read":  ["file_path"],
    "Write": ["file_path"],
}

def _build_status_line(block: ToolUseBlock) -> str:
    """Return a human-readable status string for a tool call, with relevant context."""
    status = TOOL_STATUS_MAP.get(block.name, f"Running {block.name}")
    for key in _CONTEXT_KEYS.get(block.name, []):
        val = str((block.input or {}).get(key, "")).strip()
        if val:
            return f"{status}: {val[:120]}"
    return status


def _truncate_result(text: str, limit: int = 8192) -> tuple[str, bool]:
    """Truncate `text` to at most `limit` UTF-8 bytes on a char boundary.

    Returns (truncated_text, was_truncated). Never splits a multibyte codepoint.
    """
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text, False
    cut = data[:limit]
    while cut:
        try:
            return cut.decode("utf-8"), True
        except UnicodeDecodeError:
            cut = cut[:-1]
    return "", True


def _coerce_result_content(content: object) -> str:
    """Coerce a ToolResultBlock.content value to a displayable string.

    The SDK sometimes returns a list of content blocks instead of a raw string.
    Concatenate text sub-blocks; fall back to json.dumps for anything else.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
            else:
                parts.append(json.dumps(b, default=str))
        return "".join(parts)
    return json.dumps(content, default=str)


# ── Per-session state ──────────────────────────────────────────────────────────
# Keyed by session_id (= AgentCore session_id)
_sdk_clients: dict[str, ClaudeSDKClient] = {}
_sdk_sessions: dict[str, str] = {}  # session_id → claude SDK session_id


def extract_user_id_from_context(context: Optional[RequestContext]) -> str:
    """Extract user_id (JWT sub claim) from the AgentCore RequestContext.

    AgentCore has already validated the incoming JWT's signature and claims
    against the configured authorizer, so we can safely base64-decode the
    payload without re-verifying. Returns "" when no Authorization header is
    present (e.g. IAM-auth invocations) so the caller can fall back to the
    payload-supplied user_id.

    Args:
        context: RequestContext passed by the AgentCore runtime (may be None
            for older invocations that did not forward the request context).

    Returns:
        The "sub" claim from the JWT, or "" when unavailable.
    """
    if context is None:
        return ""
    headers = getattr(context, "request_headers", None) or {}
    auth_header = headers.get("Authorization") or headers.get("authorization") or ""
    if not auth_header.startswith("Bearer "):
        return ""
    token = auth_header[len("Bearer "):]
    try:
        # JWT payload is the middle segment — base64url-decode without
        # signature verification (AgentCore already validated the token).
        import base64
        parts = token.split(".")
        if len(parts) < 2:
            return ""
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)  # pad to multiple of 4
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("sub", "") or ""
    except Exception as e:
        logger.warning("[auth] failed to decode JWT sub (non-fatal): %s", e)
        return ""


def _upsert_chat_row(thread_id: str, timeline: list[dict], user_id: str = "") -> None:
    """Persist the ordered timeline (messages + tool entries) to DynamoDB.

    Args:
        thread_id: AgentCore session_id / thread_id, used as the PK.
        timeline: ordered list of discriminated entries. Each entry is either
          {"kind":"message","id","role","content"} or
          {"kind":"tool","id","step","name","status","args",
           "result","truncated","isError"}.
        user_id: authenticated Cognito sub; overwritten every turn.

    Non-fatal on DDB errors: logs a warning and returns without raising.
    """
    if not thread_id or not timeline:
        return
    try:
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        table = dynamodb.Table(JOBS_TABLE)
        table.update_item(
            Key={"task_id": thread_id},
            UpdateExpression=(
                "SET thread_id  = :t, "
                "    user_id    = :uid, "
                "    timeline   = :tl, "
                "    updated_at = :u, "
                "    jobs       = if_not_exists(jobs, :empty_map), "
                "    created_at = if_not_exists(created_at, :u)"
            ),
            ExpressionAttributeValues={
                ":t":         thread_id,
                ":uid":       user_id,
                ":tl":        json.dumps(timeline),
                ":u":         int(time.time()),
                ":empty_map": {},
            },
        )
        logger.info("[timeline] persisted %d entries for thread %s (user_id=%r)",
                    len(timeline), thread_id, user_id)
    except Exception as e:
        logger.warning("[timeline] DDB upsert failed (non-fatal): %s", e)


app = BedrockAgentCoreApp()
# Use "strands.telemetry.tracer" so the agent.run span is picked up by the
# bedrock_agentcore_starter_toolkit evaluation pipeline, which filters spans by
# scope.name against an allowlist that includes this value.
tracer = otel_trace.get_tracer("strands.telemetry.tracer")


# ── S3 session persistence ─────────────────────────────────────────────────────

def _s3():
    import boto3
    return boto3.client("s3", region_name=AWS_REGION)


def _workspace_prefix(session_id: str) -> str:
    return f"agent-workspaces/{session_id}"


def _claude_home_prefix(session_id: str) -> str:
    return f"agent-sessions/{session_id}/claude-home"


def _sdk_session_key(session_id: str) -> str:
    return f"agent-sessions/{session_id}/sdk_session_id"


def _claude_json_key(session_id: str) -> str:
    return f"agent-sessions/{session_id}/claude.json"


_SYNC_EXCLUDE_DIRS = {
    "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", ".git", ".venv", "venv",
}


def _sync_to_s3(local_dir: Path, bucket: str, prefix: str, s3_client) -> int:
    uploaded = 0
    for fp in local_dir.rglob("*"):
        if not fp.is_file():
            continue
        rel = fp.relative_to(local_dir)
        if any(part in _SYNC_EXCLUDE_DIRS for part in rel.parts):
            continue
        try:
            s3_client.upload_file(str(fp), bucket, f"{prefix}/{rel}")
            uploaded += 1
        except Exception as e:
            logger.warning("[S3 sync] upload failed %s: %s", fp, e)
    return uploaded


def _restore_from_s3(local_dir: Path, bucket: str, prefix: str, s3_client) -> int:
    local_dir.mkdir(parents=True, exist_ok=True)
    paginator = s3_client.get_paginator("list_objects_v2")
    downloaded = 0
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(prefix) + 1:]
                if not rel or any(part in _SYNC_EXCLUDE_DIRS for part in Path(rel).parts):
                    continue
                dest = local_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    s3_client.download_file(bucket, key, str(dest))
                    downloaded += 1
                except Exception as e:
                    logger.warning("[S3 restore] download failed %s: %s", key, e)
    except Exception as e:
        logger.warning("[S3 restore] list failed prefix='%s': %s", prefix, e)
    return downloaded


def restore_session(session_id: str, workspace: Path) -> Optional[str]:
    """Restore workspace + ~/.claude/ + ~/.claude.json from S3. Returns sdk_session_id if saved."""
    if not SESSION_BUCKET:
        return None
    s3 = _s3()
    n = _restore_from_s3(workspace, SESSION_BUCKET, _workspace_prefix(session_id), s3)
    if n:
        logger.info("[S3 restore] workspace: %d files", n)
    n = _restore_from_s3(CLAUDE_HOME, SESSION_BUCKET, _claude_home_prefix(session_id), s3)
    if n:
        logger.info("[S3 restore] claude home: %d files", n)
    # ~/.claude.json is the Claude CLI auth/config file. It lives one level above ~/.claude/
    # so the directory sync above misses it — restore it explicitly.
    try:
        resp = s3.get_object(Bucket=SESSION_BUCKET, Key=_claude_json_key(session_id))
        claude_json_path = Path.home() / ".claude.json"
        claude_json_path.write_bytes(resp["Body"].read())
        logger.info("[S3 restore] claude.json restored")
    except Exception:  # nosec B110
        pass  # restore is best-effort: not yet persisted (first Turn 1) — Claude CLI will create it fresh
    try:
        resp = s3.get_object(Bucket=SESSION_BUCKET, Key=_sdk_session_key(session_id))
        sid = resp["Body"].read().decode().strip()
        logger.info("[S3 restore] sdk_session_id=%s", sid)
        return sid
    except Exception:
        return None


def sync_session(session_id: str, workspace: Path, sdk_session_id: Optional[str]) -> None:
    """Sync workspace + ~/.claude/ + ~/.claude.json to S3 after task completion."""
    if not SESSION_BUCKET:
        return
    s3 = _s3()
    n = _sync_to_s3(workspace, SESSION_BUCKET, _workspace_prefix(session_id), s3)
    logger.info("[S3 sync] workspace: %d files", n)
    if CLAUDE_HOME.exists():
        n = _sync_to_s3(CLAUDE_HOME, SESSION_BUCKET, _claude_home_prefix(session_id), s3)
        logger.info("[S3 sync] claude home: %d files", n)
    # ~/.claude.json is the Claude CLI auth/config file — it sits one level above ~/.claude/
    # so the directory sync above misses it. Upload it explicitly so a resumed container
    # has the config available and does not fall back to the backup warning path.
    claude_json_path = Path.home() / ".claude.json"
    if claude_json_path.exists():
        try:
            s3.upload_file(str(claude_json_path), SESSION_BUCKET, _claude_json_key(session_id))
            logger.info("[S3 sync] claude.json saved")
        except Exception as e:
            logger.warning("[S3 sync] failed to save claude.json: %s", e)
    if sdk_session_id:
        try:
            s3.put_object(
                Bucket=SESSION_BUCKET,
                Key=_sdk_session_key(session_id),
                Body=sdk_session_id.encode(),
                ContentType="text/plain",
            )
            logger.info("[S3 sync] sdk_session_id saved")
        except Exception as e:
            logger.warning("[S3 sync] failed to save sdk_session_id: %s", e)


# ── Telemetry helpers ────────────────────────────────────────────────────────

# NOTE: We deliberately do NOT emit per-tool `execute_tool` child spans.
#
# AgentCore Evaluations, running in split-telemetry mode here
# (OTEL_LOGS_EXPORTER=otlp + the ADOT LLOHandler), requires every execute_tool
# span to have a correlated LLO log event, AND that event must parse. In this
# setup neither can be satisfied for a manually-emitted tool span:
#   - with a gen_ai.tool.message event, the LLOHandler nests its attributes into
#     an unparseable body → SpanEventParsingException;
#   - without the event, the span has no correlated record → LogEventMissingException.
# Either way the WHOLE session fails batch evaluation, which in turn starves the
# optimizer ("No sessions were identified from input agent traces"). Sessions
# with NO execute_tool spans evaluate cleanly (the builtin judges read the
# conversation off the invoke_agent + inference spans). So we omit tool spans
# entirely. The trade-off is the confirmation-gate custom evaluator can no longer
# read tool ordering from spans (it ABSTAINs); if needed it can be re-based on
# the DynamoDB timeline, which still records every tool call. The invoke_agent
# span still carries gen_ai.tool_calls (the ordered tool-name list) for context.


# ── Dynamic skill + system-prompt hydration ─────────────────────────────────────

_skills_seeded = False


def _maybe_seed_skills(prefix: str) -> None:
    """Seed the S3 skill store from the baked container skills once per container
    (only if the S3 prefix is empty). Best-effort; guarded so it runs at most
    once regardless of how many turns the container serves."""
    global _skills_seeded
    if _skills_seeded or not SESSION_BUCKET:
        return
    _skills_seeded = True
    try:
        n = skills_loader.seed_skills_if_empty(SESSION_BUCKET, prefix, str(CLAUDE_HOME))
        if n:
            logger.info("[seed] uploaded %d baked skill(s) to s3://%s/%s", n, SESSION_BUCKET, prefix)
    except Exception as e:
        logger.warning("[seed] skill seed failed (non-fatal): %s", e)


def hydrate_skills(bucket: str, prefix: str, claude_home: str) -> Optional[str]:
    """Load skills + system prompt from the S3 store into the agent's Claude home.

    Skills are written to ``<claude_home>/skills/<name>/SKILL.md`` (the SDK
    discovers them from the filesystem via ``setting_sources``), and the system
    prompt overwrites ``<claude_home>/CLAUDE.md`` when present in S3. We do NOT
    pass a ``system_prompt`` string to ClaudeAgentOptions — the ``claude_code``
    preset must stay; the dynamic prompt rides in via the project/user CLAUDE.md.

    Must run AFTER ``restore_session`` (which repopulates ~/.claude from S3) and
    BEFORE the SDK client is created. Returns the system-prompt text if one was
    written, else ``None``. Best-effort: any failure falls back to the baked-in
    container skills/prompt (logged at warning), never raising into the turn.
    """
    if not bucket:
        return None
    try:
        skills_loader.clear_cache()
        skills = skills_loader.load_skills_s3(bucket, prefix)
        if skills:
            n = skills_loader.write_skill_files(str(Path(claude_home) / "skills"), skills)
            logger.info("[hydrate] wrote %d skill(s) from s3://%s/%s", n, bucket, prefix)
        prompt = skills_loader.load_system_prompt(bucket, prefix)
        if prompt:
            (Path(claude_home) / "CLAUDE.md").write_text(prompt, encoding="utf-8")
            logger.info("[hydrate] system prompt updated from S3 (%d chars)", len(prompt))
        return prompt
    except Exception as e:
        logger.warning("[hydrate] failed, falling back to baked-in skills: %s", e)
        return None


# ── Client lifecycle ───────────────────────────────────────────────────────────

async def _get_or_create_client(sdk_key: str, options: ClaudeAgentOptions) -> ClaudeSDKClient:
    """Return a connected client, reusing an existing subprocess when possible."""
    existing = _sdk_clients.get(sdk_key)
    if existing is not None and getattr(existing, "_query", None) is not None:
        logger.info("[Client] reusing existing client for %s", sdk_key)
        return existing

    if existing is not None:
        logger.info("[Client] discarding disconnected client for %s", sdk_key)
        try:
            await existing.disconnect()
        # Client already dead; nothing to clean up.
        except Exception:  # nosec B110
            pass
        _sdk_clients.pop(sdk_key, None)

    client = ClaudeSDKClient(options=options)
    await client.connect()
    _sdk_clients[sdk_key] = client
    logger.info("[Client] created new client for %s", sdk_key)
    return client


async def _disconnect_client(sdk_key: str) -> None:
    client = _sdk_clients.pop(sdk_key, None)
    if client:
        try:
            await client.disconnect()
            logger.info("[Client] disconnected %s", sdk_key)
        except Exception as e:
            logger.warning("[Client] disconnect failed for %s: %s", sdk_key, e)


def _get_m2m_token() -> str:
    """Obtain a Cognito M2M access token via client_credentials grant.

    Fetches the machine client secret from Secrets Manager on first call
    (ARN looked up via SSM). Tokens are cached in _M2M_TOKEN_CACHE for the
    container lifetime; a new token is requested 5 minutes before expiry.

    A fetch-in-progress marker prevents concurrent asyncio tasks from issuing
    duplicate SSM / Secrets Manager calls at the token-expiry boundary.

    Returns:
        str: Bearer access token, or empty string if Gateway is not configured
             (token endpoint not resolvable from SSM) or secret fetch fails.
    """
    gateway_config = _get_gateway_config()
    token_endpoint = gateway_config["token_endpoint"]
    if not token_endpoint:
        return ""

    cached = _M2M_TOKEN_CACHE.get("token")
    expires_at = _M2M_TOKEN_CACHE.get("expires_at", 0)
    if cached and time.time() < expires_at - 300:
        return cached

    # Resolve client secret: look up ARN from SSM, then fetch value from Secrets Manager.
    # Both calls are made only when the cached token is expired (typically once per cold start).
    # _secret_fetching flag prevents concurrent tasks from issuing duplicate AWS API calls.
    if "client_secret" not in _M2M_TOKEN_CACHE and not _M2M_TOKEN_CACHE.get("_secret_fetching"):
        _M2M_TOKEN_CACHE["_secret_fetching"] = True
        try:
            import boto3 as _boto3
            ssm_client = _boto3.client("ssm", region_name=AWS_REGION)
            secret_arn = ssm_client.get_parameter(
                Name=f"/{PROJECT_NAME}/dev/gateway/m2m-client-secret-arn",
            )["Parameter"]["Value"]
            sm_client = _boto3.client("secretsmanager", region_name=AWS_REGION)
            _M2M_TOKEN_CACHE["client_secret"] = sm_client.get_secret_value(
                SecretId=secret_arn,
            )["SecretString"]
        except Exception as e:
            # Gateway SSM params may not yet be written (e.g., first deploy before GatewayStack).
            # Log and return empty — the caller's try/except will skip Gateway MCP setup.
            # Logs the exception only — the secret value never reaches the logger.
            logger.warning(  # nosemgrep: python-logger-credential-disclosure
                "[Gateway] Failed to fetch M2M client secret: %s", e
            )
            return ""
        finally:
            _M2M_TOKEN_CACHE.pop("_secret_fetching", None)

    if "client_secret" not in _M2M_TOKEN_CACHE:
        # Another concurrent task is mid-fetch; skip token acquisition this call.
        return ""

    import requests as _requests
    resp = _requests.post(
        token_endpoint,
        data={
            "grant_type": "client_credentials",
            "client_id": gateway_config["m2m_client_id"],
            "client_secret": _M2M_TOKEN_CACHE["client_secret"],
            "scope": f"{PROJECT_NAME}-gateway/invoke",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data["access_token"]
    _M2M_TOKEN_CACHE["token"] = token
    _M2M_TOKEN_CACHE["expires_at"] = time.time() + data.get("expires_in", 3600)
    return token


# ── AgentCore entrypoint ───────────────────────────────────────────────────────

@app.entrypoint
async def run(payload: dict, context: Optional[RequestContext] = None):
    session_id = payload.get("session_id", "")
    run_id = payload.get("run_id") or str(uuid.uuid4())
    # Prefer user_id derived from the validated JWT "sub" claim in the
    # RequestContext — payload.user_id is client-controllable and therefore
    # only used as a fallback when the request is not JWT-authenticated.
    user_id: str = extract_user_id_from_context(context) or payload.get("user_id", "")
    # Propagate user_id to skill scripts running as subprocesses.
    # submit.py and HuggingFace skills read CURRENT_USER_ID to:
    #   1. Write user_id to DynamoDB for async callback attribution
    #   2. Look up per-user Token Vault entries (Phase 2)
    os.environ["CURRENT_USER_ID"] = user_id
    # The AgentCore session_id IS the thread_id. Export it so skills + MCP tool
    # calls can read it directly instead of hunting for it in the prompt.
    os.environ["CURRENT_THREAD_ID"] = session_id

    # WORKLOAD_ACCESS_TOKEN is unused on the service-linked workload identity
    # that AgentCore auto-creates for this Runtime: GetWorkloadAccessTokenForUserId
    # raises ValidationException ("WorkloadIdentity is linked to a service and
    # cannot retrieve an access token by the caller"). The HF skill already
    # falls back to the SSM token when this env var is empty, so we leave it
    # unset. Switching to a custom (non-service-linked) workload identity would
    # be required to issue per-user tokens; not wired up yet.
    # Deliberately empty — not a credential (see comment above).
    os.environ["WORKLOAD_ACCESS_TOKEN"] = ""  # nosec B105

    # Build Gateway MCP server config for ClaudeAgentOptions.mcp_servers.
    # The Python claude-agent-sdk does NOT auto-discover MCP servers from any
    # config file — they must be passed explicitly via the mcp_servers kwarg.
    # We build the config here (before ClaudeAgentOptions) so we can include the
    # current X-Ray trace ID for correlated spans across agent → Gateway → skill.
    mcp_servers: dict = {}
    gateway_mcp_url = _get_gateway_config()["mcp_url"]
    if gateway_mcp_url:
        try:
            m2m_token = _get_m2m_token()
            current_span = otel_trace.get_current_span()
            ctx = current_span.get_span_context()
            trace_id_hex = format(ctx.trace_id, "032x") if ctx.is_valid else ""
            xray_trace_id = f"Root=1-{trace_id_hex[:8]}-{trace_id_hex[8:]}" if trace_id_hex else ""

            mcp_servers["mlops-gateway"] = {
                "type": "http",
                "url": gateway_mcp_url,
                "headers": {
                    "Authorization": f"Bearer {m2m_token}",
                    "X-Amzn-Trace-Id": xray_trace_id,
                    "_user_id": user_id,
                },
            }
            logger.info("[run] Gateway MCP config built for session %s", session_id)
        except Exception as e:
            logger.warning("[run] Gateway MCP config failed (non-fatal): %s", e)

    messages = payload.get("messages", [])
    if messages:
        last_msg = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
    else:
        last_msg = payload.get("message", "")

    yield {"type": "RUN_STARTED", "session_id": session_id, "run_id": run_id}

    if not last_msg:
        logger.warning("[run] empty message — nothing to do")
        yield {"type": "RUN_FINISHED", "session_id": session_id, "run_id": run_id}
        return

    # Per-session workspace
    sdk_key = session_id
    workspace = Path(WORKSPACE_BASE) / sdk_key
    workspace.mkdir(parents=True, exist_ok=True)

    # Wrap the entire invocation in a root span so that all child operations
    # (SDK calls, S3 session sync) share one trace instead of appearing as
    # disconnected root spans in X-Ray.
    # Accumulate assistant text and tool call names so we can attach them as
    # span attributes after the SDK stream completes.
    _text_chunks: list[str] = []
    _tool_names: list[str] = []
    timeline_out: list[dict] = []
    tool_entry_by_id: dict[str, dict] = {}

    with tracer.start_as_current_span(
        "invoke_agent sample_mlops_agent",
        kind=otel_trace.SpanKind.CLIENT,
        attributes={
            gen_ai_attributes.GEN_AI_SYSTEM: "strands-agents",
            gen_ai_attributes.GEN_AI_OPERATION_NAME: "invoke_agent",
            "gen_ai.agent.name": "sample_mlops_agent",
            gen_ai_attributes.GEN_AI_REQUEST_MODEL: os.environ.get(
                "CLAUDE_MODEL_ID", "anthropic.claude-opus-4-5-20251001-v1:0"
            ),
            "session.id": session_id,
            "run.id": run_id,
            "user.id": user_id,
        },
    ) as run_span:
        # Strands-format event: content is JSON-serialized Bedrock content blocks.
        # The ADOT LLOHandler converts this to a CloudWatch log record used by
        # the evaluate API to extract user_query and agent response.
        run_span.add_event(
            "gen_ai.user.message",
            {"content": json.dumps([{"text": last_msg[:4000]}])},
        )
        # Restore or look up SDK session ID for conversation continuity
        sdk_session_id = _sdk_sessions.get(sdk_key)
        if not sdk_session_id:
            sdk_session_id = restore_session(session_id, workspace)
            if sdk_session_id:
                _sdk_sessions[sdk_key] = sdk_session_id
                logger.info("[run] session restored from S3: %s", sdk_session_id)
            else:
                logger.info("[run] starting new SDK session")
        else:
            logger.info("[run] resuming in-memory SDK session: %s", sdk_session_id)

        # Seed the S3 skill store from the baked container skills on first use
        # (no-op once the prefix is populated, so live UI edits are never
        # clobbered), then hydrate live skills + system prompt from S3 AFTER
        # restore (restore repopulates ~/.claude) and BEFORE client creation so
        # edits apply within one cache TTL without a redeploy. Both best-effort;
        # fall back to the baked-in skills on any failure.
        _skills_prefix = os.environ.get("SKILLS_S3_PREFIX", "skills/")
        _maybe_seed_skills(_skills_prefix)
        hydrate_skills(SESSION_BUCKET, _skills_prefix, str(CLAUDE_HOME))

        options = ClaudeAgentOptions(
            system_prompt={"type": "preset", "preset": "claude_code"},
            allowed_tools=ALLOWED_TOOLS,
            mcp_servers=mcp_servers,
            permission_mode="bypassPermissions",
            resume=sdk_session_id,
            cwd=str(workspace),
            max_turns=100,
            cli_path=shutil.which("claude") or "/usr/local/bin/claude",
            stderr=lambda line: logger.error("[claude stderr] %s", line),
            setting_sources=["user", "project"],
        )

        # Seed timeline with all messages from the request (full history on multi-turn)
        for m in (payload.get("messages") or []):
            timeline_out.append({
                "kind": "message",
                "id": m.get("id") or str(uuid.uuid4()),
                "role": m.get("role", "user"),
                "content": m.get("content", ""),
            })

        # QA BUG-010: persist the thread row BEFORE streaming starts. Closing
        # the browser mid-turn cancels this generator — without this early
        # write the first turn of a new task vanished entirely (no DynamoDB
        # row, so the dashboard had no card to resume the session from).
        # The end-of-turn upsert below overwrites with the full timeline.
        early_thread_id = payload.get("thread_id") or session_id
        if early_thread_id and timeline_out:
            _upsert_chat_row(early_thread_id, timeline_out, user_id=user_id)

        msg_id = str(uuid.uuid4())

        try:
            client = await _get_or_create_client(sdk_key, options)
        except CLINotFoundError as e:
            logger.exception("[run] Claude CLI not found")
            run_span.record_exception(e)
            yield {
                "type": "TEXT_MESSAGE_CHUNK",
                "message_id": msg_id,
                "delta": f"Error: Claude CLI not found. Check container setup. ({e})",
            }
            yield {"type": "RUN_FINISHED", "session_id": session_id, "run_id": run_id}
            return

        step_counter = 0
        tool_step_by_id: dict[str, int] = {}
        pending_tool_ids: set[str] = set()

        def _flush_orphans() -> list[dict[str, object]]:
            """Emit synthetic TOOL_CALL_RESULT events for orphaned tool calls.

            Also patches the in-memory timeline entry so the persisted row matches
            the streamed events. Called on the normal-exit path and from every
            except handler. Clears `pending_tool_ids` so a second call is a no-op.
            """
            out: list[dict[str, object]] = []
            for orphan_id in list(pending_tool_ids):
                e = tool_entry_by_id.get(orphan_id)
                if e is not None:
                    e["result"] = ""
                    e["isError"] = True
                    if not e["status"].endswith("(interrupted)"):
                        e["status"] = f"{e['status']} (interrupted)"
                out.append({
                    "type": "TOOL_CALL_RESULT",
                    "tool_call_id": orphan_id,
                    "parent_message_id": msg_id,
                    "content": "",
                    "truncated": False,
                    "is_error": True,
                    "step": tool_step_by_id.get(orphan_id, 0),
                })
            pending_tool_ids.clear()
            return out

        try:
            await client.query(last_msg)

            async for message in client.receive_messages():
                # Capture SDK session_id from init event
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    new_sid = message.data.get("session_id")
                    if new_sid and new_sid != _sdk_sessions.get(sdk_key):
                        _sdk_sessions[sdk_key] = new_sid
                        logger.info("[run] SDK session stored: %s", new_sid)

                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text:
                            _text_chunks.append(block.text)
                            existing = next(
                                (e for e in timeline_out
                                 if e.get("kind") == "message" and e.get("id") == msg_id),
                                None,
                            )
                            if existing is None:
                                timeline_out.append({
                                    "kind": "message", "id": msg_id,
                                    "role": "assistant", "content": block.text,
                                })
                            else:
                                existing["content"] = existing.get("content", "") + block.text
                            yield {
                                "type": "TEXT_MESSAGE_CHUNK",
                                "message_id": msg_id,
                                "delta": block.text,
                            }
                        elif isinstance(block, ToolUseBlock):
                            step_counter += 1
                            _tool_names.append(block.name)
                            status = _build_status_line(block)
                            args_str = json.dumps(block.input or {})
                            args_str, _args_trunc = _truncate_result(args_str, 8192)
                            # No per-tool execute_tool span — see the note by the
                            # telemetry helpers: any manual tool span fails
                            # AgentCore batch evaluation under split telemetry.
                            # The ordered tool-name list is still recorded on the
                            # invoke_agent span (gen_ai.tool_calls) and the
                            # DynamoDB timeline below.
                            tool_step_by_id[block.id] = step_counter
                            pending_tool_ids.add(block.id)
                            logger.info("[tool] step=%d %s", step_counter, status)
                            entry = {"kind": "tool", "id": block.id, "step": step_counter,
                                     "name": block.name, "status": status, "args": args_str,
                                     "result": None, "truncated": False, "isError": False}
                            timeline_out.append(entry)
                            tool_entry_by_id[block.id] = entry
                            yield {
                                "type": "TOOL_CALL_CHUNK",
                                "tool_call_id": block.id,
                                "tool_call_name": block.name,
                                "parent_message_id": msg_id,
                                "delta": args_str,
                                "status": status,
                                "step": step_counter,
                            }

                elif isinstance(message, UserMessage):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            raw = _coerce_result_content(block.content)
                            content_str, truncated = _truncate_result(raw, 8192)
                            is_error = bool(getattr(block, "is_error", False))
                            pending_tool_ids.discard(block.tool_use_id)
                            e = tool_entry_by_id.get(block.tool_use_id)
                            if e is not None:
                                e["result"] = content_str
                                e["truncated"] = truncated
                                e["isError"] = is_error
                            yield {
                                "type": "TOOL_CALL_RESULT",
                                "tool_call_id": block.tool_use_id,
                                "parent_message_id": msg_id,
                                "content": content_str,
                                "truncated": truncated,
                                "is_error": is_error,
                                "step": tool_step_by_id.get(block.tool_use_id, 0),
                            }

                elif isinstance(message, ResultMessage):
                    if message.session_id and message.session_id != _sdk_sessions.get(sdk_key):
                        _sdk_sessions[sdk_key] = message.session_id
                        logger.info("[run] SDK session stored from result: %s", message.session_id)
                    run_span.set_attribute("agent.tool_steps", step_counter)
                    logger.info("[run] result received after %d steps", step_counter)
                    break

            for ev in _flush_orphans():
                yield ev

        except (CLIConnectionError, CLIJSONDecodeError) as e:
            logger.exception("[run] CLI communication error")
            run_span.record_exception(e)
            await _disconnect_client(sdk_key)
            for ev in _flush_orphans():
                yield ev
            yield {
                "type": "TEXT_MESSAGE_CHUNK",
                "message_id": msg_id,
                "delta": f"Error: CLI communication failed. ({e})",
            }
        except ProcessError as e:
            logger.exception("[run] CLI process error (exit code: %s)", e.exit_code)
            run_span.record_exception(e)
            await _disconnect_client(sdk_key)
            for ev in _flush_orphans():
                yield ev
            yield {
                "type": "TEXT_MESSAGE_CHUNK",
                "message_id": msg_id,
                "delta": f"Error: {e}",
            }
        except Exception as e:
            logger.exception("[run] unexpected error")
            run_span.record_exception(e)
            await _disconnect_client(sdk_key)
            for ev in _flush_orphans():
                yield ev
            yield {
                "type": "TEXT_MESSAGE_CHUNK",
                "message_id": msg_id,
                "delta": f"Error: {e}",
            }

        final_text = "".join(_text_chunks)
        if final_text:
            run_span.add_event(
                "gen_ai.choice",
                {"message": final_text[:4000], "finish_reason": "end_turn"},
            )
        if _tool_names:
            run_span.set_attribute("gen_ai.tool_calls", json.dumps(_tool_names))

        # Persist the timeline to DynamoDB
        thread_id = payload.get("thread_id") or session_id
        if thread_id and timeline_out:
            _upsert_chat_row(thread_id, timeline_out, user_id=user_id)

        # Sync workspace + ~/.claude/ to S3 for cross-container-restart continuity
        sync_session(session_id or sdk_key, workspace, _sdk_sessions.get(sdk_key))

    yield {"type": "RUN_FINISHED", "session_id": session_id, "run_id": run_id}


if __name__ == "__main__":
    app.run()
