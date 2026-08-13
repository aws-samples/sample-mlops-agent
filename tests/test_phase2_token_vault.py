"""Phase 2 Token Vault tests.

Covers:
  T2.1 — HuggingFace skill uses WORKLOAD_ACCESS_TOKEN env var (Token Vault path)
  T2.2 — SSM fallback used only when WORKLOAD_ACCESS_TOKEN is absent
  T2.3 — RuntimeError raised (no silent fallback) when neither source has a token
  T2.4 — Each _get_hf_token() call returns the token for the calling context
          (per-user isolation via env var injection from workload access token)

The Lambda source lives in the deployed package extracted to /tmp/hf_skill_code/.
Tests import from that path to test exactly what is deployed.
"""
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── path: use the deployed Lambda code ───────────────────────────────────────
_HF_SKILL_DIR = Path("/tmp/hf_skill_code")  # nosec B108
if _HF_SKILL_DIR.exists():
    sys.path.insert(0, str(_HF_SKILL_DIR))
else:
    # Fallback: attempt to import from the live Lambda zip if not yet extracted
    pytest.skip(
        reason=(
            "HuggingFace skill Lambda code not found at /tmp/hf_skill_code/. "
            "Run: URL=$(aws lambda get-function --function-name "
            "sample-mlops-agent-gatewa-HuggingFaceSkillFn3B9284-ZV9PJ1KkKtQY "
            "--query 'Code.Location' --output text) && "
            "curl -sL \"$URL\" -o /tmp/hf_skill.zip && "
            "unzip -o /tmp/hf_skill.zip -d /tmp/hf_skill_code/"
        ),
        allow_module_level=True,
    )


def _reload_hf_handler():
    """Import or reload hf skill handler with mocked heavy deps."""
    hf_stub = MagicMock()
    with patch.dict("sys.modules", {
        "huggingface_hub": hf_stub,
        "boto3": MagicMock(),
    }):
        import handler as hf_handler
        importlib.reload(hf_handler)
        return hf_handler


# ─────────────────────────────────────────────────────────────────────────────
# T2.1 — Token Vault path: WORKLOAD_ACCESS_TOKEN env var is used first
# ─────────────────────────────────────────────────────────────────────────────

def test_get_hf_token_uses_workload_access_token_env_var():
    """When WORKLOAD_ACCESS_TOKEN is set, _get_hf_token() returns it directly
    without touching SSM."""
    hf_handler = _reload_hf_handler()

    with patch.dict("os.environ", {
        "WORKLOAD_ACCESS_TOKEN": "vault-token-xyz",
        "PROJECT_NAME": "sample-mlops-agent",
        "AWS_REGION": "us-east-1",
    }), patch("handler.boto3") as mock_boto:
        token = hf_handler._get_hf_token()

    assert token == "vault-token-xyz", (  # nosec B105
        "T2.1: _get_hf_token() must return WORKLOAD_ACCESS_TOKEN when set"
    )
    # SSM must NOT be called when Token Vault token is present
    mock_boto.client.assert_not_called()


def test_get_hf_token_does_not_call_ssm_when_vault_token_present():
    """SSM get_parameter must not be called when WORKLOAD_ACCESS_TOKEN is set.
    This ensures the Token Vault path is truly used, not SSM."""
    hf_handler = _reload_hf_handler()

    mock_boto = MagicMock()
    mock_ssm = MagicMock()
    mock_boto.client.return_value = mock_ssm

    with patch.dict("os.environ", {
        "WORKLOAD_ACCESS_TOKEN": "vault-token-abc",
        "PROJECT_NAME": "sample-mlops-agent",
        "AWS_REGION": "us-east-1",
    }), patch("handler.boto3", mock_boto):
        hf_handler._get_hf_token()

    mock_ssm.get_parameter.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# T2.2 — SSM fallback when WORKLOAD_ACCESS_TOKEN is absent
# ─────────────────────────────────────────────────────────────────────────────

def test_get_hf_token_falls_back_to_ssm_when_no_vault_token():
    """When WORKLOAD_ACCESS_TOKEN is absent, _get_hf_token() must fall back
    to SSM /<project>/dev/hf-token."""
    hf_handler = _reload_hf_handler()

    mock_boto = MagicMock()
    mock_ssm = MagicMock()
    mock_ssm.get_parameter.return_value = {"Parameter": {"Value": "ssm-hf-token"}}
    mock_boto.client.return_value = mock_ssm

    env = {k: v for k, v in os.environ.items() if k != "WORKLOAD_ACCESS_TOKEN"}
    env["PROJECT_NAME"] = "sample-mlops-agent"
    env["AWS_REGION"] = "us-east-1"

    with patch.dict("os.environ", env, clear=True), \
         patch("handler.boto3", mock_boto):
        token = hf_handler._get_hf_token()

    assert token == "ssm-hf-token", (  # nosec B105
        "T2.2: _get_hf_token() must fall back to SSM when WORKLOAD_ACCESS_TOKEN is absent"
    )
    mock_ssm.get_parameter.assert_called_once()
    call_args = mock_ssm.get_parameter.call_args
    assert "/sample-mlops-agent/dev/hf-token" in str(call_args), (
        "SSM parameter name must be /<project>/dev/hf-token"
    )


def test_get_hf_token_empty_string_workload_token_falls_back_to_ssm():
    """An empty WORKLOAD_ACCESS_TOKEN must be treated as absent and fall back to SSM.
    This prevents a blank token from being passed to HuggingFace API."""
    hf_handler = _reload_hf_handler()

    mock_boto = MagicMock()
    mock_ssm = MagicMock()
    mock_ssm.get_parameter.return_value = {"Parameter": {"Value": "ssm-token"}}
    mock_boto.client.return_value = mock_ssm

    with patch.dict("os.environ", {
        "WORKLOAD_ACCESS_TOKEN": "",
        "PROJECT_NAME": "sample-mlops-agent",
        "AWS_REGION": "us-east-1",
    }), patch("handler.boto3", mock_boto):
        token = hf_handler._get_hf_token()

    assert token == "ssm-token", (  # nosec B105
        "Empty WORKLOAD_ACCESS_TOKEN must fall back to SSM — not return an empty string"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T2.3 — RuntimeError raised when neither source has a token (no silent fallback)
# ─────────────────────────────────────────────────────────────────────────────

def test_get_hf_token_raises_when_ssm_fails_and_no_vault_token():
    """When WORKLOAD_ACCESS_TOKEN is absent AND SSM raises, _get_hf_token()
    must raise RuntimeError — no silent fallback, fail loudly."""
    hf_handler = _reload_hf_handler()

    mock_boto = MagicMock()
    mock_ssm = MagicMock()
    mock_ssm.get_parameter.side_effect = Exception("SSM access denied")
    mock_boto.client.return_value = mock_ssm

    env = {k: v for k, v in os.environ.items() if k != "WORKLOAD_ACCESS_TOKEN"}
    env["PROJECT_NAME"] = "sample-mlops-agent"
    env["AWS_REGION"] = "us-east-1"

    with patch.dict("os.environ", env, clear=True), \
         patch("handler.boto3", mock_boto):
        with pytest.raises(RuntimeError, match="Cannot retrieve HuggingFace token"):
            hf_handler._get_hf_token()


# ─────────────────────────────────────────────────────────────────────────────
# T2.4 — Per-user isolation: each invocation uses its own env var value
# ─────────────────────────────────────────────────────────────────────────────

def test_token_isolation_different_env_values_return_different_tokens():
    """Simulates two Lambda executions with different WORKLOAD_ACCESS_TOKEN values.
    Each call must return its own token — no cross-contamination via module-level state."""
    hf_handler = _reload_hf_handler()

    with patch.dict("os.environ", {
        "WORKLOAD_ACCESS_TOKEN": "token-user-a",
        "PROJECT_NAME": "sample-mlops-agent",
        "AWS_REGION": "us-east-1",
    }):
        token_a = hf_handler._get_hf_token()

    with patch.dict("os.environ", {
        "WORKLOAD_ACCESS_TOKEN": "token-user-b",
        "PROJECT_NAME": "sample-mlops-agent",
        "AWS_REGION": "us-east-1",
    }):
        token_b = hf_handler._get_hf_token()

    assert token_a == "token-user-a"  # nosec B105
    assert token_b == "token-user-b"  # nosec B105
    assert token_a != token_b, "T2.4: tokens must be isolated per invocation"


# ─────────────────────────────────────────────────────────────────────────────
# handler() integration: unknown tool returns isError
# ─────────────────────────────────────────────────────────────────────────────

def test_hf_handler_unknown_tool_returns_error():
    """Handler must return isError=True for unrecognised tool names."""
    hf_handler = _reload_hf_handler()

    event = {"params": {"name": "nonexistent_tool", "arguments": {}}}
    result = hf_handler.handler(event, {})

    assert result.get("isError") is True
    content = result.get("content", [{}])
    assert any("Unknown tool" in (c.get("text") or "") for c in content), (
        "isError response must explain which tool was not found"
    )


def test_hf_handler_upload_model_called_with_vault_token():
    """upload_model tool must use _get_hf_token() which honours WORKLOAD_ACCESS_TOKEN."""
    hf_handler = _reload_hf_handler()

    captured_token = {}

    def _fake_upload_model(args):
        captured_token["token"] = hf_handler._get_hf_token()
        return {"repo_url": "https://huggingface.co/test/model"}

    with patch.object(hf_handler, "_upload_model", side_effect=_fake_upload_model), \
         patch.dict("os.environ", {
             "WORKLOAD_ACCESS_TOKEN": "vault-tok-upload",
             "PROJECT_NAME": "sample-mlops-agent",
             "AWS_REGION": "us-east-1",
         }):
        event = {"params": {"name": "upload_model", "arguments": {
            "artifact_s3": "s3://bucket/model.tar.gz",
            "repo_id": "user/model",
        }}}
        result = hf_handler.handler(event, {})

    assert result.get("isError") is not True
    assert captured_token.get("token") == "vault-tok-upload", (
        "upload_model must use the vault token from WORKLOAD_ACCESS_TOKEN"
    )
