import sys
from unittest.mock import MagicMock

import boto3
import pytest

# Mock claude_agent_sdk if not installed
if 'claude_agent_sdk' not in sys.modules:
    mock_sdk = MagicMock()

    def _tool_decorator(name, description, schema):
        def decorator(fn):
            return fn
        return decorator

    mock_sdk.tool = _tool_decorator
    sys.modules['claude_agent_sdk'] = mock_sdk


@pytest.fixture(autouse=True)
def _restore_boto3_client():
    """Undo any test that reassigns the global ``boto3.client``.

    Several handler tests do ``h.boto3.client = lambda ...`` to stub AWS calls.
    Because ``h.boto3`` is the real, shared boto3 module, that mutation leaks into
    later tests — in particular any handler that creates a client at import time
    (e.g. the web_search skill's module-level ``bedrock-runtime`` client). Snapshot
    and restore the attribute around every test to keep imports deterministic.
    """
    original_client = boto3.client
    try:
        yield
    finally:
        boto3.client = original_client
