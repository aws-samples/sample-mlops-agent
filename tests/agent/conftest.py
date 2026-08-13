import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest


class _ToolUseBlock:
    def __init__(self, id, name, input):
        self.id, self.name, self.input = id, name, input


class _ToolResultBlock:
    def __init__(self, tool_use_id, content, is_error=False):
        self.tool_use_id, self.content, self.is_error = tool_use_id, content, is_error


class _TextBlock:
    def __init__(self, text):
        self.text = text


class _AssistantMessage:
    def __init__(self, content):
        self.content = content


class _UserMessage:
    def __init__(self, content):
        self.content = content


class _ResultMessage:
    session_id = "sdk-xyz"
    def __init__(self):
        pass


class _SystemMessage:
    subtype = "other"
    data: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _stub_claude_agent_sdk():
    """Stub the claude_agent_sdk module to avoid SDK dependency in tests."""
    original = sys.modules.get("claude_agent_sdk")
    fake = types.ModuleType("claude_agent_sdk")
    fake.AssistantMessage = _AssistantMessage
    fake.UserMessage = _UserMessage
    fake.SystemMessage = _SystemMessage
    fake.ResultMessage = _ResultMessage
    fake.TextBlock = _TextBlock
    fake.ToolUseBlock = _ToolUseBlock
    fake.ToolResultBlock = _ToolResultBlock
    fake.ClaudeAgentOptions = MagicMock
    fake.ClaudeSDKClient = MagicMock
    fake.CLIConnectionError = Exception
    fake.CLIJSONDecodeError = Exception
    fake.CLINotFoundError = Exception
    fake.ProcessError = Exception
    sys.modules["claude_agent_sdk"] = fake
    import agent.main as agent_main  # noqa: F401 — forces reload below to find module
    importlib.reload(agent_main)
    yield
    if original is not None:
        sys.modules["claude_agent_sdk"] = original
    else:
        sys.modules.pop("claude_agent_sdk", None)
    # Reload agent.main against whatever sys.modules now points at, so other
    # test files aren't left reading our fake classes.
    importlib.reload(agent_main)
