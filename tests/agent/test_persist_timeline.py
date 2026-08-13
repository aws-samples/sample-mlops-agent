import asyncio
import json
from unittest.mock import MagicMock, patch

import agent.main as agent_main
from agent.main import _upsert_chat_row


def test_upsert_writes_timeline_attribute():
    timeline = [
        {"kind": "message", "id": "u1", "role": "user", "content": "hi"},
        {"kind": "tool", "id": "tu1", "step": 1, "name": "Bash",
         "status": "Running command: ls", "args": "{}", "result": "a\n",
         "truncated": False, "isError": False},
        {"kind": "message", "id": "a1", "role": "assistant", "content": "ok"},
    ]
    with patch("agent.main.boto3") as fake_boto:
        table = MagicMock()
        fake_boto.resource.return_value.Table.return_value = table
        _upsert_chat_row("thread-1", timeline, user_id="u-sub")
    (call,) = table.update_item.call_args_list
    kwargs = call.kwargs
    expr = kwargs["UpdateExpression"]
    vals = kwargs["ExpressionAttributeValues"]
    assert "timeline = :tl" in expr or "timeline   = :tl" in expr
    assert "messages" not in expr
    assert json.loads(vals[":tl"]) == timeline
    assert vals[":uid"] == "u-sub"


def test_multi_turn_message_history_seeding(monkeypatch):
    """
    Test that run() preserves the full message history across turns.

    On turn 2+, the frontend sends the complete prior history via payload["messages"].
    The seeding block must iterate the full list, not just the latest message,
    to preserve prior turns' message entries in the timeline.
    """
    async def _async_return(val):
        return val

    async def _fake_receive():
        # Assistant responds in turn 2
        yield agent_main.AssistantMessage([
            agent_main.TextBlock("done")
        ])
        yield agent_main.ResultMessage()

    client = MagicMock()
    client.query = MagicMock(return_value=_async_return(None))
    client.receive_messages = _fake_receive
    monkeypatch.setattr(agent_main, "_get_or_create_client",
                        lambda *a, **k: _async_return(client))
    monkeypatch.setattr(agent_main, "sync_session", lambda *a, **k: None)
    monkeypatch.setattr(agent_main, "restore_session", lambda *a, **k: None)

    # Capture _upsert_chat_row calls to inspect the timeline
    upsert_calls = []
    def capture_upsert(thread_id, timeline, user_id=None):
        upsert_calls.append({"thread_id": thread_id, "timeline": timeline, "user_id": user_id})
    monkeypatch.setattr(agent_main, "_upsert_chat_row", capture_upsert)

    # Simulate turn 2: frontend sends full prior history
    payload = {
        "messages": [
            {"id": "u1", "role": "user", "content": "hello"},
            {"id": "a1", "role": "assistant", "content": "hi there"},
            {"id": "u2", "role": "user", "content": "how are you"},
        ],
        "session_id": "s1", "run_id": "r1",
    }

    events = []
    async def drain():
        async for ev in agent_main.run(payload, context=None):
            events.append(ev)

    asyncio.run(drain())

    # Verify _upsert_chat_row was called with a timeline containing all 3 payload message entries
    assert len(upsert_calls) >= 1, f"Expected at least 1 upsert call, got {len(upsert_calls)}"
    timeline = upsert_calls[-1]["timeline"]

    # Filter to just message entries (tool entries are added later)
    message_entries = [e for e in timeline if e["kind"] == "message"]

    # Should have at least the 3 messages from payload. The agent may also add its own response,
    # so we check for >= 3 and verify the first 3 match the payload.
    assert len(message_entries) >= 3, (
        f"Expected at least 3 message entries in timeline, got {len(message_entries)}: {message_entries}"
    )

    # Verify the first 3 messages match the payload in order
    assert message_entries[0]["id"] == "u1"
    assert message_entries[0]["role"] == "user"
    assert message_entries[0]["content"] == "hello"

    assert message_entries[1]["id"] == "a1"
    assert message_entries[1]["role"] == "assistant"
    assert message_entries[1]["content"] == "hi there"

    assert message_entries[2]["id"] == "u2"
    assert message_entries[2]["role"] == "user"
    assert message_entries[2]["content"] == "how are you"


def test_first_turn_persisted_before_stream_on_disconnect(monkeypatch):
    """QA BUG-010 regression: the thread row must be written BEFORE the SDK
    stream runs, so a browser closing mid-turn (generator cancelled) still
    leaves a resumable thread on the dashboard. Previously the only upsert
    ran after the stream finished — a first-turn disconnect lost the task
    entirely (no DDB row)."""
    async def _async_return(val):
        return val

    async def _dying_receive():
        # Simulate the client disconnect killing the stream immediately.
        raise asyncio.CancelledError()
        yield  # pragma: no cover — makes this an async generator

    client = MagicMock()
    client.query = MagicMock(return_value=_async_return(None))
    client.receive_messages = _dying_receive
    monkeypatch.setattr(agent_main, "_get_or_create_client",
                        lambda *a, **k: _async_return(client))
    monkeypatch.setattr(agent_main, "sync_session", lambda *a, **k: None)
    monkeypatch.setattr(agent_main, "restore_session", lambda *a, **k: None)

    upsert_calls = []
    monkeypatch.setattr(
        agent_main, "_upsert_chat_row",
        lambda thread_id, timeline, user_id=None: upsert_calls.append(
            {"thread_id": thread_id, "timeline": list(timeline)}),
    )

    payload = {
        "messages": [{"id": "u1", "role": "user", "content": "fine-tune qwen"}],
        "session_id": "s-disc", "run_id": "r1",
    }

    async def drain():
        async for _ in agent_main.run(payload, context=None):
            pass

    try:
        asyncio.run(drain())
    except asyncio.CancelledError:
        pass  # the disconnect itself — irrelevant to the assertion

    assert upsert_calls, "thread row must be persisted before the stream starts"
    first = upsert_calls[0]
    assert first["thread_id"] == "s-disc"
    assert any(e.get("content") == "fine-tune qwen" for e in first["timeline"]), \
        "seeded user prompt missing from the early persist"
