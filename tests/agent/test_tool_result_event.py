import asyncio
from unittest.mock import MagicMock

import agent.main as agent_main


async def _fake_receive():
    yield agent_main.AssistantMessage([
        agent_main.ToolUseBlock(id="tu1", name="Bash", input={"command": "ls"})
    ])
    yield agent_main.UserMessage([
        agent_main.ToolResultBlock(tool_use_id="tu1", content="a\nb\n")
    ])
    yield agent_main.AssistantMessage([agent_main.TextBlock("done")])
    yield agent_main.ResultMessage()


def test_emits_result_event_after_tool_use(monkeypatch):
    async def _async_return(val):
        return val
    client = MagicMock()
    client.query = MagicMock(return_value=_async_return(None))
    client.receive_messages = _fake_receive
    monkeypatch.setattr(agent_main, "_get_or_create_client",
                        lambda *a, **k: _async_return(client))
    monkeypatch.setattr(agent_main, "sync_session", lambda *a, **k: None)
    monkeypatch.setattr(agent_main, "restore_session", lambda *a, **k: None)
    monkeypatch.setattr(agent_main, "_upsert_chat_row", lambda *a, **k: None)

    payload = {
        "messages": [{"id": "u1", "role": "user", "content": "hi"}],
        "session_id": "s1", "run_id": "r1",
    }
    events = []
    async def drain():
        async for ev in agent_main.run(payload, context=None):
            events.append(ev)
    asyncio.run(drain())

    chunks = [e for e in events if e["type"] == "TOOL_CALL_CHUNK"]
    results = [e for e in events if e["type"] == "TOOL_CALL_RESULT"]
    assert len(chunks) == 1, f"expected 1 chunk, got {len(chunks)}: {events}"
    assert len(results) == 1, f"expected 1 result, got {len(results)}: {events}"
    assert chunks[0]["tool_call_id"] == results[0]["tool_call_id"] == "tu1"
    assert results[0]["content"] == "a\nb\n"
    assert results[0]["truncated"] is False
    assert results[0]["is_error"] is False
    assert results[0]["step"] == 1
