"""OTEL span-coverage audit.

The agent deliberately emits NO per-tool ``execute_tool`` child spans: under
split telemetry (OTEL_LOGS_EXPORTER=otlp + the ADOT LLOHandler) any manually
emitted tool span fails AgentCore batch evaluation — with a gen_ai.tool.message
event the LLO record is unparseable (SpanEventParsingException), and without it
the span has no correlated record (LogEventMissingException). Either way the
whole session fails and the optimizer reports "No sessions were identified".
Sessions with no tool spans evaluate cleanly off the invoke_agent + inference
spans, so the tool-span helpers were removed. The ordered tool-name list still
rides on the invoke_agent span's ``gen_ai.tool_calls`` attribute.
"""
import agent.main as agent_main


def test_no_manual_tool_span_helpers():
    # Guard against reintroducing per-tool execute_tool spans, which break
    # AgentCore batch evaluation in this split-telemetry setup.
    assert not hasattr(agent_main, "record_tool_span")
    assert not hasattr(agent_main, "_should_record_tool_span")


def test_module_starts_no_execute_tool_span():
    # The only start_as_current_span call left is the invoke_agent root span.
    import inspect

    src = inspect.getsource(agent_main)
    assert 'f"execute_tool' not in src
