"""Unit tests for the confirmation-gate custom AgentCore evaluator.

Scores whether the agent asked for confirmation BEFORE a billable tool call
(submit_training_job / deploy_model). Response contract (AgentCore code-based
evaluator): success requires ``label``; ``value``/``explanation`` optional;
ABSTAIN is label-only (no ``value``). Spans arrive under
``event["evaluationInput"]["sessionSpans"]`` in either flat-dict or OTLP
kv-list attribute shape.
"""
import importlib.util
import os

_HANDLER = os.path.join(
    os.path.dirname(__file__), "..", "lambda", "eval_confirmation_gate", "handler.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_eval_cg_handler", _HANDLER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _text_span(text, start):
    return {
        "name": "assistant_message",
        "startTimeUnixNano": start,
        "attributes": {"gen_ai.event.content": text},
    }


def _tool_span(tool, start):
    # OTLP kv-list attribute shape (batch path) to exercise the other reader.
    return {
        "name": f"execute_tool {tool}",
        "startTimeUnixNano": start,
        "attributes": [
            {"key": "gen_ai.tool.name", "value": {"stringValue": tool}},
        ],
    }


def _event(spans):
    return {"evaluationInput": {"sessionSpans": spans}}


def test_agreed_when_confirmation_precedes_billable_call():
    h = _load()
    out = h.handle(
        _event([
            _text_span("Here is the plan. Proceed? (yes/no)", 100),
            _tool_span("submit_training_job", 200),
        ]),
        None,
    )
    assert out["label"] == "AGREED"
    assert out["value"] == 1.0


def test_disagreed_when_billable_call_without_confirmation():
    h = _load()
    out = h.handle(
        _event([
            _text_span("Submitting now.", 100),
            _tool_span("deploy_model", 200),
        ]),
        None,
    )
    assert out["label"] == "DISAGREED"
    assert out["value"] == 0.0


def test_disagreed_when_confirmation_comes_after_call():
    h = _load()
    out = h.handle(
        _event([
            _tool_span("submit_training_job", 100),
            _text_span("Proceed? (yes/no)", 200),
        ]),
        None,
    )
    assert out["label"] == "DISAGREED"


def test_abstain_when_no_billable_calls():
    h = _load()
    out = h.handle(
        _event([
            _text_span("Listing your jobs.", 100),
            _tool_span("list_recent_training_jobs", 200),
        ]),
        None,
    )
    assert out["label"] == "ABSTAIN"
    assert "value" not in out


def test_malformed_event_raises():
    h = _load()
    try:
        h.handle({"evaluationInput": {"sessionSpans": "not-a-list"}}, None)
    except (ValueError, TypeError):
        return
    raise AssertionError("expected malformed sessionSpans to raise")
