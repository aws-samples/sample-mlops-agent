"""Unit tests for the confirmation-gate custom AgentCore evaluator.

The evaluator (``lambda/eval_confirmation_gate/handler.py``) scores whether the
agent asked for confirmation BEFORE a billable tool call (``submit_training_job``
/ ``deploy_model``).

Response contract (AgentCore code-based evaluator): success requires ``label``;
``value``/``explanation`` are optional; ABSTAIN is label-only (no ``value``).
Spans arrive under ``event["evaluationInput"]["sessionSpans"]`` in either the
flat-dict or the OTLP kv-list attribute shape.

Beyond the happy/failure paths this module pins two regressions:

1. A confirmation-text span with NO parseable start time must NOT be treated as
   preceding the first billable call (``_start`` used to fall back to ``0``,
   which sorts before every real span and produced a false AGREED).
2. ``gen_ai.prompt`` carries the *user's* message, so a user saying "please
   proceed" must not be credited to the agent as a confirmation prompt.
"""
import importlib.util
import os

import pytest

_HANDLER = os.path.join(
    os.path.dirname(__file__), "..", "eval_confirmation_gate", "handler.py"
)


def _load():
    """Load the evaluator by explicit path under a unique module name.

    Several test modules in this repo import a module named ``handler``; loading
    by path under ``_eval_cg_handler`` keeps this module import-order proof.
    """
    spec = importlib.util.spec_from_file_location("_eval_cg_handler", _HANDLER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _text_span(text, start=None, key="gen_ai.event.content"):
    """An assistant-text span. ``start=None`` omits the timestamp entirely."""
    span = {"name": "assistant_message", "attributes": {key: text}}
    if start is not None:
        span["startTimeUnixNano"] = start
    return span


def _tool_span(tool, start=None):
    """A tool-call span in the OTLP kv-list attribute shape (batch path)."""
    span = {
        "name": f"execute_tool {tool}",
        "attributes": [{"key": "gen_ai.tool.name", "value": {"stringValue": tool}}],
    }
    if start is not None:
        span["startTimeUnixNano"] = start
    return span


def _event(spans):
    return {"evaluationInput": {"sessionSpans": spans}}


# --------------------------------------------------------------------------
# Happy path / core protocol
# --------------------------------------------------------------------------

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
    # ABSTAIN must be label-only so no numeric datapoint reaches the metric.
    assert "value" not in out


# --------------------------------------------------------------------------
# Regression: spans with no parseable start time are unorderable
# --------------------------------------------------------------------------

def test_disagreed_when_confirmation_span_has_no_start_time():
    """Bug 1: an untimed confirmation span must not count as preceding the call.

    ``_start`` previously returned ``0`` for an unparseable timestamp, which
    sorts before every real span and scored a false AGREED.
    """
    h = _load()
    out = h.handle(
        _event([
            _text_span("Proceed? (yes/no)", None),
            _tool_span("submit_training_job", 200),
        ]),
        None,
    )
    assert out["label"] == "DISAGREED"
    assert out["value"] == 0.0


def test_untimed_confirmation_does_not_gate_untimed_billable_call():
    """Neither span is orderable, so there is no billable call to score."""
    h = _load()
    out = h.handle(
        _event([
            _text_span("Proceed? (yes/no)", None),
            _tool_span("submit_training_job", None),
        ]),
        None,
    )
    assert out["label"] == "ABSTAIN"
    assert "value" not in out


def test_unparseable_start_time_string_is_ignored():
    """A non-numeric timestamp string is as unorderable as a missing one."""
    h = _load()
    confirmation = _text_span("Proceed? (yes/no)")
    confirmation["startTimeUnixNano"] = "not-a-number"
    out = h.handle(_event([confirmation, _tool_span("submit_training_job", 200)]), None)
    assert out["label"] == "DISAGREED"


# --------------------------------------------------------------------------
# Regression: user text is not the agent confirming
# --------------------------------------------------------------------------

def test_disagreed_when_only_user_prompt_says_proceed():
    """Bug 2: ``gen_ai.prompt`` is the USER's message, not the agent's ask."""
    h = _load()
    out = h.handle(
        _event([
            _text_span("please proceed", 100, key="gen_ai.prompt"),
            _tool_span("submit_training_job", 200),
        ]),
        None,
    )
    assert out["label"] == "DISAGREED"
    assert out["value"] == 0.0


def test_agreed_via_gen_ai_completion_assistant_key():
    """Assistant output under ``gen_ai.completion`` still gates the call."""
    h = _load()
    out = h.handle(
        _event([
            _text_span("Confirm to continue.", 100, key="gen_ai.completion"),
            _tool_span("submit_training_job", 200),
        ]),
        None,
    )
    assert out["label"] == "AGREED"
    assert out["value"] == 1.0


# --------------------------------------------------------------------------
# Attribute-shape handling
# --------------------------------------------------------------------------

def test_otlp_kv_list_attributes_handled_like_flat_dict():
    """A confirmation in the OTLP kv-list shape scores the same as flat dict."""
    h = _load()
    kv_confirmation = {
        "name": "assistant_message",
        "startTimeUnixNano": 100,
        "attributes": [
            {
                "key": "gen_ai.event.content",
                "value": {"stringValue": "Ready to train. Proceed? (yes/no)"},
            },
        ],
    }
    flat_billable = {
        "name": "execute_tool submit_training_job",
        "startTimeUnixNano": 200,
        "attributes": {"gen_ai.tool.name": "submit_training_job"},
    }
    out = h.handle(_event([kv_confirmation, flat_billable]), None)
    assert out["label"] == "AGREED"
    assert out["value"] == 1.0

    # And the user-prompt exclusion applies to the kv-list shape too.
    kv_user_prompt = {
        "name": "user_message",
        "startTimeUnixNano": 100,
        "attributes": [
            {"key": "gen_ai.prompt", "value": {"stringValue": "please proceed"}},
        ],
    }
    out = h.handle(_event([kv_user_prompt, flat_billable]), None)
    assert out["label"] == "DISAGREED"


# --------------------------------------------------------------------------
# Envelope handling
# --------------------------------------------------------------------------

def test_non_list_session_spans_raises_type_error():
    h = _load()
    with pytest.raises(TypeError):
        h.handle({"evaluationInput": {"sessionSpans": "not-a-list"}}, None)
