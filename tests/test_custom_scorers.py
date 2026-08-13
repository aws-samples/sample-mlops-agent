"""Unit tests for R6 — custom scorer Lambdas.

Three layers, all in-process:
  1. ``handler._validate_custom_scorer_arns`` — handler-side ARN regex.
  2. ``eval/entrypoint.build_custom_scorers`` — the per-row Lambda wrapper.
  3. ``eval/reference_scorers/math_correctness/handler.handler`` — the
     reference scorer's scoring rules (sympy-equivalence + string fallback).
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _make_stub(name: str) -> ModuleType:
    """Fresh stub module whose every attribute resolves to a MagicMock."""
    mod = ModuleType(name)
    mod.__getattr__ = lambda attr, _n=name: MagicMock(name=f"{_n}.{attr}")  # type: ignore[attr-defined]
    return mod


@contextlib.contextmanager
def _stubbed_modules(*names: str):
    """Force stubs into sys.modules for the given names, restoring whatever
    was there (including real installed packages) on exit.

    Never mutate an already-imported real module — earlier revisions of this
    file overwrote the real ``botocore.config.Config`` in place, poisoning
    every later boto3 user in the pytest process.
    """
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for n in names:
            sys.modules[n] = _make_stub(n)
        yield
    finally:
        for n, orig in saved.items():
            if orig is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = orig


def _load_by_path(module_name: str, *rel_parts: str) -> ModuleType:
    """Load a module by file path under a unique name — no sys.path edits,
    no collisions with the many other test modules that import a file
    called ``handler``."""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", *rel_parts))
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        # Keep the local reference; drop the sys.modules entry so nothing
        # else in the suite can accidentally pick it up.
        sys.modules.pop(module_name, None)
    return mod


# ----- 1. handler._validate_custom_scorer_arns ----------------------------

def _import_sagemaker_handler():
    return _load_by_path("_sm_skill_handler", "lambda", "skills", "sagemaker", "handler.py")


def test_validate_arns_accepts_well_formed():
    h = _import_sagemaker_handler()
    h._validate_custom_scorer_arns([
        "arn:aws:lambda:us-east-1:123456789012:function:math-scorer",
        "arn:aws:lambda:us-west-2:123456789012:function:my-scorer:prod",
        "arn:aws:lambda:eu-west-1:123456789012:function:my-scorer:42",
        "arn:aws:lambda:eu-west-1:123456789012:function:my-scorer:$LATEST",
    ])


def test_validate_arns_rejects_garbage():
    h = _import_sagemaker_handler()
    bad = [
        "not-an-arn",
        "arn:aws:lambda:us-east-1:123:function:scorer",            # 3-digit account
        "arn:aws:s3:::bucket/key",                                  # wrong service
        "arn:aws:lambda:us-east-1:123456789012:function:bad name",  # space
    ]
    for arn in bad:
        try:
            h._validate_custom_scorer_arns([arn])
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {arn!r}")


def test_validate_arns_rejects_non_list():
    h = _import_sagemaker_handler()
    try:
        h._validate_custom_scorer_arns("arn:aws:lambda:us-east-1:123456789012:function:s")  # type: ignore[arg-type]
    except ValueError as e:
        assert "must be a list" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-list input")


# ----- 2. eval/entrypoint.build_custom_scorers ----------------------------

def _import_entrypoint():
    """Import lambda/skills/sagemaker/eval/entrypoint.py.

    Heavy ML deps (mlflow, pandas, boto3, …) are force-stubbed for the
    duration of the import — entrypoint runs ``_ensure_aws_env_vars()`` at
    module level, which must not touch real AWS credentials — and the real
    modules are restored afterwards. The returned module keeps references
    to its stubs, so tests can swap ``e.boto3`` freely without leaking.
    """
    # entrypoint.py reads several env vars at module import time. Stub them
    # all — these tests exercise build_custom_scorers in isolation.
    for k, v in {
        "MLFLOW_TRACKING_URI":  "file:///tmp/mlflow-test",
        "MLFLOW_RUN_ID":        "run-test",
        "TARGET_MODEL":         "endpoints:/test",
        "JUDGE_MODEL":          "bedrock:/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "SCORERS":              "Correctness",
        "EVAL_DATASET_S3_URI":  "s3://b/k.parquet",
        "AWS_REGION":           "us-east-1",
    }.items():
        os.environ.setdefault(k, v)
    with _stubbed_modules(
        "mlflow", "mlflow.tracking", "mlflow.exceptions", "mlflow.genai",
        "mlflow.genai.scorers", "datasets", "transformers", "torch",
        "pandas", "numpy", "evidently", "huggingface_hub",
        "boto3", "botocore", "botocore.config", "botocore.exceptions",
    ):
        # `from botocore.config import Config` / `from botocore.exceptions
        # import ClientError` need real attributes on the stub modules.
        sys.modules["botocore.config"].Config = lambda *a, **kw: MagicMock(name="Config")
        sys.modules["botocore.exceptions"].ClientError = type("ClientError", (Exception,), {})
        # entrypoint._ensure_aws_env_vars() runs at import time and writes
        # frozen-credential attributes into os.environ — those must be str.
        frozen = MagicMock(access_key="AKTEST", secret_key="SKTEST", token="TKTEST")  # nosec B106
        session = MagicMock()
        session.get_credentials.return_value.get_frozen_credentials.return_value = frozen
        sys.modules["boto3"].session = MagicMock(Session=MagicMock(return_value=session))
        return _load_by_path("_eval_entrypoint", "lambda", "skills", "sagemaker", "eval", "entrypoint.py")


def _make_invoke(*, payload: dict | None = None, status: int = 200,
                 raises: Exception | None = None,
                 function_error: str | None = None):
    """Build a MagicMock lambda client whose .invoke() returns the given body."""
    client = MagicMock()
    if raises is not None:
        client.invoke.side_effect = raises
        return client
    body = (json.dumps(payload) if payload is not None else "").encode()
    resp = {"StatusCode": status, "Payload": io.BytesIO(body)}
    if function_error:
        resp["FunctionError"] = function_error
    client.invoke.return_value = resp
    return client


def _row_score(e, client, arn="arn:aws:lambda:us-east-1:123456789012:function:math-scorer",
               inputs=None, outputs=None, expectations=None):
    """Score one row through the undecorated helper (R6 unit contract)."""
    return e._invoke_scorer_lambda(client, arn, arn.split(":function:")[-1].split(":")[0],
                                   inputs, outputs, expectations)


def test_custom_scorer_passes_payload_and_returns_score():
    e = _import_entrypoint()
    client = _make_invoke(payload={"name": "math_scorer", "score": 0.75, "reason": "ok"})
    out = _row_score(e, client, inputs={"q": "x"}, outputs={"answer": "y"},
                     expectations={"answer": "y"})
    # QA R6 closure: mlflow 3.4 scorers must return a number (or Feedback),
    # not the raw contract dict — the wrapper unwraps to the float score.
    assert out == 0.75
    sent = json.loads(client.invoke.call_args.kwargs["Payload"].decode())
    assert sent == {"inputs": {"q": "x"}, "outputs": {"answer": "y"},
                    "expectations": {"answer": "y"}}


def test_custom_scorer_invoke_exception_returns_none():
    e = _import_entrypoint()
    assert _row_score(e, _make_invoke(raises=RuntimeError("boom"))) is None


def test_custom_scorer_function_error_returns_none():
    """Lambda FunctionError ('Unhandled' / 'Handled') must be treated as failure."""
    e = _import_entrypoint()
    client = _make_invoke(payload={"errorMessage": "boom"}, function_error="Unhandled")
    assert _row_score(e, client) is None


def test_custom_scorer_non_numeric_score_returns_none():
    e = _import_entrypoint()
    assert _row_score(e, _make_invoke(payload={"name": "x", "score": "not-a-number"})) is None


def test_row_scorer_uses_arn_function_name_as_metric_name():
    """The MLflow metric key comes from the callable __name__, derived from
    the ARN function-name segment (alias/version suffix stripped)."""
    e = _import_entrypoint()
    fn = e._make_row_scorer(MagicMock(),
                            "arn:aws:lambda:us-east-1:123456789012:function:my-prod-scorer:prod")
    assert fn.__name__ == "my-prod-scorer"


# ----- 3. reference math_correctness scorer ------------------------------

def _import_math_scorer():
    return _load_by_path(
        "_math_scorer_handler",
        "lambda", "skills", "sagemaker", "eval", "reference_scorers",
        "math_correctness", "handler.py",
    )


def test_math_scorer_string_match_scores_one():
    h = _import_math_scorer()
    out = h.handler(
        {"inputs": {"q": "2+2"}, "outputs": {"output": "4"}, "expectations": {"expected_output": "4"}},
        None,
    )
    assert out["name"] == "math_correctness"
    assert out["score"] == 1.0


def test_math_scorer_string_mismatch_scores_zero():
    h = _import_math_scorer()
    out = h.handler(
        {"inputs": {}, "outputs": {"output": "5"}, "expectations": {"expected_output": "4"}},
        None,
    )
    assert out["score"] == 0.0
    assert "got=" in out["reason"]


def test_math_scorer_missing_fields_score_zero_with_reason():
    h = _import_math_scorer()
    out = h.handler({"inputs": {}, "outputs": {}, "expectations": {}}, None)
    assert out["score"] == 0.0
    assert "missing" in out["reason"]


def test_math_scorer_accepts_alternate_output_keys():
    """`completion` / `response` / `answer` are valid output keys per contract."""
    h = _import_math_scorer()
    out = h.handler(
        {"outputs": {"completion": "42"}, "expectations": {"answer": "42"}},
        None,
    )
    assert out["score"] == 1.0


def test_math_scorer_handles_non_dict_event_via_json_string():
    """API-Gateway-style invocations may deliver the body as a JSON string."""
    h = _import_math_scorer()
    body = json.dumps({"outputs": {"output": "x"}, "expectations": {"answer": "x"}})
    out = h.handler(body, None)
    assert out["score"] == 1.0


# ----- 4. BUG-002: RETRIEVER span emission for trace-based judges ---------

def test_predict_emits_retriever_span_when_context_present():
    """RetrievalGroundedness reads retrieved docs from RETRIEVER spans on the
    row trace (mlflow trace_utils.extract_retrieval_context_from_trace). The
    predict path must surface the dataset's static context as such a span —
    without it the judge returned None for every row (QA BUG-002)."""
    e = _import_entrypoint()
    spans = []

    class _Span:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def set_inputs(self, v): spans.append(("inputs", v))
        def set_outputs(self, v): spans.append(("outputs", v))

    e.mlflow.start_span = lambda **kw: (spans.append(("open", kw)) or _Span())
    e._emit_retriever_span(query="Q?", context="ctx blob")

    kinds = [k for k, _ in spans]
    assert kinds == ["open", "inputs", "outputs"]
    open_kw = spans[0][1]
    assert str(open_kw.get("span_type")).upper().endswith("RETRIEVER")
    outputs = spans[2][1]
    assert outputs[0]["page_content"] == "ctx blob", \
        "chunk must use a key _parse_chunk recognizes (page_content/content/text)"


def test_predict_skips_retriever_span_without_context():
    """No context column → no retriever span; predict must still run."""
    e = _import_entrypoint()
    called = []
    e._emit_retriever_span = lambda **kw: called.append(kw)
    # Recreate the wrapper logic exactly as main() wires it.
    prompt_inputs = {"question": "just a question"}
    prompt = e._build_prompt(prompt_inputs)
    if prompt_inputs.get("context"):
        e._emit_retriever_span(query=prompt, context=str(prompt_inputs["context"]))
    assert called == []
    assert prompt == "just a question"


def test_math_scorer_accepts_string_outputs_and_expectations():
    """R6 closure: mlflow hands simple predict_fn rows as raw strings —
    the reference scorer must handle str outputs/expectations."""
    h = _import_math_scorer()
    out = h.handler({"inputs": {}, "outputs": "4", "expectations": "2*2"}, None)
    assert out["score"] == 1.0


# ----- 5. QA-01: Correctness expectations contract ------------------------

def test_normalise_record_sets_only_expected_response():
    """QA-01: mlflow >= 3.11 ``judges.is_correct`` raises "Only one of
    expected_response or expected_facts should be provided, not both." —
    every Correctness invocation failed (20/20) because _normalise_record
    stamped BOTH canonical keys on each row. The record must carry
    ``expected_response`` and must NOT carry ``expected_facts``."""
    e = _import_entrypoint()
    rec = e._normalise_record(
        {"question": "What was FY23 revenue?", "answer": "Revenue was $10M. It grew 5%."},
        input_cols=["question"],
        expectation_cols=["answer"],
        context_cols=None,
    )
    assert rec["expectations"]["expected_response"] == "Revenue was $10M. It grew 5%."
    assert "expected_facts" not in rec["expectations"]
    # The original ground-truth column is preserved alongside the canonical key.
    assert rec["expectations"]["answer"] == "Revenue was $10M. It grew 5%."
    assert rec["inputs"] == {"question": "What was FY23 revenue?"}


def test_normalise_record_no_ground_truth_leaves_expectations_empty():
    """Rows without any expectation column must not invent canonical keys."""
    e = _import_entrypoint()
    rec = e._normalise_record(
        {"question": "ungrounded prompt"},
        input_cols=["question"],
        expectation_cols=["answer"],
        context_cols=None,
    )
    assert rec["expectations"] == {}
