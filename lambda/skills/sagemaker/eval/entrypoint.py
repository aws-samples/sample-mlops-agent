"""SageMaker Processing entrypoint — run mlflow.genai.evaluate on a HF dataset.

Consumes one of two inputs from EVAL_DATASET_S3_URI:

  1) `*.json` eval-spec authored by huggingface-skill prepare_eval_dataset —
     the container calls datasets.load_dataset(), applies the task-default
     column mapping, and feeds {inputs, expectations} records into
     mlflow.genai.evaluate. This is the primary path: no size ceiling, no
     Lambda /tmp issues, dataset IO happens where it belongs.

  2) `*.parquet` with pre-baked inputs/expectations columns — legacy path
     for callers that stage their own records directly.

Env vars (all required):
    MLFLOW_TRACKING_URI   — SageMaker MLflow App ARN.
    MLFLOW_RUN_ID         — pre-created RUNNING run from the SageMaker skill.
    TARGET_MODEL          — "bedrock:/<model_id>" or "sagemaker:/<endpoint>".
    JUDGE_MODEL           — "bedrock:/<judge_id>".
    SCORERS               — comma-separated scorer names from SCORER_CATALOG.
    EVAL_DATASET_S3_URI   — .json spec (preferred) or .parquet records.
    TASK                  — "question_answering" | "rag" | ...
"""
import json
import os
from typing import Any, Callable
from urllib.parse import urlparse

# MLflow hardening — MUST be set before `import mlflow` so the SDK picks them up
# at import time. Without these, a slow/half-ready SageMaker MLflow tracking
# server makes mlflow.genai.evaluate hang indefinitely on its first-sample
# "Testing model prediction" trace-logging step (observed: processing job
# sample-mlops-agent-eval-1786504461 sat InProgress ~26h with the container log
# dead-ending at that line). The job's 24h MaxRuntime then wastes a full day of
# instance time before SageMaker force-stops it.
#   - MLFLOW_HTTP_REQUEST_TIMEOUT: bound EVERY tracking-server HTTP call so a
#     stalled connection fails fast instead of blocking forever.
#   - MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION: skip the pre-flight first-sample
#     trace round-trip (the exact call that hung); the real eval loop still runs
#     and logs traces, so this only removes a redundant validation probe.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
os.environ.setdefault("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "True")

import boto3
import mlflow
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from mlflow.genai import evaluate, scorers as S

# Cap boto retries so a bad model ID surfaces in seconds rather than the
# default ~10-attempt adaptive backoff (which previously caused a hung eval
# job to sit in InProgress for hours — see processing job
# sample-mlops-agent-eval-1777083185). standard mode + 2 attempts total is
# enough to tolerate transient throttling without masking validation errors.
_BOTO_CONFIG = Config(retries={"max_attempts": 2, "mode": "standard"})

TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
RUN_ID       = os.environ["MLFLOW_RUN_ID"]
TARGET       = os.environ["TARGET_MODEL"]
JUDGE        = os.environ["JUDGE_MODEL"]
SCORERS      = [s.strip() for s in os.environ["SCORERS"].split(",") if s.strip()]
# R6: optional list of Lambda ARNs to invoke per row as custom MLflow
# scorers. Empty string → no custom scorers (legacy behaviour).
CUSTOM_SCORER_LAMBDA_ARNS = [
    s.strip() for s in os.environ.get("CUSTOM_SCORER_LAMBDA_ARNS", "").split(",") if s.strip()
]
DATASET_URI  = os.environ["EVAL_DATASET_S3_URI"]
TASK         = os.environ.get("TASK", "question_answering")
# SageMaker Processing containers do not auto-populate AWS_DEFAULT_REGION, so
# boto3.client("bedrock-runtime") would fail with NoRegionError. Resolve the
# region explicitly (submitter injects AWS_REGION) and fail loudly otherwise.
AWS_REGION   = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or boto3.session.Session().region_name
)
if not AWS_REGION:
    raise RuntimeError(
        "AWS region is not set — cannot create Bedrock/SageMaker runtime clients. "
        "Submitter must inject AWS_REGION into the Processing job Environment."
    )


def _ensure_aws_env_vars() -> None:
    """Freeze the task-role credentials into env vars MLflow's Bedrock
    gateway adapter reads directly.

    MLflow 3.4's builtin judge path (Correctness / RelevanceToQuery / …)
    routes through gateway_adapter → model_utils._get_provider_instance,
    which instantiates ``AWSIdAndKey(aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), …)``.
    That helper does NOT consult the boto3 credential chain or the
    SageMaker Processing container-credentials endpoint — so without these
    env vars set, every judge invocation fails with a pydantic
    ValidationError before the Bedrock call is ever attempted, and the
    scorer silently emits a null value into the MLflow run.
    Modeled on rbc-evals-demo src/tools.py:_ensure_aws_env_vars.
    """
    creds = boto3.session.Session().get_credentials()
    if creds is None:
        raise RuntimeError(
            "No AWS credentials resolvable inside the Processing container — "
            "task role is not attached or IMDS is unreachable."
        )
    frozen = creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"]     = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    os.environ["AWS_DEFAULT_REGION"]    = AWS_REGION
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token


_ensure_aws_env_vars()

mlflow.set_tracking_uri(TRACKING_URI)


# ── target / judge predict wrappers ────────────────────────────────────────
def _bedrock_predict(model_id: str, prompt: str) -> str:
    """Call Bedrock Converse and return the first text chunk."""
    rt = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=_BOTO_CONFIG)
    resp = rt.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": 0.2, "maxTokens": 1024},
    )
    return resp["output"]["message"]["content"][0]["text"]


def _sagemaker_predict(endpoint_name: str, prompt: str) -> str:
    """Call a SageMaker real-time endpoint with an OpenAI-chat-style payload."""
    sm = boto3.client("sagemaker-runtime", region_name=AWS_REGION, config=_BOTO_CONFIG)
    body = json.dumps({"messages": [{"role": "user", "content": prompt}]})
    resp = sm.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=body,
    )
    data = json.loads(resp["Body"].read())
    # Tolerate both {choices:[{message:{content}}]} and {generated_text} shapes.
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    if "generated_text" in data:
        return data["generated_text"]
    return json.dumps(data)


def build_predict_fn(spec: str) -> Callable[[str], str]:
    """Parse TARGET_MODEL spec and return a callable(prompt) -> str."""
    scheme, ref = spec.split(":/", 1)
    if scheme == "bedrock":
        return lambda q: _bedrock_predict(ref, q)
    if scheme == "sagemaker":
        return lambda q: _sagemaker_predict(ref, q)
    raise ValueError(f"Unknown target_model scheme: {scheme!r}")


# ── scorer catalog ────────────────────────────────────────────────────────
# Only MLflow 3.4.0 built-ins that are actually exported from
# mlflow.genai.scorers. Authoritative list (from the 3.4.0 wheel's
# scorers/__init__.py __all__):
#   Correctness, ExpectationsGuidelines, Guidelines, RelevanceToQuery,
#   RetrievalGroundedness, RetrievalRelevance, RetrievalSufficiency, Safety.
# Notes:
#   - Equivalence and Fluency existed in earlier 3.x and were dropped in 3.4.
#     Referencing them raises AttributeError at module import and crashes the
#     whole Processing job.
#   - Faithfulness/AnswerRelevancy/Bias/ChrfScore/BleuScore are DeepEval-only
#     (NOT in mlflow.genai.scorers) and stay out of the catalog.
SCORER_CATALOG: dict[str, Any] = {
    "Safety":                 S.Safety,
    "RelevanceToQuery":       S.RelevanceToQuery,
    "Correctness":            S.Correctness,
    "Guidelines":             S.Guidelines,
    "ExpectationsGuidelines": S.ExpectationsGuidelines,
    "RetrievalGroundedness":  S.RetrievalGroundedness,
    "RetrievalRelevance":     S.RetrievalRelevance,
    "RetrievalSufficiency":   S.RetrievalSufficiency,
}

def _resolve_scorer_name(name: str) -> str:
    """Validate that `name` is a canonical SCORER_CATALOG key.

    The container accepts canonical names only. Translation of user-facing
    vocabulary ("faithfulness", "answer relevance", …) into these names is
    the agent's job via mlflow-skill___list_scorers — NOT the container's.
    Raises ValueError listing valid names if the caller (agent) sent
    something unknown, so the failure mode is a clear error instead of a
    raw KeyError bubbling out of main().
    """
    if name in SCORER_CATALOG:
        return name
    valid = sorted(SCORER_CATALOG.keys())
    raise ValueError(
        f"Unknown scorer name {name!r}. Valid canonical names: {valid}. "
        f"The agent must call mlflow-skill___list_scorers first and pass "
        f"canonical names verbatim; the eval container does not translate "
        f"plain-English aliases."
    )

# Guidelines requires a per-instance `guidelines` string. Since the public
# API is a fixed catalog (no inline rubrics from the agent), bake one
# sensible default: answer groundedness. Add more catalog entries later if
# we need richer rubrics — do NOT expand the tool schema to accept them.
_DEFAULT_GUIDELINES = (
    "The answer must be grounded in any provided context and must not "
    "hallucinate facts beyond what the context supports."
)


def build_scorer(name: str) -> Any:
    """Instantiate scorer `name` with the judge model.

    Guidelines needs both `name` and `guidelines` kwargs on top of `model`;
    every other scorer takes `model=JUDGE` alone.

    Accepts either a canonical SCORER_CATALOG key or a SCORER_ALIASES alias
    (e.g. "faithfulness" → "RetrievalGroundedness") so agent prompts that
    use MLflow-agnostic vocabulary still run without burning an instance.
    """
    canonical = _resolve_scorer_name(name)
    cls = SCORER_CATALOG[canonical]
    if canonical == "Guidelines":
        return cls(
            name="answer_groundedness",
            guidelines=_DEFAULT_GUIDELINES,
            model=JUDGE,
        )
    return cls(model=JUDGE)


def _invoke_scorer_lambda(lam: Any, arn: str, metric_name: str,
                          inputs: Any, outputs: Any, expectations: Any) -> Any:
    """Invoke one custom-scorer Lambda for a single eval row (R6).

    5 s boto3 read timeout (set on the client) keeps a misbehaving scorer
    from stalling the whole evaluation. Non-200 responses, FunctionErrors,
    malformed bodies, or non-numeric scores are logged and returned as None
    — MLflow records a null metric for the row and the evaluation continues.

    Args:
        lam: boto3 Lambda client (short read timeout).
        arn: Scorer Lambda ARN.
        metric_name: Metric key used in logs.
        inputs / outputs / expectations: The MLflow row fields, forwarded as
            the canonical ``{"inputs", "outputs", "expectations"}`` payload.

    Returns:
        float score, or None on any failure.
    """
    payload = {"inputs": inputs, "outputs": outputs, "expectations": expectations}
    try:
        resp = lam.invoke(
            FunctionName=arn,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload, default=str).encode(),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[custom_scorer] {arn}: invoke failed: {exc}")
        return None
    status = resp.get("StatusCode", 0)
    body_raw = resp.get("Payload")
    try:
        body = json.loads(body_raw.read()) if body_raw else {}
    except Exception as exc:  # noqa: BLE001
        print(f"[custom_scorer] {arn}: non-JSON response: {exc}")
        return None
    if status >= 300 or resp.get("FunctionError"):
        print(f"[custom_scorer] {arn}: status={status} error={body!r}")
        return None
    score = body.get("score")
    if not isinstance(score, (int, float)):
        print(f"[custom_scorer] {arn}: non-numeric score={score!r}")
        return None
    reason = body.get("reason", "")
    if reason:
        print(f"[custom_scorer] {metric_name}: score={score} reason={reason!r}")
    return float(score)


def _make_row_scorer(lam: Any, arn: str) -> Callable[..., Any]:
    """Build the plain per-row scoring function for one Lambda ARN (R6).

    Kept as a separate, undecorated function so unit tests can exercise the
    payload/error contract without a real mlflow installation.
    """
    fn_part = arn.split(":function:", 1)[-1]
    metric_name = fn_part.split(":", 1)[0]

    def _scorer(inputs: Any = None, outputs: Any = None, expectations: Any = None) -> Any:
        return _invoke_scorer_lambda(lam, arn, metric_name, inputs, outputs, expectations)

    # The metric key MLflow logs comes from the callable name.
    _scorer.__name__ = metric_name
    return _scorer


def build_custom_scorers(arns: list[str]) -> list[Any]:
    """Wrap each scorer-Lambda ARN as an MLflow Scorer (R6).

    QA R6 closure: mlflow 3.4's ``validate_scorers`` REJECTS plain callables
    ("The `scorers` argument must be a list of scorers … invalid item with
    type: function" — observed on job sample-mlops-agent-eval-1785029964),
    so each row function is wrapped with the ``@scorer`` decorator, which
    yields a first-class Scorer whose metric name comes from ``__name__``.
    """
    import boto3  # noqa: PLC0415
    from botocore.config import Config  # noqa: PLC0415
    from mlflow.genai.scorers import scorer as scorer_decorator  # noqa: PLC0415

    # 5 s per-row cap. A stuck scorer must not hold up 2k-row evals.
    cfg = Config(read_timeout=5, retries={"max_attempts": 1})
    lam = boto3.client("lambda", region_name=AWS_REGION, config=cfg)
    return [scorer_decorator(_make_row_scorer(lam, arn)) for arn in arns]



# ── task mapping + row normalisation ───────────────────────────────────────
# Default {inputs, expectations, context} column-candidate lists per task.
# Moved here from the HF Lambda so dataset materialisation + mapping happen
# in one place. The Lambda only authors the spec; we apply it here.
TASK_DEFAULT_MAPPING: dict[str, dict[str, list[str]]] = {
    "question_answering": {
        "inputs":       ["question", "query", "prompt"],
        "expectations": ["answer", "answers", "ground_truth", "expected_answer"],
    },
    "rag": {
        "inputs":       ["question", "query"],
        "expectations": ["answer", "ground_truth", "expected_answer"],
        # `evidence` covers PatronusAI/financebench-style datasets that
        # carry retrieved passages as a list of structured dicts. The
        # flattening helper handles both shapes.
        "context":      ["context", "contexts", "retrieved_context", "evidence"],
    },
    "summarization": {
        "inputs":       ["document", "article", "text", "input"],
        "expectations": ["summary", "highlights", "expected_summary"],
    },
    "text_generation": {
        "inputs":       ["prompt", "instruction", "input"],
        "expectations": ["completion", "response", "output", "expected_output"],
    },
    "classification": {
        "inputs":       ["text", "sentence", "input"],
        "expectations": ["label", "class", "category"],
    },
}


_CONTEXT_TEXT_KEYS = (
    "evidence_text",
    "text",
    "passage",
    "content",
    "chunk",
    "evidence_text_full_page",
)


def _flatten_context(value: Any) -> str:
    """Coerce a dataset's context column into a single string prompt-ready blob.

    Handles three shapes produced by HuggingFace datasets:
      - plain string → returned as-is
      - list[str] → joined with blank-line separators
      - list[dict] (e.g. PatronusAI/financebench `evidence`) → pull the
        first matching text key per dict, join with blank-line separators
    Anything else is stringified as a last resort.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in _CONTEXT_TEXT_KEYS:
                    if key in item and isinstance(item[key], str) and item[key].strip():
                        parts.append(item[key])
                        break
                else:
                    # No recognised text field — stringify the dict so we
                    # don't silently drop evidence the caller supplied.
                    parts.append(json.dumps(item, default=str))
            else:
                parts.append(str(item))
        return "\n\n".join(p for p in parts if p)
    return str(value)


def _normalise_record(
    row: dict,
    *,
    input_cols: list[str],
    expectation_cols: list[str],
    context_cols: list[str] | None,
) -> dict:
    """Project a raw dataset row into the MLflow genai eval shape.

    Returns ``{"inputs": {...}, "expectations": {...}}`` where:

    - ``inputs`` carries every candidate input column that exists in the
      row, plus ``context`` for RAG tasks (so it reaches ``predict_fn``
      via the ``**inputs`` splat and the target model can ground its
      answer on the retrieved passage).
    - ``expectations`` carries the original ground-truth column AND the
      canonical ``expected_response`` key that MLflow's Equivalence and
      Correctness scorers read. Exactly ONE of ``expected_response`` /
      ``expected_facts`` may be set: mlflow >= 3.11 ``judges.is_correct``
      raises "Only one of expected_response or expected_facts should be
      provided, not both." when a row carries both (QA-01).
    """
    inputs: dict[str, Any] = {}
    for col in input_cols:
        if col in row and row[col] is not None:
            inputs[col] = row[col]
    if not inputs:
        # Better to include everything than silently drop the record.
        inputs = {k: v for k, v in row.items() if k not in expectation_cols}

    expectations: dict[str, Any] = {}
    ground_truth: Any = None
    for col in expectation_cols:
        if col in row and row[col] is not None:
            expectations[col] = row[col]
            if ground_truth is None:
                ground_truth = row[col]

    # Canonical key the built-in scorers read directly. Without it,
    # Equivalence and Correctness silently score None on every row.
    # Do NOT also set expected_facts: mlflow >= 3.11 judges.is_correct
    # rejects rows that carry both keys (QA-01 — Correctness failed 20/20),
    # and expected_response is the faithful ground truth while
    # expected_facts was only a lossy regex sentence-split of it.
    if ground_truth is not None:
        gt_str = str(ground_truth) if not isinstance(ground_truth, str) else ground_truth
        expectations.setdefault("expected_response", gt_str)

    # Surface retrieved context INSIDE inputs so the target model receives
    # it through predict_fn(**inputs). Placing it under expectations would
    # still let RetrievalGroundedness see it but blinds the generator.
    # Flatten to a string so list[dict] passages (e.g. financebench
    # `evidence`) reach Bedrock Converse as readable text, not a repr.
    if context_cols:
        for col in context_cols:
            if col in row and row[col] is not None:
                flat = _flatten_context(row[col])
                if flat:
                    inputs.setdefault("context", flat)
                break

    return {"inputs": inputs, "expectations": expectations}


def _records_from_spec(spec: dict) -> list[dict]:
    """Materialise a dataset from a prepare_eval_dataset spec.

    Loads the dataset inside the container (datasets.load_dataset), applies
    the task-default column mapping merged with any spec overrides, and
    returns a list of {inputs, expectations} records.

    R3 (spec version ≥ 2): if the spec carries ``target_schema`` different
    from the source, apply ``schema_transform`` to each row *before*
    column mapping. Source schema is either taken from the spec
    (``source_schema``) or inferred from the loaded dataset's columns.
    """
    from datasets import load_dataset  # type: ignore  # noqa: PLC0415
    import schema_transform  # type: ignore  # noqa: PLC0415 — container-local

    task_type  = spec["task_type"]
    ds_name    = spec["dataset_name"]
    split      = spec.get("split", "train")
    config     = spec.get("config") or None
    max_rows   = int(spec.get("max_rows") or 0)
    mapping    = spec.get("column_mapping") or {}
    defaults   = TASK_DEFAULT_MAPPING.get(task_type, {})

    input_cols       = mapping.get("inputs")       or defaults.get("inputs",       [])
    expectation_cols = mapping.get("expectations") or defaults.get("expectations", [])
    context_cols     = mapping.get("context")      or defaults.get("context")

    if not input_cols:
        raise ValueError(
            f"No input columns for task_type={task_type!r} — set spec.column_mapping.inputs"
        )

    # Slice at load time so we never pull more than needed.
    sliced_split = f"{split}[:{max_rows}]" if max_rows else split
    # Pin the Hub revision (default "main") so an eval can lock to an immutable
    # commit SHA — an unpinned pull would silently follow upstream repo changes.
    hf_revision = spec.get("revision") or os.environ.get("HF_HUB_REVISION", "main")
    ds = load_dataset(ds_name, name=config, split=sliced_split, revision=hf_revision)

    rows = ds.to_list()

    # R3 — schema transformation. Only activate when the caller set
    # target_schema; otherwise preserve pre-R3 behaviour bit-for-bit.
    target_schema = spec.get("target_schema")
    if target_schema:
        source_schema = spec.get("source_schema")
        if not source_schema:
            # Caller deferred inference to us; look at the loaded dataset's
            # column names rather than the empty-rows case.
            source_schema = schema_transform.infer_source_schema(
                ds.column_names if hasattr(ds, "column_names") else (rows[0].keys() if rows else [])
            )
            if source_schema == "unknown":
                raise ValueError(
                    f"source_schema could not be inferred from columns "
                    f"{list(rows[0].keys()) if rows else []!r}. Pass "
                    f"source_schema explicitly on prepare_eval_dataset."
                )
        # May raise UnsupportedTransformError for bad pairs; let it
        # bubble so the Processing job FailureReason carries the reason.
        mechanism = spec.get("transform_mechanism") or schema_transform.resolve_mechanism(
            source=source_schema, target=target_schema,
        )
        rows = schema_transform.apply_transform(rows, mechanism=mechanism)

    return [
        _normalise_record(
            row,
            input_cols=input_cols,
            expectation_cols=expectation_cols,
            context_cols=context_cols,
        )
        for row in rows
    ]


def _records_from_parquet(uri: str) -> list[dict]:
    """Read a pre-baked parquet with inputs/expectations columns (legacy)."""
    df = pd.read_parquet(uri)
    records: list[dict] = []
    for row in df.to_dict("records"):
        # Tolerate both JSON-string and native dict columns.
        ins = row.get("inputs")
        exp = row.get("expectations")
        records.append({
            "inputs":       json.loads(ins) if isinstance(ins, str) else (ins or {}),
            "expectations": json.loads(exp) if isinstance(exp, str) else (exp or {}),
        })
    return records


def _load_records(uri: str) -> list[dict]:
    """Route on object key suffix — JSON spec vs parquet records."""
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if key.endswith(".json"):
        bucket = parsed.netloc
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        spec = json.loads(obj["Body"].read())
        return _records_from_spec(spec)
    if key.endswith(".parquet"):
        return _records_from_parquet(uri)
    raise ValueError(f"Unsupported EVAL_DATASET_S3_URI suffix: {uri!r}")


# ── inputs-dict → prompt string ────────────────────────────────────────────
def _build_prompt(inputs: dict) -> str:
    """Assemble the user-facing prompt string from an inputs dict.

    Picks the primary question/prompt column and, when ``context`` is
    also present (RAG tasks), prepends it so the target model can ground
    its answer. Without this, RAG evaluations score a blind generator.
    """
    question: str | None = None
    for key in ("question", "prompt", "query", "document", "input", "text"):
        if key in inputs and inputs[key] is not None:
            question = str(inputs[key])
            break
    if question is None:
        # Last resort — stringify the whole inputs dict.
        return json.dumps(inputs)

    context = inputs.get("context")
    if context:
        return f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely."
    return question


def _emit_retriever_span(*, query: str, context: str) -> None:
    """Surface a static dataset's context column as a RETRIEVER span
    (QA BUG-002).

    MLflow 3.x's RetrievalGroundedness / RetrievalRelevance /
    RetrievalSufficiency judges read retrieved documents from spans of type
    RETRIEVER on the row's trace — they do NOT parse context out of the
    prompt string. Our predict path is a plain str→str callable, so those
    judges previously saw no retrieval step and returned None for every row
    (observed: retrieval_groundedness missing from run
    32b8462475194dac94a56bf2fa2ea2e4 while the other scorers logged fine).

    Args:
        query: The user question for this row.
        context: The flattened context blob from the dataset row.
    """
    # MLflow's extract_retrieval_context_from_trace → _parse_chunk accepts
    # dict chunks keyed page_content/content/text (verified against
    # mlflow/genai/utils/trace_utils.py) — plain dicts avoid any Document
    # serialization variance across 3.x releases.
    docs = [{"page_content": context, "metadata": {"source": "eval_dataset"}}]
    try:
        from mlflow.entities import SpanType  # noqa: PLC0415
        span_type: Any = SpanType.RETRIEVER
    except Exception:  # noqa: BLE001 — string form is accepted too
        span_type = "RETRIEVER"
    with mlflow.start_span(name="static_context_retriever", span_type=span_type) as span:
        span.set_inputs({"query": query})
        span.set_outputs(docs)


def _validate_target_model(target: str) -> None:
    """Send a tiny Converse/InvokeEndpoint ping so a bad model ID fails fast.

    Runs BEFORE mlflow.genai.evaluate, which otherwise swallows the error
    inside its ``predict_fn`` smoke call and leaves the Processing job
    hanging in InProgress until the 24h StoppingCondition fires. Capped
    retries (`_BOTO_CONFIG`) keep the total wait to seconds.

    Regression guard: processing job sample-mlops-agent-eval-1777083185 hung
    for >14h on ``bedrock:/global.amazon.nova-2-lite-v1:0`` — a UI display
    alias, not a real Bedrock model ID. This pre-flight call raises with the
    exact ClientError so the submitter sees the problem in the job's
    FailureReason immediately.
    """
    scheme, ref = target.split(":/", 1)
    try:
        if scheme == "bedrock":
            rt = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=_BOTO_CONFIG)
            rt.converse(
                modelId=ref,
                messages=[{"role": "user", "content": [{"text": "ping"}]}],
                inferenceConfig={"maxTokens": 1},
            )
            return
        if scheme == "sagemaker":
            sm = boto3.client("sagemaker", region_name=AWS_REGION, config=_BOTO_CONFIG)
            sm.describe_endpoint(EndpointName=ref)
            return
        raise ValueError(f"Unknown target_model scheme: {scheme!r}")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "ClientError")
        raise RuntimeError(
            f"TARGET_MODEL={target!r} failed pre-flight validation ({code}): {exc}. "
            "Fix the model ID before resubmitting — Bedrock model IDs must be "
            "the canonical on-demand or inference-profile ID, not a UI alias."
        ) from exc


def main() -> None:
    _validate_target_model(TARGET)
    records = _load_records(DATASET_URI)
    predict_fn = build_predict_fn(TARGET)
    scorer_objs = [build_scorer(n) for n in SCORERS]
    # R6: append custom scorer Lambdas, if any. Order is built-ins first
    # so MLflow result_df columns keep their familiar ordering.
    if CUSTOM_SCORER_LAMBDA_ARNS:
        print(f"[main] wiring {len(CUSTOM_SCORER_LAMBDA_ARNS)} custom scorer Lambda(s)")
        scorer_objs.extend(build_custom_scorers(CUSTOM_SCORER_LAMBDA_ARNS))

    def _predict_with_trace(**inputs: Any) -> str:
        """Per-row predict: prepend context to the prompt AND surface it as a
        RETRIEVER span so trace-based judges can ground against it (BUG-002)."""
        prompt = _build_prompt(inputs)
        context = inputs.get("context")
        if context:
            try:
                _emit_retriever_span(query=prompt, context=str(context))
            except Exception as exc:  # noqa: BLE001 — never fail the row over tracing
                print(f"[predict] retriever span emission failed: {exc}")
        return predict_fn(prompt)

    # QA BUG-002 (part 2): traces are created under the ACTIVE experiment,
    # which defaults to '0' — while our pre-created run lives in its own
    # experiment. The mismatch detaches every evaluation trace from the run
    # (observed: "extra traces … experiment 58 … not in the list ['0']" and
    # 403s on /v1/traces), which starves trace-based judges like
    # RetrievalGroundedness. Align the active experiment with the run's.
    run_experiment_id = mlflow.tracking.MlflowClient().get_run(RUN_ID).info.experiment_id
    mlflow.set_experiment(experiment_id=run_experiment_id)

    with mlflow.start_run(run_id=RUN_ID):
        result = evaluate(
            data=records,
            predict_fn=_predict_with_trace,
            scorers=scorer_objs,
        )
        os.makedirs("/opt/ml/processing/output", exist_ok=True)
        # Persist per-row results so generate_compliance_report can attach
        # the failing-rows appendix. MLflow 3.4's EvaluationResult exposes
        # `result_df` (a pandas DataFrame) rather than the `tables` dict the
        # older 2.x API used.
        result_df = getattr(result, "result_df", None)
        if result_df is None:
            # Back-compat for older MLflow releases that still expose `tables`.
            tables = getattr(result, "tables", None) or {}
            result_df = tables.get("eval_results_table")
        if result_df is not None:
            # MLflow 3.4's result_df includes nested struct columns (trace
            # tags, metadata) that pyarrow cannot serialize to Parquet
            # ("Cannot write struct type 'tags' with no child field").
            # Write JSON instead — lossless, human-readable, and the
            # compliance-documentation skill reads it the same way.
            out_path = "/opt/ml/processing/output/eval_results.json"
            result_df.to_json(out_path, orient="records", default_handler=str)
            mlflow.log_artifact(out_path)
        for name, value in (result.metrics or {}).items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value)


if __name__ == "__main__":
    main()
