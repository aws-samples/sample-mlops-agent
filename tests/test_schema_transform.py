"""R3 — schema transformation tests.

Covers both layers:
  - The HF-skill handler validation (handler-side refusal on unsupported
    target_schema pairs, before any S3 write).
  - The shared transformation functions that run inside the eval
    Processing container.
"""
from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch


# ── Helpers (mirrors tests/test_lambda_handler_dispatch.py:_ensure_stub) ──
def _ensure_stub(name: str) -> None:
    """Stub `name` only when the real module is not installed — a fake left
    in sys.modules shadows the genuine package for every later test module
    in the pytest session."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass
    mod = ModuleType(name)
    mod.__getattr__ = lambda attr, _m=mod: MagicMock(name=f"{name}.{attr}")  # type: ignore[attr-defined]
    sys.modules[name] = mod


for _m in ("mlflow", "mlflow.tracking", "mlflow.exceptions",
           "huggingface_hub", "requests"):
    _ensure_stub(_m)
# Set attributes only on OUR stub (no __spec__); never mutate a real package.
if getattr(sys.modules["huggingface_hub"], "__spec__", None) is None:
    sys.modules["huggingface_hub"].HfApi = MagicMock(name="HfApi")
    sys.modules["huggingface_hub"].snapshot_download = MagicMock(name="snapshot_download")


def _import_handler(skill: str):
    """Re-import a skill's handler.py from scratch."""
    lam_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "lambda", "skills", skill)
    )
    for p in list(sys.path):
        if "/lambda/skills/" in p:
            sys.path.remove(p)
    sys.path.insert(0, lam_dir)
    sys.modules.pop("handler", None)
    return importlib.import_module("handler")


# ── Handler-side validation ───────────────────────────────────────────────


def test_prepare_eval_dataset_no_target_schema_preserves_legacy_behaviour():
    """When target_schema is omitted, spec carries null transform fields
    and the handler does not raise."""
    h = _import_handler("huggingface")
    h.SESSION_BUCKET = "bucket"
    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.return_value = MagicMock()
        out = h._prepare_eval_dataset({
            "task_type": "question_answering",
            "dataset_name": "PatronusAI/financebench",
            "split": "train",
        })
    assert out["target_schema"] is None
    assert out["transform_mechanism"] is None


def test_prepare_eval_dataset_supported_pair_sets_mechanism():
    """chat → sft is supported and resolves to flatten_messages up-front."""
    h = _import_handler("huggingface")
    h.SESSION_BUCKET = "bucket"
    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.return_value = MagicMock()
        out = h._prepare_eval_dataset({
            "task_type": "text_generation",
            "dataset_name": "HuggingFaceH4/ultrachat_200k",
            "split": "test_sft",
            "source_schema": "chat",
            "target_schema": "sft",
        })
    assert out["source_schema"] == "chat"
    assert out["target_schema"] == "sft"
    assert out["transform_mechanism"] == "flatten_messages"


def test_prepare_eval_dataset_unsupported_pair_raises_before_write():
    """chat → dpo is not in the matrix; handler must raise and not call S3."""
    h = _import_handler("huggingface")
    h.SESSION_BUCKET = "bucket"
    mock_s3 = MagicMock()
    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_s3
        import pytest  # noqa: PLC0415
        with pytest.raises(ValueError) as exc:
            h._prepare_eval_dataset({
                "task_type": "text_generation",
                "dataset_name": "some/dataset",
                "source_schema": "chat",
                "target_schema": "dpo",
            })
        assert "No transformation mechanism" in str(exc.value)
    # The S3 put_object must not have happened — we fail loudly.
    mock_s3.put_object.assert_not_called()


def test_prepare_eval_dataset_identity_when_same_schema():
    """source_schema == target_schema resolves to identity."""
    h = _import_handler("huggingface")
    h.SESSION_BUCKET = "bucket"
    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.return_value = MagicMock()
        out = h._prepare_eval_dataset({
            "task_type": "text_generation",
            "dataset_name": "ds",
            "source_schema": "sft",
            "target_schema": "sft",
        })
    assert out["transform_mechanism"] == "identity"


def test_prepare_eval_dataset_invalid_schema_value_rejected():
    """target_schema must be one of the documented enum values."""
    h = _import_handler("huggingface")
    h.SESSION_BUCKET = "bucket"
    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.return_value = MagicMock()
        import pytest  # noqa: PLC0415
        with pytest.raises(ValueError) as exc:
            h._prepare_eval_dataset({
                "task_type": "text_generation",
                "dataset_name": "ds",
                "target_schema": "parquet",   # not in enum
            })
        assert "not one of" in str(exc.value)


def test_prepare_eval_dataset_defers_inference_to_container():
    """When source_schema is omitted but target_schema is set, the
    handler records target_schema in the spec with mechanism=None so
    the container resolves it at materialisation time."""
    h = _import_handler("huggingface")
    h.SESSION_BUCKET = "bucket"
    with patch.object(h, "boto3") as mock_boto:
        mock_boto.client.return_value = MagicMock()
        out = h._prepare_eval_dataset({
            "task_type": "text_generation",
            "dataset_name": "ds",
            "target_schema": "sft",
        })
    assert out["target_schema"] == "sft"
    assert out["source_schema"] is None
    assert out["transform_mechanism"] is None


# ── Container-side transforms (schema_transform.py) ───────────────────────


def _import_schema_transform():
    """Load the container-local schema_transform module directly."""
    import importlib.util  # noqa: PLC0415
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..",
                     "lambda", "skills", "sagemaker", "eval", "schema_transform.py")
    )
    spec = importlib.util.spec_from_file_location("schema_transform", path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_schema_transform_infer_from_columns():
    st = _import_schema_transform()
    assert st.infer_source_schema(["messages", "id"]) == "chat"
    assert st.infer_source_schema(["prompt", "chosen", "rejected"]) == "dpo"
    assert st.infer_source_schema(["text"]) == "sft"
    assert st.infer_source_schema(["feature_a", "feature_b", "target"]) == "tabular-csv"
    assert st.infer_source_schema(["unknown_one_col"]) == "unknown"


def test_schema_transform_resolve_mechanism_matrix():
    st = _import_schema_transform()
    assert st.resolve_mechanism("chat", "sft") == "flatten_messages"
    assert st.resolve_mechanism("sft", "chat") == "unflatten_to_messages"
    assert st.resolve_mechanism("dpo", "sft") == "drop_rejected"
    assert st.resolve_mechanism("sft", "sft") == "identity"

    import pytest  # noqa: PLC0415
    with pytest.raises(st.UnsupportedTransformError):
        st.resolve_mechanism("chat", "dpo")
    with pytest.raises(st.UnsupportedTransformError):
        st.resolve_mechanism("tabular-csv", "sft")


def test_flatten_messages_produces_text_with_markers():
    st = _import_schema_transform()
    rows = [{"messages": [
        {"role": "user",      "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]}]
    out = st.apply_transform(rows, mechanism="flatten_messages")
    assert "text" in out[0]
    txt = out[0]["text"]
    assert "<|user|>" in txt and "hi" in txt
    assert "<|assistant|>" in txt and "hello" in txt
    assert "<|end|>" in txt


def test_unflatten_to_messages_round_trips():
    st = _import_schema_transform()
    src = [{"messages": [
        {"role": "user",      "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]}]
    as_sft = st.apply_transform(src, mechanism="flatten_messages")
    back = st.apply_transform(as_sft, mechanism="unflatten_to_messages")
    assert back[0]["messages"][0]["role"] == "user"
    assert back[0]["messages"][0]["content"] == "hi"
    assert back[0]["messages"][1]["role"] == "assistant"
    assert back[0]["messages"][1]["content"] == "hello"


def test_unflatten_messages_fallback_on_marker_free_text():
    st = _import_schema_transform()
    out = st.apply_transform(
        [{"text": "raw text with no markers"}],
        mechanism="unflatten_to_messages",
    )
    # Fallback: whole text becomes a single user message.
    assert out[0]["messages"] == [{"role": "user", "content": "raw text with no markers"}]


def test_drop_rejected_keeps_prompt_and_chosen():
    st = _import_schema_transform()
    out = st.apply_transform(
        [{"prompt": "Q?", "chosen": "A.", "rejected": "WRONG"}],
        mechanism="drop_rejected",
    )
    txt = out[0]["text"]
    assert "Q?" in txt and "A." in txt
    assert "WRONG" not in txt
    assert "rejected" not in out[0]


def test_identity_is_noop():
    st = _import_schema_transform()
    rows = [{"text": "unchanged"}]
    out = st.apply_transform(rows, mechanism="identity")
    assert out == rows


def test_apply_transform_rejects_unknown_mechanism():
    st = _import_schema_transform()
    import pytest  # noqa: PLC0415
    with pytest.raises(ValueError) as exc:
        st.apply_transform([{"text": "x"}], mechanism="not_a_mechanism")
    assert "not_a_mechanism" in str(exc.value)
