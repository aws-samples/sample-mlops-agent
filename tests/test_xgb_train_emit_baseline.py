"""R1 Task 1 — xgb_train._emit_baseline helper unit tests.

Exercises the helper with synthetic frames; does NOT run SageMaker or
XGBoost. Guards against regressions in the disk-write shape and the
MLflow log_artifacts hook.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd


def _load_xgb_train():
    """Load lambda/skills/sagemaker/training/xgb_train.py directly.

    Uses importlib rather than sys.path injection because
    test_lambda_handler_dispatch.py's ``_import_handler`` helper purges
    ``/lambda/skills/`` entries from sys.path when it runs before this
    file; a stale path insert would silently import the wrong module.
    """
    path = (Path(__file__).resolve().parents[1]
            / "lambda/skills/sagemaker/training/xgb_train.py")
    spec = importlib.util.spec_from_file_location("xgb_train", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["xgb_train"] = module
    spec.loader.exec_module(module)
    return module


def test_emit_baseline_writes_both_csvs_and_stats(tmp_path):
    xgb_train = _load_xgb_train()

    X_train = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    y_train = pd.Series([0, 1, 0], name="target")
    X_eval = pd.DataFrame({"f1": [7, 8], "f2": [9, 10]})
    y_eval = pd.Series([1, 0], name="target")

    xgb_train._emit_baseline(
        X_train=X_train, y_train=y_train,
        X_eval=X_eval, y_eval=y_eval,
        target_column="target", model_dir=str(tmp_path), mlflow_module=None,
    )

    base = tmp_path / "baseline"
    assert (base / "baseline.csv").exists()
    assert (base / "eval_split.csv").exists()
    assert (base / "baseline_stats.json").exists()

    baseline_df = pd.read_csv(base / "baseline.csv")
    eval_df = pd.read_csv(base / "eval_split.csv")
    assert list(baseline_df.columns) == ["f1", "f2", "target"]
    assert list(eval_df.columns) == ["f1", "f2", "target"]
    assert len(baseline_df) == 3 and len(eval_df) == 2

    import json  # noqa: PLC0415
    stats = json.loads((base / "baseline_stats.json").read_text())
    assert stats["target_column"] == "target"
    assert stats["num_rows"] == 3
    assert stats["num_features"] == 2
    assert "f1" in stats["columns"] and "target" in stats["columns"]


def test_emit_baseline_logs_to_mlflow_when_module_provided(tmp_path):
    """Regression guard: when the caller passes a mlflow module, the
    baseline directory must be logged as artifacts under
    ``artifact_path="baseline"``."""
    xgb_train = _load_xgb_train()

    mlflow_mock = MagicMock()
    xgb_train._emit_baseline(
        X_train=pd.DataFrame({"f1": [1]}), y_train=pd.Series([0], name="target"),
        X_eval=pd.DataFrame({"f1": [2]}), y_eval=pd.Series([1], name="target"),
        target_column="target", model_dir=str(tmp_path), mlflow_module=mlflow_mock,
    )
    mlflow_mock.log_artifacts.assert_called_once()
    call = mlflow_mock.log_artifacts.call_args
    # Accept either artifact_path=kwarg or positional second arg.
    assert call.kwargs.get("artifact_path") == "baseline" or (
        len(call.args) > 1 and call.args[1] == "baseline"
    )


def test_emit_baseline_no_mlflow_log_when_module_none(tmp_path):
    """When mlflow_module=None (MLFLOW_TRACKING_URI not set), the helper
    must still write files but MUST NOT attempt any MLflow call."""
    xgb_train = _load_xgb_train()

    # Nothing to assert about a non-call — the test passes if no exception
    # is raised and the files land on disk.
    xgb_train._emit_baseline(
        X_train=pd.DataFrame({"f1": [1]}), y_train=pd.Series([0], name="target"),
        X_eval=pd.DataFrame({"f1": [2]}), y_eval=pd.Series([1], name="target"),
        target_column="target", model_dir=str(tmp_path), mlflow_module=None,
    )
    assert (tmp_path / "baseline" / "baseline.csv").exists()
