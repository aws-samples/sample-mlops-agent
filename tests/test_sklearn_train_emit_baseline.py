"""R1 Task 2 — sklearn_train._emit_baseline helper unit tests.

Mirrors the xgb_train test; the two helpers are structurally identical
but live in separate training scripts to keep each container self-
contained.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_sklearn_train():
    path = (Path(__file__).resolve().parents[1]
            / "lambda/skills/sagemaker/training/sklearn_train.py")
    spec = importlib.util.spec_from_file_location("sklearn_train", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["sklearn_train"] = module
    spec.loader.exec_module(module)
    return module


def test_emit_baseline_writes_both_csvs(tmp_path):
    sklearn_train = _load_sklearn_train()

    X_train = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    y_train = pd.Series([0, 1, 0], name="target")
    X_eval = pd.DataFrame({"f1": [7, 8], "f2": [9, 10]})
    y_eval = pd.Series([1, 0], name="target")

    sklearn_train._emit_baseline(
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
