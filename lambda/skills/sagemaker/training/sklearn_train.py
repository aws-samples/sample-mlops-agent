#!/usr/bin/env python3
"""Sklearn training entrypoint — runs inside SageMaker PyTorch 2.4.0/py311 container."""
import importlib
import json
import logging
import os
import subprocess  # nosec B404
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_dependencies():
    # Static arg list (sys.executable + exact pip pins) — no user input
    # reaches this call.
    subprocess.check_call(  # nosemgrep: dangerous-subprocess-use-audit # nosec B603
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "joblib>=1.3.0",
            "mlflow==3.11.1",
            "sagemaker-mlflow==0.5.0",
        ]
    )


def _emit_baseline(*, X_train, y_train, X_eval, y_eval, target_column, model_dir, mlflow_module):
    """Mirror of xgb_train._emit_baseline — writes baseline + eval split
    CSVs plus stats JSON under ``<model_dir>/baseline/`` for the batch-
    monitoring Processing job to consume.

    R1: Task 2. See docs/plans/2026-04-28-batch-model-monitoring.md.
    """
    baseline_dir = os.path.join(model_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_df = X_train.copy()
    baseline_df[target_column] = y_train
    baseline_df.to_csv(os.path.join(baseline_dir, "baseline.csv"), index=False)
    eval_df = X_eval.copy()
    eval_df[target_column] = y_eval
    eval_df.to_csv(os.path.join(baseline_dir, "eval_split.csv"), index=False)
    stats = {
        "num_rows":      len(baseline_df),
        "num_features":  len(baseline_df.columns) - 1,
        "target_column": target_column,
        "columns": {
            c: {
                "dtype":   str(baseline_df[c].dtype),
                "nunique": int(baseline_df[c].nunique(dropna=True)),
            }
            for c in baseline_df.columns
        },
    }
    with open(os.path.join(baseline_dir, "baseline_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    if mlflow_module is not None:
        mlflow_module.log_artifacts(baseline_dir, artifact_path="baseline")


def main():
    logger.info("Installing training dependencies...")
    install_dependencies()

    import boto3
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    bucket = os.environ["CONFIG_S3_BUCKET"]
    key = os.environ["CONFIG_S3_KEY"]
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    s3 = boto3.client("s3")
    config_data = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    cfg = config_data["config"]

    dataset_name = cfg["data"]["dataset_name"]
    if dataset_name.startswith("s3://"):
        parts = dataset_name[5:].split("/", 1)
        bucket_ds, key_ds = parts[0], parts[1] if len(parts) > 1 else ""
        local_csv = "/tmp/dataset.csv"  # nosec B108
        s3.download_file(bucket_ds, key_ds, local_csv)
        dataset_name = local_csv
    df = pd.read_csv(dataset_name)
    target_column = cfg["data"]["target_column"]
    X, y = df.drop(columns=[target_column]), df[target_column]
    # R1 Task 2: keep the eval split so we can emit it alongside the baseline.
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    sk_cfg = cfg.get("sklearn", {})
    estimator_path = sk_cfg.get(
        "estimator", "sklearn.ensemble.RandomForestClassifier"
    )
    # The estimator path comes from the job config (agent-supplied). Restrict
    # the dynamic import to the sklearn package so the config cannot be used
    # to import and execute arbitrary modules inside the training container
    # (semgrep non-literal-import).
    if not estimator_path.startswith("sklearn."):
        raise ValueError(
            f"estimator must be an sklearn.* class path, got {estimator_path!r}"
        )
    module_path, class_name = estimator_path.rsplit(".", 1)
    EstimatorClass = getattr(importlib.import_module(module_path), class_name)  # nosemgrep: non-literal-import
    estimator = EstimatorClass(**sk_cfg.get("params", {}))

    pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(model_dir, "model.joblib"))

    # R1 Task 2: emit baseline + eval split for batch monitoring. This
    # path does NOT currently start an MLflow run (no _init_mlflow in
    # sklearn_train), so we pass mlflow_module=None. The files still
    # land on disk and SageMaker auto-uploads /opt/ml/model/baseline/
    # as a sibling of model.joblib; the callback Lambda derives URIs
    # from the model artifact path regardless of MLflow.
    _emit_baseline(
        X_train=X_train, y_train=y_train,
        X_eval=X_eval, y_eval=y_eval,
        target_column=target_column,
        model_dir=model_dir,
        mlflow_module=None,
    )
    logger.info("Sklearn training complete. Model saved to %s", model_dir)


if __name__ == "__main__":
    main()
