#!/usr/bin/env python3
"""SageMaker Processing entrypoint — run Evidently drift + quality reports
against a completed xgboost/sklearn training job's baseline.

Env vars (injected by ``handler._background_submit_monitoring``):
    MLFLOW_TRACKING_URI, MLFLOW_RUN_ID
    BASELINE_S3_URI         — s3://.../baseline/baseline.csv
    CURRENT_DATA_S3_URI     — s3://.../baseline/eval_split.csv (or user-supplied)
    MODEL_ARTIFACT_S3_URI   — s3://.../output/model.tar.gz
    TARGET_COLUMN           — optional; enables ClassificationPreset when present
    AWS_REGION              — required for boto3 in Processing containers
"""
import json
import logging
import os
import tarfile
import tempfile
from urllib.parse import urlparse

import boto3
import mlflow
import pandas as pd
from botocore.config import Config


_BOTO_CONFIG = Config(retries={"max_attempts": 2, "mode": "standard"})

TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
RUN_ID = os.environ["MLFLOW_RUN_ID"]
BASELINE_URI = os.environ["BASELINE_S3_URI"]
CURRENT_URI = os.environ["CURRENT_DATA_S3_URI"]
MODEL_URI = os.environ["MODEL_ARTIFACT_S3_URI"]
TARGET_COLUMN = os.environ.get("TARGET_COLUMN", "")
AWS_REGION = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or boto3.session.Session().region_name
)
if not AWS_REGION:
    raise RuntimeError("AWS_REGION not set")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_aws_env_vars() -> None:
    """Freeze task-role creds into ``AWS_ACCESS_KEY_ID`` etc. so MLflow's
    SigV4 auth provider (which reads env vars directly) can sign requests.
    Paste-copy from ``lambda/skills/sagemaker/eval/entrypoint.py``.
    """
    creds = boto3.session.Session().get_credentials()
    if creds is None:
        raise RuntimeError("No AWS credentials resolvable inside Processing container")
    frozen = creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    os.environ["AWS_DEFAULT_REGION"] = AWS_REGION
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token


_ensure_aws_env_vars()
mlflow.set_tracking_uri(TRACKING_URI)


def _read_csv_from_s3(uri: str) -> pd.DataFrame:
    """Download a single CSV object and return it as a pandas DataFrame.

    boto3 writes the file itself, so allocate a bare temp *path* (mkstemp,
    fd closed immediately) rather than holding an open NamedTemporaryFile
    handle that download_file never writes through (semgrep
    tempfile-without-flush).
    """
    s3 = boto3.client("s3", region_name=AWS_REGION, config=_BOTO_CONFIG)
    parsed = urlparse(uri)
    fd, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    s3.download_file(parsed.netloc, parsed.path.lstrip("/"), csv_path)
    return pd.read_csv(csv_path)


def _download_model(uri: str) -> str:
    """Download a ``model.tar.gz`` artifact and extract it to a tempdir.

    Returns the absolute path to the ``.xgb`` model file found inside the
    tarball. Raises ``RuntimeError`` when no ``.xgb`` file is present —
    the monitoring container is tabular-only for v1, so a missing
    booster means the caller hit a non-xgboost training_type that
    slipped past the handler's pre-flight check.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION, config=_BOTO_CONFIG)
    parsed = urlparse(uri)
    tmp = tempfile.mkdtemp()
    tar_path = os.path.join(tmp, "model.tar.gz")
    s3.download_file(parsed.netloc, parsed.path.lstrip("/"), tar_path)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(tmp, filter="data")
    for root, _, files in os.walk(tmp):
        for f in files:
            if f.endswith(".xgb"):
                return os.path.join(root, f)
    raise RuntimeError(f"No .xgb model found in {uri}")


def _extract_metric(report_dict: dict, key: str):
    """Recursively search Evidently 0.4's ``as_dict()`` output for ``key``
    and return the first scalar value found. ``as_dict()`` shape is
    roughly ``{"metrics": [{"metric": "<Preset>", "result": {"<key>":
    <value>, ...}}, ...]}`` but nested preset outputs hide values
    deeper — defensive walk.
    """
    def _walk(node):
        if isinstance(node, dict):
            if key in node and isinstance(node[key], (int, float)):
                return node[key]
            for v in node.values():
                r = _walk(v)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = _walk(v)
                if r is not None:
                    return r
        return None

    return _walk(report_dict)


def main() -> None:
    """Load baseline + current frames, predict with the trained booster,
    run Evidently presets, log HTML + JSON + scalars to MLflow.
    """
    import xgboost as xgb  # noqa: PLC0415

    # Evidently 0.4.x imports — the public aws-samples pipeline uses this
    # major. Note:
    #   - Report lives in ``evidently.report``, NOT ``evidently``.
    #   - Presets live in ``evidently.metric_preset``. DataQualityPreset
    #     was renamed to DataSummaryPreset in 0.7+; we pin 0.4.40.
    #   - ``report.run(...)`` mutates the Report and returns None; we
    #     call ``save_html`` / ``as_dict`` on the instance itself.
    from evidently.report import Report  # noqa: PLC0415
    from evidently.metric_preset import (  # noqa: PLC0415
        DataDriftPreset,
        DataQualityPreset,
        ClassificationPreset,
    )
    from evidently.pipeline.column_mapping import ColumnMapping  # noqa: PLC0415

    baseline_df = _read_csv_from_s3(BASELINE_URI)
    current_df = _read_csv_from_s3(CURRENT_URI)
    model_path = _download_model(MODEL_URI)

    feature_cols = [c for c in baseline_df.columns if c != TARGET_COLUMN]
    booster = xgb.Booster()
    booster.load_model(model_path)

    preds = booster.predict(xgb.DMatrix(current_df[feature_cols]))
    if preds.ndim == 2:
        current_df["prediction"] = preds.argmax(axis=1)
    else:
        current_df["prediction"] = (preds >= 0.5).astype(int)

    # Evidently's ClassificationPreset needs the prediction column on the
    # reference frame too, otherwise the preset skips silently.
    if TARGET_COLUMN in baseline_df.columns:
        base_preds = booster.predict(xgb.DMatrix(baseline_df[feature_cols]))
        if base_preds.ndim == 2:
            baseline_df["prediction"] = base_preds.argmax(axis=1)
        else:
            baseline_df["prediction"] = (base_preds >= 0.5).astype(int)

    presets = [DataDriftPreset(), DataQualityPreset()]
    has_labels = bool(TARGET_COLUMN) and TARGET_COLUMN in current_df.columns
    column_mapping = ColumnMapping(
        target=TARGET_COLUMN if has_labels else None,
        prediction="prediction" if has_labels else None,
        numerical_features=[c for c in feature_cols if c != TARGET_COLUMN],
    )
    if has_labels:
        presets.append(ClassificationPreset())

    report = Report(metrics=presets)
    report.run(
        reference_data=baseline_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    out_dir = "/opt/ml/processing/output"
    os.makedirs(out_dir, exist_ok=True)
    html_path = f"{out_dir}/monitoring_report.html"
    json_path = f"{out_dir}/monitoring_report.json"
    report.save_html(html_path)
    report_dict = report.as_dict()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, default=str)

    with mlflow.start_run(run_id=RUN_ID):
        mlflow.log_artifact(html_path)
        mlflow.log_artifact(json_path)
        # Evidently 0.4 surfaces drift scalars under the
        # DataDriftPreset's result. Key names differ across minors;
        # try both, fall back to 0.0.
        drifted_share = (
            _extract_metric(report_dict, "drift_share")
            or _extract_metric(report_dict, "share_of_drifted_columns")
            or 0.0
        )
        drifted_count = _extract_metric(report_dict, "number_of_drifted_columns") or 0
        mlflow.log_metric("drifted_columns_share", float(drifted_share))
        mlflow.log_metric("drifted_columns_count", float(drifted_count))
        if has_labels:
            for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                v = _extract_metric(report_dict, metric_name)
                if isinstance(v, (int, float)):
                    mlflow.log_metric(metric_name, float(v))


if __name__ == "__main__":
    main()
