#!/usr/bin/env python3
"""XGBoost training entrypoint — runs inside SageMaker PyTorch 2.4.0/py311 container.

Dataset dispatch supports three shapes, chosen from `cfg["data"]["dataset_name"]`:

  1. ``s3://bucket/key.csv`` / ``s3://bucket/prefix/`` — downloaded and
     loaded via pandas. If the URI is a prefix we concatenate every CSV
     shard underneath it.
  2. ``sklearn:<bundled_name>`` OR a bare bundled name from
     ``_SKLEARN_BUNDLED`` (iris, breast_cancer, wine, digits) — loaded
     via ``sklearn.datasets.load_<name>`` in-process. No S3 round-trip.
  3. ``<hf-org>/<hf-repo>`` (e.g. ``scikit-learn/iris``) — fetched via
     HuggingFace ``datasets`` and coerced to pandas via ``.to_pandas()``.
     Requires the dataset to expose a tabular schema; non-tabular HF
     datasets raise a clear error instead of silently dying inside xgboost.

All three paths produce a single ``pandas.DataFrame`` that the rest of the
script then trains on. MLflow instrumentation mirrors ``sft_train.py``:
the orchestrator pre-creates a RUNNING run and injects ``MLFLOW_RUN_ID``;
we resume into it, stream per-round metrics via an XGBoost training
callback, and log the final model artifact with ``mlflow.xgboost.log_model``.
"""
import json
import logging
import os
import subprocess  # nosec B404
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_dependencies():
    """Install pinned deps. Bump ``sagemaker-mlflow`` to 0.3.0 to match the
    rest of the stack after the eval-container pin bump."""
    # Static arg list (sys.executable + exact pip pins) — no user input
    # reaches this call.
    subprocess.check_call(  # nosemgrep: dangerous-subprocess-use-audit # nosec B603
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "xgboost==2.1.3",
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "mlflow==3.11.1",
            "sagemaker-mlflow==0.5.0",
            # datasets is only needed for the HF-repo fallback, but the
            # pip download is ~40 MB so install unconditionally — negligible
            # compared to the SageMaker bootstrap time.
            "datasets>=3.1.0",
        ]
    )


_SKLEARN_BUNDLED = {
    # name -> (loader_fn_name, default_target_column)
    "iris":          ("load_iris",          "target"),
    "breast_cancer": ("load_breast_cancer", "target"),
    "wine":          ("load_wine",          "target"),
    "digits":        ("load_digits",        "target"),
}


def _init_mlflow():
    """Initialise MLflow tracking and return the active run context manager.

    Mirrors sft_train.py:_init_mlflow — we own start_run/end_run so the
    context manager closes the run FAILED on crash rather than leaving it
    stuck in RUNNING. Returns None if MLFLOW_TRACKING_URI is not set so
    the script still works in a standalone-container test.
    """
    import mlflow  # noqa: PLC0415

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    run_id = os.environ.get("MLFLOW_RUN_ID", "")
    if not tracking_uri:
        logger.warning("MLFLOW_TRACKING_URI not set — MLflow logging disabled")
        return None
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s  run_id: %s", tracking_uri, run_id or "<none>")
    return mlflow.start_run(run_id=run_id or None)


def _load_dataset_to_pandas(dataset_name: str, target_column: str | None) -> tuple:
    """Resolve `dataset_name` into a (DataFrame, target_column) pair.

    Raises RuntimeError with an actionable message when the dataset can't
    be loaded — the previous behavior was a bare ``FileNotFoundError`` that
    bubbled up with no context.
    """
    import pandas as pd  # noqa: PLC0415
    import boto3         # noqa: PLC0415

    # 1) Bundled sklearn datasets — fastest path, no network I/O.
    lookup_key = dataset_name.removeprefix("sklearn:").lower()
    if lookup_key in _SKLEARN_BUNDLED:
        loader_name, default_target = _SKLEARN_BUNDLED[lookup_key]
        import sklearn.datasets as skd  # noqa: PLC0415
        ds = getattr(skd, loader_name)(as_frame=True)
        df = ds.frame  # as_frame=True gives us a ready DataFrame with 'target' col
        return df, target_column or default_target

    # 2) S3 object or prefix.
    if dataset_name.startswith("s3://"):
        s3 = boto3.client("s3")
        bucket, _, key = dataset_name[5:].partition("/")
        if dataset_name.endswith(".csv"):
            local_csv = "/tmp/dataset.csv"  # nosec B108
            s3.download_file(bucket, key, local_csv)
            df = pd.read_csv(local_csv)
        else:
            # Prefix — concatenate every *.csv shard underneath.
            listing = s3.list_objects_v2(Bucket=bucket, Prefix=key).get("Contents", [])
            csv_keys = [o["Key"] for o in listing if o["Key"].endswith(".csv")]
            if not csv_keys:
                raise RuntimeError(
                    f"No .csv objects under {dataset_name!r}. XGBoost/sklearn "
                    f"training requires tabular CSV data."
                )
            frames = []
            for k in csv_keys:
                local = f"/tmp/{os.path.basename(k)}"  # nosec B108
                s3.download_file(bucket, k, local)
                frames.append(pd.read_csv(local))
            df = pd.concat(frames, ignore_index=True)
        if not target_column:
            raise RuntimeError(
                f"`target_column` is required when loading from {dataset_name!r} — "
                "the script cannot infer which column is the label."
            )
        return df, target_column

    # 3) HuggingFace repo fallback — org/repo form (e.g. scikit-learn/iris).
    if "/" in dataset_name:
        from datasets import load_dataset  # noqa: PLC0415
        # Pin the Hub revision (default "main") so a job can lock to an immutable
        # commit SHA — an unpinned pull would silently follow upstream repo changes.
        hf_revision = os.environ.get("HF_HUB_REVISION", "main")
        try:
            ds = load_dataset(dataset_name, split="train", revision=hf_revision)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {dataset_name!r} via HuggingFace datasets: "
                f"{type(exc).__name__}: {exc}. If this is a tabular dataset, "
                f"consider hosting it as `s3://<bucket>/<key>.csv` instead; "
                f"if it's a bundled sklearn dataset, pass "
                f"`dataset_name=iris` (or one of {list(_SKLEARN_BUNDLED)})."
            ) from exc
        df = ds.to_pandas()
        if not target_column:
            raise RuntimeError(
                f"`target_column` is required when loading from HuggingFace repo "
                f"{dataset_name!r}."
            )
        return df, target_column

    raise RuntimeError(
        f"Unrecognised dataset_name={dataset_name!r}. Supported forms: "
        f"s3://…/*.csv, s3://…/prefix/, HF repo id (org/repo), or one of "
        f"the bundled sklearn names {list(_SKLEARN_BUNDLED)}."
    )


def _make_mlflow_callback(mlflow_module):
    """Build an XGBoost TrainingCallback that mirrors each eval-set metric
    into MLflow per boosting round.

    We subclass ``xgb.callback.TrainingCallback`` rather than use
    ``after_iteration=...`` hooks so we're compatible with xgb >= 1.7 while
    remaining a simple pure-Python object (no extra deps).
    """
    import xgboost as xgb  # noqa: PLC0415

    class _MLflowCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            for eval_name, metric_map in (evals_log or {}).items():
                for metric_name, values in metric_map.items():
                    if not values:
                        continue
                    # values is a list of per-round measurements; the last
                    # entry is this round's number.
                    value = values[-1]
                    if isinstance(value, (list, tuple)):
                        value = value[0]
                    key = f"{eval_name}-{metric_name}"
                    try:
                        mlflow_module.log_metric(key, float(value), step=epoch)
                    except Exception as exc:
                        logger.warning("mlflow.log_metric(%s) failed: %s", key, exc)
            return False  # never short-circuit training

    return _MLflowCallback()


class _nullcontext:
    """contextlib.nullcontext backport for pre-3.10 parity with sft_train."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


def _emit_baseline(*, X_train, y_train, X_eval, y_eval, target_column, model_dir, mlflow_module):
    """Persist baseline + eval split CSVs plus a small stats JSON under
    ``<model_dir>/baseline/`` so SageMaker auto-uploads them alongside
    ``model.xgb``. The batch-monitoring Lambda reads these URIs from the
    DDB thread row (stamped by ``lambda/callback/handler.py`` on
    Completed) and the Processing container feeds them to Evidently.

    R1: Task 1. See docs/plans/2026-04-28-batch-model-monitoring.md.
    """
    import json  # noqa: PLC0415

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

    import mlflow            # noqa: PLC0415
    import mlflow.xgboost    # noqa: PLC0415 — triggers the XGBoost flavor registration
    import xgboost as xgb    # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    bucket = os.environ["CONFIG_S3_BUCKET"]
    key = os.environ["CONFIG_S3_KEY"]
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    import boto3
    s3 = boto3.client("s3")
    config_data = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    cfg = config_data["config"]

    dataset_name = cfg["data"]["dataset_name"]
    target_column = cfg["data"].get("target_column")
    df, target_column = _load_dataset_to_pandas(dataset_name, target_column)
    logger.info("Loaded dataset %s: %d rows, target=%r, columns=%s",
                dataset_name, len(df), target_column, list(df.columns))

    X, y = df.drop(columns=[target_column]), df[target_column]
    # Stratify only for classification targets (non-float dtype). If the
    # label is continuous-looking, skip stratify so regression tasks work.
    stratify = y if y.dtype != float else None
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify,
    )

    # Pick a sane default objective based on number of classes so `iris`
    # (3 classes) doesn't try to train as binary. Caller overrides via
    # cfg["xgb"]["objective"].
    xgb_params = cfg.get("xgb", {})
    n_classes = int(y.nunique())
    if "objective" in xgb_params:
        objective = xgb_params["objective"]
    elif n_classes == 2:
        objective = "binary:logistic"
    elif n_classes > 2:
        objective = "multi:softprob"
    else:  # continuous
        objective = "reg:squarederror"

    params = {
        "max_depth":   int(xgb_params.get("max_depth", 6)),
        "eta":         float(xgb_params.get("eta", 0.3)),
        "objective":   objective,
        "eval_metric": xgb_params.get(
            "eval_metric",
            "logloss" if objective.startswith("binary")
            else ("mlogloss" if objective.startswith("multi") else "rmse"),
        ),
    }
    if objective.startswith("multi"):
        params["num_class"] = n_classes

    num_round = int(xgb_params.get("num_round", 100))
    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    run_ctx = _init_mlflow()
    callbacks = []
    if run_ctx is not None:
        callbacks.append(_make_mlflow_callback(mlflow))

    with (run_ctx if run_ctx is not None else _nullcontext()):
        if run_ctx is not None:
            # Log resolved hyperparams so they're discoverable in the UI.
            mlflow.log_params({
                "dataset_name":  dataset_name,
                "target_column": target_column,
                "num_rows":      len(df),
                "num_features":  X.shape[1],
                "num_classes":   n_classes,
                "num_round":     num_round,
                **{f"xgb.{k}": v for k, v in params.items()},
            })
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_round,
            evals=[(dtrain, "train"), (deval, "eval")],
            verbose_eval=10,
            callbacks=callbacks,
        )
        # Save the Booster to SM_MODEL_DIR so SageMaker uploads it as the
        # output artifact (same convention as sft_train.py).
        os.makedirs(model_dir, exist_ok=True)
        model.save_model(os.path.join(model_dir, "model.xgb"))
        # R1 Task 1: emit baseline + eval split alongside the model so the
        # batch-monitoring Processing job has something to compare against.
        _emit_baseline(
            X_train=X_train, y_train=y_train,
            X_eval=X_eval, y_eval=y_eval,
            target_column=target_column,
            model_dir=model_dir,
            mlflow_module=mlflow if run_ctx is not None else None,
        )
        if run_ctx is not None:
            # Log the booster itself as an MLflow model so it's deployable
            # via mlflow.xgboost.load_model.
            try:
                mlflow.xgboost.log_model(xgb_model=model, name="model")
            except Exception as exc:
                # Older MLflow releases use artifact_path= instead of name=.
                logger.warning("log_model(name=...) failed (%s); retrying with artifact_path=", exc)
                mlflow.xgboost.log_model(xgb_model=model, artifact_path="model")

    logger.info("XGBoost training complete. Model saved to %s", model_dir)


if __name__ == "__main__":
    main()
