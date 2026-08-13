#!/usr/bin/env python3
"""DPO training entrypoint — runs inside SageMaker PyTorch 2.4.0/py311 container."""
import json
import logging
import os
import subprocess  # nosec B404
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_dependencies():
    # EXACT pins — see sft_train.py for the incident this set fixes.
    # Static arg list (sys.executable + exact pip pins) — no user input
    # reaches this call.
    subprocess.check_call(  # nosemgrep: dangerous-subprocess-use-audit # nosec B603
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "trl==0.21.0",
            "accelerate==1.7.0",
            "transformers==4.56.0",
            "datasets==3.6.0",
            "peft==0.17.0",
            "mlflow==3.11.1",
            "sagemaker-mlflow==0.5.0",
        ]
    )


def _init_mlflow():
    """Initialise MLflow tracking and return an active run context manager.

    Strategy (mlflow>=3.0 + sagemaker-mlflow==0.5.0):
    - We own start_run / end_run explicitly via the context manager returned here.
    - Per-step metrics are logged by _MlflowMetricsCallback (report_to=["none"] avoids
      MLflowCallback.setup() which unconditionally calls start_run(), crashing with
      "Run already active" when a run is already open).
    - The `with mlflow.start_run(...)` context manager guarantees end_run() is
      called even if training crashes, marking the MLflow run as FAILED rather
      than leaving it stuck in RUNNING state.

    Returns None if MLFLOW_TRACKING_URI is not set (MLflow disabled).
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


def main():
    logger.info("Installing training dependencies...")
    install_dependencies()

    import boto3
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    bucket = os.environ["CONFIG_S3_BUCKET"]
    key = os.environ["CONFIG_S3_KEY"]
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    s3 = boto3.client("s3")
    config_data = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    cfg = config_data["config"]

    # Pin the Hub revision (default "main") so a job can lock to an immutable
    # commit SHA — an unpinned pull would silently follow upstream repo changes.
    hf_revision = os.environ.get("HF_HUB_REVISION", "main")

    # DPO requires dataset with "prompt", "chosen", "rejected" columns
    train_ds = load_dataset(
        cfg["data"]["dataset_name"],
        split=cfg["data"].get("train_split", "train"),
        revision=cfg["data"].get("revision", hf_revision),
    )

    max_samples = int(cfg["data"].get("max_samples", 0))
    if max_samples > 0:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
        logger.info("Dataset capped to %d samples (max_samples=%d)", len(train_ds), max_samples)

    run_ctx = _init_mlflow()

    dpo_cfg = DPOConfig(
        output_dir=model_dir,
        max_steps=int(cfg["training"].get("max_steps", 500)),
        learning_rate=float(cfg["training"].get("learning_rate", 5e-7)),
        per_device_train_batch_size=int(cfg["training"].get("batch_size", 2)),
        bf16=bool(cfg["training"].get("bf16", True)),
        beta=float(cfg.get("dpo", {}).get("beta", 0.1)),
        max_length=int(cfg.get("dpo", {}).get("max_length", 1024)),
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        # Per-step metrics are handled by _MlflowMetricsCallback passed to the trainer.
        # report_to=["mlflow"] is avoided because MLflowCallback.setup() unconditionally
        # calls mlflow.start_run(), crashing when a run is already active.
        report_to=["none"],
    )

    with (run_ctx if run_ctx is not None else _nullcontext()):
        trainer = DPOTrainer(
            model=cfg["model"]["name"],
            args=dpo_cfg,
            train_dataset=train_ds,
            callbacks=[_MlflowMetricsCallback()] if run_ctx is not None else [],
        )
        trainer.train()
        if run_ctx is not None:
            import mlflow  # noqa: PLC0415
            # Log the final summary row (train_loss, runtime, samples/sec, steps/sec).
            final = {
                k: v
                for k, v in (trainer.state.log_history[-1] if trainer.state.log_history else {}).items()
                if isinstance(v, (int, float))
            }
            if final:
                mlflow.log_metrics(final)
            # Archive the full step-by-step history so the loss curve is
            # reproducible even after the CloudWatch log stream expires.
            mlflow.log_text(json.dumps(trainer.state.log_history, indent=2), "trainer_log_history.json")

    logger.info("DPO complete. Model saved to %s", model_dir)


class _MlflowMetricsCallback:
    """Logs Trainer per-step metrics into the already-active MLflow run.

    MLflowCallback (report_to=["mlflow"]) unconditionally calls mlflow.start_run()
    in its setup() method, which raises an exception when a run is already active.
    This lightweight callback bypasses that by calling mlflow.log_metrics() directly,
    leaving run lifecycle entirely to the _init_mlflow() context manager.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called by the Trainer on every logging_steps interval."""
        if not logs:
            return
        import mlflow  # noqa: PLC0415
        if mlflow.active_run() is None:
            return
        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if metrics:
            mlflow.log_metrics(metrics, step=state.global_step)

    def __getattr__(self, name):
        """Return a no-op for any TrainerCallback lifecycle method not explicitly defined."""
        def _noop(*args, **kwargs):
            pass
        return _noop


class _nullcontext:
    """Backport of contextlib.nullcontext for Python <3.10 compatibility."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


if __name__ == "__main__":
    main()
