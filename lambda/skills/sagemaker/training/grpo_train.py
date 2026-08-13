#!/usr/bin/env python3
"""Training entrypoint — runs inside SageMaker managed PyTorch 2.4.0/py311 container.

MLflow: HuggingFace MLflowCallback resumes the pre-created run via MLFLOW_RUN_ID env var.
Logs train_loss and other Trainer metrics every logging_steps. Ends run on_train_end.
"""
import json
import logging
import os
import pathlib
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

    import functools
    import boto3
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    bucket = os.environ["CONFIG_S3_BUCKET"]
    key = os.environ["CONFIG_S3_KEY"]
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    s3 = boto3.client("s3")
    config_data = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    cfg = config_data["config"]
    reward_code = config_data.get("reward_function", "")

    dataset_name = cfg["data"]["dataset_name"]
    train_split = cfg["data"].get("train_split", "train")
    test_split = cfg["data"].get("test_split", "test")
    # Pin the Hub revision (default "main") so a job can lock to an immutable
    # commit SHA — an unpinned pull would silently follow upstream repo changes.
    hf_revision = cfg["data"].get("revision", os.environ.get("HF_HUB_REVISION", "main"))

    # Support pre-staged S3 datasets (s3:// prefix) as well as HuggingFace Hub names
    if dataset_name.startswith("s3://"):
        from datasets import load_from_disk

        parts = dataset_name[5:].split("/", 1)
        bucket_ds, prefix_ds = parts[0], parts[1] if len(parts) > 1 else ""
        local_ds_dir = "/tmp/dataset"  # nosec B108
        s3_dl = boto3.client("s3")
        paginator = s3_dl.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_ds, Prefix=prefix_ds + "/"):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix_ds) + 1 :]
                if not rel:
                    continue
                dest = pathlib.Path(local_ds_dir) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3_dl.download_file(bucket_ds, obj["Key"], str(dest))
        full_ds = load_from_disk(local_ds_dir)
        train_ds = full_ds[train_split] if train_split in full_ds else full_ds
        eval_ds = full_ds[test_split] if test_split in full_ds else None
        max_samples = int(cfg["data"].get("max_samples", 0))
        if max_samples > 0:
            train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
            logger.info("Train dataset capped to %d samples (max_samples=%d)", len(train_ds), max_samples)
            if eval_ds is not None:
                eval_cap = max(1, max_samples // 10)
                eval_ds = eval_ds.select(range(min(eval_cap, len(eval_ds))))
                logger.info("Eval dataset capped to %d samples", len(eval_ds))
    else:
        train_ds = load_dataset(dataset_name, split=train_split, revision=hf_revision)
        try:
            eval_ds = load_dataset(dataset_name, split=test_split, revision=hf_revision)
        except Exception:
            logger.warning("Could not load eval split '%s' — skipping eval", test_split)
            eval_ds = None
        max_samples = int(cfg["data"].get("max_samples", 0))
        if max_samples > 0:
            train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
            logger.info("Train dataset capped to %d samples (max_samples=%d)", len(train_ds), max_samples)
            if eval_ds is not None:
                eval_cap = max(1, max_samples // 10)
                eval_ds = eval_ds.select(range(min(eval_cap, len(eval_ds))))
                logger.info("Eval dataset capped to %d samples", len(eval_ds))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"],
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        revision=cfg["model"].get("revision", hf_revision),
    )

    def _disable_thinking(tok):
        orig = tok.apply_chat_template
        @functools.wraps(orig)
        def patched(*a, **kw):
            kw.setdefault("enable_thinking", False)
            return orig(*a, **kw)
        tok.apply_chat_template = patched
        return tok

    tokenizer = _disable_thinking(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if reward_code:
        from reward_functions import get_reward_fn  # noqa: PLC0415
        reward_fn = get_reward_fn(reward_code, cfg, tokenizer)
    else:
        target = cfg["reward"]["target_length"]

        def reward_fn(completions, **kw):
            return [
                -(len(tokenizer.encode(c, add_special_tokens=False)) - target) ** 2 / 1000
                for c in completions
            ]

    run_ctx = _init_mlflow()

    grpo_cfg = GRPOConfig(
        output_dir=model_dir,
        max_steps=int(cfg["training"].get("max_steps", 500)),
        learning_rate=float(cfg["training"].get("learning_rate", 1e-5)),
        per_device_train_batch_size=1,
        num_generations=int(cfg.get("grpo", {}).get("num_generations", 4)),
        max_completion_length=int(cfg.get("grpo", {}).get("max_completion_length", 256)),
        fp16=bool(cfg["training"].get("fp16", True)),
        gradient_checkpointing=True,
        gradient_accumulation_steps=int(cfg["training"].get("gradient_accumulation_steps", 8)),
        logging_steps=10,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=50,
        do_eval=eval_ds is not None,
        save_strategy="steps",
        save_steps=100,
        dataloader_num_workers=0,
        # Per-step metrics are handled by _MlflowMetricsCallback passed to the trainer.
        # report_to=["mlflow"] is avoided because MLflowCallback.setup() unconditionally
        # calls mlflow.start_run(), crashing when a run is already active.
        report_to=["none"],
    )

    with (run_ctx if run_ctx is not None else _nullcontext()):
        trainer = GRPOTrainer(
            model=cfg["model"]["name"],
            reward_funcs=reward_fn,
            args=grpo_cfg,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
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

    logger.info("Training complete. Model saved to %s", model_dir)


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
