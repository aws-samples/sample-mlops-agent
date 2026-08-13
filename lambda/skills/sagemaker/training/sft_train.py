#!/usr/bin/env python3
"""SFT training entrypoint — runs inside SageMaker PyTorch 2.4.0/py311 container."""
import json
import logging
import os
import subprocess  # nosec B404
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_dependencies():
    # EXACT pins — do NOT relax these to floating ranges.
    # transformers==4.56.0 calls `Accelerator.parallelism_config`,
    # which only exists in accelerate>=1.10.0. Both 1.7.0 and 1.9.0
    # raised AttributeError inside trainer.train() on the first step.
    # An earlier comment warned off accelerate>=1.10 because it broke
    # `SFTConfig.distributed_state` on TRL 0.13/0.14 — that concern
    # does NOT apply to TRL 0.21.0 (which this script uses).
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
            "accelerate==1.10.1",
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


def _inline_chat_template(model_dir: str) -> None:
    """Copy a standalone ``chat_template.jinja`` back into
    ``tokenizer_config.json["chat_template"]`` (QA BUG-020 follow-up).

    transformers ≥ 4.5x saves the chat template to a separate
    ``chat_template.jinja`` file, but TGI's Messages API (and other servers)
    only read ``tokenizer_config.json["chat_template"]`` — without inlining,
    every OpenAI-schema request 422s with "Template error: template not
    found" and the AI Benchmark fails 30/30.

    Args:
        model_dir: The SageMaker model output dir (/opt/ml/model).
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    jinja_path = os.path.join(model_dir, "chat_template.jinja")
    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if not (os.path.exists(jinja_path) and os.path.exists(cfg_path)):
        return
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("chat_template"):
        return  # already inline — nothing to do
    with open(jinja_path, encoding="utf-8") as f:
        cfg["chat_template"] = f.read()
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[sft] inlined chat_template.jinja into {cfg_path}")


def main():
    logger.info("Installing training dependencies...")
    install_dependencies()

    import mlflow  # noqa: PLC0415 — imported after pip install
    import boto3
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    bucket = os.environ["CONFIG_S3_BUCKET"]
    key = os.environ["CONFIG_S3_KEY"]
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    s3 = boto3.client("s3")
    config_data = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    cfg = config_data["config"]

    # Pin the Hub revision (default "main") so a job can lock to an immutable
    # commit SHA — an unpinned pull would silently follow upstream repo changes.
    hf_revision = os.environ.get("HF_HUB_REVISION", "main")
    train_ds = load_dataset(
        cfg["data"]["dataset_name"],
        split=cfg["data"].get("train_split", "train"),
        revision=cfg["data"].get("revision", hf_revision),
    )

    max_samples = int(cfg["data"].get("max_samples", 0))
    if max_samples > 0:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
        logger.info("Dataset capped to %d samples (max_samples=%d)", len(train_ds), max_samples)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"],
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        revision=cfg["model"].get("revision", hf_revision),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set max sequence length on the tokenizer — TRL ≥ 0.13 removed max_seq_length
    # from both SFTConfig and SFTTrainer; setting model_max_length is the correct approach.
    tokenizer.model_max_length = int(cfg.get("sft", {}).get("max_seq_length", 1024))

    text_field = cfg["data"].get("text_field", "text")

    # If the dataset doesn't have a plain text column, convert conversation-format
    # columns (conversations / messages) to text via the tokenizer's chat template.
    # This handles datasets like teknium/OpenHermes-2.5 that store data as structs.
    if text_field not in train_ds.column_names:
        conv_col = next((c for c in ("conversations", "messages") if c in train_ds.column_names), None)
        if conv_col is None:
            raise ValueError(
                f"Dataset has no '{text_field}' column and no 'conversations'/'messages' column. "
                f"Available columns: {train_ds.column_names}"
            )
        logger.info("No '%s' column found — applying chat template from '%s'", text_field, conv_col)

        def _to_text(example):
            msgs = []
            for m in example[conv_col]:
                role = m.get("role") or m.get("from", "user")
                if role == "human":
                    role = "user"
                elif role in ("gpt", "assistant"):
                    role = "assistant"
                elif role == "system":
                    role = "system"
                content = m.get("content") or m.get("value", "")
                msgs.append({"role": role, "content": content})
            return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

        # Drop ALL non-text columns, not just the conversation column. TRL
        # 0.21's SFTTrainer auto-detects dataset format by column presence:
        # if it sees a `prompt` column it switches to prompt-completion mode
        # and then KeyErrors on the missing `completion` column. This bit
        # HuggingFaceH4/ultrachat_200k (columns: prompt, prompt_id, messages)
        # — removing only `messages` left a stray `prompt` that triggered
        # the wrong code path. Keeping only `text` forces single-field mode.
        cols_to_remove = [c for c in train_ds.column_names if c != "text"]
        train_ds = train_ds.map(_to_text, remove_columns=cols_to_remove)
        text_field = "text"

    run_ctx = _init_mlflow()

    sft_cfg = SFTConfig(
        output_dir=model_dir,
        max_steps=int(cfg["training"].get("max_steps", 500)),
        learning_rate=float(cfg["training"].get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(cfg["training"].get("batch_size", 2)),
        bf16=bool(cfg["training"].get("bf16", True)),
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        dataset_text_field=text_field,
        # Per-step metrics are handled by _MlflowMetricsCallback passed to the trainer.
        # report_to=["mlflow"] is avoided because MLflowCallback.setup() unconditionally
        # calls mlflow.start_run(), crashing when a run is already active.
        report_to=["none"],
    )

    with (run_ctx if run_ctx is not None else _nullcontext()):
        trainer = SFTTrainer(
            model=cfg["model"]["name"],
            args=sft_cfg,
            train_dataset=train_ds,
            processing_class=tokenizer,
            callbacks=[_MlflowMetricsCallback()] if run_ctx is not None else [],
        )
        trainer.train()
        # Save the final model to /opt/ml/model/ root (alongside any
        # checkpoint-N/ subfolders from save_strategy="steps"). Without this
        # the tarball SageMaker packages only contains per-checkpoint
        # subfolders, which breaks downstream serving containers (TGI,
        # vLLM, etc. look for weights at the model-dir root).
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        _inline_chat_template(model_dir)
        if run_ctx is not None:
            # Log the final summary row (train_loss, runtime, samples/sec, steps/sec)
            # that the Trainer appends to log_history after training completes.
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

    logger.info("SFT complete. Model saved to %s", model_dir)


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
