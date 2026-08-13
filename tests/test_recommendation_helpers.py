import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lambda/skills/sagemaker"))

def test_derive_serving_pytorch_training_returns_region_pinned_tgi(monkeypatch):
    """F-3 fix: derivation now returns the region-pinned TGI DLC for any
    recognisable PyTorch training image, rather than regex-building a
    hypothetical URI that may not exist in ECR."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    import importlib
    import handler as h
    importlib.reload(h)  # pick up monkeypatched AWS_REGION
    training = ("763104351884.dkr.ecr.us-east-1.amazonaws.com/"
                "pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker")
    out = h._derive_serving_from_training(training)
    assert out is not None
    assert "huggingface-pytorch-tgi-inference" in out
    assert "us-east-1" in out

def test_derive_serving_unknown_returns_none(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    import importlib
    import handler as h
    importlib.reload(h)
    assert h._derive_serving_from_training("custom.ecr/my:image") is None

def test_derive_serving_unsupported_region_returns_none(monkeypatch):
    """Regions we haven't onboarded fall back to serving_image_uri override."""
    monkeypatch.setenv("AWS_REGION", "ap-southeast-1")
    import importlib
    import handler as h
    importlib.reload(h)
    training = ("763104351884.dkr.ecr.us-east-1.amazonaws.com/"
                "pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker")
    assert h._derive_serving_from_training(training) is None

def test_tgi_default_env_shape():
    import handler as h
    env = h._tgi_default_env({"input_tokens": 500, "output_tokens": 150})
    assert env["HF_MODEL_ID"] == "/opt/ml/model"
    assert env["SM_NUM_GPUS"] == "1"
    # TGI requires MAX_INPUT_LENGTH < MAX_TOTAL_TOKENS.
    # input_tokens(500) + 256 = 756; total = 756 + 150 + 256 = 1162.
    assert int(env["MAX_INPUT_LENGTH"]) == 756
    assert int(env["MAX_TOTAL_TOKENS"]) == 1162
    assert int(env["MAX_INPUT_LENGTH"]) < int(env["MAX_TOTAL_TOKENS"])

def test_compute_workload_fingerprint_is_stable():
    import handler as h
    a = h._compute_workload_fingerprint(
        {"input_tokens": 500, "output_tokens": 150, "concurrency_levels": [1, 4]})
    b = h._compute_workload_fingerprint(
        {"concurrency_levels": [1, 4], "input_tokens": 500, "output_tokens": 150})
    assert a == b  # key order must not matter
    assert len(a) == 64  # sha256 hex

def test_estimate_wall_clock_minutes_is_nonzero():
    import handler as h
    assert h._estimate_wall_clock_minutes("ml.g6.xlarge") >= 15
    assert h._estimate_wall_clock_minutes("ml.g5.2xlarge") >= 15
