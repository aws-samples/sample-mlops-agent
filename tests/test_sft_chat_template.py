"""QA BUG-020 follow-up — chat template inlining in sft_train.

transformers >= 4.5x writes the chat template to a standalone
chat_template.jinja; TGI's Messages API only reads
tokenizer_config.json["chat_template"]. The training script must inline it
so fine-tuned artifacts are chat-servable.
"""
import importlib.util
import json
from pathlib import Path


def _load_sft_train():
    path = (Path(__file__).resolve().parents[1]
            / "lambda/skills/sagemaker/training/sft_train.py")
    spec = importlib.util.spec_from_file_location("sft_train_mod", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_inlines_jinja_into_tokenizer_config(tmp_path):
    mod = _load_sft_train()
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 1024}))
    mod._inline_chat_template(str(tmp_path))
    cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert cfg["chat_template"] == "{{ messages }}"
    assert cfg["model_max_length"] == 1024  # existing keys preserved


def test_noop_when_template_already_inline(tmp_path):
    mod = _load_sft_train()
    (tmp_path / "chat_template.jinja").write_text("NEW")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "OLD"}))
    mod._inline_chat_template(str(tmp_path))
    cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert cfg["chat_template"] == "OLD"


def test_noop_when_no_jinja_file(tmp_path):
    mod = _load_sft_train()
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({}))
    mod._inline_chat_template(str(tmp_path))
    assert "chat_template" not in json.loads((tmp_path / "tokenizer_config.json").read_text())
