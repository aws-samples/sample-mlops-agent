"""Tests for main.hydrate_skills — dynamic S3 skill + system-prompt loading.

Eng-review M1/M2: skills are written to ``<claude_home>/skills/<name>/SKILL.md``
(NOT ``<claude_home>/.claude/skills`` — CLAUDE_HOME is already ~/.claude), and
the system prompt overwrites ``<claude_home>/CLAUDE.md`` (NOT passed as
ClaudeAgentOptions(system_prompt=...) — the preset stays). Empty S3 → no writes.
"""
import boto3
from moto import mock_aws

import agent.main as agent_main
from agent import skills_loader

_BUCKET = "test-session-bucket"
_PREFIX = "skills/"


def _seed(client, key, body):
    client.put_object(Bucket=_BUCKET, Key=key, Body=body.encode())


@mock_aws
def test_hydrate_writes_skills_and_prompt(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=_BUCKET)
    _seed(client, "skills/foo/SKILL.md", "# Foo\n\nFoo does things.\n")
    _seed(client, "skills/system-prompt.md", "You are the MLOps agent.")
    skills_loader.clear_cache()

    prompt = agent_main.hydrate_skills(_BUCKET, _PREFIX, str(tmp_path))

    assert (tmp_path / "skills" / "foo" / "SKILL.md").is_file()
    # Must NOT double-nest under an extra .claude/
    assert not (tmp_path / ".claude").exists()
    # System prompt written to <claude_home>/CLAUDE.md and returned
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "You are the MLOps agent."
    assert prompt == "You are the MLOps agent."


@mock_aws
def test_hydrate_empty_s3_writes_nothing(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=_BUCKET)
    skills_loader.clear_cache()
    # Pre-existing baked CLAUDE.md must be preserved (fallback path).
    (tmp_path / "CLAUDE.md").write_text("BAKED", encoding="utf-8")

    prompt = agent_main.hydrate_skills(_BUCKET, _PREFIX, str(tmp_path))

    assert prompt is None
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "BAKED"
    assert not (tmp_path / "skills").exists()


def test_hydrate_no_bucket_is_noop(tmp_path):
    # SESSION_BUCKET unset → returns None, no exception, no writes.
    assert agent_main.hydrate_skills("", _PREFIX, str(tmp_path)) is None
    assert not (tmp_path / "skills").exists()
