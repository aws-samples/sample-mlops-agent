"""Unit tests for agent/skills_loader.py — the S3-backed skill store loader.

Covers: frontmatter parsing (present + absent/fallback), S3 catalog + load with
a 60s TTL cache, and write_skill_files. S3 is faked with moto.
"""

import boto3
from moto import mock_aws

from agent import skills_loader

_BUCKET = "test-session-bucket"
_PREFIX = "skills/"

_WITH_FM = """---
name: sagemaker
description: SageMaker training + deployment via MCP.
tools: sagemaker-skill___submit_training_job, sagemaker-skill___deploy_model
model: claude-opus-4-5
---
# SageMaker Skill

Body text here.
"""

_NO_FM = """# Git Skill

All git operations are performed via MCP tools on the mlops-gateway server.
"""


def _seed(client, key, body):
    client.put_object(Bucket=_BUCKET, Key=key, Body=body.encode())


def test_parse_skill_with_frontmatter():
    s = skills_loader.parse_skill(_WITH_FM, fallback_name="sagemaker")
    assert s["name"] == "sagemaker"
    assert s["description"].startswith("SageMaker")
    assert "sagemaker-skill___submit_training_job" in s["tools"]
    assert s["model"] == "claude-opus-4-5"
    assert s["body"].lstrip().startswith("# SageMaker Skill")


def test_parse_skill_without_frontmatter_falls_back():
    # No frontmatter: name from fallback (dir id), description from first prose line.
    s = skills_loader.parse_skill(_NO_FM, fallback_name="git")
    assert s["name"] == "git"
    assert s["description"].startswith("All git operations")
    assert s["tools"] == []
    assert s["model"] is None
    assert s["body"].startswith("# Git Skill")


@mock_aws
def test_catalog_s3_lists_and_parses():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=_BUCKET)
    _seed(client, "skills/sagemaker/SKILL.md", _WITH_FM)
    _seed(client, "skills/git/SKILL.md", _NO_FM)
    _seed(client, "skills/system-prompt.md", "IGNORE ME")  # not a skill dir
    skills_loader.clear_cache()
    cat = skills_loader.catalog_s3(_BUCKET, _PREFIX)
    names = sorted(c["name"] for c in cat)
    assert names == ["git", "sagemaker"]


@mock_aws
def test_ttl_cache_avoids_second_s3_hit(monkeypatch):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=_BUCKET)
    _seed(client, "skills/git/SKILL.md", _NO_FM)
    skills_loader.clear_cache()
    calls = {"n": 0}
    real_list = skills_loader._list_skill_keys

    def counting(bucket, prefix):
        calls["n"] += 1
        return real_list(bucket, prefix)

    monkeypatch.setattr(skills_loader, "_list_skill_keys", counting)
    skills_loader.catalog_s3(_BUCKET, _PREFIX)
    skills_loader.catalog_s3(_BUCKET, _PREFIX)  # within TTL → cached
    assert calls["n"] == 1


@mock_aws
def test_write_skill_files(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=_BUCKET)
    _seed(client, "skills/git/SKILL.md", _NO_FM)
    skills_loader.clear_cache()
    skills = skills_loader.load_skills_s3(_BUCKET, _PREFIX)
    skills_loader.write_skill_files(str(tmp_path), skills)
    written = tmp_path / "git" / "SKILL.md"
    assert written.is_file()
    assert written.read_text(encoding="utf-8").startswith("# Git Skill")


@mock_aws
def test_load_system_prompt(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=_BUCKET)
    _seed(client, "skills/system-prompt.md", "You are the MLOps agent.")
    skills_loader.clear_cache()
    assert skills_loader.load_system_prompt(_BUCKET, _PREFIX) == "You are the MLOps agent."
    # Absent → None
    skills_loader.clear_cache()
    assert skills_loader.load_system_prompt(_BUCKET, "other/") is None


def test_normalize_frontmatter_prepends_yaml():
    out = skills_loader.normalize_frontmatter("# Git Skill\n\nAll git ops via MCP.\n", name="git")
    assert out.startswith("---\n")
    assert "name: git" in out
    assert "description: All git ops via MCP." in out
    assert "# Git Skill" in out
    # Already-frontmattered content is returned unchanged.
    already = "---\nname: git\n---\n# X\n"
    assert skills_loader.normalize_frontmatter(already, name="git") == already


@mock_aws
def test_seed_skills_if_empty(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=_BUCKET)
    # Baked source: plain markdown skills + a CLAUDE.md system prompt.
    src = tmp_path / ".claude"
    (src / "skills" / "git").mkdir(parents=True)
    (src / "skills" / "git" / "SKILL.md").write_text("# Git Skill\n\nAll git ops.\n")
    (src / "CLAUDE.md").write_text("You are the agent.")
    skills_loader.clear_cache()

    n = skills_loader.seed_skills_if_empty(_BUCKET, _PREFIX, str(src))
    assert n == 1
    body = client.get_object(Bucket=_BUCKET, Key="skills/git/SKILL.md")["Body"].read().decode()
    assert "name: git" in body  # frontmatter normalized on seed
    sp = client.get_object(Bucket=_BUCKET, Key="skills/system-prompt.md")["Body"].read().decode()
    assert sp == "You are the agent."

    # Second call is a no-op (prefix non-empty) — must not clobber live edits.
    client.put_object(Bucket=_BUCKET, Key="skills/git/SKILL.md", Body=b"EDITED")
    skills_loader.clear_cache()
    assert skills_loader.seed_skills_if_empty(_BUCKET, _PREFIX, str(src)) == 0
    body2 = client.get_object(Bucket=_BUCKET, Key="skills/git/SKILL.md")["Body"].read().decode()
    assert body2 == "EDITED"
