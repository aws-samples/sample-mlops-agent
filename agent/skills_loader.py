"""S3-backed skill store loader for the MLOps agent.

Skills live at ``s3://<session-bucket>/<prefix><name>/SKILL.md`` (prefix defaults
to ``skills/``). Each object is Markdown, optionally prefixed with a YAML-ish
frontmatter block (``name``, ``description``, ``tools``, ``model``). Our baked-in
container skills carry NO frontmatter, so parsing falls back to the directory id
for the name and the first prose line for the description.

The agent hydrates these into ``~/.claude/skills/`` at turn start (see
``main.hydrate_skills``) so UI edits take effect within one cache TTL without a
container redeploy. ``catalog_s3`` is cached in-process for ``_S3_TTL_SECONDS``
to keep the per-turn S3 cost bounded.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import boto3

_S3_TTL_SECONDS = 60
# (bucket, prefix) -> {"at": epoch_seconds, "skills": list[dict]}
_S3_CACHE: dict[tuple[str, str], dict[str, Any]] = {}

_META_KEYS = ("name", "description", "tools", "model")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def clear_cache() -> None:
    """Drop the in-process catalog cache (used by tests and forced refreshes)."""
    _S3_CACHE.clear()


def _first_prose_line(text: str) -> str:
    """Return the first non-heading, non-blockquote, non-blank line — used as the
    description when a skill file has no frontmatter."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ">")):
            continue
        return stripped
    return ""


def parse_skill(md: str, *, fallback_name: str) -> dict[str, Any]:
    """Parse one SKILL.md into ``{name, description, tools, model, body}``.

    Frontmatter is optional. When absent, ``name`` falls back to
    ``fallback_name`` (the directory id) and ``description`` to the first prose
    line. Raises ``ValueError`` on malformed frontmatter (fail loud).
    """
    m = _FRONTMATTER_RE.match(md)
    meta: dict[str, Any] = {}
    body = md
    if m:
        raw, body = m.group(1), m.group(2)
        for line in raw.splitlines():
            if not line.strip():
                continue
            if ":" not in line:
                raise ValueError(f"malformed frontmatter line: {line!r}")
            key, _, val = line.partition(":")
            key = key.strip()
            if key in _META_KEYS:
                meta[key] = val.strip()
    tools_raw = meta.get("tools", "")
    tools = [t.strip() for t in tools_raw.split(",") if t.strip()] if tools_raw else []
    return {
        "name": meta.get("name") or fallback_name,
        "description": meta.get("description") or _first_prose_line(body),
        "tools": tools,
        "model": meta.get("model") or None,
        "body": body,
    }


def _s3_client():
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _list_skill_keys(bucket: str, prefix: str) -> list[str]:
    """Return every ``<prefix><name>/SKILL.md`` key under the prefix."""
    client = _s3_client()
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/SKILL.md"):
                keys.append(key)
    return keys


def _skill_name_from_key(key: str, prefix: str) -> str:
    """``skills/git/SKILL.md`` -> ``git`` (the directory id under the prefix)."""
    rel = key[len(prefix):] if key.startswith(prefix) else key
    return rel.split("/", 1)[0]


def catalog_s3(bucket: str, prefix: str) -> list[dict[str, Any]]:
    """List + parse all skills under the prefix, cached for ``_S3_TTL_SECONDS``."""
    cache_key = (bucket, prefix)
    cached = _S3_CACHE.get(cache_key)
    if cached and (time.time() - cached["at"]) < _S3_TTL_SECONDS:
        return cached["skills"]
    client = _s3_client()
    skills: list[dict[str, Any]] = []
    for key in _list_skill_keys(bucket, prefix):
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        skills.append(parse_skill(body, fallback_name=_skill_name_from_key(key, prefix)))
    _S3_CACHE[cache_key] = {"at": time.time(), "skills": skills}
    return skills


def load_skills_s3(
    bucket: str, prefix: str, names: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """Return parsed skills (optionally filtered to ``names``)."""
    cat = catalog_s3(bucket, prefix)
    if names is None:
        return cat
    wanted = set(names)
    return [s for s in cat if s["name"] in wanted]


def write_skill_files(dest_dir: str, skills: list[dict[str, Any]]) -> int:
    """Write each skill to ``<dest_dir>/<name>/SKILL.md``. Returns count written."""
    dest = Path(dest_dir)
    written = 0
    for s in skills:
        target = dest / s["name"] / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(s["body"], encoding="utf-8")
        written += 1
    return written


def normalize_frontmatter(md: str, *, name: str) -> str:
    """Prepend YAML frontmatter (name + first-prose-line description) to a plain
    Markdown skill. Content that already has frontmatter is returned unchanged."""
    if _FRONTMATTER_RE.match(md):
        return md
    description = _first_prose_line(md)
    return f"---\nname: {name}\ndescription: {description}\n---\n{md}"


def seed_skills_if_empty(bucket: str, prefix: str, baked_claude_dir: str) -> int:
    """Seed the S3 skill store from the baked container skills, ONLY if the
    prefix is currently empty (so live UI edits are never clobbered).

    Reads ``<baked_claude_dir>/skills/<name>/SKILL.md`` + ``<baked_claude_dir>/
    CLAUDE.md``, normalizes frontmatter, and uploads to ``<prefix>``. Returns the
    number of skill files uploaded (0 if the prefix already had objects).
    """
    if not bucket:
        return 0
    client = _s3_client()
    existing = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    if existing.get("KeyCount", 0) > 0:
        return 0
    baked = Path(baked_claude_dir)
    uploaded = 0
    skills_root = baked / "skills"
    if skills_root.is_dir():
        for skill_dir in sorted(p for p in skills_root.iterdir() if p.is_dir()):
            src = skill_dir / "SKILL.md"
            if not src.is_file():
                continue
            body = normalize_frontmatter(src.read_text(encoding="utf-8"), name=skill_dir.name)
            client.put_object(
                Bucket=bucket,
                Key=f"{prefix}{skill_dir.name}/SKILL.md",
                Body=body.encode("utf-8"),
                ContentType="text/markdown",
            )
            uploaded += 1
    prompt_src = baked / "CLAUDE.md"
    if prompt_src.is_file():
        client.put_object(
            Bucket=bucket,
            Key=f"{prefix}system-prompt.md",
            Body=prompt_src.read_text(encoding="utf-8").encode("utf-8"),
            ContentType="text/markdown",
        )
    return uploaded


def load_system_prompt(bucket: str, prefix: str) -> Optional[str]:
    """Return the text of ``<prefix>system-prompt.md`` or ``None`` if absent."""
    client = _s3_client()
    key = f"{prefix}system-prompt.md"
    try:
        return client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    except client.exceptions.NoSuchKey:
        return None
    except Exception:
        return None
