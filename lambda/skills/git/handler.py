"""Git skill Lambda — Gateway MCP target.

Tools: commit_experiment.
GitHub token read from SSM (shared service credential — not per-user).
"""
import json
import os
import tempfile
from typing import Any

import boto3

PROJECT_NAME = os.environ.get("PROJECT_NAME", "sample-mlops-agent")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def _get_github_token() -> str:
    """Fetch GitHub personal access token from SSM.

    Returns:
        str: GitHub token.

    Raises:
        RuntimeError: If SSM lookup fails.
    """
    try:
        return boto3.client("ssm", region_name=AWS_REGION).get_parameter(
            Name=f"/{PROJECT_NAME}/dev/github-token", WithDecryption=True
        )["Parameter"]["Value"]
    except Exception as e:
        raise RuntimeError(f"Cannot retrieve GitHub token: {e}") from e


def _get_experiment_repo() -> str:
    """Fetch experiment repo URL from SSM.

    Returns:
        str: Git repo URL.

    Raises:
        RuntimeError: If SSM lookup fails.
    """
    try:
        return boto3.client("ssm", region_name=AWS_REGION).get_parameter(
            Name=f"/{PROJECT_NAME}/dev/git-experiment-repo"
        )["Parameter"]["Value"]
    except Exception as e:
        raise RuntimeError(f"Cannot retrieve experiment repo: {e}") from e


def _commit_experiment(args: dict) -> dict:
    """Commit experiment config files to the experiment git repo.

    Args:
        args: Required: files (dict[str, str] path→content), commit_message.
              Optional: branch (default main).

    Returns:
        dict: commit_sha, repo_url.
    """
    import git  # gitpython

    files: dict[str, str] = args["files"]
    commit_message = args.get("commit_message", "Add experiment config")
    branch = args.get("branch", "main")

    token = _get_github_token()
    repo_url = _get_experiment_repo()
    # Inject token into HTTPS URL for authentication
    authed_url = repo_url.replace("https://", f"https://{token}@")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = git.Repo.clone_from(authed_url, tmpdir, branch=branch, depth=1)
        for rel_path, content in files.items():
            full_path = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            repo.index.add([rel_path])
        repo.index.commit(commit_message)
        origin = repo.remotes.origin
        origin.push()
        commit_sha: str = repo.head.commit.hexsha

    return {"commit_sha": commit_sha, "repo_url": repo_url}


_DISPATCH: dict[str, Any] = {"commit_experiment": _commit_experiment}


def handler(event: dict, context: Any) -> dict:
    """Gateway MCP tool dispatcher for Git skills.

    Args:
        event: AgentCore Gateway passes the tool arguments map as `event`.
        context: Lambda context; tool name is in
            `context.client_context.custom['bedrockAgentCoreToolName']`,
            formatted as `${target_name}___${tool_name}`.

    Returns:
        dict: MCP content response.
    """
    raw_tool = context.client_context.custom.get("bedrockAgentCoreToolName", "") if getattr(context, "client_context", None) else ""
    tool_name = raw_tool.split("___", 1)[1] if "___" in raw_tool else raw_tool
    arguments = event or {}
    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}
    try:
        return {"content": [{"type": "text", "text": json.dumps(fn(arguments))}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
