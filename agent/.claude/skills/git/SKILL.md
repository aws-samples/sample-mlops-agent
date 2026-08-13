# Git Skill

All git operations are performed via MCP tools on the `mlops-gateway` server.

> See `agent/.claude/skills/planning/SKILL.md` for cross-skill rules (EULA, one-decision-per-turn, routing, confirmation gate).

## Confirmation Required Before State-Changing Actions

`git-skill___commit_experiment` pushes a commit to the remote repo — an
externally visible side effect. Before invoking it, summarize the commit
message and the files/config being persisted, ask for explicit confirmation,
and only invoke after an affirmative reply.

| Tool                            | Kind           |
| ------------------------------- | -------------- |
| `git-skill___commit_experiment` | state-changing |

## Commit Experiment

Use MCP tool: `git-skill___commit_experiment`

Parameters:

- `files` (object): map of relative path → file content to commit,
  e.g. `{ "experiments/run-123/config.json": "{...}" }`
- `commit_message` (string): git commit message
- `branch` (string): target branch (default: main)
- `_user_id` (string): always pass CURRENT_USER_ID

Returns: `{ "commit_sha": "abc123...", "repo_url": "https://github.com/..." }`
