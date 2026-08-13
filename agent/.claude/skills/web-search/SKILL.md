# Web Search Skill

> See `agent/.claude/skills/planning/SKILL.md` for cross-skill rules (EULA, one-decision-per-turn, routing, confirmation gate).

## Confirmation Rule (Read-Only Skill)

All web-search tools are read-only — external HTTP GETs with no side effects.
No confirmation is required before invoking them. Do NOT over-confirm: users
asking "search for X" expect you to search, not to first ask "should I search?".

| Tool                            | Kind      |
| ------------------------------- | --------- |
| `web-search-skill___web_search` | read-only |
| `web-search-skill___fetch`      | read-only |

All web-search operations are performed via MCP tools on the `mlops-gateway`
server (target name `web-search-skill`). The Lambda fronts **Amazon Nova Web
Grounding** via the Bedrock Converse API's `systemTool` mechanism, with a
**DuckDuckGo** fallback that fires automatically on any Nova Grounding
exception (throttling, service errors, timeouts).

## When to use it

- The user asks about **current facts**: library versions, pricing, release
  notes, changelogs, new model IDs, API deprecations, "what's the latest X?"
- You need to **validate an assumption** against something outside the
  codebase before committing the agent to a billable action — e.g. "does
  Bedrock support this model in us-east-1?"
- You need to **read a specific URL** the user mentioned, or a citation
  returned by a previous `web_search` call.

Do NOT use this tool when:

- The user wants a **HuggingFace dataset or model ID** — use the
  `huggingface` skill, which is already authenticated and knows the Hub API.
- You need **MLflow scorer names** — use `mlflow-skill___list_scorers`.
- You need **AWS SDK / API reference** info — prefer the
  `aws-documentation-mcp-server` tools if they are available; fall back to
  `web_search` only if they aren't.

## Web Search

Use MCP tool: `web-search-skill___web_search`

Runs Nova Grounding first. If Nova raises (after its adaptive-retry loop
gives up), transparently falls back to DDG and sets `fallback: "ddg"` on
the response so you can downgrade the caller's trust accordingly.

Parameters:

- `query` (string): natural-language search query.

Returns:

- Nova path: `{ "results": [{ "index": 1, "title": "...", "snippet":
"<synthesised answer>", "sources": ["https://...", ...], "link":
"<first source>" }] }`. **Cite at least one URL from `sources` when
  you surface the answer to the user** — Nova Grounding's grounding is
  worthless if you strip the citations.
- DDG path: `{ "results": [{ "index": N, "title", "snippet", "link" }, ...],
"fallback": "ddg" }` — five results max, no synthesis, no citations beyond
  the top-level `link` each.

## Fetch

Use MCP tool: `web-search-skill___fetch`

Fetches a specific URL and returns readable text. HTML is stripped (scripts,
styles, nav chrome removed); JSON / plain text pass through unchanged.

Parameters:

- `url` (string): absolute URL to fetch.
- `max_length` (int, optional, default 5000): truncation cap in characters.
  On truncation the response carries `truncated: true` and a `note` telling
  you how to get more.

Returns: `{ "content": "...", "url": "...", "truncated?": bool, "note?": "..." }`

### URL restrictions (SSRF guard)

`fetch` only reaches **public http/https** targets. It rejects — with an
MCP error, before connecting — any other scheme (`file:`, `ftp:`, …) and
any host that resolves to loopback, link-local (incl. the EC2 metadata
IP), private, reserved, multicast, or unspecified address space. Redirect
targets are re-validated hop by hop, so a public URL that 3xx-redirects
into a blocked range also fails. These rejections are **permanent policy,
not transient errors**: do not retry with a different scheme or an IP
literal — tell the user the URL is not fetchable from this environment.

## Rules

- **Never search for the same thing twice in a single turn.** If the first
  call didn't answer your question, change the query, not the tool.
- **Always cite.** If Nova Grounding returns citations, include at least
  one URL in your reply; if DDG fallback fired, call that out ("per a
  DuckDuckGo search, …") so the user knows the answer is ungrounded.
- **Prefer `fetch` over a second `web_search`** when you already have a URL
  that looks promising — it's cheaper, faster, and deterministic.
- **Do not use this for large downloads.** `fetch` truncates at 5 000
  chars by default and 50 000 is a sensible hard ceiling. For datasets
  or model weights, use the HuggingFace skill instead.
