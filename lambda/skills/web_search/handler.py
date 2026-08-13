"""Web search skill Lambda — Gateway MCP target.

Tools:
  - web_search — Nova Web Grounding (primary) with DuckDuckGo fallback.
                 Returns grounded, cited search results for a natural-language
                 query.
  - fetch      — Fetch a URL and return its content as readable text (HTML
                 is stripped to plain text; JSON / plain text pass through).

Ported from /Users/huthmac/Documents/AWS/00_workspace/nova-web-grounding-mcp
but rewritten to use boto3 bedrock-runtime.converse() directly instead of
the Strands Agent wrapper — Strands isn't used by any other skill Lambda
in this project and dragging it in would inflate the image without benefit.

Nova Web Grounding is invoked as a systemTool via additional request fields
on the Converse API; Nova handles the search internally and the response
carries citations in citationsContent blocks on assistant messages.
"""
import ipaddress
import json
import logging
import os
import re
import socket
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.config import Config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
NOVA_GROUNDING_MODEL_ID = os.environ.get(
    "NOVA_GROUNDING_MODEL_ID", "us.amazon.nova-2-lite-v1:0"
)

_NOVA_SYSTEM_PROMPT = (
    "You are a concise web search assistant. Search for accurate, up-to-date "
    "information and respond with a clear, well-structured summary. "
    "Always include source URLs to support your response."
)

# Module-level Bedrock runtime client — reused across warm invocations.
# read_timeout=3600 is required by the Nova Grounding docs; the systemTool
# path can issue multiple internal searches and take longer than the default.
_bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(
        retries={"max_attempts": 5, "mode": "adaptive"},
        read_timeout=3600,
    ),
)


def _nova_grounding_search(query: str) -> dict:
    """Run a single Nova Web Grounding search via Converse + systemTool.

    Returns the synthesised answer plus the list of citation URLs pulled
    from citationsContent blocks. Raises on any bedrock-runtime exception
    so the caller can fall back to DuckDuckGo.
    """
    resp = _bedrock.converse(
        modelId=NOVA_GROUNDING_MODEL_ID,
        system=[{"text": _NOVA_SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": query}]}],
        inferenceConfig={"temperature": 0.3, "maxTokens": 4096},
        additionalModelRequestFields={
            "toolConfig": {"tools": [{"systemTool": {"name": "nova_grounding"}}]}
        },
    )

    # Extract synthesised answer + citation URLs from the assistant message.
    content_blocks = (
        resp.get("output", {}).get("message", {}).get("content", [])
    )
    text_parts: list[str] = []
    citation_urls: list[str] = []
    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        cc = block.get("citationsContent") or {}
        for citation in cc.get("citations", []) or []:
            url = (
                citation.get("location", {}).get("web", {}).get("url")
                or citation.get("location", {}).get("webLocation", {}).get("url")
            )
            if url and url not in citation_urls:
                citation_urls.append(url)

    snippet = "".join(text_parts).strip()
    logger.info(
        "Nova Grounding search completed for %r (%d citations)",
        query[:50], len(citation_urls),
    )
    return {
        "results": [
            {
                "index":   1,
                "title":   "Web Search Results (Nova Grounding)",
                "snippet": snippet,
                "sources": citation_urls,
                "link":    citation_urls[0] if citation_urls else "Grounded web sources",
            }
        ]
    }


def _ddg_search(query: str) -> dict:
    """Fallback search via DuckDuckGo (no auth needed)."""
    # `ddgs` is the maintained fork of the older duckduckgo-search package.
    from ddgs import DDGS  # noqa: PLC0415

    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=5))
    results = [
        {
            "index":   idx + 1,
            "title":   r.get("title", "No title"),
            "snippet": r.get("body", "No snippet"),
            "link":    r.get("href", "No link"),
        }
        for idx, r in enumerate(raw)
    ]
    logger.info("DuckDuckGo search: %d results for %r", len(results), query[:50])
    return {"results": results}


def _web_search(args: dict) -> dict:
    """Search the web for `query`. Nova Grounding first, DDG fallback on error.

    Args:
        args: Required: query (string).

    Returns:
        dict: {results: [...], fallback?: "ddg"}
    """
    query = (args.get("query") or "").strip()
    if not query:
        raise ValueError("web_search requires a non-empty 'query' argument")
    try:
        return _nova_grounding_search(query)
    except Exception as exc:
        logger.warning(
            "Nova Grounding failed for %r (%s: %s) — falling back to DuckDuckGo",
            query[:50], type(exc).__name__, exc,
        )
        result = _ddg_search(query)
        result["fallback"] = "ddg"
        return result


class _TextExtractor(HTMLParser):
    """Minimal HTML → plain-text converter using stdlib only."""

    _SKIP_TAGS = {"script", "style", "head", "noscript", "svg", "iframe"}
    _BLOCK_TAGS = {
        "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
        "tr", "blockquote", "pre",
    }

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, _attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of 3+ blank lines to exactly 2 for readability.
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def _assert_safe_url(url: str) -> None:
    """Reject URLs that could drive a server-side request forgery (SSRF).

    Allows only http/https and resolves the host to make sure it does not point at
    loopback, link-local (incl. the EC2 metadata IP 169.254.169.254), private,
    reserved, multicast, or unspecified address space.

    Args:
        url: The URL requested by the caller (agent/model-influenced, untrusted).

    Raises:
        ValueError: If the scheme is not http/https, the host is missing, DNS does
            not resolve, or any resolved address is in a blocked range.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"fetch only allows http/https URLs, got scheme {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("fetch requires a URL with a host")

    def _check(ip_str: str) -> None:
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_private
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(f"fetch refuses to reach non-public address {ip} (host {host!r})")

    # A literal IP (v4/v6) needs no DNS — check it directly. This avoids a
    # needless (and, in constrained/sandboxed environments, flaky) getaddrinfo
    # call for the common literal-IP path; hostnames still resolve below.
    try:
        _check(host.strip("[]"))
        return
    except ValueError as exc:
        if "does not appear to be an IPv4 or IPv6 address" not in str(exc):
            raise  # a real "blocked address" rejection — propagate it

    try:
        # Resolve every address the host maps to — a single safe-looking A record is
        # not enough if an AAAA (or a second A) points at a blocked range.
        infos = socket.getaddrinfo(host, parsed.port or 0, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"fetch could not resolve host {host!r}: {exc}") from exc
    for info in infos:
        _check(info[4][0])


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect target so a safe host cannot 3xx-hop to a blocked one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, D102
        _assert_safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# Opener that enforces the SSRF guard on redirects; the initial URL is validated in _fetch.
_SAFE_OPENER = urllib.request.build_opener(_SafeRedirectHandler())


def _fetch(args: dict) -> dict:
    """Fetch a URL and return its content as plain text.

    Args:
        args: Required: url.
              Optional: max_length (int, default 5000, truncation cap).

    Returns:
        dict: {content, url, truncated?, note?}
    """
    url = (args.get("url") or "").strip()
    if not url:
        raise ValueError("fetch requires a non-empty 'url' argument")
    # SSRF guard: block metadata / loopback / private targets before connecting.
    _assert_safe_url(url)
    max_length = int(args.get("max_length", 5000))

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; sample-mlops-agent/1.0; "
            "+https://github.com/aws-samples)"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        # Use the SSRF-aware opener so redirects are re-validated, not just the first hop.
        with _SAFE_OPENER.open(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            # Read extra to allow for HTML markup overhead before stripping.
            raw_bytes = resp.read(max_length * 5)
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            raw_text = raw_bytes.decode(charset, errors="replace")
            if "text/html" in content_type or "application/xhtml" in content_type:
                parser = _TextExtractor()
                parser.feed(raw_text)
                content = parser.get_text()
            else:
                content = raw_text
        truncated = len(content) > max_length
        content = content[:max_length]
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error: {exc.reason}") from exc

    result: dict[str, Any] = {"content": content, "url": url}
    if truncated:
        result["truncated"] = True
        result["note"] = (
            f"Content truncated to {max_length} chars. Pass a larger "
            f"max_length to retrieve more."
        )
    return result


_DISPATCH: dict[str, Any] = {
    "web_search": _web_search,
    "fetch":      _fetch,
}


def handler(event: dict, context: Any) -> dict:
    """Gateway MCP tool dispatcher for the web_search skill.

    Args:
        event: AgentCore Gateway passes the tool arguments map as `event`.
        context: Lambda context; tool name is in
            `context.client_context.custom['bedrockAgentCoreToolName']`,
            formatted as `${target_name}___${tool_name}`.

    Returns:
        dict: MCP content response.
    """
    raw_tool = (
        context.client_context.custom.get("bedrockAgentCoreToolName", "")
        if getattr(context, "client_context", None)
        else ""
    )
    tool_name = raw_tool.split("___", 1)[1] if "___" in raw_tool else raw_tool
    arguments = event or {}
    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}
    try:
        return {"content": [{"type": "text", "text": json.dumps(fn(arguments))}]}
    except Exception as e:
        logger.exception("[handler] tool=%s raised", tool_name)
        return {"content": [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}], "isError": True}
