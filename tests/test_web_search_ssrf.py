"""Security regression — web-search fetch SSRF guard (CSO 2026-07-25, HIGH).

Covers the fix in lambda/skills/web_search/handler.py: _assert_safe_url rejects
non-http(s) schemes and any host resolving to metadata / loopback / private /
reserved address space. Literal-IP URLs are used so no live DNS is needed.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest


def _import_web_search():
    """Load the web_search handler under a unique module name (collision-proof)."""
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "lambda", "skills", "web_search", "handler.py"
        )
    )
    spec = importlib.util.spec_from_file_location("_web_search_handler_ssrf", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_web_search_handler_ssrf"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",  # EC2 IMDS
        "http://127.0.0.1/",  # loopback
        "http://[::1]/",  # ipv6 loopback
        "http://10.0.0.5/internal",  # private
        "http://192.168.1.1/",  # private
        "file:///etc/passwd",  # non-http scheme
        "ftp://example.com/x",  # non-http scheme
        "https:///no-host",  # missing host
    ],
)
def test_assert_safe_url_rejects_ssrf_targets(url):
    """Metadata, loopback, private ranges, and non-http schemes are all refused."""
    h = _import_web_search()
    with pytest.raises(ValueError):
        h._assert_safe_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://93.184.216.34/",  # public literal IPv4
        "https://93.184.216.34/path?q=1",
    ],
)
def test_assert_safe_url_allows_public(url):
    """A public address over http/https passes the guard."""
    h = _import_web_search()
    # Should not raise.
    h._assert_safe_url(url)


def test_fetch_blocks_metadata_before_connecting():
    """_fetch raises on the IMDS URL without ever opening a socket."""
    h = _import_web_search()
    called = {"opened": False}
    h._SAFE_OPENER.open = lambda *a, **k: called.__setitem__("opened", True)  # type: ignore[assignment]
    with pytest.raises(ValueError):
        h._fetch({"url": "http://169.254.169.254/latest/meta-data/"})
    assert called["opened"] is False
