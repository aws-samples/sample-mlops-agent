from agent.main import _truncate_result, _coerce_result_content


def test_truncate_under_limit():
    assert _truncate_result("hello", 8192) == ("hello", False)


def test_truncate_at_exact_limit():
    s = "x" * 8192
    assert _truncate_result(s, 8192) == (s, False)


def test_truncate_over_limit():
    s = "x" * 9000
    out, trunc = _truncate_result(s, 8192)
    assert trunc is True and len(out.encode("utf-8")) <= 8192


def test_truncate_preserves_utf8_boundary():
    # "€" is 3 bytes; fill 8191 ascii then one euro so the cut lands mid-codepoint
    s = "a" * 8191 + "€€"
    out, trunc = _truncate_result(s, 8192)
    assert trunc is True
    out.encode("utf-8")  # must not raise


def test_coerce_string_passthrough():
    assert _coerce_result_content("hi") == "hi"


def test_coerce_list_of_text_blocks():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert _coerce_result_content(blocks) == "ab"


def test_coerce_fallback_to_json():
    assert _coerce_result_content({"k": 1}) == '{"k": 1}'
