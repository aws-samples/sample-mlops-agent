"""Reference math-correctness scorer Lambda (R6).

Canonical implementation of the custom-scorer contract documented in
``agent/.claude/skills/mlflow/SKILL.md`` — copy this directory into your
own service and adapt as needed.

Contract (see SKILL.md "Custom Scorers"):
  Input  body: {"inputs": {...}, "outputs": {...}, "expectations": {...}}
  Output body: {"name": "math_correctness", "score": 0.0|1.0,
                "reason": "<free text>"}
  Timeout: 5 s per row (enforced by the eval container's boto3 cfg).

Scoring rules:
  * Read the model's answer from ``outputs.output`` (preferred) or
    ``outputs.completion`` / ``outputs.response``. Fall back to the
    stringified outputs map.
  * Read the expected answer from ``expectations.expected_output`` (or
    ``expectations.answer`` / ``expectations.ground_truth``).
  * Try to parse both as ``sympy`` expressions. If they're symbolically
    equivalent (``sympy.simplify(a - b) == 0``), score=1.0.
  * Otherwise, fall back to a normalised string compare (whitespace +
    case stripped). On match, score=1.0; otherwise 0.0.
  * Any exception surfaces as score=0.0 with the reason field carrying
    the exception type — the eval container treats non-200 as a null
    metric anyway, so we deliberately stay 200 + score=0.0 to keep the
    metric populated.
"""
import json
from typing import Any


_OUTPUT_KEYS = ("output", "completion", "response", "answer")
_EXPECTED_KEYS = ("expected_output", "answer", "ground_truth", "expected_answer")


def _first_string(d: dict | None, keys: tuple[str, ...]) -> str:
    """Return the first non-empty stringified value from `d` matching keys.

    Returns "" if `d` is falsy or no key resolves to a non-empty value.
    """
    if not isinstance(d, dict):
        return ""
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _normalize(s: str) -> str:
    """Lowercase + collapse whitespace for the string-compare fallback."""
    return " ".join(s.lower().split())


def _sympy_equivalent(a: str, b: str) -> bool:
    """Return True iff sympy parses both sides and ``simplify(a-b) == 0``.

    Any parse / simplify failure returns False so the caller falls back to
    string compare. Imported lazily so the cold-start cost is paid only on
    the first invocation per container.
    """
    try:
        from sympy import simplify  # noqa: PLC0415
        from sympy.parsing.sympy_parser import parse_expr  # noqa: PLC0415

        ea = parse_expr(a, evaluate=True)
        eb = parse_expr(b, evaluate=True)
        return simplify(ea - eb) == 0
    except Exception:  # noqa: BLE001
        return False


def handler(event: Any, _context: Any) -> dict:
    """Score one (inputs, outputs, expectations) row.

    The eval container invokes us with ``InvocationType=RequestResponse``
    and a JSON body. AWS Lambda may deliver the body either parsed (when
    the function URL / API gateway proxy mode is used) or as a raw dict.
    We accept both — `event` is the JSON payload.
    """
    try:
        if isinstance(event, (bytes, bytearray)):
            event = json.loads(event)
        elif isinstance(event, str):
            event = json.loads(event)
        outputs = event.get("outputs") or {}
        expectations = event.get("expectations") or {}

        # R6 closure: mlflow.genai.evaluate hands scorers the raw row values —
        # for simple predict_fns `outputs` is the model's response STRING and
        # `expectations` is often a plain string too, not the documented dict
        # shape. Accept both.
        got = outputs.strip() if isinstance(outputs, str) else _first_string(outputs, _OUTPUT_KEYS)
        want = (expectations.strip() if isinstance(expectations, str)
                else _first_string(expectations, _EXPECTED_KEYS))
        if not got or not want:
            return {
                "name":   "math_correctness",
                "score":  0.0,
                "reason": "missing model output or expected answer",
            }

        if _sympy_equivalent(got, want) or _normalize(got) == _normalize(want):
            return {"name": "math_correctness", "score": 1.0, "reason": "match"}
        return {
            "name":   "math_correctness",
            "score":  0.0,
            "reason": f"got={got!r} want={want!r}",
        }
    except Exception as exc:  # noqa: BLE001
        # Never raise — keep the metric populated. The eval container would
        # still survive (it treats non-200 as null), but a populated zero
        # is more informative than a null on a misbehaving scorer.
        return {
            "name":   "math_correctness",
            "score":  0.0,
            "reason": f"{type(exc).__name__}: {exc}",
        }
