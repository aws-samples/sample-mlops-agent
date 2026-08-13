"""Schema transformation for eval datasets.

Applies a (source_schema, target_schema) conversion on top of a list of
records loaded from HuggingFace datasets. Scope v1: used only by the
eval Processing container (``entrypoint.py``). Training containers expect
already-shaped data today; if a future use case needs training-time
transformation this module is safe to import from there too.

Schemas
-------
- ``chat``        — OpenAI-style: each row has a ``messages`` column containing
                    a list of ``{"role": ..., "content": ...}`` dicts.
- ``sft``         — supervised-fine-tuning: each row has a single ``text``
                    column with the prompt + response encoded in one string.
- ``dpo``         — direct preference optimisation: each row has ``prompt``,
                    ``chosen``, ``rejected`` string columns.
- ``tabular-csv`` — ``pandas``-loadable CSV with named feature columns plus
                    a target column.
- ``auto``        — the caller did not specify a source; infer from columns.

Supported transformations (see R3 plan)
---------------------------------------
| Source       | Target       | Mechanism               |
| ------------ | ------------ | ----------------------- |
| chat         | sft          | flatten_messages        |
| sft          | chat         | unflatten_to_messages   |
| dpo          | sft          | drop_rejected           |
| any          | identity     | identity (noop)         |

Any other pair (e.g. chat → dpo, tabular-csv → anything, dpo → chat)
raises ``UnsupportedTransformError``. chat → dpo specifically requires
human-provided chosen/rejected pairings the agent cannot infer.
"""
from __future__ import annotations

from typing import Any, Iterable

# Role → chat-template markers used by flatten_messages /
# unflatten_to_messages. We keep a single canonical dialect
# (<|role|> ... <|end|>) so round-tripping a single row is lossless and
# agnostic of whatever tokenizer a downstream trainer applies later.
_ROLE_MARKERS = {
    "system":    "<|system|>",
    "user":      "<|user|>",
    "assistant": "<|assistant|>",
}
_END_MARKER = "<|end|>"


class UnsupportedTransformError(ValueError):
    """Raised when (source_schema, target_schema) has no defined mechanism.

    The message echoes back the matrix row verbatim so the caller (Lambda
    handler pre-flight, or the container) can surface the refusal to the
    agent unchanged.
    """


def infer_source_schema(columns: Iterable[str]) -> str:
    """Infer the source schema from a dataset's top-level column names.

    Returns one of: ``chat``, ``sft``, ``dpo``, ``tabular-csv``, or
    ``"unknown"`` when no heuristic matches. ``unknown`` is *not* an
    error — callers should ask the user to pass ``source_schema``
    explicitly rather than guess.
    """
    cols = {c.lower() for c in columns}
    if "messages" in cols:
        return "chat"
    if {"prompt", "chosen", "rejected"}.issubset(cols):
        return "dpo"
    if "text" in cols:
        return "sft"
    # Anything else with more than one column we call tabular; single-column
    # datasets without ``text``/``messages`` are ambiguous, return unknown.
    if len(cols) >= 2:
        return "tabular-csv"
    return "unknown"


def resolve_mechanism(source: str, target: str) -> str:
    """Return the transformation mechanism name for the pair, or raise.

    Args:
        source: inferred or caller-supplied source schema.
        target: the desired target schema.

    Returns:
        One of: ``identity``, ``flatten_messages``, ``unflatten_to_messages``,
        ``drop_rejected``.

    Raises:
        UnsupportedTransformError: for any pair not in the v1 matrix.
    """
    if source == target:
        return "identity"
    matrix = {
        ("chat", "sft"):       "flatten_messages",
        ("sft", "chat"):       "unflatten_to_messages",
        ("dpo", "sft"):        "drop_rejected",
    }
    mech = matrix.get((source, target))
    if mech is None:
        raise UnsupportedTransformError(
            f"No transformation mechanism for source_schema={source!r} → "
            f"target_schema={target!r}. Supported pairs: "
            f"{sorted(matrix.keys())} + identity."
        )
    return mech


# ── Mechanism implementations ──────────────────────────────────────────────


def _flatten_messages(row: dict[str, Any]) -> dict[str, Any]:
    """chat → sft: collapse the messages list into a single ``text`` column.

    Uses the canonical ``<|role|> content <|end|>`` template. Rows with no
    ``messages`` key are passed through unchanged so callers can decide
    whether to drop them (we return the original row verbatim; the eval
    container's downstream ``_normalise_record`` will raise if the row is
    malformed).
    """
    messages = row.get("messages")
    if not isinstance(messages, list):
        return row
    parts: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = msg.get("content") or ""
        marker = _ROLE_MARKERS.get(role, f"<|{role}|>")
        parts.append(f"{marker}\n{content}\n{_END_MARKER}")
    return {"text": "\n".join(parts)}


def _unflatten_to_messages(row: dict[str, Any]) -> dict[str, Any]:
    """sft → chat: split a ``text`` string on the role markers back into
    the messages list. Best-effort: text that was not produced by
    ``flatten_messages`` may not split cleanly; in that case we return a
    single ``user``-role message carrying the raw text.
    """
    text = row.get("text")
    if not isinstance(text, str) or not text:
        return row
    # Split on the end marker; each chunk looks like "<|role|>\n...".
    chunks = [c.strip() for c in text.split(_END_MARKER) if c.strip()]
    messages: list[dict[str, str]] = []
    for chunk in chunks:
        if not chunk.startswith("<|"):
            # Fallback: whole text was a single user turn that never had
            # markers inserted.
            return {"messages": [{"role": "user", "content": text}]}
        close = chunk.find("|>")
        if close < 0:
            return {"messages": [{"role": "user", "content": text}]}
        role = chunk[2:close]
        content = chunk[close + 2:].lstrip("\n")
        messages.append({"role": role, "content": content})
    return {"messages": messages}


def _drop_rejected(row: dict[str, Any]) -> dict[str, Any]:
    """dpo → sft: keep ``prompt`` + ``chosen``; drop ``rejected``. The
    chosen response becomes the SFT target, joined via the user/assistant
    markers so the result is consistent with ``flatten_messages`` output.
    """
    prompt = row.get("prompt") or ""
    chosen = row.get("chosen") or ""
    text = (
        f"{_ROLE_MARKERS['user']}\n{prompt}\n{_END_MARKER}\n"
        f"{_ROLE_MARKERS['assistant']}\n{chosen}\n{_END_MARKER}"
    )
    return {"text": text}


_MECHANISMS = {
    "identity":               lambda row: row,
    "flatten_messages":       _flatten_messages,
    "unflatten_to_messages":  _unflatten_to_messages,
    "drop_rejected":          _drop_rejected,
}


def apply_transform(rows: list[dict], mechanism: str) -> list[dict]:
    """Apply a named mechanism to every row in place. Unknown mechanism
    raises ``ValueError`` — the caller should always resolve the mechanism
    via ``resolve_mechanism`` first, so an unknown value means the handler
    and the container are out of sync.
    """
    fn = _MECHANISMS.get(mechanism)
    if fn is None:
        raise ValueError(
            f"Unknown mechanism={mechanism!r}. Known: {sorted(_MECHANISMS.keys())}."
        )
    return [fn(row) for row in rows]
