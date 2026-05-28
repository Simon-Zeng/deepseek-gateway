"""Tool call ID normalization utilities for cross-format compatibility.

Anthropic uses the ``toolu_`` prefix for tool call IDs.
DeepSeek and OpenAI Chat/Responses use the ``call_`` prefix.

These functions normalize between the two schemes.
"""

from __future__ import annotations

from app.utils.sse import generate_id


def to_toolu(tc_id: str) -> str:
    """Normalize tool call ID to Anthropic ``toolu_`` format.

    DeepSeek uses ``call_`` prefix, Anthropic uses ``toolu_`` prefix.

    Args:
        tc_id: Raw tool call ID (may be empty, ``call_xxx``, ``toolu_xxx``, or other).

    Returns:
        An Anthropic-style ID with ``toolu_`` prefix.
    """
    if not tc_id:
        return f"toolu_{generate_id()}"
    if tc_id.startswith("call_"):
        return f"toolu_{tc_id[5:]}"
    if not tc_id.startswith("toolu_"):
        return f"toolu_{tc_id}"
    return tc_id


def to_call(tc_id: str) -> str:
    """Normalize tool call ID to ``call_`` format (DeepSeek / OpenAI).

    Handles these cases:
    - empty/missing → generates new ``call_xxx`` id
    - ``toolu_xxx`` → converts to ``call_xxx`` (Anthropic → OpenAI format)
    - ``call_xxx`` → pass through
    - other → prepends ``call_``

    Args:
        tc_id: Raw tool call ID.

    Returns:
        An ID with ``call_`` prefix.
    """
    if not tc_id:
        return f"call_{generate_id()}"
    if tc_id.startswith("toolu_"):
        return f"call_{tc_id[6:]}"
    if not tc_id.startswith("call_"):
        return f"call_{tc_id}"
    return tc_id
