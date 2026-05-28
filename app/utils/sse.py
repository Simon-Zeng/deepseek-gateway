"""SSE parsing and generation utilities."""

from __future__ import annotations

import json
import time
from typing import AsyncIterator, Optional

import httpx


async def parse_sse_stream(response: httpx.Response) -> AsyncIterator[dict]:
    """Parse an SSE stream from an httpx response, yielding parsed JSON chunks.

    Handles the standard SSE format:
        data: {"key": "value"}\\n\\n
        data: [DONE]\\n\\n

    Args:
        response: The httpx streaming response.

    Yields:
        Parsed JSON objects from each SSE data line.
    """
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")

            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    # Skip malformed JSON
                    continue
            elif line.startswith("data:"):
                # Handle "data:" without space
                data = line[5:]
                if data.strip() == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue
            # Ignore comment lines (starting with :) and event: lines for parsing

    # Process any remaining data in buffer
    if buffer.strip():
        remaining = buffer.strip()
        if remaining.startswith("data: "):
            data = remaining[6:]
            if data.strip() != "[DONE]":
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    pass


def format_sse_event(data: dict | str, event: Optional[str] = None) -> str:
    """Format data as an SSE event string.

    Args:
        data: The data to send (dict will be JSON-encoded, str used as-is).
        event: Optional SSE event type name.

    Returns:
        Formatted SSE string ending with double newline.
    """
    result = ""
    if event:
        result += f"event: {event}\n"

    if isinstance(data, dict):
        result += f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    else:
        result += f"data: {data}\n\n"

    return result


def format_sse_done() -> str:
    """Format the SSE stream termination signal."""
    return "data: [DONE]\n\n"


def format_sse_comment(comment: str = "") -> str:
    """Format an SSE comment (used for keep-alive)."""
    return f": {comment}\n\n"


_id_counter: int = 0


def generate_id(prefix: str = "chatcmpl") -> str:
    """Generate a unique-ish ID for responses with counter to avoid collisions."""
    global _id_counter
    _id_counter += 1
    return f"{prefix}-{int(time.time() * 1000)}-{_id_counter}"
