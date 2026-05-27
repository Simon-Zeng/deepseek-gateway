"""Anthropic Messages API streaming converter."""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from typing import AsyncIterator, Optional

from app.config import get_settings
from app.models.common import ModelType
from app.utils.sse import format_sse_done, format_sse_event, generate_id, format_sse_comment

logger = logging.getLogger(__name__)


class StreamPhase(str, Enum):
    """State machine phases for streaming."""

    IDLE = "idle"
    REASONING = "reasoning"
    CONTENT = "content"
    DONE = "done"


class AnthropicStreamState:
    """Mutable state for Anthropic streaming conversion.

    This avoids class-level attribute hacks and makes the state
    management explicit and clean.
    """

    def __init__(self, original_model: str, is_reasoner: bool, ping_interval: int):
        self.original_model = original_model
        self.is_reasoner = is_reasoner
        self.ping_interval = ping_interval
        self.message_id = f"msg_{generate_id()}"
        self.phase = StreamPhase.IDLE
        self.content_block_index = 0
        self.last_ping_time = time.time()
        self.input_tokens = 0
        self.output_tokens = 0
        self.text_block_started = False  # Track if we emitted content_block_start for text
        # Tool call tracking (for streaming tool_calls from DeepSeek)
        self.pending_tool_calls: dict[int, dict] = {}  # index -> accumulated tool call data


async def stream_anthropic(
    deepseek_stream: AsyncIterator[dict],
    original_model: str,
    model_type: ModelType,
) -> AsyncIterator[str]:
    """Convert a DeepSeek SSE stream to Anthropic Messages API SSE format.

    The Anthropic SSE format uses named events with specific types:
    - event: message_start
    - event: content_block_start
    - event: content_block_delta
    - event: content_block_stop
    - event: message_delta
    - event: message_stop
    - event: ping (keep-alive)

    For R1 models, reasoning_content is mapped to thinking blocks.

    Args:
        deepseek_stream: Async iterator of parsed SSE chunks from DeepSeek.
        original_model: The original model name requested by the client.
        model_type: Whether the model is a chat or reasoner.

    Yields:
        Formatted SSE event strings with event type headers.
    """
    settings = get_settings()
    state = AnthropicStreamState(
        original_model=original_model,
        is_reasoner=model_type == ModelType.REASONER,
        ping_interval=settings.gateway.ping_interval,
    )

    # ── Emit message_start ──
    yield format_sse_event(
        {
            "type": "message_start",
            "message": {
                "id": state.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": original_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 1},
            },
        },
        event="message_start",
    )

    # Initial ping
    yield format_sse_event({}, event="ping")

    async for chunk in deepseek_stream:
        # Keep-alive ping
        now = time.time()
        if now - state.last_ping_time >= state.ping_interval:
            yield format_sse_event({}, event="ping")
            state.last_ping_time = now

        # Extract usage if present
        if chunk.get("usage"):
            state.input_tokens = chunk["usage"].get("prompt_tokens", state.input_tokens)
            state.output_tokens = chunk["usage"].get("completion_tokens", state.output_tokens)

        if not chunk.get("choices"):
            continue

        for choice in chunk["choices"]:
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # ── Reasoning content ──
            reasoning_text = delta.get("reasoning_content")
            if reasoning_text is not None:
                async for event in _handle_reasoning(state, reasoning_text):
                    yield event

            # ── Tool calls (function calling) ──
            tool_calls = delta.get("tool_calls")
            if tool_calls is not None:
                async for event in _handle_tool_calls(state, tool_calls):
                    yield event

            # ── Text content ──
            content_text = delta.get("content")
            if content_text is not None:
                async for event in _handle_content(state, content_text):
                    yield event

            # ── Finish ──
            if finish_reason:
                async for event in _handle_finish(state, finish_reason, chunk):
                    yield event

    # Clean up: if stream ended without proper finish
    if state.phase != StreamPhase.DONE:
        if state.phase in (StreamPhase.REASONING, StreamPhase.CONTENT):
            yield format_sse_event(
                {"type": "content_block_stop", "index": state.content_block_index},
                event="content_block_stop",
            )
        yield format_sse_event(
            {
                "type": "error",
                "error": {"type": "api_error", "message": "Stream ended unexpectedly"},
            },
            event="error",
        )


async def _handle_reasoning(state: AnthropicStreamState, reasoning_text: str) -> AsyncIterator[str]:
    """Handle a reasoning_content chunk."""
    if state.phase == StreamPhase.IDLE:
        state.phase = StreamPhase.REASONING
        # Start thinking content block
        yield format_sse_event(
            {
                "type": "content_block_start",
                "index": state.content_block_index,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            event="content_block_start",
        )

    # Emit thinking delta
    yield format_sse_event(
        {
            "type": "content_block_delta",
            "index": state.content_block_index,
            "delta": {"type": "thinking_delta", "thinking": reasoning_text},
        },
        event="content_block_delta",
    )


async def _handle_tool_calls(state: AnthropicStreamState, tool_calls: list[dict]) -> AsyncIterator[str]:
    """Handle streaming tool_calls from DeepSeek, converting to Anthropic tool_use blocks.

    DeepSeek streams tool calls incrementally:
        delta.tool_calls[0] = {index: 0, id: "call_abc", function: {name: "get_weather", arguments: ""}}
        delta.tool_calls[0] = {index: 0, function: {arguments: '{"ci'}}
        delta.tool_calls[0] = {index: 0, function: {arguments: 'ty": "Beijing"}'}}

    Anthropic expects:
        event: content_block_start  {index: N, content_block: {type: "tool_use", id: "...", name: "...", input: {}}}
        event: content_block_delta  {index: N, delta: {type: "input_json_delta", partial_json: "..."}}
        event: content_block_stop   {index: N}
    """
    # Close any open text/reasoning block first
    if state.phase == StreamPhase.REASONING:
        yield format_sse_event(
            {"type": "content_block_stop", "index": state.content_block_index},
            event="content_block_stop",
        )
        state.content_block_index += 1
    elif state.phase == StreamPhase.CONTENT and state.text_block_started:
        yield format_sse_event(
            {"type": "content_block_stop", "index": state.content_block_index},
            event="content_block_stop",
        )
        state.content_block_index += 1

    state.phase = StreamPhase.CONTENT  # Reuse CONTENT phase for tool calls

    for tc_delta in tool_calls:
        tc_index = tc_delta.get("index", 0)

        # Initialize or update tool call data
        if tc_index not in state.pending_tool_calls:
            state.pending_tool_calls[tc_index] = {
                "id": tc_delta.get("id", f"toolu_{generate_id()}"),
                "name": "",
                "arguments": "",
                "started": False,
            }

        tc = state.pending_tool_calls[tc_index]

        # Update name if present
        func_delta = tc_delta.get("function", {})
        if func_delta.get("name"):
            tc["name"] = func_delta["name"]

        # Accumulate arguments
        if func_delta.get("arguments"):
            tc["arguments"] += func_delta["arguments"]

        # Update ID if present
        if tc_delta.get("id"):
            tc["id"] = tc_delta["id"]

        # Emit content_block_start on first chunk for this tool call
        if not tc["started"]:
            tc["started"] = True
            # Map tool call index to Anthropic content block index
            tc["block_index"] = state.content_block_index
            yield format_sse_event(
                {
                    "type": "content_block_start",
                    "index": state.content_block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": {},
                    },
                },
                event="content_block_start",
            )
            state.content_block_index += 1

        # Emit input_json_delta for arguments
        if func_delta.get("arguments"):
            yield format_sse_event(
                {
                    "type": "content_block_delta",
                    "index": tc.get("block_index", state.content_block_index - 1),
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": func_delta["arguments"],
                    },
                },
                event="content_block_delta",
            )



    """Handle a content chunk."""
    # Transition from reasoning to content
    if state.phase == StreamPhase.REASONING:
        # Close thinking block
        yield format_sse_event(
            {"type": "content_block_stop", "index": state.content_block_index},
            event="content_block_stop",
        )
        state.content_block_index += 1
        state.phase = StreamPhase.CONTENT
        state.text_block_started = False

    elif state.phase == StreamPhase.IDLE:
        state.phase = StreamPhase.CONTENT
        state.text_block_started = False

    # Start text content block on first content chunk
    if not state.text_block_started:
        state.text_block_started = True
        yield format_sse_event(
            {
                "type": "content_block_start",
                "index": state.content_block_index,
                "content_block": {"type": "text", "text": ""},
            },
            event="content_block_start",
        )

    # Emit text delta
    yield format_sse_event(
        {
            "type": "content_block_delta",
            "index": state.content_block_index,
            "delta": {"type": "text_delta", "text": content_text},
        },
        event="content_block_delta",
    )


async def _handle_finish(
    state: AnthropicStreamState,
    finish_reason: str,
    chunk: dict,
) -> AsyncIterator[str]:
    """Handle stream finish."""
    # Close any open content block
    if state.phase == StreamPhase.REASONING:
        # Close thinking block
        yield format_sse_event(
            {"type": "content_block_stop", "index": state.content_block_index},
            event="content_block_stop",
        )
        state.content_block_index += 1
    elif state.phase == StreamPhase.CONTENT:
        # Close text block if open
        if state.text_block_started:
            yield format_sse_event(
                {"type": "content_block_stop", "index": state.content_block_index},
                event="content_block_stop",
            )

        # Close any tool_use blocks that are still open
        for tc_index, tc in sorted(state.pending_tool_calls.items()):
            if tc.get("started"):
                yield format_sse_event(
                    {"type": "content_block_stop", "index": tc["block_index"]},
                    event="content_block_stop",
                )

    # Map finish_reason
    stop_reason = _map_finish_reason(finish_reason)

    # Update usage from final chunk
    if chunk.get("usage"):
        state.input_tokens = chunk["usage"].get("prompt_tokens", state.input_tokens)
        state.output_tokens = chunk["usage"].get("completion_tokens", state.output_tokens)

    # Emit message_delta
    yield format_sse_event(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": state.output_tokens},
        },
        event="message_delta",
    )

    # Emit message_stop
    yield format_sse_event(
        {"type": "message_stop"},
        event="message_stop",
    )

    state.phase = StreamPhase.DONE


def _map_finish_reason(finish_reason: Optional[str]) -> str:
    """Map DeepSeek/OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(finish_reason or "stop", "end_turn")
