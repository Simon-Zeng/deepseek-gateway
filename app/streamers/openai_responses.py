"""OpenAI Responses API streaming converter."""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from typing import AsyncIterator, Optional

from app.models.common import ModelType
from app.utils.sse import format_sse_done, format_sse_event, generate_id

logger = logging.getLogger(__name__)


class StreamPhase(str, Enum):
    """State machine phases for streaming."""

    IDLE = "idle"
    REASONING = "reasoning"
    CONTENT = "content"
    DONE = "done"


async def stream_openai_responses(
    deepseek_stream: AsyncIterator[dict],
    original_model: str,
    model_type: ModelType,
) -> AsyncIterator[str]:
    """Convert a DeepSeek SSE stream to OpenAI Responses API SSE format.

    The Responses API uses a different SSE event format with named events
    and a specific lifecycle:
    - response.created
    - response.in_progress
    - response.output_item.added
    - response.content_part.added
    - response.output_text.delta (or response.reasoning.delta)
    - response.output_text.done
    - response.content_part.done
    - response.output_item.done
    - response.completed

    Args:
        deepseek_stream: Async iterator of parsed SSE chunks from DeepSeek.
        original_model: The original model name requested by the client.
        model_type: Whether the model is a chat or reasoner.

    Yields:
        Formatted SSE event strings.
    """
    is_reasoner = model_type == ModelType.REASONER
    response_id = f"resp_{generate_id()}"
    reasoning_id = f"rs_{generate_id()}"
    message_id = f"msg_{generate_id()}"
    created_at = int(time.time())
    phase = StreamPhase.IDLE
    accumulated_reasoning = ""
    accumulated_content = ""
    seq = 0

    # Emit initial events
    yield format_sse_event({
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "queued",
            "model": original_model,
            "output": [],
        },
    })

    yield format_sse_event({
        "type": "response.in_progress",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": original_model,
            "output": [],
        },
    })

    async for chunk in deepseek_stream:
        if not chunk.get("choices"):
            continue

        for choice in chunk["choices"]:
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Handle reasoning_content
            reasoning_text = delta.get("reasoning_content")
            if reasoning_text is not None:
                if phase == StreamPhase.IDLE:
                    phase = StreamPhase.REASONING
                    # Emit reasoning output item added
                    yield format_sse_event({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": reasoning_id,
                            "summary": [],
                        },
                    })
                    yield format_sse_event({
                        "type": "response.content_part.added",
                        "part": {"type": "summary_text", "text": ""},
                        "output_index": 0,
                        "content_index": 0,
                    })

                accumulated_reasoning += reasoning_text
                # Emit reasoning delta
                yield format_sse_event({
                    "type": "response.reasoning.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": reasoning_text,
                })

            # Handle content
            content_text = delta.get("content")
            if content_text is not None:
                # Transition from reasoning to content
                if phase == StreamPhase.REASONING:
                    # Close reasoning
                    yield format_sse_event({
                        "type": "response.reasoning.done",
                        "output_index": 0,
                        "content_index": 0,
                        "text": accumulated_reasoning,
                    })
                    yield format_sse_event({
                        "type": "response.content_part.done",
                        "part": {"type": "summary_text", "text": accumulated_reasoning},
                        "output_index": 0,
                        "content_index": 0,
                    })
                    yield format_sse_event({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": reasoning_id,
                            "summary": [{"type": "summary_text", "text": accumulated_reasoning}],
                        },
                    })

                    phase = StreamPhase.CONTENT

                    # Add message output item
                    msg_output_index = 1 if is_reasoner else 0
                    yield format_sse_event({
                        "type": "response.output_item.added",
                        "output_index": msg_output_index,
                        "item": {
                            "type": "message",
                            "id": message_id,
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    })
                    yield format_sse_event({
                        "type": "response.content_part.added",
                        "part": {"type": "output_text", "text": ""},
                        "output_index": msg_output_index,
                        "content_index": 0,
                    })

                elif phase == StreamPhase.IDLE:
                    # No reasoning, start content directly
                    phase = StreamPhase.CONTENT
                    yield format_sse_event({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": message_id,
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    })
                    yield format_sse_event({
                        "type": "response.content_part.added",
                        "part": {"type": "output_text", "text": ""},
                        "output_index": 0,
                        "content_index": 0,
                    })

                accumulated_content += content_text
                msg_output_index = 1 if is_reasoner and phase == StreamPhase.CONTENT else 0

                yield format_sse_event({
                    "type": "response.output_text.delta",
                    "output_index": msg_output_index,
                    "content_index": 0,
                    "delta": content_text,
                })

            # Handle finish
            if finish_reason:
                msg_output_index = 1 if is_reasoner and accumulated_reasoning else 0

                # Close content if not already done
                if phase in (StreamPhase.CONTENT, StreamPhase.REASONING):
                    # If still in reasoning (no content came), close it
                    if phase == StreamPhase.REASONING:
                        yield format_sse_event({
                            "type": "response.reasoning.done",
                            "output_index": 0,
                            "content_index": 0,
                            "text": accumulated_reasoning,
                        })
                        yield format_sse_event({
                            "type": "response.content_part.done",
                            "part": {"type": "summary_text", "text": accumulated_reasoning},
                            "output_index": 0,
                            "content_index": 0,
                        })
                        yield format_sse_event({
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {
                                "type": "reasoning",
                                "id": reasoning_id,
                                "summary": [{"type": "summary_text", "text": accumulated_reasoning}],
                            },
                        })

                    # Close message output
                    yield format_sse_event({
                        "type": "response.output_text.done",
                        "output_index": msg_output_index,
                        "content_index": 0,
                        "text": accumulated_content,
                    })
                    yield format_sse_event({
                        "type": "response.content_part.done",
                        "part": {"type": "output_text", "text": accumulated_content},
                        "output_index": msg_output_index,
                        "content_index": 0,
                    })
                    yield format_sse_event({
                        "type": "response.output_item.done",
                        "output_index": msg_output_index,
                        "item": {
                            "type": "message",
                            "id": message_id,
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": accumulated_content, "annotations": []}],
                        },
                    })

                phase = StreamPhase.DONE

                # Get usage from the chunk if available
                usage_data = chunk.get("usage", {})

                # Emit completion
                yield format_sse_event({
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created_at,
                        "status": "completed",
                        "model": original_model,
                        "output": _build_output(
                            is_reasoner, accumulated_reasoning, reasoning_id,
                            accumulated_content, message_id,
                        ),
                        "usage": {
                            "input_tokens": usage_data.get("prompt_tokens", 0),
                            "output_tokens": usage_data.get("completion_tokens", 0),
                            "total_tokens": usage_data.get("total_tokens", 0),
                        },
                    },
                })

    # If stream ended without finish_reason
    if phase != StreamPhase.DONE:
        yield format_sse_event({
            "type": "response.failed",
            "response": {
                "id": response_id,
                "status": "failed",
                "model": original_model,
            },
        })

    yield format_sse_done()


def _build_output(
    is_reasoner: bool,
    reasoning: str,
    reasoning_id: str,
    content: str,
    message_id: str,
) -> list:
    """Build the output array for the completed response."""
    output = []
    if is_reasoner and reasoning:
        output.append({
            "type": "reasoning",
            "id": reasoning_id,
            "summary": [{"type": "summary_text", "text": reasoning}],
        })
    output.append({
        "type": "message",
        "id": message_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    })
    return output
