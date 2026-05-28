"""OpenAI Chat Completions streaming converter."""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import AsyncIterator, Optional

from app.config import get_settings
from app.models.common import ModelType
from app.utils.sse import format_sse_done, format_sse_event

logger = logging.getLogger(__name__)


class StreamPhase(str, Enum):
    """State machine phases for streaming."""

    IDLE = "idle"
    REASONING = "reasoning"
    CONTENT = "content"
    DONE = "done"


async def stream_openai_chat(
    deepseek_stream: AsyncIterator[dict],
    original_model: str,
    model_type: ModelType,
    response_id: str,
    created: int,
) -> AsyncIterator[str]:
    """Convert a DeepSeek SSE stream to OpenAI Chat SSE format.

    This is a near pass-through since DeepSeek uses the same format.
    The main differences:
    - Replace model name with original
    - Handle reasoning_content based on REASONING_MODE setting

    Args:
        deepseek_stream: Async iterator of parsed SSE chunks from DeepSeek.
        original_model: The original model name requested by the client.
        model_type: Whether the model is a chat or reasoner.
        response_id: The response ID from the first chunk.
        created: The creation timestamp.

    Yields:
        Formatted SSE event strings.
    """
    settings = get_settings()
    reasoning_mode = settings.gateway.reasoning_mode
    is_reasoner = model_type == ModelType.REASONER
    phase = StreamPhase.IDLE
    prepend_marker = settings.gateway.reasoning_prepend_marker

    async for chunk in deepseek_stream:
        # Replace model name with original
        chunk["model"] = original_model

        # Use the ID from the first chunk if not set
        if not response_id and chunk.get("id"):
            response_id = chunk["id"]

        # Process choices
        if chunk.get("choices"):
            for choice in chunk["choices"]:
                delta = choice.get("delta", {})

                # Handle reasoning_content
                if delta.get("reasoning_content") is not None:
                    reasoning_text = delta.pop("reasoning_content")

                    if phase == StreamPhase.IDLE:
                        phase = StreamPhase.REASONING

                    if reasoning_mode == "drop":
                        # Skip this chunk's reasoning entirely
                        # But still check if there's content in the same chunk
                        if not delta.get("content") and not choice.get("finish_reason"):
                            continue
                    elif reasoning_mode == "prepend":
                        # Convert reasoning to content with marker
                        if phase == StreamPhase.REASONING:
                            # Check if there's also content in this same delta
                            existing_content = delta.get("content")
                            if existing_content:
                                # Both reasoning and content in same chunk — combine
                                delta["content"] = f"{prepend_marker}{reasoning_text}{existing_content}"
                            else:
                                delta["content"] = f"{prepend_marker}{reasoning_text}"
                            phase = StreamPhase.CONTENT
                        else:
                            # Subsequent reasoning after the first — just append as content
                            existing_content = delta.get("content")
                            if existing_content:
                                delta["content"] = f"{reasoning_text}{existing_content}"
                            else:
                                delta["content"] = reasoning_text
                    elif reasoning_mode == "custom_field":
                        # Include as non-standard field
                        delta["reasoning_content"] = reasoning_text

                # Track content phase
                if delta.get("content") is not None:
                    if phase == StreamPhase.IDLE:
                        phase = StreamPhase.CONTENT

                # tool_calls pass-through: DeepSeek and OpenAI use the same streaming
                # format for tool_calls (delta.tool_calls[].index/id/function), so no
                # conversion is needed — they are preserved in the delta and emitted
                # with the rest of the chunk after model name replacement above.

                # Track finish
                if choice.get("finish_reason"):
                    phase = StreamPhase.DONE

        # Yield the modified chunk
        yield format_sse_event(chunk)

    # Stream complete
    yield format_sse_done()
