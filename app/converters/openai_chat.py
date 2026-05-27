"""OpenAI Chat Completions protocol converter.

Handles Xcode agent compatibility:
- Content arrays ([{"type": "text", "text": "..."}]) → string
- image_url content parts → discarded (DeepSeek is text-only)
- tool_calls / tool_call_id / name fields → preserved for DeepSeek function calling
- tool role messages → content flattened to string
"""

from __future__ import annotations

import json
import time
import logging
from typing import Optional

from app.config import get_settings
from app.models.deepseek import (
    DeepSeekChoice,
    DeepSeekMessage,
    DeepSeekMessageResponse,
    DeepSeekRequest,
    DeepSeekResponse,
)
from app.models.openai_chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)
from app.utils.sse import generate_id

logger = logging.getLogger(__name__)


def _flatten_content(content) -> Optional[str]:
    """Flatten content to a string that DeepSeek accepts.

    Handles these formats:
    - str → pass through
    - list of content parts → extract text, discard image_url
    - None → None

    Xcode sends content as arrays like:
        [{"type": "text", "text": "Hello"}, {"type": "image_url", "image_url": {...}}]
    DeepSeek requires content as a plain string.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type", "")
                if part_type == "text":
                    text_parts.append(part.get("text", ""))
                elif part_type == "image_url":
                    # DeepSeek doesn't support vision — log and skip
                    logger.debug("Discarding image_url content part (DeepSeek is text-only)")
                else:
                    # Unknown type — try to extract text if present
                    if "text" in part:
                        text_parts.append(part["text"])
        return "\n".join(text_parts) if text_parts else None

    # Fallback: convert to string
    return str(content)


def convert_request(
    request: ChatCompletionRequest,
    target_model: str,
) -> DeepSeekRequest:
    """Convert an OpenAI Chat request to a DeepSeek request.

    Handles Xcode agent compatibility:
    - Flattens content arrays to strings
    - Preserves tool_calls, tool_call_id, name for function calling
    - Passes through tools definition for DeepSeek function calling

    Args:
        request: The incoming OpenAI Chat request.
        target_model: The DeepSeek model name to use.

    Returns:
        DeepSeekRequest ready to send to the DeepSeek API.
    """
    messages = []
    for msg in request.messages:
        # Flatten content array to string (Xcode compatibility)
        content = _flatten_content(msg.content)

        # Build message with all relevant fields
        ds_msg = DeepSeekMessage(
            role=msg.role,
            content=content,
        )

        # Preserve tool calling fields
        if msg.tool_calls:
            ds_msg.tool_calls = msg.tool_calls
        if msg.tool_call_id:
            ds_msg.tool_call_id = msg.tool_call_id
        if msg.name:
            ds_msg.name = msg.name

        # Copy any extra fields from model_extra
        if hasattr(msg, "model_extra") and msg.model_extra:
            for key, value in msg.model_extra.items():
                if key not in ("role", "content", "tool_calls", "tool_call_id",
                               "name", "reasoning_content"):
                    setattr(ds_msg, key, value)

        messages.append(ds_msg)

    # Map max_completion_tokens -> max_tokens
    max_tokens = request.max_tokens or request.max_completion_tokens

    # Build stop sequences
    stop = request.stop

    # Build DeepSeek request — pass tools through (DeepSeek supports function calling)
    deepseek_req = DeepSeekRequest(
        model=target_model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=max_tokens,
        stream=request.stream,
        stop=stop,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        n=request.n,
        tools=request.tools,
        tool_choice=request.tool_choice,
        response_format=request.response_format,
    )

    # Copy any extra request fields
    if hasattr(request, "model_extra") and request.model_extra:
        for key, value in request.model_extra.items():
            if key not in DeepSeekRequest.model_fields:
                setattr(deepseek_req, key, value)

    return deepseek_req


def convert_response(
    deepseek_response: DeepSeekResponse,
    original_model: str,
    is_reasoner: bool = False,
) -> ChatCompletionResponse:
    """Convert a DeepSeek response to an OpenAI Chat response.

    Args:
        deepseek_response: The response from DeepSeek.
        original_model: The original model name requested by the client.
        is_reasoner: Whether the target model is a reasoner (R1).

    Returns:
        ChatCompletionResponse in OpenAI format.
    """
    settings = get_settings()
    reasoning_mode = settings.gateway.reasoning_mode

    choices = []
    for choice in deepseek_response.choices:
        message = ChatCompletionMessage(
            role=choice.message.role or "assistant",
            content=_convert_content(choice.message, reasoning_mode, is_reasoner),
            tool_calls=choice.message.tool_calls,
        )
        choices.append(ChatCompletionChoice(
            index=choice.index,
            message=message,
            finish_reason=choice.finish_reason,
        ))

    usage = None
    if deepseek_response.usage:
        usage = ChatCompletionUsage(
            prompt_tokens=deepseek_response.usage.prompt_tokens,
            completion_tokens=deepseek_response.usage.completion_tokens,
            total_tokens=deepseek_response.usage.total_tokens,
        )

    return ChatCompletionResponse(
        id=deepseek_response.id,
        created=deepseek_response.created or int(time.time()),
        model=original_model,  # Return original model name, not DeepSeek's
        choices=choices,
        usage=usage,
    )


def _convert_content(
    message: DeepSeekMessageResponse,
    reasoning_mode: str,
    is_reasoner: bool,
) -> Optional[str]:
    """Convert message content based on reasoning mode.

    Args:
        message: The DeepSeek response message.
        reasoning_mode: How to handle reasoning_content ("drop", "prepend", "custom_field").
        is_reasoner: Whether the target model is a reasoner.

    Returns:
        The content string (or None).
    """
    content = message.content
    reasoning = message.reasoning_content

    if not reasoning or not is_reasoner:
        return content

    if reasoning_mode == "prepend":
        settings = get_settings()
        marker = settings.gateway.reasoning_prepend_marker
        if content:
            return f"{marker}{content}"
        return marker

    # Default: "drop" - just return content without reasoning
    return content
