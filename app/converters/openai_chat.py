"""OpenAI Chat Completions protocol converter."""

from __future__ import annotations

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


def convert_request(
    request: ChatCompletionRequest,
    target_model: str,
) -> DeepSeekRequest:
    """Convert an OpenAI Chat request to a DeepSeek request.

    Args:
        request: The incoming OpenAI Chat request.
        target_model: The DeepSeek model name to use.

    Returns:
        DeepSeekRequest ready to send to the DeepSeek API.
    """
    # Convert messages - strip reasoning_content from any message
    messages = []
    for msg in request.messages:
        content = msg.content
        # Handle content that might be a list of content parts
        if isinstance(content, list):
            # Extract text from content parts
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts) if text_parts else None

        # Strip reasoning_content from incoming messages (DeepSeek doesn't accept it in history)
        messages.append(DeepSeekMessage(
            role=msg.role,
            content=content,
        ))

    # Map max_completion_tokens -> max_tokens
    max_tokens = request.max_tokens or request.max_completion_tokens

    # Build stop sequences
    stop = request.stop

    # Build DeepSeek request
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
