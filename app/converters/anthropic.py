"""Anthropic Messages API protocol converter."""

from __future__ import annotations

import logging
from typing import Optional

from app.models.deepseek import (
    DeepSeekMessage,
    DeepSeekRequest,
    DeepSeekResponse,
)
from app.models.anthropic import (
    AnthropicRequest,
    AnthropicResponse,
    AnthropicUsage,
    TextResponseBlock,
    ThinkingResponseBlock,
)
from app.utils.sse import generate_id

logger = logging.getLogger(__name__)


def convert_request(
    request: AnthropicRequest,
    target_model: str,
) -> DeepSeekRequest:
    """Convert an Anthropic Messages API request to a DeepSeek request.

    Args:
        request: The incoming Anthropic request.
        target_model: The DeepSeek model name to use.

    Returns:
        DeepSeekRequest ready to send to the DeepSeek API.
    """
    messages = []

    # Add system message if present
    if request.system:
        system_content = _extract_text_content(request.system)
        if system_content:
            messages.append(DeepSeekMessage(
                role="system",
                content=system_content,
            ))

    # Convert messages
    for msg in request.messages:
        content = _extract_text_content(msg.content)
        # Strip reasoning_content from incoming messages
        messages.append(DeepSeekMessage(
            role=msg.role,
            content=content,
        ))

    # Map Anthropic-specific fields to DeepSeek equivalents
    return DeepSeekRequest(
        model=target_model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
        stop=request.stop_sequences,
        tools=request.tools,
        tool_choice=request.tool_choice if request.tools else None,
    )


def convert_response(
    deepseek_response: DeepSeekResponse,
    original_model: str,
    is_reasoner: bool = False,
) -> AnthropicResponse:
    """Convert a DeepSeek response to an Anthropic Messages API response.

    Args:
        deepseek_response: The response from DeepSeek.
        original_model: The original model name requested by the client.
        is_reasoner: Whether the target model is a reasoner (R1).

    Returns:
        AnthropicResponse in Anthropic Messages format.
    """
    content_blocks = []

    if deepseek_response.choices:
        choice = deepseek_response.choices[0]
        message = choice.message

        # Add thinking block for R1 models with reasoning content
        if is_reasoner and message.reasoning_content:
            content_blocks.append(ThinkingResponseBlock(
                thinking=message.reasoning_content,
            ))

        # Add text content block
        if message.content:
            content_blocks.append(TextResponseBlock(
                text=message.content,
            ))

    # Map finish_reason
    stop_reason = None
    if deepseek_response.choices:
        finish = deepseek_response.choices[0].finish_reason
        stop_reason = _map_finish_reason(finish)

    # Map usage
    usage = AnthropicUsage()
    if deepseek_response.usage:
        usage = AnthropicUsage(
            input_tokens=deepseek_response.usage.prompt_tokens,
            output_tokens=deepseek_response.usage.completion_tokens,
        )

    return AnthropicResponse(
        id=deepseek_response.id or f"msg_{generate_id()}",
        model=original_model,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=usage,
    )


def _extract_text_content(content) -> Optional[str]:
    """Extract text content from Anthropic-style content (string or block array).

    Args:
        content: A string, list of content blocks, or None.

    Returns:
        Extracted text string, or None.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_result":
                    # Extract text from tool result content
                    result_content = block.get("content")
                    if isinstance(result_content, str):
                        text_parts.append(result_content)
                    elif isinstance(result_content, list):
                        for sub in result_content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                text_parts.append(sub.get("text", ""))
                elif block_type in ("image",):
                    logger.warning("Image content blocks not supported by DeepSeek, skipping")
                elif block_type == "tool_use":
                    logger.warning("Tool use content blocks not fully supported, skipping")
                elif block_type == "thinking":
                    # Skip thinking blocks from previous messages
                    pass
            elif hasattr(block, "type"):
                # Pydantic model
                if block.type == "text":
                    text_parts.append(block.text if hasattr(block, "text") else str(block))
        return "\n".join(text_parts) if text_parts else None
    return str(content)


def _map_finish_reason(finish_reason: Optional[str]) -> Optional[str]:
    """Map DeepSeek/OpenAI finish_reason to Anthropic stop_reason.

    Mapping:
    - "stop" -> "end_turn"
    - "length" -> "max_tokens"
    - "tool_calls" -> "tool_use"
    - Others -> pass through
    """
    if finish_reason is None:
        return None
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(finish_reason, finish_reason)
