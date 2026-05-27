"""Anthropic Messages API protocol converter.

Handles Xcode agent compatibility:
- Anthropic tool_use blocks → DeepSeek tool_calls format
- Anthropic tool_result blocks → DeepSeek tool role messages
- Content arrays → flattened strings where needed
- image blocks → discarded (DeepSeek is text-only)
"""

from __future__ import annotations

import json
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

    Handles:
    - System message extraction
    - tool_use content blocks → DeepSeek tool_calls format
    - tool_result content blocks → DeepSeek tool role messages
    - Content arrays → flattened strings

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
        # Check if message contains tool_use blocks (assistant role)
        # Anthropic puts tool calls inside content array; DeepSeek uses tool_calls field
        tool_calls, text_content = _extract_tool_calls_and_text(msg.content)

        if tool_calls:
            # Assistant message with tool calls → DeepSeek format
            messages.append(DeepSeekMessage(
                role=msg.role,
                content=text_content,
                tool_calls=tool_calls,
            ))
        elif msg.role == "user" and _has_tool_result(msg.content):
            # User message with tool_result → DeepSeek tool role messages
            for tc_msg in _convert_tool_results(msg.content):
                messages.append(tc_msg)
        else:
            # Regular message — flatten content to string
            content = _extract_text_content(msg.content)
            messages.append(DeepSeekMessage(
                role=msg.role,
                content=content,
            ))

    # Map Anthropic-specific fields to DeepSeek equivalents
    # Convert Anthropic tool definitions to OpenAI format if present
    deepseek_tools = _convert_tools(request.tools) if request.tools else None

    return DeepSeekRequest(
        model=target_model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
        stop=request.stop_sequences,
        tools=deepseek_tools,
        tool_choice=_convert_tool_choice(request.tool_choice) if deepseek_tools else None,
    )


def convert_response(
    deepseek_response: DeepSeekResponse,
    original_model: str,
    is_reasoner: bool = False,
) -> AnthropicResponse:
    """Convert a DeepSeek response to an Anthropic Messages API response.

    Handles:
    - DeepSeek tool_calls → Anthropic tool_use content blocks
    - reasoning_content → thinking blocks
    - finish_reason mapping (tool_calls → tool_use)

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

        # Add tool_use blocks from DeepSeek tool_calls
        if message.tool_calls:
            for tc in message.tool_calls:
                tc_id = tc.get("id", f"toolu_{generate_id()}")
                tc_function = tc.get("function", {})
                tc_name = tc_function.get("name", "")
                tc_input = tc_function.get("arguments", {})
                # Parse arguments if it's a string
                if isinstance(tc_input, str):
                    try:
                        tc_input = json.loads(tc_input)
                    except json.JSONDecodeError:
                        tc_input = {"raw": tc_input}

                content_blocks.append({
                    "type": "tool_use",
                    "id": tc_id,
                    "name": tc_name,
                    "input": tc_input,
                })

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


def _extract_tool_calls_and_text(content) -> tuple[Optional[list[dict]], Optional[str]]:
    """Extract tool_calls and text from Anthropic content array.

    Anthropic puts tool_use inside content blocks:
        [{"type": "tool_use", "id": "...", "name": "...", "input": {...}}]
    DeepSeek uses the tool_calls field:
        {"tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]}

    Returns:
        Tuple of (tool_calls list or None, text content or None)
    """
    if content is None or isinstance(content, str):
        return None, content

    if not isinstance(content, list):
        return None, _extract_text_content(content)

    tool_calls = []
    text_parts = []

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "")
            if block_type == "tool_use":
                # Convert Anthropic tool_use → OpenAI tool_calls format
                tool_calls.append({
                    "id": block.get("id", f"call_{generate_id()}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                })
            elif block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "thinking":
                # Skip thinking blocks from previous messages
                pass
            elif block_type in ("image",):
                logger.debug("Discarding image content block (DeepSeek is text-only)")
        elif isinstance(block, str):
            text_parts.append(block)

    text = "\n".join(text_parts) if text_parts else None
    return (tool_calls if tool_calls else None), text


def _has_tool_result(content) -> bool:
    """Check if content contains tool_result blocks."""
    if content is None or isinstance(content, str):
        return False
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return True
    return False


def _convert_tool_results(content) -> list[DeepSeekMessage]:
    """Convert Anthropic tool_result blocks to DeepSeek tool role messages.

    Anthropic tool_result:
        {"type": "tool_result", "tool_use_id": "...", "content": "result text"}
    DeepSeek tool message:
        {"role": "tool", "tool_call_id": "...", "content": "result text"}

    There can be multiple tool_results in a single user message.
    """
    results = []
    if not isinstance(content, list):
        return results

    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tool_use_id = block.get("tool_use_id", "")
            # Extract text from result content
            result_content = block.get("content")
            result_text = _extract_text_content(result_content)

            results.append(DeepSeekMessage(
                role="tool",
                tool_call_id=tool_use_id,
                content=result_text,
            ))
        elif isinstance(block, dict) and block.get("type") == "text":
            # Regular text in between tool results — include as user message
            text = block.get("text", "")
            if text:
                results.append(DeepSeekMessage(
                    role="user",
                    content=text,
                ))

    return results


def _convert_tools(tools: Optional[list[dict]]) -> Optional[list[dict]]:
    """Convert Anthropic tool definitions to OpenAI format.

    Anthropic:
        {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI/DeepSeek:
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    if not tools:
        return None

    converted = []
    for tool in tools:
        # Already in OpenAI format?
        if tool.get("type") == "function":
            converted.append(tool)
            continue

        # Anthropic format — convert
        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("input_schema", {})

        converted.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema,
            },
        })

    return converted


def _convert_tool_choice(tool_choice) -> Optional[dict | str]:
    """Convert Anthropic tool_choice to DeepSeek format.

    Anthropic: {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "..."}
    DeepSeek: "auto" | "required" | {"type": "function", "function": {"name": "..."}}
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        # Already in OpenAI format
        return tool_choice

    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type", "")
        if tc_type == "auto":
            return "auto"
        elif tc_type == "any":
            return "required"
        elif tc_type == "tool":
            return {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }

    return "auto"


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
                    logger.debug("Discarding image content block (DeepSeek is text-only)")
                elif block_type == "tool_use":
                    # Skip tool_use blocks (handled separately)
                    pass
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
