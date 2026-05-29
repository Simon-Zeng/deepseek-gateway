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
from app.services.model_mapper import map_reasoning_effort
from app.utils.sse import generate_id
from app.utils.tool_call_ids import to_toolu, to_call

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
        # Map role: Anthropic uses "developer" but DeepSeek expects "system"
        role = msg.role
        if role == "developer":
            role = "system"

        # Check if message contains tool_use blocks (assistant role)
        # Anthropic puts tool calls inside content array; DeepSeek uses tool_calls field
        tool_calls, text_content, reasoning = _extract_tool_calls_and_text(msg.content)

        if tool_calls:
            # Assistant message with tool calls → DeepSeek format
            # DeepSeek supports BOTH content and tool_calls in one message.
            # Anthropic messages can have text blocks alongside tool_use blocks —
            # preserve the text content instead of discarding it.
            ds_msg = DeepSeekMessage(
                role=role,
                content=text_content,
                tool_calls=tool_calls,
            )
            # In thinking mode, DeepSeek requires reasoning_content to be passed
            # back verbatim in multi-turn conversations.
            if reasoning:
                ds_msg.reasoning_content = reasoning
            messages.append(ds_msg)
        elif role == "user" and _has_tool_result(msg.content):
            # User message with tool_result → DeepSeek tool role messages
            for tc_msg in _convert_tool_results(msg.content):
                messages.append(tc_msg)
        else:
            # Regular message — flatten content to string.
            # In thinking mode, DeepSeek requires reasoning_content to be
            # passed back verbatim on assistant messages in multi-turn.
            content = _extract_text_content(msg.content)
            ds_msg = DeepSeekMessage(
                role=role,
                content=content,
            )
            if reasoning and role == "assistant":
                ds_msg.reasoning_content = reasoning
            messages.append(ds_msg)

    # Map Anthropic-specific fields to DeepSeek equivalents
    # Convert Anthropic tool definitions to OpenAI format if present
    deepseek_tools = _convert_tools(request.tools) if request.tools else None

    deepseek_req = DeepSeekRequest(
        model=target_model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
        stop=request.stop_sequences if request.stop_sequences else None,
        tools=deepseek_tools,
        tool_choice=_convert_tool_choice(request.tool_choice) if deepseek_tools else None,
    )

    # ── Forward thinking config to DeepSeek ──
    # Anthropic has two thinking formats (both in model_extra since Pydantic
    # doesn't define them as formal fields):
    #
    # Old format (deprecated on Opus 4.7):
    #   thinking: {"type": "enabled", "budget_tokens": <int>}
    #
    # New format (recommended for all Opus 4.6+ and Sonnet 4.6+):
    #   thinking: {"type": "adaptive"}
    #   output_config: {"effort": "low"|"medium"|"high"|"max"}
    extra = request.model_extra or {}
    thinking_config = extra.get("thinking")
    if isinstance(thinking_config, dict):
        thinking_type = thinking_config.get("type")
        if thinking_type == "enabled":
            # Old format: determine effort from budget_tokens
            budget = thinking_config.get("budget_tokens", 0)
            raw_effort = "high" if budget >= 10000 else "low"
            ds_effort = map_reasoning_effort(raw_effort)
            deepseek_req.reasoning_effort = ds_effort
            deepseek_req.thinking = {"type": "enabled"}
            logger.debug(
                "Forwarded Anthropic thinking (enabled, budget=%d) -> "
                "DeepSeek reasoning_effort=%s",
                budget, ds_effort,
            )
        elif thinking_type == "adaptive":
            # New format: extract effort from output_config (top-level extra)
            output_config = extra.get("output_config", {})
            raw_effort = output_config.get("effort", "high") if isinstance(output_config, dict) else "high"
            ds_effort = map_reasoning_effort(raw_effort)
            deepseek_req.reasoning_effort = ds_effort
            deepseek_req.thinking = {"type": "enabled"}
            logger.debug(
                "Forwarded Anthropic thinking (adaptive) + output_config.effort=%s -> "
                "DeepSeek reasoning_effort=%s",
                raw_effort, ds_effort,
            )

    return deepseek_req


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
                tc_id = to_toolu(tc.get("id", ""))
                tc_function = tc.get("function") or {}
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

    # DeepSeek returned no content — fall back to an empty text block.
    logger.warning("DeepSeek returned empty response with no content, reasoning, or tool calls")
    # Ensure at least one content block (Anthropic requires non-empty content)
    if not content_blocks:
        content_blocks.append(TextResponseBlock(text=""))

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


def _extract_tool_calls_and_text(content) -> tuple[Optional[list[dict]], Optional[str], Optional[str]]:
    """Extract tool_calls, text, and reasoning from Anthropic content array.

    Anthropic puts tool_use and thinking inside content blocks:
        [{"type": "thinking", "thinking": "..."},
         {"type": "tool_use", "id": "...", "name": "...", "input": {...}}]
    DeepSeek uses the tool_calls field and reasoning_content field:
        {"tool_calls": [{"id": "...", "type": "function", "function": {...}}],
         "reasoning_content": "..."}

    In thinking mode, DeepSeek requires ``reasoning_content`` to be passed
    back verbatim in multi-turn conversations. We extract it from the
    ``thinking`` blocks (joined with newlines if multiple) and return it as the third element.

    Returns:
        Tuple of (tool_calls list or None, text content or None, reasoning text or None)
    """
    if content is None or isinstance(content, str):
        return None, content, None

    if not isinstance(content, list):
        return None, _extract_text_content(content), None

    tool_calls = []
    text_parts = []
    reasoning_parts = []

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "")
            if block_type == "tool_use":
                # Convert Anthropic tool_use → OpenAI tool_calls format
                # Normalize the ID: Anthropic toolu_ → DeepSeek call_
                ds_id = to_call(block.get("id", ""))

                tool_calls.append({
                    "id": ds_id,
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                })
            elif block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "thinking":
                # Extract reasoning text to pass back to DeepSeek
                if "thinking" in block:
                    reasoning_parts.append(block["thinking"])
            elif block_type == "signature":
                # Skip signature blocks (extended thinking verification)
                pass
            elif block_type in ("image",):
                logger.debug("Discarding image content block (DeepSeek is text-only)")
        elif isinstance(block, str):
            text_parts.append(block)

    text = "\n".join(text_parts) if text_parts else None
    reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
    return (tool_calls if tool_calls else None), text, reasoning


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

    If text blocks are interleaved with tool_result blocks, the text is
    appended to the nearest tool_result's content (preceding the tool_result).
    This avoids inserting user messages between tool messages, which would
    violate DeepSeek API ordering constraints.
    """
    results = []
    if not isinstance(content, list):
        return results

    # First pass: collect all tool_results and text blocks
    tool_results = []
    pending_text_parts = []

    for block in content:
        if isinstance(block, str):
            # Raw string block — treat as text
            if block.strip():
                pending_text_parts.append(block)
        elif isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "tool_result":
                # Any pending text gets appended to the previous tool_result
                # or prepended to this one
                tool_use_id = to_call(block.get("tool_use_id") or "")

                result_content = block.get("content")
                result_text = _extract_text_content(result_content) or ""

                # If there's pending text, prepend it to this tool result
                if pending_text_parts:
                    prefix = "\n".join(pending_text_parts)
                    result_text = f"{prefix}\n{result_text}" if result_text else prefix
                    pending_text_parts = []

                tool_results.append(DeepSeekMessage(
                    role="tool",
                    tool_call_id=tool_use_id,
                    content=result_text,
                ))
            elif block_type == "text":
                # Accumulate text — will be merged with adjacent tool_result
                text = block.get("text", "")
                if text:
                    pending_text_parts.append(text)

    # Any remaining text after all tool_results → append as user message
    # This must come after all tool messages per DeepSeek API constraints
    if pending_text_parts:
        results.append(DeepSeekMessage(
            role="user",
            content="\n".join(pending_text_parts),
        ))

    # All tool results come first, then any trailing user message
    results = tool_results + results

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
        elif tc_type == "none":
            return "none"

    return "auto"


def _extract_text_content(content) -> Optional[str]:
    """Extract text content from Anthropic-style content (string or block array).

    NOTE: This function should only be called for messages that do NOT contain
    tool_result blocks. For messages with tool_result, use _convert_tool_results()
    instead. This function skips tool_result and tool_use blocks explicitly.

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
                    # Skip — handled separately by _convert_tool_results
                    pass
                elif block_type == "tool_use":
                    # Skip — handled separately by _extract_tool_calls_and_text
                    pass
                elif block_type == "thinking":
                    # Skip thinking blocks from previous messages
                    pass
                elif block_type == "signature":
                    # Skip signature blocks (extended thinking verification)
                    pass
                elif block_type in ("image",):
                    logger.debug("Discarding image content block (DeepSeek is text-only)")
                elif "text" in block:
                    # Unknown block type with text — extract it
                    text_parts.append(block["text"])
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
        "error": "end_turn",
    }
    return mapping.get(finish_reason, finish_reason)
