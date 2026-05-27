"""OpenAI Responses API protocol converter."""

from __future__ import annotations

import time
import logging
from typing import Optional

from app.models.deepseek import (
    DeepSeekMessage,
    DeepSeekRequest,
    DeepSeekResponse,
)
from app.models.openai_responses import (
    MessageOutputItem,
    OutputText,
    ReasoningOutputItem,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
    SummaryText,
)
from app.utils.sse import generate_id

logger = logging.getLogger(__name__)


def _flatten_content(content) -> Optional[str]:
    """Flatten content to a string for DeepSeek.

    Handles:
    - str -> pass through
    - list of content parts -> extract text from input_text/text/output_text parts
    - None -> None
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
                if part_type in ("input_text", "text", "output_text"):
                    text_parts.append(part.get("text", ""))
                elif "text" in part:
                    text_parts.append(part["text"])
        return "\n".join(text_parts) if text_parts else None

    # Fallback
    return str(content) if content else None


def _convert_input_item(item, messages: list):
    """Convert a single input item (dict or Pydantic model) to DeepSeekMessage(s).

    Preserves tool_calls, tool_call_id, and name fields for multi-turn
    function calling conversations.
    """
    if isinstance(item, dict):
        role = item.get("role", "user")
        content = _flatten_content(item.get("content"))
        tool_calls = item.get("tool_calls")
        tool_call_id = item.get("tool_call_id")
        name = item.get("name")
    elif hasattr(item, "role"):
        # Pydantic model (ResponseInputMessage)
        role = item.role
        content = _flatten_content(item.content)
        tool_calls = getattr(item, "tool_calls", None)
        tool_call_id = getattr(item, "tool_call_id", None)
        name = getattr(item, "name", None)
    else:
        logger.warning("Unknown input item type, skipping: %s", type(item).__name__)
        return

    # Build the message
    msg = DeepSeekMessage(role=role, content=content)

    # Preserve tool calling fields for multi-turn conversations
    if tool_calls:
        msg.tool_calls = tool_calls
    if tool_call_id:
        msg.tool_call_id = tool_call_id
    if name:
        msg.name = name

    messages.append(msg)


def convert_request(
    request: ResponsesRequest,
    target_model: str,
) -> DeepSeekRequest:
    """Convert an OpenAI Responses API request to a DeepSeek request.

    Args:
        request: The incoming Responses API request.
        target_model: The DeepSeek model name to use.

    Returns:
        DeepSeekRequest ready to send to the DeepSeek API.
    """
    messages = []

    # Add system instructions if present
    if request.instructions:
        messages.append(DeepSeekMessage(
            role="system",
            content=request.instructions,
        ))

    # Convert input
    if isinstance(request.input, str):
        # Simple string input -> single user message
        messages.append(DeepSeekMessage(
            role="user",
            content=request.input,
        ))
    elif isinstance(request.input, list):
        for item in request.input:
            _convert_input_item(item, messages)

    # Log summary for debugging
    msg_roles = [f"{m.role}({len(m.content) if m.content else 0}c)" for m in messages]
    logger.info(
        "Converted %d messages [%s], %d tools for DeepSeek (model=%s, stream=%s, max_tokens=%s)",
        len(messages),
        ", ".join(msg_roles),
        len(converted_tools) if converted_tools else 0,
        target_model,
        request.stream,
        request.max_output_tokens,
    )

    # Log warnings for unsupported features
    if request.previous_response_id:
        logger.warning("previous_response_id not supported, ignoring")
    # Convert tools from Responses API format to Chat Completions format
    # Responses API: {type: "function", name: "...", description: "...", input_schema: {...}}
    # Chat Completions: {type: "function", function: {name: "...", description: "...", parameters: {...}}}
    converted_tools = None
    converted_tool_choice = None
    if request.tools:
        converted_tools = []
        for tool in request.tools:
            tool_type = tool.get("type")
            if tool_type == "function":
                # Convert from Responses format to Chat Completions format
                converted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                    }
                }
                # Map input_schema -> parameters
                if "input_schema" in tool:
                    converted_tool["function"]["parameters"] = tool["input_schema"]
                elif "parameters" in tool:
                    converted_tool["function"]["parameters"] = tool["parameters"]
                # Map function key if already in that format
                if "function" in tool:
                    converted_tool["function"] = tool["function"]
                converted_tools.append(converted_tool)
            elif tool_type == "web_search_preview":
                logger.warning("web_search_preview tool not supported by DeepSeek, skipping")
            elif tool_type == "file_search":
                logger.warning("file_search tool not supported by DeepSeek, skipping")
            else:
                logger.warning("Unknown tool type '%s', skipping", tool_type)

        # Convert tool_choice from Responses format to Chat Completions format
        # Responses: {"type": "function", "name": "func_name"} 
        # Chat: {"type": "function", "function": {"name": "func_name"}}
        if request.tool_choice:
            if isinstance(request.tool_choice, dict):
                tc = dict(request.tool_choice)
                if "name" in tc and "function" not in tc:
                    converted_tool_choice = {
                        "type": tc.get("type", "function"),
                        "function": {"name": tc["name"]},
                    }
                else:
                    converted_tool_choice = tc
            else:
                converted_tool_choice = request.tool_choice

        logger.info("Converted %d tools for DeepSeek", len(converted_tools) if converted_tools else 0)

    return DeepSeekRequest(
        model=target_model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        stream=request.stream,
        tools=converted_tools if converted_tools else None,
        tool_choice=converted_tool_choice if converted_tools else None,
    )


def convert_response(
    deepseek_response: DeepSeekResponse,
    original_model: str,
    is_reasoner: bool = False,
) -> ResponsesResponse:
    """Convert a DeepSeek response to an OpenAI Responses API response.

    Args:
        deepseek_response: The response from DeepSeek.
        original_model: The original model name requested by the client.
        is_reasoner: Whether the target model is a reasoner (R1).

    Returns:
        ResponsesResponse in OpenAI Responses format.
    """
    output_items = []

    if deepseek_response.choices:
        choice = deepseek_response.choices[0]
        message = choice.message

        # Add reasoning output item if present (R1 models)
        if is_reasoner and message.reasoning_content:
            reasoning_id = generate_id("rs")
            output_items.append(ReasoningOutputItem(
                id=reasoning_id,
                summary=[SummaryText(text=message.reasoning_content)],
            ))

        # Add message output item
        msg_id = generate_id("msg")
        content_parts = []
        if message.content:
            content_parts.append(OutputText(text=message.content))

        output_items.append(MessageOutputItem(
            id=msg_id,
            content=content_parts,
        ))

    # Build usage
    usage = None
    if deepseek_response.usage:
        # Estimate reasoning tokens for R1
        reasoning_tokens = 0
        if is_reasoner and deepseek_response.choices:
            reasoning = deepseek_response.choices[0].message.reasoning_content or ""
            # Rough heuristic: ~4 chars per token
            reasoning_tokens = len(reasoning) // 4

        usage = ResponsesUsage(
            input_tokens=deepseek_response.usage.prompt_tokens,
            output_tokens=deepseek_response.usage.completion_tokens,
            output_tokens_details={"reasoning_tokens": reasoning_tokens} if reasoning_tokens else None,
            total_tokens=deepseek_response.usage.total_tokens,
        )

    # Ensure ID has resp_ prefix
    resp_id = deepseek_response.id
    if not resp_id.startswith("resp_"):
        resp_id = f"resp_{resp_id}"

    return ResponsesResponse(
        id=resp_id,
        created_at=deepseek_response.created or int(time.time()),
        model=original_model,
        output=output_items,
        usage=usage,
    )
