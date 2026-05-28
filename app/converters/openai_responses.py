"""OpenAI Responses API protocol converter."""

from __future__ import annotations

import json
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
from app.utils.tool_call_ids import to_call

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
                elif part_type == "image_url":
                    # DeepSeek doesn't support vision — log and skip
                    logger.debug("Discarding image_url content part (DeepSeek is text-only)")
                elif part_type == "refusal":
                    # Treat refusal as text content with a marker to preserve semantics
                    refusal_text = part.get("refusal", "")
                    if refusal_text:
                        text_parts.append(f"[refusal: {refusal_text}]")
                elif "text" in part:
                    text_parts.append(part["text"])
        return "\n".join(text_parts) if text_parts else None

    # Fallback — should not normally be reached
    logger.warning("Unexpected content type %s, falling back to str()", type(content).__name__)
    return str(content) if content else None



def _get_item_attr(item, key, default=None):
    """Get an attribute from a dict or Pydantic model.

    When Pydantic parses the ``input`` list of a ResponsesRequest, items
    with ``type: "function_call"`` or ``type: "function_call_output"`` are
    still coerced into ``ResponseInputMessage`` models (because the union
    type includes ``list[ResponseInputMessage]`` and ``extra="allow"``).
    The original ``type``, ``call_id``, ``name``, ``arguments`` etc. are
    preserved as extra fields accessible via ``getattr()``.

    This helper abstracts the difference between dicts and Pydantic models
    so that ``_convert_input_item`` can handle both uniformly.
    """
    if isinstance(item, dict):
        return item.get(key, default)
    # Pydantic model — try getattr (covers model fields + extra fields)
    return getattr(item, key, default)


def _convert_input_item(item, messages: list, last_reasoning: list[str] | None = None):
    """Convert a single input item (dict or Pydantic model) to DeepSeekMessage(s).

    Handles these Responses API input item types:
    - "message": standard user/assistant/system message
    - "function_call": assistant tool call → DeepSeek tool_calls format
    - "function_call_output": tool result → DeepSeek tool role message
    - "reasoning": captures text for inclusion in the next assistant message
      (DeepSeek thinking mode requires ``reasoning_content`` to be passed
      back verbatim in multi-turn conversations)

    Works with both raw dicts and Pydantic ``ResponseInputMessage`` models.
    When Pydantic parses the ``input`` list, items with ``type: "function_call"``
    or ``type: "function_call_output"`` are still coerced into
    ``ResponseInputMessage`` models (extra fields preserved via ``extra="allow"``).
    We use ``_get_item_attr()`` to abstract dict vs model access.

    Args:
        item: A dict or Pydantic model from the Responses API input list.
        messages: Output list of DeepSeekMessage (mutated in place).
        last_reasoning: Optional mutable container for tracking reasoning text
            between items. ``last_reasoning[0]`` is consumed and cleared when
            a ``function_call`` item creates an assistant message.
    """
    item_type = _get_item_attr(item, "type")

    # ── function_call: assistant tool call ──
    # Responses API: {"type": "function_call", "id": "call_xxx", "call_id": "call_xxx",
    #                 "name": "func_name", "arguments": "{...}"}
    # DeepSeek: {"role": "assistant", "tool_calls": [{"id": "call_xxx", "type": "function",
    #             "function": {"name": "func_name", "arguments": "{...}"}}]}
    if item_type == "function_call":
        call_id = _get_item_attr(item, "call_id")
        if call_id is None:
            call_id = _get_item_attr(item, "id", "")
        func_name = _get_item_attr(item, "name", "")
        arguments = _get_item_attr(item, "arguments", "")
        # Ensure arguments is a string
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)

        ds_msg = DeepSeekMessage(
            role="assistant",
            content=None,
            tool_calls=[{
                "id": to_call(call_id),
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": arguments,
                },
            }],
        )

        # In thinking mode, DeepSeek requires reasoning_content to be passed
        # back verbatim in multi-turn conversations.
        if last_reasoning is not None and last_reasoning[0]:
            ds_msg.reasoning_content = last_reasoning[0]
            last_reasoning[0] = ""

        messages.append(ds_msg)
        return

    # ── function_call_output: tool result ──
    # Responses API: {"type": "function_call_output", "call_id": "call_xxx", "output": "result text"}
    # DeepSeek: {"role": "tool", "tool_call_id": "call_xxx", "content": "result text"}
    if item_type == "function_call_output":
        call_id = _get_item_attr(item, "call_id", "")
        output = _get_item_attr(item, "output", "")
        # output can be a string or a dict — flatten to string for DeepSeek
        if isinstance(output, dict):
            output = json.dumps(output, ensure_ascii=False)

        messages.append(DeepSeekMessage(
            role="tool",
            tool_call_id=to_call(call_id),
            content=output,
        ))
        return

    # ── reasoning: capture text for inclusion in next assistant message ──
    # DeepSeek thinking mode requires reasoning_content to be passed back
    # verbatim in multi-turn conversations. We capture it here and attach
    # it when the next function_call (assistant) item is processed.
    if item_type == "reasoning":
        if last_reasoning is not None:
            summary = _get_item_attr(item, "summary", [])
            if summary and isinstance(summary, list):
                text = summary[0].get("text", "") if isinstance(summary[0], dict) else ""
                if text:
                    last_reasoning[0] = text
        return

    # ── Regular message type ──
    if item_type and item_type not in ("message", None):
        logger.warning("Unexpected input item type '%s', treating as message", item_type)

    role = _get_item_attr(item, "role", "user")
    # Responses API uses "developer" role — map to "system" for Chat API
    if role == "developer":
        role = "system"

    content = _flatten_content(_get_item_attr(item, "content"))
    tool_calls = _get_item_attr(item, "tool_calls")
    tool_call_id = _get_item_attr(item, "tool_call_id")
    name = _get_item_attr(item, "name")

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
        last_reasoning = [""]  # Mutable container to track reasoning text across items
        for item in request.input:
            _convert_input_item(item, messages, last_reasoning)

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
                # If already in Chat Completions format, use directly
                if "function" in tool:
                    converted_tools.append(tool)
                    continue

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

    # Forward response_format from model_extra if present (Responses API supports it
    # as an extra parameter for JSON mode / structured output)
    response_format = None
    if hasattr(request, "model_extra") and request.model_extra:
        response_format = request.model_extra.get("response_format")

    deepseek_req = DeepSeekRequest(
        model=target_model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        stream=request.stream,
        tools=converted_tools if converted_tools else None,
        tool_choice=converted_tool_choice if converted_tools else None,
        response_format=response_format,
    )

    return deepseek_req


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
    finish_reason = None

    if deepseek_response.choices:
        choice = deepseek_response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason

        # Add reasoning output item if present (R1 models)
        if is_reasoner and message.reasoning_content:
            reasoning_id = generate_id("rs")
            output_items.append(ReasoningOutputItem(
                id=reasoning_id,
                summary=[SummaryText(text=message.reasoning_content)],
            ))

        # Add message output item first (before function_call items,
        # matching streaming output order)
        if message.content or not message.tool_calls:
            msg_id = generate_id("msg")
            content_parts = []
            if message.content:
                content_parts.append(OutputText(text=message.content))

            output_items.append(MessageOutputItem(
                id=msg_id,
                content=content_parts,
            ))

        # Add function_call output items from tool_calls
        if message.tool_calls:
            for tc in message.tool_calls:
                tc_id = to_call(tc.get("id", ""))
                tc_function = tc.get("function", {})
                tc_name = tc_function.get("name", "")
                tc_arguments = tc_function.get("arguments", "")
                output_items.append({
                    "type": "function_call",
                    "id": tc_id,
                    "call_id": tc_id,
                    "name": tc_name,
                    "arguments": tc_arguments,
                })

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

    # Determine status and incomplete_details from finish_reason
    status = "completed"
    incomplete_details = None
    if finish_reason == "length":
        status = "incomplete"
        incomplete_details = {"reason": "max_tokens"}
    elif finish_reason == "content_filter":
        status = "incomplete"
        incomplete_details = {"reason": "content_filter"}

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
        status=status,
        incomplete_details=incomplete_details,
    )
