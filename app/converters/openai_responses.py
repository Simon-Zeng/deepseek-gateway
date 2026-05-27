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
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                # Handle content that might be a list
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "input_text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "\n".join(text_parts) if text_parts else ""
                messages.append(DeepSeekMessage(role=role, content=str(content) if content else None))
            elif hasattr(item, "role"):
                # Pydantic model
                content = item.content
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in ("input_text", "text"):
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "\n".join(text_parts) if text_parts else None
                messages.append(DeepSeekMessage(role=item.role, content=content))

    # Log warnings for unsupported features
    if request.previous_response_id:
        logger.warning("previous_response_id not supported, ignoring")
    if request.tools:
        logger.warning("tools not fully supported by DeepSeek, passing through")

    return DeepSeekRequest(
        model=target_model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        stream=request.stream,
        tools=request.tools,
        tool_choice=request.tool_choice if request.tools else None,
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
