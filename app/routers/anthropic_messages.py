"""Anthropic Messages API router."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.dependencies import verify_api_key
from app.converters.anthropic import convert_request, convert_response
from app.models.common import ModelType
from app.models.deepseek import DeepSeekRequest
from app.models.anthropic import AnthropicRequest, AnthropicResponse
from app.services.deepseek_client import DeepSeekClient, get_client
from app.services.model_mapper import ModelMapper, MappingResult
from app.streamers.anthropic import stream_anthropic
from app.utils.errors import create_anthropic_error
from app.utils.sse import generate_id

logger = logging.getLogger(__name__)

router = APIRouter()

# Threshold for Anthropic thinking budget_tokens to be treated as "high" reasoning effort
# If budget_tokens >= this value, force use of pro model
ANTHROPIC_THINKING_BUDGET_HIGH_THRESHOLD = 10000


@router.post("/v1/messages")
async def anthropic_messages(
    request: AnthropicRequest,
    req: Request,
    api_key: str = Depends(verify_api_key),
    anthropic_version: Optional[str] = Header(None),
):
    """Handle Anthropic Messages API requests.

    Converts the request to DeepSeek format, forwards it, and converts
    the response back to Anthropic Messages format.

    Supports reasoning effort override: when thinking.budget_tokens is set to a
    high value (>= threshold), forces use of deepseek-v4-pro regardless of model_mapping.
    """
    client: DeepSeekClient = get_client()
    mapper: ModelMapper = req.app.state.model_mapper

    # ── Model Mapping (with thinking budget override) ──
    reasoning_effort = _extract_reasoning_effort(request)

    mapping = mapper.map_model(request.model, reasoning_effort=reasoning_effort)
    logger.info(
        "Anthropic messages: model='%s' -> '%s' stream=%s effort=%s",
        request.model, mapping.target_model, request.stream, reasoning_effort,
    )

    # ── Convert Request ──
    try:
        deepseek_request = convert_request(request, mapping.target_model)
    except Exception as e:
        logger.error("Request conversion error: %s", e)
        error_resp, _ = create_anthropic_error(
            message=f"Failed to convert request: {e}",
            error_type="invalid_request_error",
            status_code=400,
        )
        return error_resp

    # ── Call DeepSeek ──
    if request.stream:
        return await _handle_streaming(client, deepseek_request, api_key, request.model, mapping)
    else:
        return await _handle_non_streaming(client, deepseek_request, api_key, request.model, mapping)


async def _handle_non_streaming(
    client: DeepSeekClient,
    deepseek_request: DeepSeekRequest,
    api_key: str,
    original_model: str,
    mapping: MappingResult,
) -> JSONResponse:
    """Handle a non-streaming Anthropic messages request."""
    try:
        deepseek_response = await client.chat_completion(deepseek_request, api_key)
    except Exception as e:
        logger.error("DeepSeek API error: %s", e)
        error_resp, _ = create_anthropic_error(
            message=f"Upstream API error: {e}",
            error_type="api_error",
            status_code=502,
        )
        return error_resp

    response = convert_response(
        deepseek_response,
        original_model=original_model,
        is_reasoner=mapping.model_type == ModelType.REASONER,
    )

    return JSONResponse(content=response.model_dump(exclude_none=True, mode="json"))


async def _handle_streaming(
    client: DeepSeekClient,
    deepseek_request: DeepSeekRequest,
    api_key: str,
    original_model: str,
    mapping: MappingResult,
) -> StreamingResponse:
    """Handle a streaming Anthropic messages request."""
    deepseek_stream = client.chat_completion_stream(deepseek_request, api_key)

    async def response_generator():
        async for chunk_str in stream_anthropic(
            deepseek_stream,
            original_model=original_model,
            model_type=mapping.model_type,
        ):
            yield chunk_str

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _extract_reasoning_effort(request: AnthropicRequest) -> Optional[str]:
    """Extract reasoning effort signal from Anthropic request.

    Anthropic doesn't have an explicit reasoning.effort field, but it uses
    `thinking.budget_tokens` to control extended thinking. When budget_tokens
    is set to a high value, we interpret it as "high" reasoning effort.

    Also checks for a non-standard `reasoning_effort` extra field that some
    clients may send.

    Returns:
        "high" if thinking budget is significant, the explicit effort value,
        or None if no reasoning signal is present.
    """
    # Check for explicit reasoning_effort field (non-standard but some clients send it)
    extra = request.model_extra or {}
    if "reasoning_effort" in extra:
        return extra["reasoning_effort"]

    # Check thinking configuration
    if "thinking" in extra:
        thinking = extra["thinking"]
        if isinstance(thinking, dict):
            # If thinking is enabled with type "enabled", treat as at least "medium"
            if thinking.get("type") == "enabled":
                budget_tokens = thinking.get("budget_tokens", 0)
                if budget_tokens >= ANTHROPIC_THINKING_BUDGET_HIGH_THRESHOLD:
                    return "high"
                elif budget_tokens > 0:
                    return "medium"

    return None
