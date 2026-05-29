"""Anthropic Messages API router."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
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
from app.utils.sse import generate_id, format_sse_event, format_sse_done

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

    # ── Auto-enable thinking for REASONER models if not already set ──
    # The Anthropic converter only enables thinking when the client sends
    # a thinking config block. But Claude Code using a REASONER model should
    # always get reasoning_content back from DeepSeek. Enable it with default
    # "high" effort if the model is a thinking model and thinking wasn't set.
    if mapper.is_thinking_model(mapping.target_model) and mapping.model_type == ModelType.REASONER:
        if not deepseek_request.thinking:
            deepseek_request.thinking = {"type": "enabled"}
            deepseek_request.reasoning_effort = "high"
            logger.debug(
                "Auto-enabled thinking (high) for Anthropic REASONER model %s (no client thinking config)",
                mapping.target_model,
            )

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

    async def safe_response_generator():
        """Wrap stream to handle upstream errors gracefully."""
        try:
            async for chunk_str in stream_anthropic(
                deepseek_stream,
                original_model=original_model,
                model_type=mapping.model_type,
            ):
                yield chunk_str
        except Exception as e:
            logger.error("Anthropic stream error: %s", e)
            yield format_sse_event(
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": "Upstream stream failed"},
                },
                event="error",
            )
            yield format_sse_event(
                {"type": "message_stop"},
                event="message_stop",
            )
        yield format_sse_done()

    return StreamingResponse(
        safe_response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _extract_reasoning_effort(request: AnthropicRequest) -> Optional[str]:
    """Extract reasoning effort signal from Anthropic request.

    Anthropic has no explicit ``reasoning.effort`` field (unlike OpenAI Responses),
    but uses a ``thinking`` config block instead. Two formats exist:

    Old format (``type: "enabled"`` + ``budget_tokens``):
      - budget_tokens >= threshold (10000) → returns "high"
      - budget_tokens > 0                → returns "medium"
      - (type: "enabled" without budget  → left for converter to handle)

    New format (``type: "adaptive"`` + ``output_config.effort``):
      - effort = "low"  → returns "low"
      - effort = "medium" → returns "medium"
      - effort = "high" → returns "high"
      - effort = "max"  → returns "max" (Opus-only)
      - (no effort      → defaults to "high")

    Also checks for a non-standard ``reasoning_effort`` extra field that some
    clients may send (highest priority).

    Returns:
        The raw Anthropic/OpenAI effort value (low/medium/high/max),
        or None if no reasoning signal is present.
        This is used upstream for model selection override
        (``mapper.map_model(reasoning_effort=...)``).
        The actual value-to-DeepSeek mapping is done inside
        ``converters.anthropic.convert_request()`` using
        ``map_reasoning_effort()``.
    """
    # 1. Check for explicit reasoning_effort field (non-standard but some clients send it)
    extra = request.model_extra or {}
    if "reasoning_effort" in extra:
        return extra["reasoning_effort"]

    # 2. Check thinking configuration
    if "thinking" in extra:
        thinking = extra["thinking"]
        if isinstance(thinking, dict):
            ttype = thinking.get("type")

            # Old format: type="enabled" with budget_tokens
            if ttype == "enabled":
                budget_tokens = thinking.get("budget_tokens", 0)
                if budget_tokens >= ANTHROPIC_THINKING_BUDGET_HIGH_THRESHOLD:
                    return "high"
                elif budget_tokens > 0:
                    return "medium"

            # New format: type="adaptive" — effort is in output_config.effort
            elif ttype == "adaptive":
                output_config = extra.get("output_config", {})
                if isinstance(output_config, dict):
                    effort = output_config.get("effort", "high")
                    return effort
                return "high"

    return None
