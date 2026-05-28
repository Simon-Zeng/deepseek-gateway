"""OpenAI Responses API router."""

from __future__ import annotations

import time
import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.dependencies import verify_api_key
from app.converters.openai_responses import convert_request, convert_response
from app.models.common import ModelType
from app.models.deepseek import DeepSeekRequest
from app.models.openai_responses import ResponsesRequest, ResponsesResponse
from app.services.deepseek_client import DeepSeekClient, get_client
from app.services.model_mapper import ModelMapper, MappingResult, map_reasoning_effort
from app.streamers.openai_responses import stream_openai_responses
from app.utils.errors import create_openai_error
from app.utils.sse import generate_id, format_sse_event, format_sse_done

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/responses")
async def responses(
    request: ResponsesRequest,
    req: Request,
    api_key: str = Depends(verify_api_key),
):
    """Handle OpenAI Responses API requests.

    Converts the request to DeepSeek format, forwards it, and converts
    the response back to OpenAI Responses format.

    Supports reasoning effort override: when reasoning.effort is "high" or above,
    forces use of deepseek-v4-pro regardless of model_mapping.
    """
    client: DeepSeekClient = get_client()
    mapper: ModelMapper = req.app.state.model_mapper

    # ── Model Mapping (with reasoning effort override) ──
    reasoning_effort = None
    if request.reasoning and request.reasoning.effort:
        reasoning_effort = request.reasoning.effort

    mapping = mapper.map_model(request.model, reasoning_effort=reasoning_effort)
    logger.info(
        "Responses: model='%s' -> '%s' stream=%s effort=%s",
        request.model, mapping.target_model, request.stream, reasoning_effort,
    )

    # ── Convert Request ──
    try:
        deepseek_request = convert_request(request, mapping.target_model)
    except Exception as e:
        logger.error("Request conversion error: %s", e)
        error_resp, _ = create_openai_error(
            message="Failed to convert request",
            error_type="invalid_request_error",
            status_code=400,
        )
        return error_resp

    # ── Forward reasoning.effort ONLY if target model supports thinking ──
    # Setting thinking on a non-thinking model (e.g. deepseek-v4-flash) causes
    # DeepSeek API to return 400 Bad Request.
    if request.reasoning and request.reasoning.effort:
        if mapper.is_thinking_model(mapping.target_model):
            ds_effort = map_reasoning_effort(request.reasoning.effort)
            if ds_effort:
                deepseek_request.reasoning_effort = ds_effort
                deepseek_request.thinking = {"type": "enabled"}
                logger.debug(
                    "Forwarded reasoning.effort=%s -> %s to %s (thinking model)",
                    request.reasoning.effort, ds_effort, mapping.target_model,
                )
        else:
            logger.debug(
                "Skipping reasoning.effort=%s: model %s does not support thinking",
                request.reasoning.effort, mapping.target_model,
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
    """Handle a non-streaming responses request."""
    try:
        deepseek_response = await client.chat_completion(deepseek_request, api_key)
    except Exception as e:
        logger.error("DeepSeek API error: %s", e)
        error_resp, _ = create_openai_error(
            message="Upstream API error",
            error_type="server_error",
            status_code=502,
            code="upstream_error",
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
    """Handle a streaming responses request."""
    # Log the payload being sent to DeepSeek for debugging
    payload_preview = deepseek_request.model_dump(exclude_none=True, mode="json")
    logger.info(
        "Sending to DeepSeek: model=%s tools=%d tool_choice=%s thinking=%s effort=%s msg_count=%d",
        payload_preview.get("model"),
        len(payload_preview.get("tools", [])),
        payload_preview.get("tool_choice"),
        payload_preview.get("thinking"),
        payload_preview.get("reasoning_effort"),
        len(payload_preview.get("messages", [])),
    )

    deepseek_stream = client.chat_completion_stream(deepseek_request, api_key)

    response_id = generate_id("resp")
    created_at = int(time.time())

    async def safe_response_generator():
        """Wrap stream to handle upstream errors gracefully."""
        emitted = False
        try:
            async for chunk_str in stream_openai_responses(
                deepseek_stream,
                original_model=original_model,
                model_type=mapping.model_type,
                response_id=response_id,
            ):
                emitted = True
                yield chunk_str
        except Exception as e:
            logger.error("Stream error (sending failed completion): %s", e)
            if not emitted:
                # Emit clean failure events so client doesn't get broken stream
                yield format_sse_event({
                    "type": "response.in_progress",
                    "response": {"id": response_id, "status": "in_progress", "model": original_model, "output": [], "created_at": created_at},
                })
                yield format_sse_event({
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "created_at": created_at,
                        "status": "failed",
                        "model": original_model,
                        "output": [],
                        "error": {"message": "Upstream stream failed", "type": "upstream_error"},
                    },
                })
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
