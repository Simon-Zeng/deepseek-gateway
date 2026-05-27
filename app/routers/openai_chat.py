"""OpenAI Chat Completions API router."""

from __future__ import annotations

import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.converters.openai_chat import convert_request, convert_response
from app.models.common import ModelType
from app.models.deepseek import DeepSeekRequest
from app.models.openai_chat import ChatCompletionRequest, ChatCompletionResponse
from app.services.deepseek_client import DeepSeekClient, get_client
from app.services.model_mapper import ModelMapper, MappingResult
from app.streamers.openai_chat import stream_openai_chat
from app.utils.errors import convert_deepseek_error, create_openai_error
from app.utils.sse import generate_id

logger = logging.getLogger(__name__)

router = APIRouter()


def get_mapper() -> ModelMapper:
    """Get the model mapper instance from app state."""
    return router.app.state.model_mapper


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    req: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    """Handle OpenAI Chat Completions API requests.

    Converts the request to DeepSeek format, forwards it, and converts
    the response back to OpenAI Chat format.
    """
    settings = get_settings()
    client: DeepSeekClient = get_client()
    mapper: ModelMapper = req.app.state.model_mapper

    # ── Auth ──
    api_key = _resolve_api_key(authorization, x_api_key, settings)
    if not api_key:
        error_resp, status = create_openai_error(
            message="Missing API key. Provide Authorization: Bearer <key> or x-api-key header.",
            error_type="authentication_error",
            status_code=401,
            code="invalid_api_key",
        )
        return error_resp

    # ── Model Mapping (with reasoning effort override) ──
    # Some OpenAI clients send reasoning_effort as an extra field
    reasoning_effort = None
    if hasattr(request, "model_extra") and request.model_extra:
        reasoning_effort = request.model_extra.get("reasoning_effort")

    mapping = mapper.map_model(request.model, reasoning_effort=reasoning_effort)
    logger.info(
        "Chat completion: model='%s' -> '%s' stream=%s effort=%s",
        request.model, mapping.target_model, request.stream, reasoning_effort,
    )

    # ── Convert Request ──
    try:
        deepseek_request = convert_request(request, mapping.target_model)
    except Exception as e:
        logger.error("Request conversion error: %s", e)
        error_resp, _ = create_openai_error(
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
    """Handle a non-streaming chat completion request."""
    try:
        deepseek_response = await client.chat_completion(deepseek_request, api_key)
    except Exception as e:
        logger.error("DeepSeek API error: %s", e)
        error_resp, _ = create_openai_error(
            message=f"DeepSeek API error: {e}",
            error_type="server_error",
            status_code=502,
            code="upstream_error",
        )
        return error_resp

    # ── Convert Response ──
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
    """Handle a streaming chat completion request."""
    response_id = generate_id()
    created = int(time.time())

    # Get the streaming generator from DeepSeek
    deepseek_stream = client.chat_completion_stream(deepseek_request, api_key)

    # Create the converted stream
    async def response_generator():
        async for chunk_str in stream_openai_chat(
            deepseek_stream,
            original_model=original_model,
            model_type=mapping.model_type,
            response_id=response_id,
            created=created,
        ):
            yield chunk_str

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx: disable buffering
        },
    )


def _resolve_api_key(
    authorization: Optional[str],
    x_api_key: Optional[str],
    settings,
) -> Optional[str]:
    """Resolve the API key to use for DeepSeek.

    Logic:
    - If GATEWAY_API_KEY is set, validate client key against it, return DEEPSEEK_API_KEY
    - Otherwise, forward the client's key to DeepSeek
    """
    # Extract bearer token
    bearer_token = None
    if authorization:
        if authorization.startswith("Bearer "):
            bearer_token = authorization[7:].strip()
        else:
            bearer_token = authorization.strip()

    # Also check x-api-key header (Anthropic-style)
    client_key = bearer_token or x_api_key

    if settings.gateway.api_key:
        # Gateway key mode
        if client_key == settings.gateway.api_key:
            return settings.deepseek.api_key
        return None  # Invalid key
    else:
        # Key forwarding mode
        return client_key
