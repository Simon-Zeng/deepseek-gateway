"""Models listing endpoint — proxies DeepSeek's /v1/models."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.dependencies import verify_api_key
from app.services.deepseek_client import get_client
from app.services.model_mapper import ModelMapper
from app.utils.errors import create_openai_error

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/models")
async def list_models(req: Request, api_key: str = Depends(verify_api_key)):
    """List available models from the DeepSeek API.

    Proxies the request to DeepSeek's /v1/models endpoint,
    returning only models that DeepSeek actually provides.
    """
    try:
        client = get_client()
        result = await client.list_models(api_key)
        return result
    except Exception as e:
        logger.error("Failed to fetch models from DeepSeek: %s", e)
        error_resp, _ = create_openai_error(
            message="Failed to fetch models from upstream",
            error_type="upstream_error",
            status_code=502,
        )
        return error_resp


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str, req: Request, api_key: str = Depends(verify_api_key)):
    """Get details about a specific model."""
    mapper: ModelMapper = req.app.state.model_mapper
    mapping = mapper.map_model(model_id)

    return {
        "id": model_id,
        "object": "model",
        "owned_by": "deepseek-gateway",
        "target_model": mapping.target_model,
        "model_type": mapping.model_type.value,
    }
