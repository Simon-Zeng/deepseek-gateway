"""Models listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.services.model_mapper import ModelMapper

router = APIRouter()


@router.get("/v1/models")
async def list_models(req: Request):
    """List available models.

    Returns a list of models that can be used with this gateway.
    Includes both DeepSeek models and their aliases.
    """
    mapper: ModelMapper = req.app.state.model_mapper
    models = mapper.available_models

    return {
        "object": "list",
        "data": models,
    }


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str, req: Request):
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
