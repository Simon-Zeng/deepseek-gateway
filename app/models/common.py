"""Common shared models and types."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ModelType(str, Enum):
    """Classification of model capabilities."""

    CHAT = "chat"
    REASONER = "reasoner"
    AUTO = "auto"


class ModelMappingRule(BaseModel):
    """A single model name mapping rule."""

    pattern: str  # Regex pattern to match against incoming model name
    target: str  # Target DeepSeek model name
    type: ModelType = ModelType.CHAT


class ModelMappingConfig(BaseModel):
    """Full model mapping configuration."""

    defaults: dict[str, str] = {"chat": "deepseek-v4-flash", "reasoner": "deepseek-v4-pro"}
    mapping: list[ModelMappingRule] = []
    thinking: dict = {"models": ["deepseek-v4-pro"], "min_length": 0}


class ErrorResponse(BaseModel):
    """Error response format (OpenAI-style)."""

    error: ErrorDetail


class ErrorDetail(BaseModel):
    """Error detail object."""

    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class AnthropicErrorResponse(BaseModel):
    """Error response format (Anthropic-style)."""

    type: str = "error"
    error: AnthropicErrorDetail


class AnthropicErrorDetail(BaseModel):
    """Anthropic error detail object."""

    type: str = "invalid_request_error"
    message: str
