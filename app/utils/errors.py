"""Error conversion utilities for protocol-aware error formatting."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from app.models.common import (
    AnthropicErrorDetail,
    AnthropicErrorResponse,
    ErrorDetail,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

# Mapping of DeepSeek/OpenAI error codes to Anthropic error types
DEEPSEEK_TO_ANTHROPIC_ERROR = {
    "insufficient_quota": "permission_error",
    "rate_limit_exceeded": "rate_limit_error",
    "model_not_found": "not_found_error",
    "context_length_exceeded": "invalid_request_error",
    "server_error": "api_error",
    "invalid_request_error": "invalid_request_error",
    "authentication_error": "authentication_error",
}


def create_openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    status_code: int = 400,
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> tuple[JSONResponse, int]:
    """Create an OpenAI-format error response.

    Returns:
        Tuple of (JSONResponse, status_code)
    """
    error = ErrorResponse(
        error=ErrorDetail(
            message=message,
            type=error_type,
            param=param,
            code=code,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump(exclude_none=True),
    ), status_code


def create_anthropic_error(
    message: str,
    error_type: str = "invalid_request_error",
    status_code: int = 400,
) -> tuple[JSONResponse, int]:
    """Create an Anthropic-format error response.

    Returns:
        Tuple of (JSONResponse, status_code)
    """
    error = AnthropicErrorResponse(
        error=AnthropicErrorDetail(
            type=error_type,
            message=message,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump(exclude_none=True),
    ), status_code


def convert_deepseek_error(
    deepseek_error: dict,
    target_protocol: str = "openai",
) -> tuple[JSONResponse, int]:
    """Convert a DeepSeek API error to the target protocol's format.

    Args:
        deepseek_error: The error dict from DeepSeek API response.
        target_protocol: "openai" or "anthropic".

    Returns:
        Tuple of (JSONResponse, status_code)
    """
    error_obj = deepseek_error.get("error", deepseek_error)
    message = error_obj.get("message", "Unknown error")
    error_type = error_obj.get("type", "server_error")
    code = error_obj.get("code", "")

    # Determine HTTP status
    status_code = _error_code_to_status(code, error_type)

    if target_protocol == "anthropic":
        anthropic_type = DEEPSEEK_TO_ANTHROPIC_ERROR.get(code, "api_error")
        return create_anthropic_error(
            message=message,
            error_type=anthropic_type,
            status_code=status_code,
        )

    return create_openai_error(
        message=message,
        error_type=error_type,
        status_code=status_code,
        code=code,
    )


def _error_code_to_status(code: str, error_type: str = "") -> int:
    """Map an error code or type to an HTTP status code."""
    mapping = {
        "insufficient_quota": 402,
        "rate_limit_exceeded": 429,
        "model_not_found": 404,
        "context_length_exceeded": 400,
        "invalid_api_key": 401,
        "authentication_error": 401,
        "server_error": 500,
    }
    if code in mapping:
        return mapping[code]
    if error_type in mapping:
        return mapping[error_type]
    return 500


def get_protocol_from_path(path: str) -> str:
    """Determine the target protocol from the request path.

    Args:
        path: The request URL path.

    Returns:
        "anthropic" for /v1/messages, "openai" otherwise.
    """
    if "/v1/messages" in path:
        return "anthropic"
    return "openai"
