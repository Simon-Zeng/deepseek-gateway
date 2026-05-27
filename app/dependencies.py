"""FastAPI dependency injection."""

from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException

from app.config import get_settings


async def verify_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> str:
    """Verify the API key and return the key to use for DeepSeek.

    Logic:
    - If GATEWAY_API_KEY is set: validate client key against it, return DEEPSEEK_API_KEY
    - Otherwise: forward the client's key to DeepSeek

    Args:
        authorization: The Authorization header (Bearer token).
        x_api_key: The x-api-key header (Anthropic convention).

    Returns:
        The API key to use when calling DeepSeek.

    Raises:
        HTTPException: If authentication fails.
    """
    settings = get_settings()

    # Extract bearer token
    bearer_token = None
    if authorization:
        if authorization.startswith("Bearer "):
            bearer_token = authorization[7:].strip()
        else:
            bearer_token = authorization.strip()

    # Use whichever key is provided
    client_key = bearer_token or x_api_key

    if not client_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide Authorization: Bearer <key> or x-api-key header.",
        )

    if settings.gateway.api_key:
        # Gateway key mode: validate against gateway key
        if client_key != settings.gateway.api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
            )
        return settings.deepseek.api_key
    else:
        # Key forwarding mode: use client's key
        return client_key
