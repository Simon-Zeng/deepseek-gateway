"""Async HTTP client for the DeepSeek API."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

from app.config import get_settings
from app.models.deepseek import DeepSeekRequest, DeepSeekResponse, DeepSeekStreamChunk
from app.utils.sse import parse_sse_stream

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Async HTTP client for DeepSeek API with connection pooling and retries."""

    def __init__(self):
        settings = get_settings()
        self._base_url = settings.deepseek.base_url
        self._timeout = settings.deepseek.timeout
        self._max_retries = settings.deepseek.max_retries
        self._retry_delay = settings.deepseek.retry_delay
        self._pool_size = settings.deepseek.connection_pool_size
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self):
        """Initialize the HTTP client (call during app lifespan startup)."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(self._timeout),
                write=30.0,
                pool=30.0,
            ),
            limits=httpx.Limits(
                max_connections=self._pool_size,
                max_keepalive_connections=self._pool_size // 2,
            ),
            http2=True,
        )
        logger.info("DeepSeek client initialized (base_url=%s)", self._base_url)

    async def close(self):
        """Close the HTTP client (call during app lifespan shutdown)."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("DeepSeek client closed")

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the active HTTP client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError("DeepSeek client not initialized. Call start() first.")
        return self._client

    async def chat_completion(
        self,
        request: DeepSeekRequest,
        api_key: str,
    ) -> DeepSeekResponse:
        """Send a non-streaming chat completion request.

        Args:
            request: The DeepSeek-formatted request.
            api_key: API key to use for authentication.

        Returns:
            DeepSeekResponse with the completion result.
        """
        payload = request.model_dump(exclude_none=True, mode="json")
        # Ensure stream is False for non-streaming
        payload["stream"] = False
        # DeepSeek requires content field on all messages — null is not accepted
        for msg in payload.get("messages", []):
            if "content" not in msg:
                msg["content"] = ""

        headers = self._build_headers(api_key)

        response = await self._request_with_retry(
            "POST",
            "/chat/completions",
            json=payload,
            headers=headers,
        )

        return DeepSeekResponse(**response.json())

    async def chat_completion_stream(
        self,
        request: DeepSeekRequest,
        api_key: str,
    ) -> AsyncIterator[dict]:
        """Send a streaming chat completion request.

        Args:
            request: The DeepSeek-formatted request.
            api_key: API key to use for authentication.

        Returns:
            AsyncIterator of parsed SSE chunks (dicts).
        """
        payload = request.model_dump(exclude_none=True, mode="json")
        payload["stream"] = True
        # DeepSeek requires content field on all messages — null is not accepted
        for msg in payload.get("messages", []):
            if "content" not in msg:
                msg["content"] = ""

        headers = self._build_headers(api_key)
        # For streaming, we need to read the response as it comes
        headers["Accept"] = "text/event-stream"

        # Log the full outgoing payload at DEBUG level
        logger.debug(
            "Sending streaming request to DeepSeek: payload=%s",
            json.dumps(payload, ensure_ascii=False),
        )

        response = await self._request_with_retry(
            "POST",
            "/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
        )

        async for chunk in parse_sse_stream(response):
            yield chunk

    async def list_models(self, api_key: str) -> dict[str, Any]:
        """Fetch available models from the DeepSeek API.

        Args:
            api_key: API key to use for authentication.

        Returns:
            The raw JSON response from DeepSeek's /models endpoint.
            Example: {"object": "list", "data": [{"id": "deepseek-v4-flash", ...}]}
        """
        headers = self._build_headers(api_key)
        response = await self._request_with_retry(
            "GET",
            "/models",
            headers=headers,
        )
        return response.json()

    def _build_headers(self, api_key: str) -> dict[str, str]:
        """Build request headers with API key."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        stream: bool = False,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> httpx.Response:
        """Send an HTTP request with exponential backoff retry.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL path (relative to base_url)
            stream: Whether to stream the response
            max_retries: Override max retries
            **kwargs: Additional arguments passed to httpx request

        Returns:
            httpx.Response

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors
            httpx.TimeoutException: After all retries exhausted
        """
        retries = max_retries if max_retries is not None else self._max_retries
        last_exception = None

        for attempt in range(retries + 1):
            try:
                if stream:
                    request = self.client.build_request(method, url, **kwargs)
                    response = await self.client.send(request, stream=True)
                else:
                    response = await self.client.request(method, url, **kwargs)

                # Raise for non-retryable client errors (4xx)
                if response.status_code >= 400:
                    # Don't retry client errors (except 429 rate limit)
                    if response.status_code == 429:
                        if attempt < retries:
                            wait = self._retry_delay * (2 ** attempt)
                            logger.warning(
                                "Rate limited by DeepSeek, retrying in %.1fs (attempt %d/%d)",
                                wait, attempt + 1, retries + 1,
                            )
                            await asyncio.sleep(wait)
                            continue
                    if response.status_code < 500:
                        # Log request body on client errors for debugging
                        try:
                            req_body = response.request.content
                            if req_body:
                                body_str = req_body.decode("utf-8", errors="replace")
                                if len(body_str) > 12000:
                                    body_str = body_str[:12000] + "...[truncated]"
                                logger.error(
                                    "DeepSeek %d request failed. Request body: %s",
                                    response.status_code, body_str,
                                )
                            # For streaming responses, read the body explicitly
                            if stream:
                                resp_body_bytes = await response.aread()
                                resp_body_text = resp_body_bytes.decode("utf-8", errors="replace")[:4000]
                            else:
                                resp_body_text = response.text[:4000] if response.text else "(empty)"
                            logger.error("DeepSeek %d response body: %s", response.status_code, resp_body_text)
                        except Exception as e:
                            logger.error("Failed to log DeepSeek error details: %s", e)
                        response.raise_for_status()

                    # Server error - retry
                    if attempt < retries:
                        wait = self._retry_delay * (2 ** attempt)
                        logger.warning(
                            "DeepSeek server error %d, retrying in %.1fs (attempt %d/%d)",
                            response.status_code, wait, attempt + 1, retries + 1,
                        )
                        await asyncio.sleep(wait)
                        continue

                    response.raise_for_status()

                return response

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                if attempt < retries:
                    wait = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "DeepSeek connection error: %s, retrying in %.1fs (attempt %d/%d)",
                        type(e).__name__, wait, attempt + 1, retries + 1,
                    )
                    await asyncio.sleep(wait)
                    continue

        raise last_exception or httpx.TimeoutException("All retries exhausted")


# Singleton client instance
_client: Optional[DeepSeekClient] = None


def get_client() -> DeepSeekClient:
    """Get the global DeepSeek client instance."""
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client
