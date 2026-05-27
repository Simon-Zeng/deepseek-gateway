"""FastAPI application entry point."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import anthropic_messages, models, openai_chat, openai_responses
from app.services.deepseek_client import get_client
from app.services.model_mapper import ModelMapper
from app.utils.errors import get_protocol_from_path
from app.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - startup and shutdown."""
    # ── Startup ──
    setup_logging()
    logger.info("Starting DeepSeek Gateway...")

    settings = get_settings()

    # Initialize DeepSeek client
    client = get_client()
    await client.start()
    logger.info("DeepSeek client connected to %s", settings.deepseek.base_url)

    # Initialize model mapper
    mapper = ModelMapper(settings.gateway.model_mapping_path)
    app.state.model_mapper = mapper
    logger.info("Model mapper initialized with %d rules", len(mapper._compiled_patterns))

    yield

    # ── Shutdown ──
    logger.info("Shutting down DeepSeek Gateway...")
    await client.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="DeepSeek Gateway",
        description="AI model proxy gateway - routes OpenAI/Anthropic requests to DeepSeek",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── Middleware ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ──
    app.include_router(openai_chat.router)
    app.include_router(openai_responses.router)
    app.include_router(anthropic_messages.router)
    app.include_router(models.router)

    # ── Health Check ──
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "deepseek-gateway"}

    # ── Error Handler ──
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler with protocol-aware error formatting."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.error(
            "Unhandled exception [request_id=%s]: %s",
            request_id,
            exc,
            exc_info=True,
        )

        protocol = get_protocol_from_path(request.url.path)

        if protocol == "anthropic":
            return JSONResponse(
                status_code=500,
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "Internal server error",
                    },
                },
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Internal server error",
                        "type": "server_error",
                        "param": None,
                        "code": "internal_error",
                    },
                },
            )

    # ── Request ID Middleware ──
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        """Add request ID and timing to all requests."""
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.time()

        # Process the request
        response = await call_next(request)

        # Add timing and request ID headers
        duration_ms = round((time.time() - start_time) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        logger.info(
            "%s %s -> %d (%dms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        return response

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.gateway.host,
        port=settings.gateway.port,
        workers=settings.gateway.workers,
        log_level=settings.gateway.log_level,
    )
