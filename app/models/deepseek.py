"""Pydantic models for DeepSeek API format (OpenAI-compatible)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Request Models ──


class DeepSeekMessage(BaseModel):
    """A single message in the DeepSeek chat format.

    Supports all OpenAI message roles:
    - system, user, assistant: content is string
    - assistant with tool_calls: tool_calls list + optional content
    - tool (result): tool_call_id + content (string)
    """

    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = Field(None, exclude=True)  # Strip from outgoing

    model_config = {"extra": "allow"}


class DeepSeekRequest(BaseModel):
    """DeepSeek chat completion request (OpenAI-compatible)."""

    model: str
    messages: list[DeepSeekMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[list[str] | str] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[str | dict] = None
    response_format: Optional[dict] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None

    model_config = {"extra": "allow"}


# ── Response Models ──


class DeepSeekUsage(BaseModel):
    """Token usage from DeepSeek response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    model_config = {"extra": "allow"}


class DeepSeekMessageResponse(BaseModel):
    """Message object in a DeepSeek response choice."""

    role: str = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None

    model_config = {"extra": "allow"}


class DeepSeekChoice(BaseModel):
    """A single choice in a DeepSeek response."""

    index: int = 0
    message: DeepSeekMessageResponse
    finish_reason: Optional[str] = None

    model_config = {"extra": "allow"}


class DeepSeekResponse(BaseModel):
    """DeepSeek chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[DeepSeekChoice]
    usage: Optional[DeepSeekUsage] = None

    model_config = {"extra": "allow"}


# ── Streaming Chunk Models ──


class DeepSeekDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None

    model_config = {"extra": "allow"}


class DeepSeekStreamChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: DeepSeekDelta
    finish_reason: Optional[str] = None

    model_config = {"extra": "allow"}


class DeepSeekStreamChunk(BaseModel):
    """A single SSE chunk from DeepSeek streaming response."""

    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[DeepSeekStreamChoice] = []
    usage: Optional[DeepSeekUsage] = None

    model_config = {"extra": "allow"}
