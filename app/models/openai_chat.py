"""Pydantic models for OpenAI Chat Completions API."""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field


# ── Request Models ──


class ChatMessage(BaseModel):
    """A message in the OpenAI Chat format."""

    role: str
    content: Optional[Union[str, list]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = Field(None, exclude=True)  # Strip from outgoing

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completions request."""

    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    user: Optional[str] = None
    stream_options: Optional[dict] = None

    model_config = {"extra": "allow"}


# ── Response Models ──


class ChatCompletionUsage(BaseModel):
    """Token usage in a chat completion response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    model_config = {"extra": "allow"}


class ChatCompletionMessage(BaseModel):
    """Message in a chat completion response choice."""

    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None

    model_config = {"extra": "allow"}


class ChatCompletionChoice(BaseModel):
    """A choice in a chat completion response."""

    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None

    model_config = {"extra": "allow"}


class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completions response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None

    model_config = {"extra": "allow"}


# ── Streaming Models ──


class ChatCompletionDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None

    model_config = {"extra": "allow"}


class ChatCompletionStreamChoice(BaseModel):
    """A choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: Optional[str] = None

    model_config = {"extra": "allow"}


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI Chat Completions streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice] = []
    usage: Optional[ChatCompletionUsage] = None

    model_config = {"extra": "allow"}
