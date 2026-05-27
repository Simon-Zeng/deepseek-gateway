"""Pydantic models for Anthropic Messages API."""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field


# ── Request Models ──


class TextContentBlock(BaseModel):
    """Text content block."""

    type: str = "text"
    text: str


class ImageContentBlock(BaseModel):
    """Image content block (not supported by DeepSeek)."""

    type: str = "image"
    source: dict = Field(default_factory=dict)


class ToolUseContentBlock(BaseModel):
    """Tool use content block."""

    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = Field(default_factory=dict)


class ToolResultContentBlock(BaseModel):
    """Tool result content block."""

    type: str = "tool_result"
    tool_use_id: str = ""
    content: Optional[Union[str, list]] = None


class ThinkingContentBlock(BaseModel):
    """Thinking content block (extended thinking)."""

    type: str = "thinking"
    thinking: str


class AnthropicMessage(BaseModel):
    """A message in the Anthropic format."""

    role: str
    content: Union[str, list[dict], None] = None

    model_config = {"extra": "allow"}


class AnthropicRequest(BaseModel):
    """Anthropic Messages API request."""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 4096
    system: Optional[Union[str, list[dict]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: bool = False
    stop_sequences: Optional[list[str]] = None
    metadata: Optional[dict] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None

    model_config = {"extra": "allow"}


# ── Response Models ──


class TextResponseBlock(BaseModel):
    """Text content block in response."""

    type: str = "text"
    text: str


class ThinkingResponseBlock(BaseModel):
    """Thinking content block in response."""

    type: str = "thinking"
    thinking: str


class AnthropicUsage(BaseModel):
    """Token usage in an Anthropic response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str
    type: str = "message"
    role: str = "assistant"
    model: str
    content: list[Union[ThinkingResponseBlock, TextResponseBlock, dict]] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)

    model_config = {"extra": "allow"}
