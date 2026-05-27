"""Pydantic models for OpenAI Responses API."""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field


# ── Request Models ──


class ResponseInputMessage(BaseModel):
    """A message in the Responses API input."""

    role: str = "user"
    content: Optional[Union[str, list]] = None

    model_config = {"extra": "allow"}


class ReasoningConfig(BaseModel):
    """Reasoning configuration."""

    effort: Optional[str] = None  # "low", "medium", "high"
    summary: Optional[str] = None

    model_config = {"extra": "allow"}


class ResponsesRequest(BaseModel):
    """OpenAI Responses API request."""

    model: str
    input: Union[str, list[ResponseInputMessage], list[dict]]
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    reasoning: Optional[ReasoningConfig] = None
    previous_response_id: Optional[str] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    metadata: Optional[dict] = None

    model_config = {"extra": "allow"}


# ── Response Models ──


class OutputText(BaseModel):
    """Text output content part."""

    type: str = "output_text"
    text: str
    annotations: list = Field(default_factory=list)


class SummaryText(BaseModel):
    """Summary text in a reasoning output item."""

    type: str = "summary_text"
    text: str


class ReasoningOutputItem(BaseModel):
    """Reasoning output item (for R1 models)."""

    type: str = "reasoning"
    id: str
    summary: list[SummaryText] = Field(default_factory=list)


class MessageOutputItem(BaseModel):
    """Message output item."""

    type: str = "message"
    id: str
    role: str = "assistant"
    status: str = "completed"
    content: list[Union[OutputText, dict]] = Field(default_factory=list)


class ResponsesUsage(BaseModel):
    """Token usage in a Responses API response."""

    input_tokens: int = 0
    output_tokens: int = 0
    output_tokens_details: Optional[dict] = None
    total_tokens: int = 0


class ResponsesResponse(BaseModel):
    """OpenAI Responses API response."""

    id: str
    object: str = "response"
    created_at: int
    status: str = "completed"
    model: str
    output: list[Union[ReasoningOutputItem, MessageOutputItem, dict]] = Field(default_factory=list)
    usage: Optional[ResponsesUsage] = None
    metadata: Optional[dict] = None

    model_config = {"extra": "allow"}
