from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class OpenAIErrorPayload(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorPayload


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "openmanus"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelObject]


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[Any]]] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    stream: bool = False

    # Accepted but currently not fully implemented (kept for compatibility)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatCompletionChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionChoiceMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunkResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
