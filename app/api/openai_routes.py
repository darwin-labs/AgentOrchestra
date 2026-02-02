import json
import time
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.agent.manus import Manus
from app.api.auth import require_openai_auth
from app.api.openai_schemas import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChunkResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ModelListResponse,
    ModelObject,
    OpenAIErrorPayload,
    OpenAIErrorResponse,
)
from app.config import config

router = APIRouter(prefix="/v1")


def _now() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _error_response(
    message: str, status_code: int, error_type: str = "invalid_request_error"
) -> JSONResponse:
    payload = OpenAIErrorResponse(
        error=OpenAIErrorPayload(message=message, type=error_type)
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _extract_best_output(agent: Manus, fallback: str) -> str:
    assistant_contents: list[str] = []
    tool_contents: list[str] = []

    for msg in agent.memory.messages:
        role = str(msg.role)
        if role == "assistant" and msg.content:
            assistant_contents.append(str(msg.content))
        if (
            role == "tool"
            and getattr(msg, "name", None) == "create_chat_completion"
            and msg.content
        ):
            tool_contents.append(str(msg.content))

    if assistant_contents:
        return assistant_contents[-1]

    if tool_contents:
        text = tool_contents[-1]
        marker = "Observed output of cmd `create_chat_completion` executed:\n"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return text

    return fallback


def _prompt_from_messages(req: ChatCompletionRequest) -> str:
    last_user: Optional[str] = None
    system_parts: list[str] = []

    for m in req.messages:
        if m.role == "system" and m.content:
            system_parts.append(str(m.content))
        if m.role == "user" and m.content:
            last_user = str(m.content)

    if last_user is None:
        raise HTTPException(status_code=400, detail="No user message found")

    if system_parts:
        return "\n\n".join(system_parts + [last_user])

    return last_user


@router.get("/models", dependencies=[Depends(require_openai_auth)])
async def list_models() -> ModelListResponse:
    created = _now()
    model_ids = {settings.model for settings in config.llm.values() if settings.model}
    if not model_ids:
        model_ids = {"openmanus"}
    models = [
        ModelObject(id=model_id, created=created) for model_id in sorted(model_ids)
    ]
    return ModelListResponse(data=models)


@router.post("/chat/completions", dependencies=[Depends(require_openai_auth)])
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return _error_response("'messages' must not be empty", status_code=400)

    try:
        prompt = _prompt_from_messages(req)
    except HTTPException as e:
        return _error_response(str(e.detail), status_code=e.status_code)

    created = _now()
    completion_id = _new_id("chatcmpl")

    # Handle dynamic configuration overrides
    token = None
    if req.groq_api_key or req.daytona_api_key or req.model:
        new_config = config.current_config.model_copy()

        if req.groq_api_key:
            new_llm = new_config.llm.copy()
            for name, settings in new_llm.items():
                if settings.api_type == "groq":
                    new_llm[name] = settings.model_copy(
                        update={"api_key": req.groq_api_key}
                    )
            new_config.llm = new_llm

        if req.model:
            new_llm = new_config.llm.copy()
            # Also update 'default' to use the requested model
            if "default" in new_llm:
                new_llm["default"] = new_llm["default"].model_copy(
                    update={"model": req.model}
                )
            new_config.llm = new_llm

        if req.daytona_api_key and new_config.daytona_config:
            new_config.daytona_config = new_config.daytona_config.model_copy(
                update={"daytona_api_key": req.daytona_api_key}
            )

        token = config.set_context_config(new_config)

    try:
        if req.stream:
            # True streaming: yield chunks as each step completes
            async def event_stream() -> AsyncIterator[bytes]:
                agent = await Manus.create()
                accumulated_content = ""

                # Send initial role chunk
                chunk1 = ChatCompletionChunkResponse(
                    id=completion_id,
                    created=created,
                    model=req.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(role="assistant"),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {json.dumps(chunk1.model_dump())}\n\n".encode("utf-8")

                # Stream each step as it completes
                try:
                    async for step_result in agent.run_stream(prompt):
                        accumulated_content += step_result + "\n"
                        chunk = ChatCompletionChunkResponse(
                            id=completion_id,
                            created=created,
                            model=req.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(
                                        content=step_result + "\n"
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {json.dumps(chunk.model_dump())}\n\n".encode(
                            "utf-8"
                        )
                finally:
                    await agent.cleanup()

                # Extract the best final output for the finish chunk
                final_content = _extract_best_output(
                    agent, fallback=accumulated_content.strip()
                )
                if final_content and final_content != accumulated_content.strip():
                    # Send the final extracted content if different from accumulated
                    final_chunk = ChatCompletionChunkResponse(
                        id=completion_id,
                        created=created,
                        model=req.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    content="\n\n---\n" + final_content
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {json.dumps(final_chunk.model_dump())}\n\n".encode(
                        "utf-8"
                    )

                # Send finish chunk
                finish_chunk = ChatCompletionChunkResponse(
                    id=completion_id,
                    created=created,
                    model=req.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {json.dumps(finish_chunk.model_dump())}\n\n".encode(
                    "utf-8"
                )
                yield b"data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # Non-streaming path
        agent = await Manus.create()
        try:
            result = await agent.run(prompt)
            content = _extract_best_output(agent, fallback=result)
        finally:
            await agent.cleanup()

        usage = ChatCompletionUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
        resp = ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionChoiceMessage(content=content),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )
        return resp
    except Exception as e:
        return _error_response(str(e), status_code=500, error_type="server_error")
    finally:
        if token:
            config.reset_context_config(token)
