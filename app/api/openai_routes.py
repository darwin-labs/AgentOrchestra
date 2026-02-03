import base64
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

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
from app.schema import Message
from app.sandbox.client import SANDBOX_CLIENT
from app.utils.files_utils import should_exclude_file

_ATTACHMENT_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
_UPLOAD_DIR = Path(config.workspace_root) / "uploads"
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB
_MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024  # 25MB

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
        if m.content is None:
            continue
        if isinstance(m.content, list):
            text_parts = []
            for item in m.content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
            content_text = "\n".join(text_parts).strip()
        else:
            content_text = str(m.content)

        if not content_text:
            continue

        if m.role == "system":
            system_parts.append(content_text)
        if m.role == "user":
            last_user = content_text

    if last_user is None:
        raise HTTPException(status_code=400, detail="No user message found")

    if system_parts:
        return "\n\n".join(system_parts + [last_user])

    return last_user


def _is_image_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    part_type = part.get("type")
    return (
        part_type in {"image_url", "input_image"}
        or "image_url" in part
        or "image" in part
    )


def _is_file_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    part_type = part.get("type")
    return (
        part_type in {"file", "input_file"}
        or "file" in part
        or "file_id" in part
        or "file_data" in part
        or "data" in part
    )


def _decode_data_url(data_url: str) -> Optional[bytes]:
    if not data_url.startswith("data:"):
        return None
    if "base64," not in data_url:
        return None
    encoded = data_url.split("base64,", 1)[1]
    return base64.b64decode(encoded)


def _extract_file_payload(part: dict) -> Tuple[str, str, bytes]:
    file_obj = part.get("file") if isinstance(part.get("file"), dict) else part

    filename = (
        file_obj.get("filename")
        or file_obj.get("name")
        or part.get("filename")
        or part.get("name")
        or "uploaded_file"
    )
    filename = os.path.basename(filename) or "uploaded_file"

    mime_type = file_obj.get("mime_type") or part.get("mime_type") or "application/octet-stream"

    data = (
        file_obj.get("data")
        or file_obj.get("file_data")
        or file_obj.get("base64")
        or part.get("data")
        or part.get("file_data")
        or part.get("base64")
    )

    if not data and isinstance(file_obj.get("url"), str):
        decoded = _decode_data_url(file_obj.get("url"))
        if decoded is not None:
            return filename, mime_type, decoded

    if not data and isinstance(part.get("url"), str):
        decoded = _decode_data_url(part.get("url"))
        if decoded is not None:
            return filename, mime_type, decoded

    if data:
        if isinstance(data, str) and data.startswith("data:"):
            decoded = _decode_data_url(data)
            if decoded is None:
                raise ValueError("Invalid data URL for file upload")
            return filename, mime_type, decoded
        try:
            return filename, mime_type, base64.b64decode(data)
        except Exception as exc:
            raise ValueError("Invalid base64 data for file upload") from exc

    if file_obj.get("file_id") or part.get("file_id"):
        raise ValueError("file_id attachments are not supported without file data")

    raise ValueError("No file data provided")


def _unique_upload_path(filename: str) -> Path:
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    candidate = _UPLOAD_DIR / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        alt = _UPLOAD_DIR / f"{stem}_{counter}{suffix}"
        if not alt.exists():
            return alt
        counter += 1


def _normalize_messages(
    req: ChatCompletionRequest,
) -> Tuple[list[Message], bool]:
    has_user = False
    has_attachment = False
    messages: list[Message] = []

    for msg in req.messages:
        content = msg.content
        if msg.role == "user" and msg.content:
            has_user = True

        if isinstance(content, list):
            normalized_parts = []
            for part in content:
                if isinstance(part, str):
                    normalized_parts.append({"type": "text", "text": part})
                    continue

                if _is_image_part(part):
                    has_attachment = True
                    normalized_parts.append(part)
                    continue

                if _is_file_part(part):
                    has_attachment = True
                    try:
                        filename, mime_type, data = _extract_file_payload(part)
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc

                    if len(data) > _MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File '{filename}' exceeds max upload size ({_MAX_UPLOAD_BYTES} bytes)",
                        )

                    save_path = _unique_upload_path(filename)
                    save_path.write_bytes(data)
                    rel_path = str(save_path.relative_to(config.workspace_root))
                    normalized_parts.append(
                        {
                            "type": "text",
                            "text": f"[User uploaded file saved to {rel_path} ({mime_type})]",
                        }
                    )
                    continue

                normalized_parts.append(part)

            content = normalized_parts

        messages.append(
            Message(
                role=msg.role,
                content=content,
                name=msg.name,
            )
        )

    if not has_user:
        raise HTTPException(status_code=400, detail="No user message found")

    return messages, has_attachment


def _apply_attachment_model_override(
    new_config, groq_api_key: Optional[str]
) -> None:
    groq_settings = None
    for _, settings in new_config.llm.items():
        if settings.api_type == "groq":
            groq_settings = settings
            break

    if groq_settings:
        updated = groq_settings.model_copy(
            update={
                "model": _ATTACHMENT_MODEL,
                "api_key": groq_api_key or groq_settings.api_key,
            }
        )
    else:
        default_settings = new_config.llm.get("default")
        if default_settings is None:
            raise HTTPException(status_code=500, detail="No default model configuration")
        updated = default_settings.model_copy(
            update={
                "model": _ATTACHMENT_MODEL,
                "api_type": "groq",
                "base_url": "https://api.groq.com/openai/v1",
                "api_key": groq_api_key or default_settings.api_key,
            }
        )

    new_llm = new_config.llm.copy()
    new_llm["default"] = updated
    new_config.llm = new_llm


def _clean_rel_path(path: Optional[str]) -> str:
    rel = (path or "").lstrip("/")
    parts = Path(rel).parts
    if any(part == ".." for part in parts):
        raise HTTPException(status_code=400, detail="Invalid path")
    return rel


def _resolve_local_path(rel_path: str) -> Path:
    root = Path(config.workspace_root).resolve()
    if not rel_path:
        return root
    resolved = (root / rel_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid path") from exc
    return resolved


def _rel_from_root(path: Path) -> str:
    return path.resolve().relative_to(Path(config.workspace_root).resolve()).as_posix()


def _entry_dict(path: Path, rel_path: str) -> dict:
    stat = path.stat()
    return {
        "path": rel_path,
        "is_dir": path.is_dir(),
        "size": stat.st_size,
        "modified": stat.st_mtime,
    }


def _list_local_entries(path: Path, recursive: bool) -> list[dict]:
    root = Path(config.workspace_root).resolve()
    entries: list[dict] = []
    if path.is_file():
        rel_path = _rel_from_root(path)
        if should_exclude_file(rel_path):
            return entries
        return [_entry_dict(path, rel_path)]

    if recursive:
        for dir_path, dir_names, file_names in os.walk(path):
            rel_dir = Path(dir_path).resolve().relative_to(root).as_posix()
            dir_names[:] = [
                d
                for d in dir_names
                if not should_exclude_file(
                    os.path.join(rel_dir, d) if rel_dir else d
                )
            ]
            for name in dir_names:
                rel_path = os.path.join(rel_dir, name) if rel_dir else name
                entry_path = Path(dir_path) / name
                entries.append(_entry_dict(entry_path, rel_path))
            for name in file_names:
                rel_path = os.path.join(rel_dir, name) if rel_dir else name
                if should_exclude_file(rel_path):
                    continue
                entry_path = Path(dir_path) / name
                entries.append(_entry_dict(entry_path, rel_path))
    else:
        for entry in path.iterdir():
            rel_path = _rel_from_root(entry)
            if should_exclude_file(rel_path):
                continue
            entries.append(_entry_dict(entry, rel_path))

    entries.sort(key=lambda item: (not item["is_dir"], item["path"]))
    return entries


async def _sandbox_stat(path: str) -> dict:
    if not SANDBOX_CLIENT.sandbox:
        await SANDBOX_CLIENT.create(config=config.sandbox)
    script = (
        "import json, os\n"
        f"path = {path!r}\n"
        "if not os.path.exists(path):\n"
        "    print(json.dumps({'exists': False}))\n"
        "else:\n"
        "    st = os.stat(path)\n"
        "    print(json.dumps({'exists': True, 'is_dir': os.path.isdir(path), 'size': st.st_size, 'modified': st.st_mtime}))\n"
    )
    output = await SANDBOX_CLIENT.run_command(f"python -c {script!r}")
    return json.loads(output.strip() or "{}")


async def _list_sandbox_entries(path: str, recursive: bool) -> list[dict]:
    if not SANDBOX_CLIENT.sandbox:
        await SANDBOX_CLIENT.create(config=config.sandbox)
    workspace_root = config.sandbox.work_dir
    script = (
        "import json, os\n"
        f"workspace_root = {workspace_root!r}\n"
        f"target = {path!r}\n"
        f"recursive = {str(recursive)}\n"
        "entries = []\n"
        "if not os.path.exists(target):\n"
        "    print(json.dumps({'error': 'not_found'}))\n"
        "    raise SystemExit(0)\n"
        "def add_entry(full_path):\n"
        "    rel = os.path.relpath(full_path, workspace_root)\n"
        "    st = os.stat(full_path)\n"
        "    entries.append({'path': rel.replace('\\\\', '/'), 'is_dir': os.path.isdir(full_path), 'size': st.st_size, 'modified': st.st_mtime})\n"
        "if os.path.isfile(target):\n"
        "    add_entry(target)\n"
        "else:\n"
        "    if recursive:\n"
        "        for root, dirs, files in os.walk(target):\n"
        "            for name in dirs:\n"
        "                add_entry(os.path.join(root, name))\n"
        "            for name in files:\n"
        "                add_entry(os.path.join(root, name))\n"
        "    else:\n"
        "        for name in os.listdir(target):\n"
        "            add_entry(os.path.join(target, name))\n"
        "print(json.dumps({'entries': entries}))\n"
    )
    output = await SANDBOX_CLIENT.run_command(f"python -c {script!r}")
    payload = json.loads(output.strip() or "{}")
    if payload.get("error") == "not_found":
        raise HTTPException(status_code=404, detail="Path not found")
    entries = payload.get("entries", [])
    filtered = [
        entry for entry in entries if not should_exclude_file(entry.get("path", ""))
    ]
    filtered.sort(key=lambda item: (not item["is_dir"], item["path"]))
    return filtered


async def _sandbox_file_response(path: str, filename: str, size: int):
    if size > _MAX_DOWNLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds max download size ({_MAX_DOWNLOAD_BYTES} bytes)",
        )
    if not SANDBOX_CLIENT.sandbox:
        await SANDBOX_CLIENT.create(config=config.sandbox)
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.close()
    await SANDBOX_CLIENT.copy_from(path, tmp_file.name)
    return FileResponse(
        tmp_file.name,
        filename=filename,
        media_type="application/octet-stream",
        background=BackgroundTask(lambda: os.remove(tmp_file.name)),
    )


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


@router.get("/workspace/files", dependencies=[Depends(require_openai_auth)])
async def list_workspace_files(
    path: Optional[str] = Query(default=None),
    recursive: bool = Query(default=False),
):
    rel_path = _clean_rel_path(path)
    if rel_path and should_exclude_file(rel_path):
        raise HTTPException(status_code=404, detail="Path not found")

    if config.sandbox and config.sandbox.use_sandbox:
        sandbox_root = config.sandbox.work_dir
        target = os.path.join(sandbox_root, rel_path) if rel_path else sandbox_root
        entries = await _list_sandbox_entries(target, recursive)
    else:
        target = _resolve_local_path(rel_path)
        if not target.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        entries = _list_local_entries(target, recursive)

    return {"path": rel_path or "", "files": entries}


@router.get("/workspace/file", dependencies=[Depends(require_openai_auth)])
async def get_workspace_file(path: str = Query(...)):
    rel_path = _clean_rel_path(path)
    if should_exclude_file(rel_path):
        raise HTTPException(status_code=404, detail="Path not found")

    if config.sandbox and config.sandbox.use_sandbox:
        sandbox_root = config.sandbox.work_dir
        target = os.path.join(sandbox_root, rel_path)
        stat = await _sandbox_stat(target)
        if not stat.get("exists"):
            raise HTTPException(status_code=404, detail="Path not found")
        if stat.get("is_dir"):
            raise HTTPException(status_code=400, detail="Path is a directory")
        return await _sandbox_file_response(target, os.path.basename(rel_path), stat.get("size", 0))

    local_path = _resolve_local_path(rel_path)
    if not local_path.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if local_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    size = local_path.stat().st_size
    if size > _MAX_DOWNLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds max download size ({_MAX_DOWNLOAD_BYTES} bytes)",
        )
    return FileResponse(local_path, filename=local_path.name)


@router.post("/chat/completions", dependencies=[Depends(require_openai_auth)])
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return _error_response("'messages' must not be empty", status_code=400)

    try:
        normalized_messages, has_attachment = _normalize_messages(req)
    except HTTPException as e:
        return _error_response(str(e.detail), status_code=e.status_code)

    created = _now()
    completion_id = _new_id("chatcmpl")
    effective_model = req.model

    # Handle dynamic configuration overrides
    token = None
    if req.groq_api_key or req.daytona_api_key or req.model or has_attachment:
        new_config = config.current_config.model_copy()

        if req.groq_api_key:
            new_llm = new_config.llm.copy()
            for name, settings in new_llm.items():
                if settings.api_type == "groq":
                    new_llm[name] = settings.model_copy(
                        update={"api_key": req.groq_api_key}
                    )
            new_config.llm = new_llm

        if req.model and not has_attachment:
            new_llm = new_config.llm.copy()
            # Also update 'default' to use the requested model
            if "default" in new_llm:
                new_llm["default"] = new_llm["default"].model_copy(
                    update={"model": req.model}
                )
            new_config.llm = new_llm
            effective_model = req.model

        if has_attachment:
            _apply_attachment_model_override(new_config, req.groq_api_key)
            effective_model = _ATTACHMENT_MODEL

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
                    model=effective_model,
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
                    agent.messages = normalized_messages
                    async for step_result in agent.run_stream():
                        accumulated_content += step_result + "\n"
                        chunk = ChatCompletionChunkResponse(
                            id=completion_id,
                            created=created,
                            model=effective_model,
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
                        model=effective_model,
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
                    model=effective_model,
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
            agent.messages = normalized_messages
            result = await agent.run()
            content = _extract_best_output(agent, fallback=result)
        finally:
            await agent.cleanup()

        usage = ChatCompletionUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
        resp = ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=effective_model,
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
