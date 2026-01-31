from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.openai_routes import router as openai_router
from app.api.openai_schemas import OpenAIErrorPayload, OpenAIErrorResponse


def create_app() -> FastAPI:
    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        payload = OpenAIErrorResponse(
            error=OpenAIErrorPayload(
                message=str(exc.detail),
                type="invalid_request_error",
            )
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        payload = OpenAIErrorResponse(
            error=OpenAIErrorPayload(
                message="Invalid request",
                type="invalid_request_error",
            )
        )
        return JSONResponse(status_code=400, content=payload.model_dump())

    @app.get("/")
    async def root():
        return {"status": "ok"}

    app.include_router(openai_router)
    return app


app = create_app()
