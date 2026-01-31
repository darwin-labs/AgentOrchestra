import os
from typing import Optional

from fastapi import Header, HTTPException, status


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None

    parts = authorization.split(" ", 1)
    if len(parts) != 2:
        return None

    scheme, token = parts[0].strip(), parts[1].strip()
    if scheme.lower() != "bearer" or not token:
        return None

    return token


def require_openai_auth(authorization: Optional[str] = Header(default=None)) -> None:
    expected = os.environ.get("OPENMANUS_API_KEY")
    if not expected:
        return

    token = _extract_bearer_token(authorization)
    if token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
