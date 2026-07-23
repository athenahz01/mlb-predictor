from __future__ import annotations

from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException, status

from athena_api.settings import Settings, get_settings


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str | None = None


def current_user(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> AuthUser:
    if not authorization:
        if settings.auth_required:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in required")
        return AuthUser(id="local-development", email="local@athena.invalid")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    if not settings.supabase_jwt_secret:
        raise HTTPException(status_code=503, detail="Supabase JWT verification is not configured")
    try:
        claims = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired session") from exc
    subject = claims.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="Session is missing a user id")
    return AuthUser(id=subject, email=claims.get("email"))
