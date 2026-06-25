"""Optional JWT authentication with role-based access (RBAC).

Demo users are configured via ``CXR_USERS``:
    clinician:secret:clinician,admin:secret:admin

When ``CXR_JWT_SECRET`` is empty, auth is disabled and all routes remain open
(for local demos). Enable in staging/production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

try:
    import jwt
except ImportError:  # pragma: no cover
    jwt = None  # type: ignore

_bearer = HTTPBearer(auto_error=False)

ROLES = {"viewer", "clinician", "admin"}


@dataclass
class User:
    username: str
    role: str


def auth_enabled() -> bool:
    return bool(os.getenv("CXR_JWT_SECRET", "").strip())


def _parse_users() -> dict[str, tuple[str, str]]:
    raw = os.getenv(
        "CXR_USERS",
        "clinician:changeme:clinician,admin:changeme:admin",
    )
    users: dict[str, tuple[str, str]] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 3:
            continue
        username, password, role = bits
        if role in ROLES:
            users[username] = (password, role)
    return users


def authenticate(username: str, password: str) -> User | None:
    creds = _parse_users().get(username)
    if creds and creds[0] == password:
        return User(username=username, role=creds[1])
    return None


def create_token(user: User, hours: int = 8) -> str:
    secret = os.getenv("CXR_JWT_SECRET", "")
    if not jwt or not secret:
        raise RuntimeError("JWT support requires pyjwt and CXR_JWT_SECRET")
    payload = {
        "sub": user.username,
        "role": user.role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=hours),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str) -> User:
    if not jwt:
        raise HTTPException(status_code=501, detail="pyjwt not installed")
    secret = os.getenv("CXR_JWT_SECRET", "")
    try:
        data = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc
    return User(username=data["sub"], role=data.get("role", "viewer"))


def get_optional_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    x_api_key: str | None = Header(default=None),
) -> User | None:
    if not auth_enabled():
        return User(username="anonymous", role="admin")
    if credentials and credentials.credentials:
        return decode_token(credentials.credentials)
    keys = {k for k in os.getenv("CXR_API_KEYS", "").split(",") if k}
    if keys and x_api_key in keys:
        return User(username="api-key", role="clinician")
    return None


def require_role(*roles: str):
    def _dep(user: Annotated[User | None, Depends(get_optional_user)]) -> User:
        if not auth_enabled():
            return user or User(username="anonymous", role="admin")
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        if user.role not in roles and user.role != "admin":
            raise HTTPException(status_code=403, detail=f"Requires role: {', '.join(roles)}")
        return user

    return _dep
