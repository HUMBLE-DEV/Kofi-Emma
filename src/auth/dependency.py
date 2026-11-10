from typing import List
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.requests import Request
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from jose import ExpiredSignatureError
from sqlmodel.ext.asyncio.session import AsyncSession

from src.auth.utility import decode_token
from src.auth.service import AuthService
from src.auth.models import User
from src.db.main import get_session
from src.errors import (
    InvalidToken,
    RevokedToken,
    AccessTokenRequired,
    RefreshTokenRequired,
    AccountNotVerified,
    InsufficientPermission,
)
from src.config import settings

import redis.asyncio as aioredis


auth_service = AuthService()

# Redis connection with short timeouts so a broker outage never blocks auth requests.
# These are separate from the guardrails Redis pool (which has its own timeouts).
_token_store = None


def _get_token_store():
    global _token_store
    if _token_store is None:
        _token_store = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            socket_keepalive=True,
            health_check_interval=30,
        )
    return _token_store


def get_token_store():
    """Access the Redis client used for token revocation checks."""
    return _get_token_store()

logger = logging.getLogger("ScholarGraph-auth")


async def is_token_revoked(jti: str) -> bool:
    """Check whether a JTI has been revoked (present in the blocklist).

    If Redis is unreachable the check fails open so a broker outage does not
    lock every user out of the platform. Set TOKEN_BLOCKLIST_STRICT=true to
    fail closed instead.
    """
    if not jti:
        return False
    try:
        result = await get_token_store().get(jti)
        return result is not None
    except Exception as exc:
        logger.warning("Redis unreachable during revocation check: %s", exc)
        if settings.TOKEN_BLOCKLIST_STRICT:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Token validation is temporarily unavailable",
            )
        return False 


async def add_jti_to_blocklist(jti: str) -> None:
    """Persist a revoked JTI in Redis with a TTL matching the token's lifetime."""
    try:
        await get_token_store().set(name=jti, value="1", ex=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    except Exception as exc:
        logger.warning("Could not write to the token blocklist: %s", exc)


class TokenBearer(HTTPBearer):
    """Base HTTPBearer that validates JWT tokens and checks revocation."""

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> dict | None:
        creds: HTTPAuthorizationCredentials = await super().__call__(request)
        if creds is None:
            return None

        token = creds.credentials

        try:
            token_data = decode_token(token)
        except ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="token_expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if token_data is None:
            raise InvalidToken()

        if await is_token_revoked(token_data.get("jti")):
            raise RevokedToken()

        self.verify_token_type(token_data)
        return token_data

    def verify_token_type(self, token_data: dict) -> None:
        """Override in subclasses to enforce token type (access vs refresh)."""
        pass


class AccessTokenBearer(TokenBearer):
    """Rejects refresh tokens — only valid for access-protected routes."""

    def verify_token_type(self, token_data: dict) -> None:
        if token_data.get("refresh"):
            raise AccessTokenRequired()


class RefreshTokenBearer(TokenBearer):
    """Rejects access tokens — only valid for the refresh endpoint."""

    def verify_token_type(self, token_data: dict) -> None:
        if not token_data.get("refresh"):
            raise RefreshTokenRequired()


async def get_current_user(
    token_data: dict = Depends(AccessTokenBearer()),
    session: AsyncSession = Depends(get_session),
) -> User:
    """Resolve the authenticated user from a valid access token."""
    email = token_data["user"]["email"]
    user = await auth_service.get_user_by_email(email, session)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user

def get_checkpointer(request: Request) -> AsyncSqliteSaver:
    return request.app.state.checkpointer


class RoleChecker:
    """Dependency that enforces role membership and account verification."""

    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)):
        if not current_user.is_verified:
            raise AccountNotVerified()
        if current_user.role not in self.allowed_roles:
            raise InsufficientPermission()
        return True