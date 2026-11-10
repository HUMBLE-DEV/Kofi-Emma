from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import hashlib
import uuid

import bcrypt
from jose import JWTError, jwt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from src.config import settings

serializer = URLSafeTimedSerializer(
    secret_key=settings.SECRET_KEY ,
    salt="email-verify"
)

BCRYPT_ROUNDS = 12


def _prepare_password(password: str) -> bytes:
    """Pre-hash with SHA-256 before bcrypt.

    bcrypt only considers the first 72 bytes of its input, so hashing the raw
    password would silently treat two different long passwords sharing a prefix
    as identical. SHA-256 gives us a fixed 64-byte digest, preserving the full
    entropy of the original password (the same approach Django uses).
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest().encode("ascii")


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt (cost 12)."""
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(_prepare_password(password), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(
            _prepare_password(plain_password),
            hashed_password.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


def create_access_token(
    user_data: Dict[str, Any],
    refresh: bool = False,
    expiry: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token with JTI for revocation support."""
    to_encode: Dict[str, Any] = {
        "user": user_data.copy(),
        "refresh": refresh,
        "iat": int(datetime.now(timezone.utc).timestamp()),
    }

    if expiry is None:
        if refresh:
            expiry = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        else:
            expiry = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode["exp"] = int((datetime.now(timezone.utc) + expiry).timestamp())
    to_encode["jti"] = str(uuid.uuid4())

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT token. Returns payload dict or None."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None


def create_url_safe_token(
    data: Dict[str, Any]
) -> str:
    """Create a time-limited URL-safe signed token (for email verification / reset)."""
    return serializer.dumps(data)


def decode_url_safe_token(
    token: str,
    max_age: int = 3600,
) -> Optional[Dict[str, Any]]:
    """Decode a URL-safe signed token. Returns data dict or None if invalid/expired."""
    try:
        return serializer.loads(token, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None