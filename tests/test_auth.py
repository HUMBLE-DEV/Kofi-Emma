"""
End-to-end verification of the ScholarGraph auth stack.

Run with:  uv run python tests/test_auth.py
"""

import os
import sys
import asyncio
import uuid

# Use a dedicated throwaway database so the suite is repeatable.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./tests/test_research_agent.db"
os.environ["REDIS_URL"] = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from src.app import app
from src.db.main import init_db, engine
from src.auth import dependency as auth_dependency
from src.auth.utility import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
    create_url_safe_token,
    decode_url_safe_token,
)
from src.config import settings

PASSED, FAILED = [], []


class FakeRedis:
    """Minimal async stand-in for redis.asyncio, so the blocklist logic can be
    exercised without a live Redis server."""

    def __init__(self):
        self.store = {}

    async def get(self, name):
        return self.store.get(name)

    async def set(self, name, value, ex=None):
        self.store[name] = value
        return True


def check(name, condition, extra=""):
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name} {extra}")


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ----------------------------------------------------------------------
section("1. Password hashing (bcrypt + SHA-256 pre-hash)")
# ----------------------------------------------------------------------
h = hash_password("testpassword123")
check("hash is bcrypt format", h.startswith("$2b$"), h[:10])
check("correct password verifies", verify_password("testpassword123", h))
check("wrong password rejected", not verify_password("wrongpassword", h))
check("hash is salted (2 hashes differ)", hash_password("same") != hash_password("same"))

long_pw = "A" * 200 + "B"
lh = hash_password(long_pw)
check("long password verifies", verify_password(long_pw, lh))
check("long password not truncated (different suffix fails)",
      not verify_password("A" * 200 + "C", lh))

# ----------------------------------------------------------------------
section("2. JWT access / refresh tokens")
# ----------------------------------------------------------------------
user_data = {"email": "a@b.com", "user_id": "123", "role": "student", "name": "Test"}
access = create_access_token(user_data)
refresh = create_access_token(user_data, refresh=True)

d_access = decode_token(access)
d_refresh = decode_token(refresh)
check("access token decodes", d_access is not None)
check("user payload nested under 'user'", d_access["user"]["email"] == "a@b.com")
check("access token flagged refresh=False", d_access["refresh"] is False)
check("refresh token flagged refresh=True", d_refresh["refresh"] is True)
check("tokens have unique jti", d_access["jti"] != d_refresh["jti"])
check("exp > iat", d_access["exp"] > d_access["iat"])
check("garbage token returns None", decode_token("not.a.jwt") is None)

# ----------------------------------------------------------------------
section("3. itsdangerous URL-safe tokens")
# ----------------------------------------------------------------------
vtok = create_url_safe_token({"email": "a@b.com"}, settings.SECRET_KEY)
vt = decode_url_safe_token(vtok, settings.SECRET_KEY)
check("verification token decodes", vt is not None and vt["email"] == "a@b.com")
check("wrong secret rejected", decode_url_safe_token(vtok, "totally-wrong") is None)
check("garbage token rejected", decode_url_safe_token("garbage", settings.SECRET_KEY) is None)
expired = decode_url_safe_token(vtok, settings.SECRET_KEY, max_age=-1)
check("expired token rejected", expired is None)

# ----------------------------------------------------------------------
section("4. API endpoints (TestClient)")
# ----------------------------------------------------------------------
asyncio.run(init_db())

email = f"student_{uuid.uuid4().hex[:10]}@example.com"
username = f"Student {uuid.uuid4().hex[:8]}"
password = "ResearchPass123!"

with TestClient(app) as client:
    r = client.get("/health")
    check("GET /health returns 200", r.status_code == 200, r.text)
    r = client.get("/api/v1/docs")
    check("GET /api/v1/docs returns 200", r.status_code == 200)

    # -- signup --
    r = client.post("/api/v1/auth/signup", json={
        "email": email, "username": username, "password": password,
    })
    check("POST /signup returns 201", r.status_code == 201, r.text[:300])
    if r.status_code == 201:
        body = r.json()
        check("signup returns user payload", body["user"]["email"] == email)
        check("signup defaults role to student", body["user"]["role"] == "student")

    # -- duplicate signup --
    r = client.post("/api/v1/auth/signup", json={
        "email": email, "username": username, "password": password,
    })
    check("duplicate signup returns 409", r.status_code == 409, r.text[:200])

    # -- duplicate username (different email) --
    r = client.post("/api/v1/auth/signup", json={
        "email": f"other_{uuid.uuid4().hex[:8]}@example.com", "username": username, "password": password,
    })
    check("duplicate username returns 409", r.status_code == 409, r.text[:200])

    # -- weak password rejected --
    r = client.post("/api/v1/auth/signup", json={
        "email": f"weak_{uuid.uuid4().hex[:6]}@example.com", "username": "W", "password": "short",
    })
    check("short password rejected (422)", r.status_code == 422, r.text[:200])

    # -- login with wrong password --
    r = client.post("/api/v1/auth/login", json={"email": email, "password": "wrong"})
    check("login with wrong password returns 401", r.status_code == 401, r.text[:200])

    # -- login unknown user --
    r = client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": password})
    check("login unknown user returns 401", r.status_code == 401, r.text[:200])

    # -- login (correct) --
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    check("POST /login returns 200", r.status_code == 200, r.text[:300])
    access_token = refresh_token = None
    if r.status_code == 200:
        data = r.json()
        access_token = data["access_token"]
        refresh_token = data["refresh_token"]
        check("login returns access_token", bool(access_token))
        check("login returns refresh_token", bool(refresh_token))
        check("login returns token_type bearer", data["token_type"] == "bearer")
        check("login returns is_verified False", data["user"]["is_verified"] is False)

    # -- protected route without token --
    r = client.get("/api/v1/auth/me")
    check("GET /me without token returns 403/401", r.status_code in (401, 403), r.text[:200])

    # -- protected route with garbage token --
    r = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer garbage.token.here"})
    check("GET /me with garbage token returns 401", r.status_code == 401, r.text[:200])

    # -- protected route with valid token --
    r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    check("GET /me with valid token returns 200", r.status_code == 200, r.text[:300])
    if r.status_code == 200:
        check("/me returns correct email", r.json()["email"] == email)
        check("/me returns is_verified False", r.json()["is_verified"] is False)

    # -- refresh endpoint rejects access token --
    r = client.get("/api/v1/auth/refresh", headers={"Authorization": f"Bearer {access_token}"})
    check("GET /refresh with access token returns 401", r.status_code == 401, r.text[:200])

    # -- refresh endpoint accepts refresh token --
    r = client.get("/api/v1/auth/refresh", headers={"Authorization": f"Bearer {refresh_token}"})
    check("GET /refresh with refresh token returns 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        new_access = r.json()["access_token"]
        r2 = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {new_access}"})
        check("refreshed access token works on /me", r2.status_code == 200, r2.text[:200])

    # -- email verification --
    vtok = create_url_safe_token({"email": email}, settings.SECRET_KEY)
    r = client.get(f"/api/v1/auth/verify?token={vtok}")
    check("GET /verify with valid token returns 200", r.status_code == 200, r.text[:200])

    r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    check("is_verified True after verification", r.json()["is_verified"] is True)

    r = client.get("/api/v1/auth/verify?token=invalidtoken")
    check("GET /verify with invalid token returns 400", r.status_code == 400, r.text[:200])

    # -- forgot password (known email) --
    r = client.post("/api/v1/auth/forgot_password", json={"email": email})
    check("POST /forgot_password returns 200", r.status_code == 200, r.text[:200])

    # -- forgot password (unknown email) -- must be identical, no enumeration --
    r_unknown = client.post("/api/v1/auth/forgot_password", json={"email": "ghost@example.com"})
    check("POST /forgot_password unknown email returns 200", r_unknown.status_code == 200)
    check("forgot_password responses identical (no user enumeration)",
          r.json()["message"] == r_unknown.json()["message"])

    # -- reset password --
    rtok = create_url_safe_token({"email": email}, settings.SECRET_KEY)
    r = client.post(f"/api/v1/auth/reset_password?token={rtok}", json={
        "new_password": "NewResearchPass456!", "confirm_password": "NewResearchPass456!",
    })
    check("POST /reset_password returns 200", r.status_code == 200, r.text[:200])

    # -- login with new password --
    r = client.post("/api/v1/auth/login", json={"email": email, "password": "NewResearchPass456!"})
    check("login works after password reset", r.status_code == 200, r.text[:200])
    new_access = r.json()["access_token"] if r.status_code == 200 else None

    # -- mismatched confirmation --
    mtok = create_url_safe_token({"email": email}, settings.SECRET_KEY)
    r = client.post(f"/api/v1/auth/reset_password?token={mtok}", json={
        "new_password": "AnotherPass789!", "confirm_password": "DifferentPass789!",
    })
    check("mismatched passwords return 400", r.status_code == 400, r.text[:200])

    # -- logout revokes the token --
    # Swap in an in-memory blocklist so revocation is exercised even when no
    # Redis server is running locally.
    fake_redis = FakeRedis()
    original_store = auth_dependency.token_store
    auth_dependency.token_store = fake_redis
    try:
        r = client.get("/api/v1/auth/logout", headers={"Authorization": f"Bearer {new_access}"})
        check("GET /logout returns 200", r.status_code == 200, r.text[:200])
        check("logout wrote the jti to the blocklist", len(fake_redis.store) == 1,
              str(fake_redis.store))
        r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {new_access}"})
        check("token rejected after logout (revoked)", r.status_code == 401, r.text[:200])

        # -- a revoked jti is detected by the revocation check --
        jti = next(iter(fake_redis.store))
        check("is_token_revoked detects the revoked jti",
              asyncio.run(auth_dependency.is_token_revoked(jti)) is True)
        check("is_token_revoked returns False for an unknown jti",
              asyncio.run(auth_dependency.is_token_revoked("never-seen")) is False)
    finally:
        auth_dependency.token_store = original_store

# ----------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("Failed checks:")
    for f in FAILED:
        print(f"  - {f}")
print("=" * 60)
sys.exit(1 if FAILED else 0)
