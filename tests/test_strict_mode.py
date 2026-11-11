"""Verify TOKEN_BLOCKLIST_STRICT: with Redis down, strict mode must fail closed."""
import os, sys, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKEN_BLOCKLIST_STRICT"] = "True"

from fastapi.testclient import TestClient
from src.app import app
from src.auth.utility import create_access_token
from src.auth import dependency as auth_dependency

# Simulate Redis being unavailable so strict mode triggers.
class FakeRedisDown:
    async def get(self, name):
        raise ConnectionError("Redis is unavailable")

auth_dependency._token_store = FakeRedisDown()

tok = create_access_token({"email": "student@example.com", "user_id": "1", "role": "student", "name": "Test Student"})

with TestClient(app, raise_server_exceptions=False) as client:
    r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tok}"})
    print("strict mode, Redis down -> HTTP", r.status_code, r.text[:200])
    assert r.status_code == 503, f"expected 503, got {r.status_code}"
    print("PASS: strict mode fails closed when Redis is unreachable")
