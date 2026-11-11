# 🎓 ScholarGraph

An AI research assistant for students, built on **LangGraph** with a production-grade
authentication and authorization layer.

ChatGPT-style conversational research help, gated behind secure JWT auth with email
verification, password reset, and background email delivery via Celery + Redis.

---

## Architecture

```
┌────────────────────┐        ┌──────────────────────────────┐
│  Streamlit (8501)  │───────▶│  FastAPI backend (8000)      │
│  Chat UI + login   │  HTTP  │  /api/v1/auth/*              │
└────────────────────┘        │  JWT auth + RBAC             │
                              └──────┬───────────────┬───────┘
                                     │               │
                              ┌──────▼──────┐  ┌─────▼──────────┐
                              │  Neon Postgres  │  │  Redis         │
                              │  (users + docs) │  │  blocklist +   │
                              └─────────────┘  └─────┬──────────┘
                                                     │ background task
                                              ┌──────▼──────┐
                                              │  Celery     │
                                              │  Worker     │──▶ SMTP (email)
                                              └─────────────┘
```

### Project layout

```
src/
├── config.py            # Settings (env-driven via pydantic-settings)
├── errors.py            # Typed HTTP exceptions (401/403/404/409)
├── logging_config.py    # Shared logger factory
├── mail.py              # fastapi-mail connection config
├── celery_worker.py     # Celery app + Send_email task
├── app.py               # FastAPI application assembly
├── db/
│   └── main.py          # Async SQLModel engine + session dependency
└── auth/
    ├── models.py        # SQLModel User table
    ├── schema.py        # Pydantic request/response schemas
    ├── utility.py       # Password hashing, JWT, itsdangerous tokens
    ├── service.py       # AuthService — business logic
    ├── dependency.py    # HTTPBearer token dependencies + RoleChecker
    └── routers.py       # Auth endpoints
backend/app.py           # LangGraph chatbot agent (research nodes read papers via doc_service)
backend/router.py        # HTTP routes: chat, uploads, documents, paper search
backend/services/        # OpenAlex search, citation chaining, LLM calls
backend/doc_models.py    # Document + SavedPaper SQLModel tables
backend/services/doc_service.py  # Document & saved-paper database helpers
main.py                  # Streamlit frontend (chat page + paper-search page)
tests/                   # Auth test suite
docker-compose.yml       # Full stack orchestration
Dockerfile               # Container build
.github/workflows/       # CI/CD pipeline
.env                     # Local development secrets (gitignored)
.env.example             # Example environment file (commit this)
```

---

## Security model

| Concern | Implementation |
|---|---|
| Password storage | **bcrypt** (cost 12) with SHA-256 pre-hash, so long passwords are never silently truncated at bcrypt's 72-byte limit |
| Access tokens | JWT (HS256), 60 min TTL, `HTTPBearer` protected |
| Refresh tokens | JWT, 7 day TTL, rejected by access-protected routes and vice-versa |
| Logout | Token `jti` written to a Redis blocklist with a TTL matching the token's life |
| Email verification | `itsdangerous` URL-safe signed token, 1 hour expiry, **separate secret** from password reset |
| Password reset | `itsdangerous` URL-safe signed token, 1 hour expiry, **separate secret** from verification |
| User enumeration | `/forgot_password` returns an identical response whether or not the email exists |
| Duplicate accounts | Unique on both `email` and `username`; duplicates return `409`, never a `500` |
| Role checks | `RoleChecker` requires a verified account **and** a matching role |

### Token format

```jsonc
{
  "user": { "email": "...", "user_id": "...", "role": "student", "name": "..." },
  "refresh": false,          // true for refresh tokens
  "iat": 1763000000,
  "exp": 1763003600,
  "jti": "8f14e45f-..."       // blocklist key
}
```

---

## API endpoints

All routes are prefixed with `/api/v1/auth`.

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/signup` | — | Register; sends a verification email |
| `GET` | `/verify?token=…` | — | Activate the account |
| `POST` | `/login` | — | Returns access + refresh tokens |
| `GET` | `/me` | Bearer | Current user profile |
| `GET` | `/refresh` | Refresh | Mint a new access token |
| `GET` | `/logout` | Bearer | Revoke the token via Redis blocklist |
| `POST` | `/forgot_password` | — | Email a reset link (no user enumeration) |
| `POST` | `/reset_password?token=…` | — | Set a new password |

Interactive docs: `http://localhost:8000/api/v1/docs`

### Paper search endpoints

All routes are prefixed with `/api/v1`.

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/papers/search` | Bearer | Search OpenAlex for papers by topic + optional year range |
| `POST` | `/papers/save` | Bearer | Save an OpenAlex paper to your account (available in every chat) |
| `GET` | `/papers` | Bearer | List all papers saved to your account |
| `DELETE` | `/papers/{paper_id}` | Bearer | Remove a saved paper |

Saved papers are scoped to the **user**, not a chat thread — once saved, a paper is available to
the research nodes in every conversation. The agent reads both uploaded documents and saved
OpenAlex papers through the same `get_thread_papers()` helper.

---

## Setup

### 1. Environment

```bash
cp .env.example .env    # then fill in the values
```

Generate real secrets — never ship the placeholders:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"   # SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(48))"   # PASSWORD_RESET_TOKEN_SECRET
python -c "import secrets; print(secrets.token_urlsafe(48))"   # FRONTEND_BASE_URL (Streamlit app URL)
```

### 2. Install

```bash
uv sync
```

### 3. Run (development)

Three processes — the API, the Celery worker, Redis, and the Streamlit frontend:

```bash
# Redis
docker run -d --name ScholarGraph-redis -p 6379:6379 redis:7-alpine

# FastAPI backend
uv run uvicorn src.app:app --reload --port 8000

# Celery worker (background email delivery)
uv run celery -A src.celery_worker.celery_app worker --loglevel=info

# Streamlit frontend
uv run streamlit run main.py
```

Or bring up the whole backend stack at once:

```bash
docker compose up -d
```

### 4. Test

```bash
uv run python tests/test_auth.py        # 54 checks across the whole auth surface
uv run python tests/test_strict_mode.py # fail-closed behaviour when Redis is down
```

---

## Configuration

All settings come from `.env` (see `src/config.py`).

| Variable | Default | Notes |
|---|---|---|
| `SECRET_KEY` | *required* | Signs JWTs. Rotating it invalidates all tokens. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Access token lifetime |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | Refresh token lifetime |
| `PUBLIC_BASE_URL` | `http://localhost:8000` | Base for email links (API) |
| `FRONTEND_BASE_URL` | `http://localhost:8501` | Base for password reset links (Streamlit frontend) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./research_agent.db` | Local: relative path. Docker: `sqlite+aiosqlite:////app/data/research_agent.db` (on app_data volume). Production: `postgresql+asyncpg://user:pass@host:5432/db` for Neon Postgres |
| `REDIS_URL` | `redis://localhost:6379/0` | Blocklist + Celery broker |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | |
| `TOKEN_BLOCKLIST_STRICT` | `false` | `true` = fail closed when Redis is down |
| `DEFAULT_MODEL` | `gemini-2.5-flash` | Default Gemini model |
| `GROQ_API_KEY` | *required* | Used when every Gemini model is rate-limited |
| `FALLBACK_GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model used in the fallback chain |
| `MAIL_*` | — | SMTP credentials, server, port, sender |

### Redis availability

The token blocklist **fails open** by default: if Redis is unreachable, requests are
still served (a warning is logged) so a broker outage never locks students out of
their research. Set `TOKEN_BLOCKLIST_STRICT=true` to fail closed with a `503` instead —
recommended in production, where losing revocation tracking is a real risk.

### LLM fallback chain

Every LLM call goes through `backend/services/llm_call.py`:

```
selected Gemini model → next Gemini model → Groq (openai/gpt-oss-120b)
```

The chain catches **every** exception type, so a `GoogleRateLimitError` on `gemini-2.5-flash`
falls through to Groq instead of failing the turn. If every model is exhausted, the student
sees *"The AI model's daily quota has been exceeded. Please try again later."* rather than a
raw Google error dump. Set `FALLBACK_GROQ_MODEL` in `.env` to change the Groq model.

---

## Email delivery

Verification and reset emails are handed to Celery and sent by the worker, so an API
request never waits on SMTP. The request path is deliberately decoupled:

1. The API signs the token and returns immediately.
2. A FastAPI background task publishes `Send_email` to Redis.
3. The Celery worker picks it up and delivers the mail.

If Redis is unavailable the API falls back to sending the mail directly rather than
failing the signup. The Celery app publishes with **no result backend** — task results
are never consumed, and keeping one configured forces a result-store consumer channel
open on every publish (which stalls for ~100 s against a dead Redis).

---

## Roles

`User.role` defaults to `student`. `RoleChecker` gates protected routes:

```python
from src.auth.dependency import RoleChecker

@router.get("/admin/users", dependencies=[Depends(RoleChecker(["admin"]))])
async def list_users(): ...
```

`RoleChecker` requires a **verified** account as well as a matching role, so an
unverified student cannot reach a student-only route either.

---

## Deployment — AWS EC2

This guide deploys ScholarGraph on a single EC2 instance using Docker Compose.

### Prerequisites

- AWS account with an EC2 instance (Ubuntu 22.04+ recommended)
- SSH access to the instance
- Neon Postgres database URL (production database)
- SMTP credentials (Gmail, SendGrid, Mailgun, etc.) for email verification / password reset
- GitHub repository with the code pushed

### 1. Launch the EC2 instance

| Setting | Recommended value |
|---|---|
| AMI | Ubuntu 22.04 LTS |
| Instance type | t3.medium (2 vCPU, 4 GiB) minimum; t3.large for production |
| Storage | 30 GB gp3 |
| Security group | SSH (22), HTTP (80), HTTPS (443), Custom TCP 8000 (API), Custom TCP 8501 (Streamlit) |

```bash
ssh -i your-key.pem ubuntu@<your-ec2-public-ip>
```

### 2. Install dependencies on the instance

```bash
sudo apt-get update && sudo apt-get upgrade -y

# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Docker Compose (v2 plugin is included with Docker on Ubuntu)

# Git
sudo apt-get install -y git

# uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Log out and back in so the `docker` group takes effect.

### 3. Clone the repository

```bash
git clone https://github.com/<your-org>/scholargraph.git
cd scholargraph
```

### 4. Create the `.env` file

Copy the example and fill in all values:

```bash
cp .env.example .env
nano .env
```

Required production values:

| Variable | Example |
|---|---|
| `SECRET_KEY` | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `PASSWORD_RESET_TOKEN_SECRET` | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `EMAIL_VERIFICATION_TOKEN_SECRET` | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `PUBLIC_BASE_URL` | `https://<your-domain-or-ec2-ip>` |
| `FRONTEND_BASE_URL` | `https://<your-domain-or-ec2-ip>` |
| `DATABASE_URL` | `postgresql+asyncpg://user:pass@<neon-host>:5432/db` |
| `REDIS_URL` | `redis://localhost:6379/0` (handled by docker-compose) |
| `GROQ_API_KEY` | `gsk_...` (fallback when Gemini is rate-limited) |
| `MAIL_USERNAME` | your SMTP username |
| `MAIL_PASSWORD` | your SMTP password |
| `MAIL_FROM` | `noreply@your-domain.com` |
| `MAIL_FROM_NAME` | `ScholarGraph` |

### 5. Deploy with Docker Compose

```bash
docker compose up -d --build
```

This starts:

| Service | Port | Container |
|---|---|---|
| Redis | 6379 | `scholargraph-redis` |
| FastAPI backend | 8000 | `scholargraph-backend` |
| Celery worker | — | `scholargraph-celery` |
| Celery Flower | 5555 | `scholargraph-flower` |
| Streamlit frontend | 8501 | `scholargraph-frontend` |

Verify:

```bash
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

### 6. Set up a reverse proxy (Nginx) + SSL

For production, put Nginx in front with HTTPS:

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

Example `/etc/nginx/sites-available/scholargraph`:

```nginx
server {
    listen 80;
    server_name <your-domain>;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name <your-domain>;

    ssl_certificate /etc/letsencrypt/live/<your-domain>/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/<your-domain>/privkey.pem;

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable the site and get a certificate:

```bash
sudo ln -s /etc/nginx/sites-available/scholargraph /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
sudo certbot --nginx -d <your-domain>
```

Update `PUBLIC_BASE_URL` and `FRONTEND_BASE_URL` in `.env` to `https://<your-domain>` and restart:

```bash
docker compose restart
```

### 7. Auto-start on reboot

```bash
sudo systemctl enable docker
```

Add a cron job or systemd service to run `docker compose up -d` on boot if needed.

### 8. Maintenance

| Task | Command |
|---|---|
| View logs | `docker compose logs -f [service]` |
| Restart all | `docker compose restart` |
| Rebuild after code changes | `git pull && docker compose up -d --build` |
| Check disk | `docker system df` |
| Prune unused | `docker system prune -f` |

### Security checklist before going live

- [ ] `SECRET_KEY` and token secrets are strong, random strings (not placeholders)
- [ ] `TOKEN_BLOCKLIST_STRICT=true` (fail closed when Redis is down)
- [ ] Nginx + Certbot HTTPS in place
- [ ] Security group restricts SSH to your IP only
- [ ] Neon Postgres password is not in git history
- [ ] `.env` is in `.gitignore` (it is by default)
