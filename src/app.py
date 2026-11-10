import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.router import ai_routers
from src.auth.routers import auth_router
from src.config import settings
from src.db.main import async_session_maker, init_db
from src.logging_config import setup_app_logging
from src.middleware import register_middleware

logger = setup_app_logging(name="ScholarGraph-backend")

# Create the folders at import time so the very first request cannot fail on a missing one.
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path("data").mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ScholarGraph backend...")
    await init_db()
    logger.info("Database tables created/verified.")

    conn = await aiosqlite.connect("data/langgraph_checkpoints.sqlite")
    app.state.checkpointer = AsyncSqliteSaver(conn)

    # Check the configured Gemini models in the background so startup is not delayed.
    # A retired model name shows up in the logs on deploy day, not in a student's chat.
    from backend.services.llm_call import check_models

    model_check = asyncio.create_task(check_models())
    try:
        yield
    finally:
        model_check.cancel()
        await conn.close()
        logger.info("Shut down cleanly.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.1.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    lifespan=lifespan,
)

register_middleware(app)

app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(ai_routers, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return {"name": settings.PROJECT_NAME, "status": "running", "docs": "/api/v1/docs"}


@app.get("/health", tags=["Health"])
async def health():
    """Liveness: the process is up. Used by the Docker healthcheck."""
    return {"status": "healthy"}


@app.get("/health/ready", tags=["Health"])
async def ready():
    """Readiness: the database answers."""
    from sqlalchemy import text

    async with async_session_maker() as session:
        await session.exec(text("SELECT 1"))
    return {"status": "ready"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)