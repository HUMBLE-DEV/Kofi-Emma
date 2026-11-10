from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import SQLModel

from src.config import settings


_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

_engine_kwargs: dict = {"echo": False, "future": True, "pool_pre_ping": True}
if _is_sqlite:
    # The API and the Celery worker may both write. Wait up to 30 s instead of failing
    # instantly with "database is locked".
    _engine_kwargs["connect_args"] = {"timeout": 30}
else:
    _engine_kwargs.update(pool_size=10, max_overflow=10, pool_recycle=1800)

engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session


async def init_db() -> None:
    # Import every module that defines tables so create_all sees them.
    from backend import doc_models  # noqa: F401
    from src.auth import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
