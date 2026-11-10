
import asyncio
import uuid
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.doc_errors import PermanentRejection
from backend.doc_models import Document, utcnow
from backend.services.doc_service import save_extractions

GENERIC_FAILURE = "We couldn't process this file right now. Please try uploading it again."


@asynccontextmanager
async def worker_session():
    from src.config import settings  # adjust if your settings object is named differently

    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session
    finally:
        await engine.dispose()


def _process_file_sync(file_path: str, thread_id: str, document_id: str, user_id: str) -> dict:
    """Heavy, blocking work: read file, check it, embed chunks, extract fields."""
    from backend.extraction import extract_fields
    from backend.rag import add_document_to_rag

    result = add_document_to_rag(file_path, thread_id, document_id, user_id)
    fields = extract_fields(result["text"])
    return {"chunks": result["chunks"], "fields": fields}

async def run_document_pipeline(document_id: str, attempt: int, max_attempts: int) -> None:
    """Raises only for a temporary failure that still has retries left. Everything else is
    recorded in the database and returns normally."""
    async with worker_session() as session:
        doc = await session.get(Document, uuid.UUID(document_id))
        if doc is None:
            return
        doc.status, doc.attempts, doc.updated_at = "processing", attempt, utcnow()
        session.add(doc)
        await session.commit()

        try:
            outcome = await asyncio.to_thread(
                _process_file_sync, doc.file_path, doc.thread_id, str(doc.id),str(doc.user_uid)
            )
        except PermanentRejection as rejection:  # will never work, so do not retry
            doc.status, doc.reason_for_student = "failed", str(rejection)
            doc.error_detail, doc.updated_at = None, utcnow()
            session.add(doc)
            await session.commit()
            return
        except Exception as exc:  # noqa: BLE001  temporary until we run out of attempts
            doc.error_detail, doc.updated_at = f"{type(exc).__name__}: {exc}"[:1000], utcnow()
            if attempt >= max_attempts:
                doc.status, doc.reason_for_student = "failed", GENERIC_FAILURE
                session.add(doc)
                await session.commit()
                return
            session.add(doc)
            await session.commit()
            raise  # Celery will retry

        await save_extractions(session, doc.id, outcome["fields"])
        doc.status, doc.chunk_count = "ready", outcome["chunks"]
        doc.reason_for_student = doc.error_detail = None
        doc.updated_at = utcnow()
        session.add(doc)
        await session.commit()
