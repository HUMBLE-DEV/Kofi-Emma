"""Database helpers for documents. Same style as backend/service.py."""
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlmodel import asc, delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.doc_models import Document, DocumentExtraction
from src.auth.models import Conversation

STUCK_AFTER = timedelta(minutes=5)


async def assert_thread_available(session: AsyncSession, thread_id: str, user_uid: uuid.UUID) -> None:
    """If this thread already exists and belongs to someone else, pretend it does not exist."""
    result = await session.exec(select(Conversation).where(Conversation.thread_id == thread_id))
    conversation = result.first()
    if conversation is not None and conversation.user_uid != user_uid:
        raise HTTPException(status_code=404, detail="Chat not found")


async def create_document(
    session: AsyncSession,
    user_uid: uuid.UUID,
    thread_id: str,
    filename: str,
    file_path: str,
    status: str = "queued",
) -> Document:
    doc = Document(
        user_uid=user_uid, thread_id=thread_id, filename=filename, file_path=file_path, status=status
    )
    session.add(doc)
    await session.commit()  # commit BEFORE sending the job, so the worker always finds the row
    await session.refresh(doc)
    return doc


async def get_user_document(
    session: AsyncSession, document_id: uuid.UUID, user_uid: uuid.UUID
) -> Document | None:
    doc = await session.get(Document, document_id)
    if doc is None or doc.user_uid != user_uid:
        return None  # the router turns this into a 404
    return doc


async def list_thread_documents(
    session: AsyncSession, thread_id: str, user_uid: uuid.UUID
) -> list[Document]:
    if isinstance(user_uid, str):
        user_uid = uuid.UUID(user_uid)
    result = await session.exec(
        select(Document)
        .where(Document.thread_id == thread_id, Document.user_uid == user_uid)
        .order_by(asc(Document.created_at))
    )
    documents = result.all()

    print(f"<<<<<<found {len(documents)} documents>>>>>>")

    return list(documents)


async def get_extractions(session: AsyncSession, document_id: uuid.UUID) -> list[DocumentExtraction]:
    result = await session.exec(
        select(DocumentExtraction).where(DocumentExtraction.document_id == document_id)
    )
    document_extract = result.all()
    
    print(f"<<<<<<found {len(document_extract)} documents extracted >>>>>>")

    return list(document_extract)


async def save_extractions(
    session: AsyncSession, document_id: uuid.UUID, fields: dict[str, dict]
) -> None:
    await session.exec(delete(DocumentExtraction).where(DocumentExtraction.document_id == document_id))
    for name, item in fields.items():
        session.add(
            DocumentExtraction(
                document_id=document_id,
                field=name,
                value=item.get("value"),
                page=item.get("page"),
                quote=item.get("quote"),
                verified=bool(item.get("verified")),
            )
        )


def present(doc: Document) -> dict:
    """What the student sees. Never includes error_detail or file_path."""
    status, reason = doc.status, doc.reason_for_student

    updated = doc.updated_at
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    if status in ("queued", "processing") and datetime.now(timezone.utc) - updated > STUCK_AFTER:
        status = "failed"
        reason = "Processing took too long. Please upload the file again."

    return {
        "id": str(doc.id),
        "thread_id": doc.thread_id,
        "filename": doc.filename,
        "status": status,
        "reason": reason,
        "retrying": status == "processing" and doc.attempts > 1,
        "chunk_count": doc.chunk_count,
        "created_at": doc.created_at.isoformat(),
    }


async def get_thread_papers(session: AsyncSession, thread_id: str, user_uid) -> dict:
    """What the research nodes need: this student's finished papers (with their extracted
    fields) plus how many are still processing or failed."""
    print(f"THREAD_ID={thread_id}")
    print(f"USER_ID={user_uid}")
    if not isinstance(user_uid, uuid.UUID):
        user_uid = uuid.UUID(str(user_uid))

    papers, pending, failed = [], 0, 0
    for doc in await list_thread_documents(session, thread_id, user_uid):
        status = present(doc)["status"]
        if status == "ready":
            rows = await get_extractions(session, doc.id)
            if rows:  # CSV files are "ready" but have no extracted fields
                papers.append(
                    {
                        "document_id": str(doc.id),
                        "filename": doc.filename,
                        "fields": {r.field: r.value for r in rows},
                    }
                )
        elif status == "failed":
            failed += 1
        else:
            pending += 1

    # ── Saved papers from OpenAlex paper search ──
    # Saved papers are scoped to the user, so they appear in every chat thread.
    for paper in await list_saved_paper(session, user_uid):
        papers.append(
            {
                "document_id": str(paper.id),
                "filename": paper.title,
                "fields": {
                    "title": paper.title,
                    "authors": paper.authors or "",
                    "year": str(paper.year) if paper.year else "",
                    "doi": paper.doi or "",
                    "abstract": paper.abstract or "",
                    "source": "OpenAlex",
                },
            }
        )

    print(f"{len(papers)}papers found is pending{pending} times and failed {failed} times")
    return {"papers": papers, "pending": pending, "failed": failed}


def match_document(docs: list, name: str):
    """Find the student's document whose file name matches what they typed."""
    wanted = Path(name or "").stem.lower().strip()
    if not wanted:
        return None
    exact = [d for d in docs if Path(d.filename).stem.lower() == wanted]
    if exact:
        return exact[-1]
    partial = [d for d in docs if wanted in d.filename.lower()]
    return partial[-1] if partial else None


def papers_to_prompt(papers: list[dict]) -> str:
    """Turn the papers into a text block for a prompt."""
    order = ["title", "authors", "year", "methodology", "dataset", "results", "limitations", "future_work"]
    blocks = []
    for number, paper in enumerate(papers, start=1):
        lines = [f"[Paper {number}: {paper['filename']}]"]
        for field in order:
            lines.append(f"{field}: {paper['fields'].get(field) or 'not stated'}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def pick_paper(papers: list[dict], message: str) -> dict | None:
    """For single-paper tasks (study snapshot): use the paper whose file name the student
    mentioned, otherwise the most recently uploaded one."""
    text = message.lower()
    for paper in papers:
        stem = Path(paper["filename"]).stem.lower()
        if stem and stem in text:
            return paper
    return papers[-1] if papers else None


async def list_saved_paper(
    session: AsyncSession, user_uid: uuid.UUID
) -> list:
    """All OpenAlex papers the student has saved to their account.

    Saved papers are scoped to the user, not to a chat thread, so they are
    available in every conversation.
    """
    from backend.doc_models import SavedPaper

    result = await session.exec(
        select(SavedPaper).where(
            SavedPaper.user_uid == user_uid
        ).order_by(SavedPaper.created_at.desc())
    )
    return result.all()


async def create_saved_paper(
    session: AsyncSession,
    user_uid: uuid.UUID,
    openalex_id: str,
    title: str,
    authors: str | None = None,
    year: int | None = None,
    doi: str | None = None,
    abstract: str | None = None,
) -> dict:
    """Create a SavedPaper row (idempotent by user_uid + openalex_id)."""
    from backend.doc_models import SavedPaper

    existing = await session.exec(
        select(SavedPaper).where(
            SavedPaper.user_uid == user_uid,
            SavedPaper.openalex_id == openalex_id,
        )
    )
    if existing.first():
        return None  # already saved

    paper = SavedPaper(
        user_uid=user_uid,
        openalex_id=openalex_id,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        abstract=abstract,
    )
    session.add(paper)
    await session.commit()
    await session.refresh(paper)
    return paper


async def delete_saved_paper(
    session: AsyncSession, paper_id: uuid.UUID, user_uid: uuid.UUID
) -> bool:
    """Delete a saved paper the student owns."""
    from backend.doc_models import SavedPaper

    paper = await session.get(SavedPaper, paper_id)
    if paper is None or paper.user_uid != user_uid:
        return False
    await session.delete(paper)
    await session.commit()
    return True
