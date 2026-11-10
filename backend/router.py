
import asyncio
from datetime import datetime, timezone
import json
import re
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel
from sqlmodel import delete
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.agent import get_agent
from backend.doc_models import DocumentExtraction
from backend.services.doc_service import (
    assert_thread_available,
    create_document,
    create_saved_paper,
    delete_saved_paper,
    get_extractions,
    get_user_document,
    list_saved_paper,

    list_thread_documents,
    present,
)
from backend.guardrails import friendly_error, rate_limit, validate_message, validate_thread_id
from backend.services.chat_service import (
    create_or_update_conversation,
    delete_conversation_fully,
    get_chat_history,
    list_user_conversations,
    save_chat_message,
)
from backend.services.llm_call import allowed_models, normalize_model_name
from backend.services.openalex_service import search_openalex_structured
from src.auth.dependency import RoleChecker, get_checkpointer, get_current_user
from src.config import settings
from src.db.main import async_session_maker, get_session
from src.logging_config import setup_app_logging

logger = setup_app_logging(name="ScholarGraph-router")

ai_routers = APIRouter(tags=["AI Agent Integration"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
# Streamed text is accepted only from these graph nodes (research nodes call the LLM internally;
# their raw tokens must not leak into the answer, the synthesizer sends the finished text).
STREAMING_NODES = {"chatbot", "research_synthesizer"}

admin_required = RoleChecker(["admin"])

user_required = RoleChecker([
    "user",
    "admin"
])

# =========================================================================
# Small helpers
# =========================================================================
def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def should_stream_chunk(chunk, metadata) -> bool:
    node = str((metadata or {}).get("langgraph_node", "")).lower()
    if node not in STREAMING_NODES:
        return False
    if isinstance(chunk, ToolMessage) or not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False
    if getattr(chunk, "tool_calls", None) or getattr(chunk, "invalid_tool_calls", None):
        return False
    if (getattr(chunk, "additional_kwargs", {}) or {}).get("tool_calls"):
        return False
    return True


def extract_text_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", "")
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in (None, "text") and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _pending_interrupts(state) -> list:
    print("<<<>>>>>>>>>>>>< Interupt is checking  >>>>>>>>>>><<<<>><>><>")
    return [i for task in getattr(state, "tasks", []) or [] for i in getattr(task, "interrupts", ()) or ()]


# =========================================================================
# Models, conversations, history
# =========================================================================
@ai_routers.get("/models")
async def list_models(current_user=Depends(get_current_user)):
    """The frontend builds its model dropdown from this, so a retired model never lingers in the UI."""
    return {"models": allowed_models(), "default": normalize_model_name(None)}


@ai_routers.get("/conversations")
async def conversations(current_user=Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    items = await list_user_conversations(session, current_user.uid)
    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
            }
            for item in items
        ]
    }


@ai_routers.get("/history/{thread_id}")
async def history(
    thread_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    messages = await get_chat_history(session, validate_thread_id(thread_id))
    return {"messages": [{"role": m.role, "content": m.content} for m in messages]}


@ai_routers.delete("/conversations/{thread_id}")
async def delete_conversation(
    thread_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    checkpointer=Depends(get_checkpointer),
):
    deleted = await delete_conversation_fully(
        session, validate_thread_id(thread_id), current_user.uid, checkpointer
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}


# =========================================================================
# Documents
# =========================================================================
@ai_routers.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    _ = Depends(user_required)
):
    thread_id = validate_thread_id(thread_id)
    await rate_limit(current_user.uid, "upload", settings.UPLOAD_RATE_LIMIT_PER_HOUR, 3600)

    filename = Path(file.filename or "").name  # drop any folder part the client sent
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        raise HTTPException(400, "CSV analysis is not available yet. Upload research papers (PDF, DOCX, TXT or MD).")
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type. Upload PDF, DOCX, TXT or MD.")

    await assert_thread_available(session, thread_id, current_user.uid)
    existing = await list_thread_documents(session, thread_id, current_user.uid)
    if len(existing) >= settings.MAX_DOCS_PER_THREAD:
        raise HTTPException(400, f"A chat can hold at most {settings.MAX_DOCS_PER_THREAD} documents.")

    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    data = await file.read(limit + 1)
    if not data:
        raise HTTPException(400, "That file is empty.")
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {settings.MAX_UPLOAD_MB} MB.")
    if suffix == ".pdf" and not data.startswith(b"%PDF-"):
        raise HTTPException(400, "This does not look like a real PDF file.")
    if suffix == ".docx" and not data.startswith(b"PK"):
        raise HTTPException(400, "This does not look like a real Word (.docx) file.")

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)[:100]
    folder = Path(settings.UPLOAD_DIR) / str(current_user.uid)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = str(folder / f"{uuid.uuid4()}_{safe_name}")
    async with aiofiles.open(file_path, "wb") as handle:
        await handle.write(data)

    # The row is committed BEFORE the job is queued, so the worker always finds it.
    doc = await create_document(session, current_user.uid, thread_id, filename, file_path)
    await create_or_update_conversation(session, thread_id, current_user.uid, f"Uploaded file: {filename}")

    try:
        from src.celery_worker import process_document

        process_document.delay(document_id=str(doc.id))
    except Exception as exc:  # noqa: BLE001  e.g. Redis is down
        logger.error("Could not queue document %s: %s", doc.id, exc)
        doc.status = "failed"
        doc.reason_for_student = "Our processing service is unavailable right now. Please try again in a few minutes."
        doc.error_detail = f"queue error: {exc}"[:500]
        session.add(doc)
        await session.commit()
        return JSONResponse(
            {"success": False, "message": doc.reason_for_student, "document": present(doc)}, status_code=503
        )

    return JSONResponse(
        {"success": True, "message": "Uploaded. Checking and processing it now.", "document": present(doc)}
    )


@ai_routers.get("/documents")
async def documents(
    thread_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    docs = await list_thread_documents(session, validate_thread_id(thread_id), current_user.uid)
    return {"documents": [present(d) for d in docs]}


@ai_routers.get("/documents/{document_id}")
async def document_detail(
    document_id: uuid.UUID,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    doc = await get_user_document(session, document_id, current_user.uid)
    if doc is None:
        raise HTTPException(404, "Document not found")
    fields = await get_extractions(session, doc.id)
    return {
        **present(doc),
        "fields": [
            {"field": f.field, "value": f.value, "page": f.page, "quote": f.quote, "verified": f.verified}
            for f in fields
        ],
    }


@ai_routers.delete("/documents/{document_id}")
async def delete_document(
    document_id: uuid.UUID,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    from backend import rag

    doc = await get_user_document(session, document_id, current_user.uid)
    if doc is None:
        raise HTTPException(404, "Document not found")
    path, doc_id = doc.file_path, str(doc.id)
    await session.exec(delete(DocumentExtraction).where(DocumentExtraction.document_id == doc.id))
    await session.delete(doc)
    await session.commit()
    await asyncio.to_thread(rag.delete_document_chunks, doc_id)
    Path(path).unlink(missing_ok=True)
    return {"success": True}


# =========================================================================
# Paper search (OpenAlex)
# =========================================================================
@ai_routers.post("/papers/search")
async def search_papers(
    topic: str = Form(...),
    from_year: int | None = Form(None),
    to_year: int | None = Form(None),
    limit: int = Form(8),
    current_user=Depends(get_current_user),
):
    """Search OpenAlex for recent papers on a topic, with optional year filtering.

    Returns structured metadata (title, authors, year, abstract) so the frontend
    can show a "Save to this chat" button for each result. The saved papers are
    then readable by the research nodes through the same get_thread_papers() helper.
    """
    await rate_limit(
        current_user.uid, 
        bucket="paper_search", 
        limit=settings.PAPER_SEARCH_RATE_LIMIT_PER_MIN, 
        window_seconds=60
    )
    if not topic or not topic.strip():
        raise HTTPException(400, "Please enter a search topic.")
    if limit < 1 or limit > 20:
        raise HTTPException(400, "Limit must be between 1 and 20.")

    current_year = datetime.now(timezone.utc).year
    if from_year and (from_year < 1900 or from_year > current_year):
        raise HTTPException(400, f"from_year must be between 1900 and {current_year}.")
    if to_year and (to_year < 1900 or to_year > current_year):
        raise HTTPException(400, f"to_year must be between 1900 and {current_year}.")
    if from_year and to_year and from_year > to_year:
        raise HTTPException(400, "from_year cannot be greater than to_year.")

    works = await search_openalex_structured(topic, from_year=from_year, to_year=to_year, limit=limit)
    return {"papers": works}

class SavePaperRequest(BaseModel):
    paper: dict


@ai_routers.post("/papers/save")
async def save_paper(
    payload: SavePaperRequest,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Save an OpenAlex paper result to the student's account.

    Saved papers are scoped to the user, not to a chat thread, so they are
    available in every conversation.
    """

    paper = payload.paper
    await rate_limit(
        current_user.uid, 
        bucket="save_paper", # New bucket
        limit=settings.SAVE_PAPER_RATE_LIMIT_PER_HOUR, 
        window_seconds=3600
    )

    openalex_id = (paper.get("id") or "").strip()
    title = (paper.get("title") or "").strip()
    if not openalex_id or not title:
        raise HTTPException(400, "A paper id and title are required.")


    

    created = await create_saved_paper(
        session,
        user_uid=current_user.uid,
        openalex_id=openalex_id,
        title=title,
        authors=paper.get("authors"),
        year=paper.get("year"),
        doi=paper.get("doi"),
        abstract=paper.get("abstract"),
    )
    if created is None:
        return {"success": True, "message": "Already saved.", "paper": None}

    return {"success": True, "paper": {
        "id": str(created.id),
        "title": created.title,
        "authors": created.authors,
        "year": created.year,
        "doi": created.doi,
    }}


@ai_routers.get("/papers")
async def list_saved_papers(
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """List the OpenAlex papers the student has saved to their account."""
    papers = await list_saved_paper(session, current_user.uid)
    return {"papers": [
        {
            "id": str(p.id),
            "title": p.title,
            "authors": p.authors,
            "year": p.year,
            "doi": p.doi,
            "created_at": p.created_at.isoformat(),
        }
        for p in papers
    ]}


@ai_routers.delete("/papers/{paper_id}")
async def delete_saved_paper(
    paper_id: uuid.UUID,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    ok = await delete_saved_paper(session, paper_id, current_user.uid)
    if not ok:
        raise HTTPException(404, "Paper not found")
    return {"success": True}


# =========================================================================
# Chat
# =========================================================================
@ai_routers.post("/chat/stream")
async def chat_stream(
    request: Request,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    checkpointer=Depends(get_checkpointer),
):
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Invalid JSON body.")
    if not isinstance(data, dict):
        raise HTTPException(400, "Invalid JSON body.")

    # ---- checks that cost nothing, done before any LLM call ----
    user_message = validate_message(data.get("message"))
    thread_id = validate_thread_id(data.get("thread_id"))
    model = normalize_model_name(data.get("model"))
    await rate_limit(current_user.uid, "chat", settings.CHAT_RATE_LIMIT_PER_MIN, 60)

    # Claims a brand-new thread for this user, or verifies they own it (404 otherwise).
    await create_or_update_conversation(session, thread_id, current_user.uid, user_message)
    await save_chat_message(session, thread_id, "user", user_message)

    config = {
        "configurable": {"thread_id": thread_id, "user_id": str(current_user.uid), "model_name": model},
        "recursion_limit": settings.GRAPH_RECURSION_LIMIT,  # hard stop for runaway loops
    }
    agent = get_agent(model, checkpointer=checkpointer)

    # If the graph is paused waiting for "yes/no", a yes/no answer resumes it.
    # Any other message is treated as a NEW question, so the student is never stuck.
    resume = bool(_pending_interrupts(await agent.aget_state(config)))
    inputs = Command(resume=user_message) if resume else {"messages": [HumanMessage(content=user_message)]}

    async def event_generator():
        final_answer = ""
        try:
            async with asyncio.timeout(settings.TURN_TIMEOUT_SECONDS):
                async for chunk, metadata in agent.astream(inputs, config=config, stream_mode="messages"):
                    if not should_stream_chunk(chunk, metadata):
                        continue
                    token = extract_text_from_chunk(chunk)
                    if token:
                        yield sse_data({"token": token})
                        final_answer += token
                        

                # Did the graph pause to ask the student something? Show them the question.
                pending = _pending_interrupts(await agent.aget_state(config))
                if pending:
                    question = str(getattr(pending[0], "value", "") or "")
                    if question:
                        yield sse_data({"token": question})
                        final_answer += question
                        
        except Exception as exc:  # noqa: BLE001
            message, _ = friendly_error(exc)
            yield sse_data({"error": message})

        if final_answer.strip():
            try:
                async with async_session_maker() as save_session:  # own session: the request's may be closed
                    await save_chat_message(save_session, thread_id, "assistant", final_answer)
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not save assistant message for %s: %s", thread_id, exc)
        yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
