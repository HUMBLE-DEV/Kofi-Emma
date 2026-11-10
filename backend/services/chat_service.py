import asyncio
from pathlib import Path
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlmodel import col, select, desc, asc,delete
from sqlmodel.ext.asyncio.session import AsyncSession
from src.auth.models import LongTermMemory, Conversation, ChatMessage
from src.logging_config import setup_app_logging
logger = setup_app_logging(name="ScholarGraph-service")

# =========================================================================
# 1. CREATE OR UPDATE CONVERSATION (With explicit User Mapping)
# =========================================================================
async def create_or_update_conversation(
    session: AsyncSession, 
    thread_id: str, 
    user_uid: uuid.UUID, # CRITICAL: Forces user mapping on creation
    first_message: Optional[str] = None
) -> Conversation:
    
    # Correct SQLModel Async Select statement syntax
    statement = select(Conversation).where(Conversation.thread_id == thread_id)
    result = await session.exec(statement)
    conversation = result.first()

    current_time = datetime.now(timezone.utc)

    if not conversation:
        title = "New Chat"
        if first_message:
            title = first_message.strip()[:40]
            if len(first_message.strip()) > 40:
                title += "..."

        conversation = Conversation(
            thread_id=thread_id,
            title=title,
            user_uid=user_uid, # Binds the thread strictly to this student
            created_at=current_time,
            updated_at=current_time
        )
        session.add(conversation)
    else:
        conversation.updated_at = current_time

    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation




# =========================================================================
# 2. LIST CONVERSATIONS (Fully Isolated Per Student)
# =========================================================================
async def list_user_conversations(session: AsyncSession, user_uid: uuid.UUID) -> List[Conversation]:
    """Fetches threads belonging ONLY to the designated user."""
    statement = (
        select(Conversation)
        .where(Conversation.user_uid == user_uid) # FIXED: Security filter added
        .order_by(desc(Conversation.updated_at))
    )
    result = await session.exec(statement)
    return list(result.all())


# =========================================================================
# 3. SAVE CHAT MESSAGE
# =========================================================================
async def save_chat_message(session: AsyncSession, thread_id: str, role: str, content: str) -> ChatMessage:
    current_time = datetime.now(timezone.utc)
    
    msg = ChatMessage(
        conversation_thread_id=thread_id, # Maps cleanly to the corrected Foreign Key
        role=role,
        content=content,
        created_at=current_time
    )
    session.add(msg)

    # Update parent conversation timestamp
    statement = select(Conversation).where(Conversation.thread_id == thread_id)
    result = await session.exec(statement)
    conversation = result.first()
    
    if conversation:
        conversation.updated_at = current_time
        session.add(conversation)

    await session.commit()
    await session.refresh(msg)
    return msg

async def delete_chat_message(
    session: AsyncSession,
    thread_id: str,
    user_uid: uuid.UUID
):
    stmt = delete(Conversation).where(
        Conversation.thread_id == thread_id,
        Conversation.user_uid == user_uid
    )

    await session.exec(stmt)
    await session.commit()



# =========================================================================
# 4. FETCH HISTORY
# =========================================================================
async def get_chat_history(session: AsyncSession, thread_id: str) -> List[ChatMessage]:
    statement = (
        select(ChatMessage)
        .where(ChatMessage.conversation_thread_id == thread_id)
        .order_by(asc(ChatMessage.created_at))
    )
    result = await session.exec(statement)
    return list(result.all())



# =========================================================================
# Delete: remove EVERYTHING that belongs to the conversation
# =========================================================================
async def delete_conversation_fully(
    session: AsyncSession, thread_id: str, user_uid: uuid.UUID, checkpointer=None
) -> bool:
    """Messages, memories, documents (rows, files, vectors) and the agent's saved state."""
    from backend import rag
    from backend.doc_models import Document, DocumentExtraction

    docs = (
        await session.exec(
            select(Document).where(Document.thread_id == thread_id, Document.user_uid == user_uid)
        )
    ).all()
    doc_ids = [d.id for d in docs]
    file_paths = [d.file_path for d in docs]

    if doc_ids:
        await session.exec(delete(DocumentExtraction).where(DocumentExtraction.document_id.in_(doc_ids)))
        await session.exec(delete(Document).where(Document.id.in_(doc_ids)))
    await session.exec(delete(ChatMessage).where(ChatMessage.conversation_thread_id == thread_id))
    await session.exec(delete(LongTermMemory).where(LongTermMemory.conversation_thread_id == thread_id))
    await session.exec(
        delete(Conversation).where(Conversation.thread_id == thread_id, Conversation.user_uid == user_uid)
    )
    await session.commit()

    # Best-effort cleanup outside the database transaction.
    for doc_id in doc_ids:
        await asyncio.to_thread(rag.delete_document_chunks, str(doc_id))
    for path in file_paths:
        Path(path).unlink(missing_ok=True)
    if checkpointer is not None and hasattr(checkpointer, "adelete_thread"):
        try:
            await checkpointer.adelete_thread(thread_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not delete checkpoints for %s: %s", thread_id, exc)
    return True







# =========================================================================
# 5. LONG TERM AGENT MEMORIES
# =========================================================================

async def save_memory(thread_id: str, memory_text: str, session: AsyncSession) -> str:
    """Asynchronously saves an agent long term memory block."""
    try:
        item = LongTermMemory(
            conversation_thread_id=thread_id,
            memory=memory_text,
            created_at=datetime.now(timezone.utc)
        )
        session.add(item)
        await session.commit()
        return "Memory saved successfully."
    except Exception as e:
        logger.error("save_memory failed: %s", exc)
        return f"I couldn't save that memory right now."

async def search_memory(
    thread_id: str,
    session: AsyncSession,
    query: Optional[str] = None,
) -> str:
    """Asynchronously retrieves the last 20 memories for this thread.

    When a `query` is supplied the memories are ranked by simple keyword
    overlap so the most relevant block surfaces first.
    """
    try:
        statement = (
            select(LongTermMemory)
            .where(LongTermMemory.conversation_thread_id == thread_id)
            .order_by(desc(LongTermMemory.created_at))
            .limit(20)
        )
        result = await session.exec(statement)
        memories = result.all()

        if not memories:
            return "No saved memory found."

        if query:
            terms = {t.lower() for t in query.split() if len(t) > 2}
            if terms:
                memories = sorted(
                    memories,
                    key=lambda m: sum(t in m.memory.lower() for t in terms),
                    reverse=True,
                )

        return "\n".join([f"- {m.memory}" for m in memories])
    except Exception as exc:
        logger.error("search_memory failed: %s", exc)
        return "I couldn't recall memories right now."




PROJECT_PREFIX = "Final-year project:"


async def get_project_topic(user_uid: uuid.UUID, session: AsyncSession) -> str | None:
    """The student's most recently saved project topic (from any of their chats), or None."""
    result = await session.exec(
        select(LongTermMemory)
        .join(Conversation, Conversation.thread_id == LongTermMemory.conversation_thread_id)
        .where(Conversation.user_uid == user_uid, col(LongTermMemory.memory).startswith(PROJECT_PREFIX))
        .order_by(desc(LongTermMemory.created_at))
        .limit(1)
    )
    row = result.first()
    print(f"saved project : {row} ")
    return row.memory[len(PROJECT_PREFIX):].strip() if row else None