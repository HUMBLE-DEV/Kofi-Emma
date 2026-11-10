import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _time_column() -> Column:
    return Column(DateTime(timezone=True), nullable=False)


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_uid: uuid.UUID = Field(index=True)  # the owner
    thread_id: str = Field(index=True)  # which chat it belongs to
    filename: str
    file_path: str
    status: str = Field(default="queued")  # queued | processing | ready | failed
    reason_for_student: str | None = Field(default=None, sa_column=Column(Text))
    error_detail: str | None = Field(default=None, sa_column=Column(Text))  # technical, never shown
    attempts: int = 0
    chunk_count: int | None = None
    created_at: datetime = Field(default_factory=utcnow, sa_column=_time_column())
    updated_at: datetime = Field(default_factory=utcnow, sa_column=_time_column())


class DocumentExtraction(SQLModel, table=True):
    """One extracted field of one document, with the proof it came from."""

    __tablename__ = "document_extractions"
    __table_args__ = (UniqueConstraint("document_id", "field"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(foreign_key="documents.id", index=True)
    field: str
    value: str | None = Field(default=None, sa_column=Column(Text))
    page: int | None = None
    quote: str | None = Field(default=None, sa_column=Column(Text))
    verified: bool = False  # True when the quote was found word-for-word in the document
    edited_by_user: bool = False


class SavedPaper(SQLModel, table=True):
    """A paper the student found via the Paper Search panel and saved to their account.

    Unlike uploaded documents, these have no file on disk — they are metadata-only
    (title, authors, year, abstract) fetched from OpenAlex. Saved papers are scoped
    to the user, not to a chat thread, so they are available in every conversation.
    """

    __tablename__ = "saved_papers"
    __table_args__ = (UniqueConstraint("user_uid", "openalex_id"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_uid: uuid.UUID = Field(index=True)
    openalex_id: str = Field(index=True)
    title: str
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    abstract: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(default_factory=utcnow, sa_column=_time_column())
