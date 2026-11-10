import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend import doc_worker
from backend.doc_errors import PermanentRejection
from backend.doc_models import Document, utcnow
from backend.extraction import FieldEvidence, PaperFields, clean_fields, quote_in_text
from backend.services import doc_service
from src.auth.models import Conversation

ANA, KOFI = uuid.uuid4(), uuid.uuid4()


@pytest.fixture()
def db(tmp_path, monkeypatch):
    url = f"sqlite+aiosqlite:///{tmp_path}/t.db"

    async def setup():
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        await engine.dispose()

    asyncio.run(setup())

    @asynccontextmanager
    async def session_factory():
        engine = create_async_engine(url)
        try:
            async with AsyncSession(engine, expire_on_commit=False) as session:
                yield session
        finally:
            await engine.dispose()

    monkeypatch.setattr(doc_worker, "worker_session", session_factory)
    return session_factory


def run(coro):
    return asyncio.run(coro)


async def new_doc(db, user=ANA, thread="t1"):
    async with db() as s:
        return await doc_service.create_document(s, user, thread, "a.pdf", "uploads/a.pdf")


async def reload(db, doc_id):
    async with db() as s:
        return await s.get(Document, doc_id)


def fake_result():
    return {
        "chunks": 7,
        "fields": {"methodology": {"value": "CNN", "page": 2, "quote": "we used a CNN", "verified": True}},
    }


def test_success_marks_ready_and_saves_extractions(db, monkeypatch):
    monkeypatch.setattr(doc_worker, "_process_file_sync", lambda *a: fake_result())
    doc = run(new_doc(db))
    assert doc.status == "queued"
    run(doc_worker.run_document_pipeline(str(doc.id), attempt=1, max_attempts=4))
    done = run(reload(db, doc.id))
    assert (done.status, done.chunk_count, done.attempts) == ("ready", 7, 1)

    async def fields():
        async with db() as s:
            return await doc_service.get_extractions(s, doc.id)

    rows = run(fields())
    assert [(r.field, r.value, r.page, r.verified) for r in rows] == [("methodology", "CNN", 2, True)]


def test_rejection_is_permanent_and_never_retried(db, monkeypatch):
    def reject(*a):
        raise PermanentRejection("This does not look like a research paper.")

    monkeypatch.setattr(doc_worker, "_process_file_sync", reject)
    doc = run(new_doc(db))
    run(doc_worker.run_document_pipeline(str(doc.id), attempt=1, max_attempts=4))  # does not raise
    failed = run(reload(db, doc.id))
    assert failed.status == "failed"
    assert failed.reason_for_student == "This does not look like a research paper."


def test_temporary_failure_raises_while_retries_remain_then_fails_cleanly(db, monkeypatch):
    def boom(*a):
        raise RuntimeError("chroma is busy")

    monkeypatch.setattr(doc_worker, "_process_file_sync", boom)
    doc = run(new_doc(db))

    with pytest.raises(RuntimeError):
        run(doc_worker.run_document_pipeline(str(doc.id), attempt=1, max_attempts=4))
    mid = run(reload(db, doc.id))
    assert mid.status == "processing"  # still trying, not rejected
    assert "chroma is busy" in mid.error_detail
    assert present_flag(mid)["retrying"] is False

    with pytest.raises(RuntimeError):
        run(doc_worker.run_document_pipeline(str(doc.id), attempt=2, max_attempts=4))
    assert present_flag(run(reload(db, doc.id)))["retrying"] is True

    run(doc_worker.run_document_pipeline(str(doc.id), attempt=4, max_attempts=4))  # last attempt
    final = run(reload(db, doc.id))
    assert final.status == "failed"
    assert final.reason_for_student == doc_worker.GENERIC_FAILURE
    shown = doc_service.present(final)
    assert "chroma" not in str(shown)  # technical detail never reaches the student


def present_flag(doc):
    return doc_service.present(doc)


def test_stuck_documents_show_as_failed(db):
    doc = run(new_doc(db))
    doc.status, doc.updated_at = "processing", utcnow() - timedelta(minutes=30)
    shown = doc_service.present(doc)
    assert shown["status"] == "failed"
    assert "too long" in shown["reason"]
    fresh = Document(user_uid=ANA, thread_id="t", filename="x", file_path="p", status="processing")
    assert doc_service.present(fresh)["status"] == "processing"


def test_only_the_owner_can_load_or_list_a_document(db):
    doc = run(new_doc(db, ANA, "t1"))

    async def check():
        async with db() as s:
            assert await doc_service.get_user_document(s, doc.id, ANA) is not None
            assert await doc_service.get_user_document(s, doc.id, KOFI) is None
            assert len(await doc_service.list_thread_documents(s, "t1", ANA)) == 1
            assert await doc_service.list_thread_documents(s, "t1", KOFI) == []

    run(check())


def test_cannot_upload_into_someone_elses_thread(db):
    async def check():
        async with db() as s:
            s.add(Conversation(thread_id="ana-chat", user_uid=ANA))
            await s.commit()
            await doc_service.assert_thread_available(s, "ana-chat", ANA)  # owner is fine
            await doc_service.assert_thread_available(s, "brand-new", KOFI)  # unused id is fine
            with pytest.raises(HTTPException) as err:
                await doc_service.assert_thread_available(s, "ana-chat", KOFI)
            assert err.value.status_code == 404

    run(check())


def test_quote_check_and_field_cleaning():
    text = "[Page 1]\nWe trained a ResNet-50\non 12,000 images.\n\n[Page 2]\nA limitation is size."
    assert quote_in_text("we trained a resnet-50 on 12,000 images.", text)
    assert not quote_in_text("we trained a vgg", text)
    assert not quote_in_text(None, text)

    result = PaperFields(
        year=FieldEvidence(value="Published in 2022"),
        methodology=FieldEvidence(value="ResNet", page=1, quote="We trained a ResNet-50 on 12,000 images."),
        results=FieldEvidence(value="94%", page=2, quote="made up sentence"),
    )
    fields = clean_fields(result, text)
    assert fields["year"]["value"] == "2022"
    assert fields["methodology"]["verified"] is True
    assert fields["results"]["verified"] is False
    assert fields["dataset"] == {"value": None, "page": None, "quote": None, "verified": False}
