from celery import Celery
from fastapi_mail import FastMail, ConnectionConfig, MessageSchema, MessageType
from src.config import settings


celery_app = Celery(
    "ScholarGraph-auth",
    broker=settings.REDIS_URL,
    # No result backend: we only publish tasks and never read their results.
    # With a result backend configured, Celery's `send_task` opens a consumer
    # channel to the result store on every publish, and its reconnect intervals
    # are hardcoded — which stalls the request for ~100s against a dead Redis.
    backend=None,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=False,
    # Task events are published to the broker (not the result backend), so
    # Flower can still show live task activity.
    worker_send_task_events=True,
    task_acks_late=True,
    # Fail fast on a dead broker instead of stalling the whole process.
    broker_connection_retry_on_startup=False,
    broker_connection_max_retries=0,
    broker_transport_options={
        "max_retries": 0,
        "socket_connect_timeout": 2,
        "socket_timeout": 2,
        "retry_on_timeout": False,
    },
    worker_prefetch_multiplier=1,
    # A hung job must not block a worker forever.
    task_soft_time_limit=420,
    task_time_limit=480,
)


def _make_mail() -> FastMail:
    """Create a FastMail instance inside the Celery worker process."""
    conf = ConnectionConfig(
        MAIL_USERNAME=settings.MAIL_USERNAME,
        MAIL_PASSWORD=settings.MAIL_PASSWORD,
        MAIL_PORT=settings.MAIL_PORT,
        MAIL_SERVER=settings.MAIL_SERVER,
        MAIL_STARTTLS=settings.MAIL_STARTTLS,
        MAIL_SSL_TLS=settings.MAIL_SSL_TLS,
        MAIL_FROM=settings.MAIL_FROM,
        MAIL_FROM_NAME=settings.MAIL_FROM_NAME,
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=True,
    )
    return FastMail(config=conf)


@celery_app.task(bind=True, name="Send_email")
def Send_email(self, recipients: list, subject: str, body: str) -> str:
    """Background task: send an HTML email via fastapi-mail."""
    import asyncio

    mail = _make_mail()
    message = MessageSchema(
        recipients=recipients,
        subject=subject,
        body=body,
        subtype=MessageType.html,
    )

    try:
        asyncio.run(mail.send_message(message))
        return f"Email sent to {', '.join(recipients)}"
    except Exception as exc:
        self.retry(exc=exc, countdown=60, max_retries=3)
        raise


@celery_app.task(bind=True, name="process_document", max_retries=2)
def process_document(self,document_id: str) -> str:
    """Background task: extract text from an uploaded PDF/DOCX/TXT file,
    split it into chunks and embed them into the Chroma vector store.

    This runs inside the Celery worker process so the API request thread
    is never blocked by slow LLM-embedding calls.
    """
    try:
        # Imports live inside the worker so the agent/LLM heavy-lifting
        # stack is only loaded in the worker, not on the API import path.
        from backend.rag import add_document_to_rag
        import asyncio
        from backend.doc_worker import run_document_pipeline
        asyncio.run(
                run_document_pipeline(
                    document_id,
                    attempt=self.request.retries + 1,
                    max_attempts=self.max_retries + 1,
                )
            )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))