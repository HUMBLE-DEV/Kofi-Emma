from pathlib import Path
from typing import List
import os
import certifi
from backend.doc_errors import PermanentRejection
from backend.security.deterministic import is_research_document_heuristic

from src.logging_config import setup_app_logging
from src.config import settings
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


from pypdf import PdfReader
import docx2txt

logger = setup_app_logging(name="ScholarGraph-rag")

COLLECTION = "agentic_chatbot_docs"
NO_RESULTS = "No relevant uploaded document content found."

# Module-level singletons for synchronous usage (chat/retrieval)
# These are NOT used by the Celery task, which recreates its own instances.
_embeddings = None
_vectorstore = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def _get_store():
    """One Chroma handle per process.

    With CHROMA_HOST set we talk to a Chroma SERVER. That is what you want when a Celery
    worker writes and the API reads: two processes must not share one Chroma folder.
    """
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma

        kwargs: dict = {
            "collection_name": COLLECTION,
            "embedding_function": _get_embeddings(),
            "collection_metadata": {"hnsw:space": "cosine"},  # same metric for readers and writers
        }
        if settings.CHROMA_HOST:
            import chromadb

            kwargs["client"] = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
        else:
            Path(settings.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
            kwargs["persist_directory"] = settings.CHROMA_DIR
        _vectorstore = Chroma(**kwargs)
    return _vectorstore


def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"
        return text

    if suffix == ".docx":
        return docx2txt.process(file_path)

    if suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise PermanentRejection("Unsupported file type. Upload PDF, DOCX, TXT or MD.")

def add_document_to_rag(file_path: str, thread_id: str, document_id: str | None = None, user_id: str | None = None):
        text = read_file_text(file_path)
        if len(text.replace("[Page", "").strip()) < 200:

            raise PermanentRejection(
                "No readable text found. If this is a scanned PDF, scans are not supported yet."
            )
        print(f"🔍 [GATEKEEPER] Analyzing document structure: {Path(file_path).name}...")
        is_valid, reason = is_research_document_heuristic(text)
        if not is_valid:
            print(f"🛑 [GATEKEEPER] Rejected: {reason}")
            raise PermanentRejection(reason)

        print(f"✅ [GATEKEEPER] Document accepted.")
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(text)
        docs = [
            Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "user_id": str(user_id or ""),
                "document_id": document_id or "",
                "source": Path(file_path).name,
            },
        )
        for chunk in chunks
    ]
        # Same ids on every retry, so a retry overwrites instead of duplicating chunks.
        ids = [f"{document_id}:{i}" for i in range(len(docs))] if document_id else None

        
        _get_store().add_documents(docs, ids=ids)
        return {"filename": Path(file_path).name, "chunks": len(docs), "text": text}


def _where(thread_id: str, user_id: str | None, document_id: str | None) -> dict:
    conditions = [{"thread_id": thread_id}]
    if user_id:
        conditions.append({"user_id": str(user_id)})
    if document_id:
        conditions.append({"document_id": document_id})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}



def retrieve_from_rag(
        query: str, thread_id: str, k: int = 4,
        user_id: str | None = None,
        document_id: str | None = None,         
                      
                      ) -> str:
    """Synchronous retrieval using the module-level vectorstore (for chat streaming)."""
    print(f"🔍 [RAG SEARCH] Query: '{query}' | Thread: {thread_id} | User: {user_id}")
    vectorstore = _get_store()
    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter=_where(thread_id,user_id,document_id),
    )
    print(f"📚 [RAG RESULTS] Found {len(docs)} chunks.")

    if not docs:
        return "No relevant uploaded document content found."
    print(f"Retrieved {len(docs)} chunks")
    results = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "uploaded document")
        results.append(f"[Source {i}: {source}]\n{doc.page_content}")

    return "\n\n".join(results)



def delete_document_chunks(document_id: str) -> None:
    """Remove every chunk of one document from the vector store."""
    try:
        _get_store().delete(where={"document_id": document_id})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete chunks for %s: %s", document_id, exc)



