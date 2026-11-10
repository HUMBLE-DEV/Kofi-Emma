import asyncio
import math
import os
import uuid

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from backend.guardrails import untrusted
from backend.services.chat_service import save_memory, search_memory
from backend.rag import NO_RESULTS, retrieve_from_rag
from langchain_experimental.tools import PythonREPLTool
from backend.services.doc_service import get_extractions, get_thread_papers, list_thread_documents, match_document, papers_to_prompt, pick_paper
from src.db.main import async_session_maker

from src.logging_config import setup_app_logging


logger = setup_app_logging(name="ScholarGraph-tools")


_tavily = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


@tool
async def web_search(query: str) -> str:
    """Search the web for current information: news, prices, recent releases, anything time-sensitive."""
    try:
        data = await _tavily.ainvoke({"query": query[:300]})
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search failed: %s", exc)
        return "Web search is unavailable right now."
    results = data.get("results", []) if isinstance(data, dict) else []
    if not results:
        return "No web results found."
    lines = [
        f"- {r.get('title', 'Untitled')} ({r.get('url', '')}): {str(r.get('content', ''))[:500]}"
        for r in results[:5]
    ]
    return untrusted("web search results", "\n".join(lines), limit=4000)


@tool
async def get_paper_overview(config: RunnableConfig, filename: str = "") -> str:
    """Get the pre-extracted summary of a paper: title, authors, year, methodology, dataset, results, limitations and future work. 
        No search query is needed.
    Pass `filename` only if the student named one file.

    USE THIS TOOL WHEN: The user asks for a summary, the methodology, the dataset, the results, or the limitations of the paper. 
    This pulls from the structured database and is the fastest way to answer broad questions."""

    cfg = config.get("configurable", {})
    async with async_session_maker() as session:
        data = await get_thread_papers(session, cfg.get("thread_id", ""), cfg.get("user_id", ""))
    if data["pending"]:
        return f"{data['pending']} document(s) are still being processed. Tell the student to try again in a minute."
    if not data["papers"]:
        return "There are no processed research papers in this chat yet."
    paper = pick_paper(data["papers"], filename)
    return untrusted("paper overview", f"File: {paper['filename']}\n" + papers_to_prompt([paper]), limit=3000)



@tool
async def search_uploaded_documents(query: str, config: RunnableConfig, filename: str = "") -> str:
    """Search uploaded documents using semantic retrieval.

Use this tool when the user asks about content contained in their uploaded
documents  (PDF, DOCX, TXT, files).

Rules:
- Use the user's original question as the retrieval query whenever possible.
- Invoke this tool only once per user request.
- Pass `filename` when the user references a specific document.
- Search all uploaded documents when `filename` is empty.
- If relevant information cannot be found, use `get_paper_overview`
  instead of performing additional searches.

Only rewrite the query when:
- The user explicitly references previous messages using pronouns
  (e.g., "that paper", "it", "this method").
- Important context must be added for retrieval.
- The question is too short or ambiguous

  """

    print(" <><><>  calling Rag Search <><>")
    if not query or len(query.strip()) < 3:
        return "ERROR: You must provide a specific search query. If you want a summary, use the get_paper_overview tool instead."
    cfg = config.get("configurable", {})
    thread_id, user_id = cfg.get("thread_id", ""), cfg.get("user_id")

    document_id = None
    if filename.strip():
        try:
            uid = uuid.UUID(str(user_id))
        except (ValueError, TypeError):
            return "System error: Invalid user session."
        async with async_session_maker() as session:
            docs = await list_thread_documents(session, thread_id, uid)
        match = match_document(docs, filename)
        if match is None:
            names = ", ".join(d.filename for d in docs) or "none"
            return f"I couldn't find a document named '{filename}' in this chat. Documents here: {names}."
        document_id = str(match.id)

    text = await asyncio.to_thread(retrieve_from_rag, query, thread_id, 4, user_id, document_id)
    if text == NO_RESULTS:
        return "I couldn't find anything relevant in the uploaded documents."
    print( f" <><><> Rag Search   {len(text) }   chucks <><>")
    return untrusted("uploaded documents", text, limit=4000)




@tool
async def remember_this(memory: str, config: RunnableConfig) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """
    print("🚨 remember_this tool called")
    # 1. Extract the thread context metrics from the hidden runtime properties
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id", "default")

    # 2. Extract the active database session we passed down from the API router!
    session = configurable.get("db_session")
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return "I couldn't save that memory right now."
    async with async_session_maker() as session:
        return await save_memory(thread_id=thread_id, memory_text=memory, session=session)


@tool
async def recall_memory(query: str, config: RunnableConfig) -> str:
    """Recall things the student asked you to remember, from any of their conversations."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "I couldn't recall memories right now."
    import uuid

    async with async_session_maker() as session:
        return await search_memory(user_uid=uuid.UUID(str(user_id)), session=session, query=query)


tools = [
    get_paper_overview,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search,
]