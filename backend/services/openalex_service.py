import asyncio
from datetime import datetime, timezone

import httpx

from src.config import settings
from src.logging_config import setup_app_logging

logger = setup_app_logging(name="ScholarGraph-openalex")

OPENALEX_URL = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Rebuild an abstract from OpenAlex's inverted index."""
    if not inverted_index:
        return "No abstract available."
    try:
        positions = {}
        for word, spots in inverted_index.items():
            for spot in spots:
                positions[spot] = word
        return " ".join(positions[i] for i in sorted(positions))
    except Exception:  # noqa: BLE001
        return "Abstract unavailable."


def clean_query(topic: str) -> str:
    """Commas, colons and pipes break OpenAlex filter syntax, so keep plain words."""
    return " ".join("".join(ch if ch.isalnum() or ch in " -" else " " for ch in topic).split())[:200]


def format_work(work: dict, number: int) -> str:
    authorships = work.get("authorships") or []
    authors = ", ".join(a["author"]["display_name"] for a in authorships[:3] if a.get("author"))
    if len(authorships) > 3:
        authors += " et al."
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    if len(abstract) > 1200:
        abstract = abstract[:1200] + "... [truncated]"
    return (
        f"[Source {number}]\nTitle: {work.get('title') or 'Untitled'}\n"
        f"Authors: {authors or 'Unknown'}\nYear: {work.get('publication_year') or 'n.d.'}\n"
        f"Abstract: {abstract}"
    )



async def search_openalex_structured(
    topic: str,
    from_year: int | None = None,
    to_year: int | None = None,
    limit: int = 8,
) -> list[dict]:
    """Return a list of raw OpenAlex work dicts for the given topic and optional year range.

    Each dict contains: id, doi, title, publication_year, authors, abstract_inverted_index.
    Returns [] on error or empty results.
    """
    query = clean_query(topic)
    if not query:
        return []

    filter_parts = ["has_abstract:true"]
    if from_year:
        filter_parts.append(f"from_publication_date:{from_year}-01-01")
    if to_year:
        filter_parts.append(f"to_publication_date:{to_year}-12-31")

    params = {
        "search": query,
        "filter": ",".join(filter_parts),
        "per-page": limit,
        "mailto": str(settings.MAIL_FROM),
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        for attempt in range(3):
            try:
                response = await client.get(OPENALEX_URL, params=params)
            except httpx.HTTPError as exc:
                logger.warning("OpenAlex connection error (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(2)
                continue
            if response.status_code == 200:
                works = response.json().get("results", [])[:limit]
                return [
                    {
                        "id": str(w.get("id", "")).split("/")[-1],
                        "doi": w.get("doi"),
                        "title": w.get("title") or "Untitled",
                        "year": w.get("publication_year"),
                        "authors": ", ".join(
                            a["author"]["display_name"] for a in (w.get("authorships") or [])[:3] if a.get("author")
                        ),
                        "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
                    }
                    for w in works
                ]
            if response.status_code == 429:
                await asyncio.sleep(3)
                continue
            logger.warning("OpenAlex returned %s", response.status_code)
            break
    return []