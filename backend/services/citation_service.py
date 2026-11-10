
import asyncio
import re

import httpx

from src.config import settings

S2 = "https://api.semanticscholar.org/graph/v1"
OA = "https://api.openalex.org"
ANCHOR_FIELDS = "title,year,authors,externalIds,citationCount,venue"
LIST_FIELDS = "title,year,citationCount,externalIds"
OA_FIELDS = "id,title,publication_year,doi,cited_by_count"
_DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)

Section = tuple[list[dict], str]  # (items, status)

MESSAGES = {
    "none": "No entries are indexed for this paper. This can happen with smaller journals; "
    "check the paper's own reference list.",
    "busy": "The citation services are busy right now. Please try again in a minute.",
    "error": "The citation service returned an unexpected answer. Please try again later.",
}


def extract_doi(text: str) -> str | None:
    match = _DOI.search(text or "")
    return match.group(0).rstrip(".,;)") if match else None


def _classify(response) -> str:
    if response is None:  # every retry failed: rate limit or connection problem
        return "busy"
    if response.status_code == 200:
        return "ok"
    if response.status_code == 404:
        return "none"
    if response.status_code == 429 or response.status_code >= 500:
        return "busy"
    return "error"


async def _get(client: httpx.AsyncClient, url: str, params: dict, headers: dict | None = None):
    """GET with a few polite retries. Returns the response, or None if we could not get one."""
    for attempt in range(3):
        try:
            response = await client.get(url, params=params, headers=headers or {"Accept": "application/json"})
        except httpx.HTTPError:
            await asyncio.sleep(1 + attempt)
            continue
        if response.status_code == 429:
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        return response
    return None


async def _get_s2(client, url, params):
    headers = {"Accept": "application/json"}
    if settings.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY
    return await _get(client, url, params, headers)


async def _get_oa(client, url, params):
    return await _get(client, url, {**params, "mailto": str(settings.MAIL_FROM)})


def _top(items: list[dict], key: str, limit: int) -> list[dict]:
    papers = [i[key] for i in items if isinstance(i.get(key), dict) and i[key].get("title")]
    papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    return papers[:limit]




async def _s2_chain(client, identifier: str, limit: int):
    doi = extract_doi(identifier)
    if doi:
        response = await _get_s2(client, f"{S2}/paper/DOI:{doi}", {"fields": ANCHOR_FIELDS})
        anchor = response.json() if _classify(response) == "ok" else None
        status = _classify(response)
    else:
        response = await _get_s2(
            client, f"{S2}/paper/search/match", {"query": identifier.strip()[:300], "fields": ANCHOR_FIELDS}
        )
        status = _classify(response)
        data = response.json().get("data") if status == "ok" else None
        anchor = data[0] if data else None
        if status == "ok" and anchor is None:
            status = "none"
    if not anchor or not anchor.get("paperId"):
        return None, ([], status), ([], status)

    pid = anchor["paperId"]
    refs, cites = await asyncio.gather(
        _get_s2(client, f"{S2}/paper/{pid}/references", {"fields": LIST_FIELDS, "limit": 100}),
        _get_s2(client, f"{S2}/paper/{pid}/citations", {"fields": LIST_FIELDS, "limit": 100}),
    )

    def section(response, key) -> Section:
        status = _classify(response)
        if status != "ok":
            return [], status
        items = _top(response.json().get("data") or [], key, limit)
        return (items, "ok") if items else ([], "none")

    return anchor, section(refs, "citedPaper"), section(cites, "citingPaper")  # the API wraps each item




def _short_id(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def _oa_paper(work: dict) -> dict:
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    return {
        "title": work.get("title") or work.get("display_name"),
        "year": work.get("publication_year"),
        "citationCount": work.get("cited_by_count") or 0,
        "externalIds": {"DOI": doi} if doi else {},
        "url": work.get("id"),
    }


async def _oa_list(client, filter_value: str, limit: int) -> Section:
    response = await _get_oa(
        client,
        f"{OA}/works",
        {"filter": filter_value, "select": OA_FIELDS, "sort": "cited_by_count:desc", "per-page": limit},
    )
    status = _classify(response)
    if status != "ok":
        return [], status
    items = [_oa_paper(w) for w in response.json().get("results", []) if w.get("title")]
    return (items, "ok") if items else ([], "none")


async def _oa_chain(client, doi: str | None, limit: int):
    if not doi:
        return None, ([], "none"), ([], "none")
    response = await _get_oa(
        client, f"{OA}/works/doi:{doi}", {"select": "id,title,publication_year,doi,referenced_works"}
    )
    status = _classify(response)
    if status != "ok":
        return None, ([], status), ([], status)
    work = response.json()
    anchor = {"title": work.get("title"), "year": work.get("publication_year")}
    ref_ids = [_short_id(r) for r in (work.get("referenced_works") or [])][:50]

    async def no_refs() -> Section:
        return [], "none"

    refs, cites = await asyncio.gather(
        _oa_list(client, "openalex:" + "|".join(ref_ids), limit) if ref_ids else no_refs(),
        _oa_list(client, f"cites:{_short_id(work['id'])}", limit),
    )
    return anchor, refs, cites




def _merge(primary: Section, backup: Section) -> tuple[list[dict], str, str]:
    """(items, where they came from, status to explain if there are none)."""
    if primary[0]:
        return primary[0], "Semantic Scholar", "ok"
    if backup[0]:
        return backup[0], "OpenAlex", "ok"
    statuses = {primary[1], backup[1]}
    return [], "", "busy" if "busy" in statuses else "error" if "error" in statuses else "none"


def _line(paper: dict) -> str:
    year = paper.get("year") or "n.d."
    cites = paper.get("citationCount") or 0
    doi = (paper.get("externalIds") or {}).get("DOI")
    if doi:
        link = f"https://doi.org/{doi}"
    elif paper.get("url"):
        link = paper["url"]
    else:
        link = f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
    return f"- [{paper['title']} ({year})]({link}), cited {cites} times"


def _render(title: str, year, refs, cites) -> str:
    lines = [f"## Citation chain: {title} ({year or 'n.d.'})"]
    for heading, (items, source, status) in (
        ("### Backward: the most cited works this paper builds on", refs),
        ("### Forward: the most cited works that cite it", cites),
    ):
        lines += ["", heading]
        if items:
            lines += [_line(p) for p in items]
            lines.append(f"_Source: {source}_")
        else:
            lines.append(f"- {MESSAGES[status]}")
    return "\n".join(lines)


async def citation_chain(identifier: str, limit: int = 10) -> str:
    """Markdown for the paper's citation chain, or a plain explanation of what went wrong."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        anchor, s2_refs, s2_cites = await _s2_chain(client, identifier, limit)
        doi = extract_doi(identifier) or ((anchor or {}).get("externalIds") or {}).get("DOI")
        backup_needed = anchor is None or not s2_refs[0] or not s2_cites[0]
        if backup_needed:
            oa_anchor, oa_refs, oa_cites = await _oa_chain(client, doi, limit)
        else:
            oa_anchor, oa_refs, oa_cites = None, ([], "none"), ([], "none")

    found = anchor or oa_anchor
    if found is None:
        statuses = {s2_refs[1], oa_refs[1] if backup_needed else "none"}
        if "busy" in statuses:
            return MESSAGES["busy"]
        return (
            f"I couldn't find '{identifier}'. Check the DOI, or paste the exact paper title."
            if "error" not in statuses
            else MESSAGES["error"]
        )
    return _render(found.get("title") or identifier, found.get("year"), _merge(s2_refs, oa_refs), _merge(s2_cites, oa_cites))