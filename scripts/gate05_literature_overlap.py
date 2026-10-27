"""Bounded citation search for Gate 0.5. This is not a systematic review."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "gate05" / "literature"
USER_AGENT = "p8-coding-agent-uncertainty-audit/gate05 (literature audit)"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=60) as response:
        status = getattr(response, "status", 200)
        payload = json.loads(response.read().decode("utf-8"))
    return {"http_status": status, "url": url, "payload": payload}


def _arxiv(query: str) -> dict:
    import time
    from urllib.error import HTTPError

    url = (
        "https://export.arxiv.org/api/query?search_query="
        + quote(query)
        + "&start=0&max_results=15"
    )
    request = Request(url, headers={"User-Agent": USER_AGENT})
    last_error = "not attempted"
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                status = getattr(response, "status", 200)
                body = response.read().decode("utf-8", errors="replace")
            break
        except HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            time.sleep(5 * (attempt + 1))
        except Exception as exc:
            last_error = type(exc).__name__
            time.sleep(5 * (attempt + 1))
    else:
        return {"query": query, "error": last_error, "titles": []}
    time.sleep(3)
    titles = []
    current = None
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("<title>") and current is None and "ArXiv Query" not in stripped:
            current = stripped.removeprefix("<title>").removesuffix("</title>")
        elif stripped == "<title>" or stripped.startswith("<title>"):
            current = stripped.removeprefix("<title>")
        elif current is not None and not stripped.startswith("<"):
            current = (current + " " + stripped).strip()
        if current is not None and "</title>" in stripped:
            titles.append(current.replace("</title>", "").strip())
            current = None
        if len(titles) >= 15:
            break
    return {"http_status": status, "query": query, "titles": titles, "body_bytes": len(body)}


def _openalex(query: str) -> dict:
    url = "https://api.openalex.org/works?per-page=10&search=" + quote(query)
    result = _get_json(url)
    works = []
    for item in (result["payload"].get("results") or [])[:10]:
        works.append(
            {
                "id": item.get("id"),
                "title": item.get("display_name") or item.get("title"),
                "publication_year": item.get("publication_year"),
                "doi": item.get("doi"),
                "cited_by_count": item.get("cited_by_count"),
                "type": item.get("type"),
            }
        )
    return {
        "query": query,
        "http_status": result["http_status"],
        "count": result["payload"].get("meta", {}).get("count"),
        "works": works,
    }


def _local_paper_hits() -> dict:
    text_dir = ROOT / "logs" / "papers" / "text"
    hits = {}
    if not text_dir.exists():
        return {"status": "text directory missing"}
    for path in sorted(text_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        hits[path.name] = text.lower().count("open-swe-traces")
    return hits


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arxiv_queries = [
        'all:"Open-SWE-Traces"',
        'all:"Open-SWE-Traces" AND all:calibration',
        'all:"Open-SWE-Traces" AND all:failure',
        'all:"repository-held-out" AND all:"coding agent"',
        'all:"execution feedback" AND all:"coding agent"',
    ]
    openalex_queries = [
        "Open-SWE-Traces",
        "Open-SWE-Traces trajectory reliability",
        "Open-SWE-Traces failure prediction",
        "Open-SWE-Traces uncertainty calibration",
        "repository-held-out coding agent reliability",
        "execution feedback coding agent outcome prediction",
    ]
    github_url = (
        "https://api.github.com/search/repositories?q="
        + quote("Open-SWE-Traces trajectory reliability")
        + "&per_page=10"
    )
    try:
        github = _get_json(github_url)
        github_summary = {
            "http_status": github["http_status"],
            "total_count": github["payload"].get("total_count"),
            "items": [
                {"full_name": item.get("full_name"), "html_url": item.get("html_url")}
                for item in (github["payload"].get("items") or [])[:10]
            ],
        }
    except Exception as exc:
        github_summary = {"error": type(exc).__name__, "detail": str(exc)[:300]}
    def safe(function, argument):
        try:
            return function(argument)
        except Exception as exc:
            return {"query": argument, "error": type(exc).__name__, "detail": str(exc)[:300]}

    payload = {
        "generated_at_utc": _now(),
        "scope": "Bounded search. Not a systematic review and not a novelty proof.",
        "local_pdf_text_open_swe_traces_mentions": _local_paper_hits(),
        "arxiv": [safe(_arxiv, query) for query in arxiv_queries],
        "openalex": [safe(_openalex, query) for query in openalex_queries],
        "github_repository_search": github_summary,
    }
    destination = OUT_DIR / "literature_search.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"LITERATURE_DONE {destination}", flush=True)


if __name__ == "__main__":
    main()
