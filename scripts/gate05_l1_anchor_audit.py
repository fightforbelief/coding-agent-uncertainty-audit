"""Source audit for the two L1 candidates. Does not run either experiment."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "gate05" / "l1"
MAX_DOWNLOAD = 80 * 1024 * 1024
USER_AGENT = "p8-coding-agent-uncertainty-audit/gate05 (source audit)"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get(url: str, timeout: int = 60) -> tuple[int, bytes, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        content_type = response.headers.get("Content-Type", "")
        return status, response.read(), content_type


def _json(url: str) -> dict:
    status, body, content_type = _get(url)
    payload = json.loads(body.decode("utf-8"))
    payload["_http_status"] = status
    payload["_content_type"] = content_type
    payload["_url"] = url
    return payload


def _zenodo() -> dict:
    record = _json("https://zenodo.org/api/records/19251470")
    files = []
    for item in record.get("files") or []:
        files.append(
            {
                "key": item.get("key"),
                "size": item.get("size"),
                "checksum": item.get("checksum"),
                "link": (item.get("links") or {}).get("self"),
            }
        )
    metadata = record.get("metadata") or {}
    downloaded = []
    for item in files:
        size = int(item["size"] or 0)
        link = item["link"]
        entry = {"key": item["key"], "size": size, "downloaded": False, "reason": ""}
        if not link:
            entry["reason"] = "no download link"
        elif size > MAX_DOWNLOAD:
            entry["reason"] = "larger than 80 MiB; not downloaded"
        else:
            status, body, _content_type = _get(link, timeout=180)
            destination = OUT_DIR / "c2_files" / Path(item["key"]).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(body)
            entry["downloaded"] = True
            entry["http_status"] = status
            entry["saved_to"] = str(destination)
            if item["key"].lower().endswith(".zip"):
                with zipfile.ZipFile(BytesIO(body)) as archive:
                    names = archive.namelist()
                    entry["zip_members"] = names[:400]
                    entry["zip_member_count"] = len(names)
                    previews = []
                    for name in names:
                        lower = name.lower()
                        if lower.endswith((".md", ".txt", ".py", ".yml", ".yaml", ".toml", ".csv")):
                            info = archive.getinfo(name)
                            if info.file_size > 200_000:
                                previews.append({"name": name, "bytes": info.file_size, "preview": "skipped large member"})
                                continue
                            text = archive.read(name).decode("utf-8", errors="replace")[:4000]
                            previews.append({"name": name, "bytes": info.file_size, "preview": text})
                        if len(previews) >= 25:
                            break
                    entry["text_previews"] = previews
        downloaded.append(entry)
    return {
        "requested_record": 19251470,
        "api_id": record.get("id"),
        "doi": (metadata.get("doi") or record.get("doi")),
        "title": metadata.get("title"),
        "publication_date": metadata.get("publication_date"),
        "license": metadata.get("license"),
        "description_head": (metadata.get("description") or "")[:2000],
        "related_identifiers": metadata.get("related_identifiers"),
        "files": files,
        "download_audit": downloaded,
    }


def _github() -> dict:
    commit = "9e3825b7eea89ee9dc88f1ccd11f277f8e10beb3"
    repo = _json("https://api.github.com/repos/bennidict23/calibration-is-not-control")
    license_payload = _json("https://api.github.com/repos/bennidict23/calibration-is-not-control/license")
    commit_payload = _json(
        f"https://api.github.com/repos/bennidict23/calibration-is-not-control/commits/{commit}"
    )
    tree = _json(
        "https://api.github.com/repos/bennidict23/calibration-is-not-control/git/trees/"
        f"{commit}?recursive=1"
    )
    texts = {}
    for path in ("README.md", "requirements.txt", "LICENSE"):
        url = (
            "https://raw.githubusercontent.com/bennidict23/calibration-is-not-control/"
            f"{commit}/{path}"
        )
        try:
            status, body, _content_type = _get(url)
            texts[path] = {"http_status": status, "text": body.decode("utf-8", errors="replace")[:20000]}
        except Exception as exc:
            texts[path] = {"error": type(exc).__name__}
    entries = []
    for item in tree.get("tree") or []:
        entries.append({"path": item.get("path"), "type": item.get("type"), "size": item.get("size")})
    return {
        "commit": commit,
        "repo_license": (repo.get("license") or {}).get("spdx_id"),
        "license_api": (license_payload.get("license") or {}).get("spdx_id"),
        "default_branch": repo.get("default_branch"),
        "pushed_at": repo.get("pushed_at"),
        "commit_date": ((commit_payload.get("commit") or {}).get("committer") or {}).get("date"),
        "commit_message": ((commit_payload.get("commit") or {}).get("message") or "")[:300],
        "tree_count": len(entries),
        "tree": entries,
        "texts": texts,
    }


def _paper_mentions() -> dict:
    text_dir = ROOT / "logs" / "papers" / "text"
    found = {}
    needles = ("github.com", "zenodo", "10.5281", "reproduc", "supplement")
    for name in ("C2_agent_tests.txt", "C5_calibration_control.txt"):
        path = text_dir / name
        if not path.exists():
            found[name] = {"status": "MISSING"}
            continue
        lines = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            lower = line.lower()
            if any(needle in lower for needle in needles):
                lines.append(line.strip()[:300])
            if len(lines) >= 40:
                break
        found[name] = {"matching_lines": lines}
    return found


def _section(name: str, function) -> dict:
    try:
        return {"ok": True, "result": function()}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:800]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": _now(),
        "experiment_executed": False,
        "zenodo": _section("zenodo", _zenodo),
        "github_c5": _section("github", _github),
        "paper_text_mentions": _section("papers", _paper_mentions),
    }
    destination = OUT_DIR / "l1_evidence.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"L1_AUDIT_DONE {destination}", flush=True)


if __name__ == "__main__":
    main()
