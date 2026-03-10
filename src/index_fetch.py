from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

DEFAULT_INDEX_FILE = "data/index.json"
DEFAULT_SOURCE_DIR = "data/raw-bulletin-officiel"
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PDFs listed in an index.json file")
    parser.add_argument("--index-file", default=DEFAULT_INDEX_FILE, help="Path to index JSON file")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="Directory where PDFs are stored")
    parser.add_argument("--limit", type=int, default=None, help="Max number of entries to process")
    parser.add_argument(
        "--language",
        action="append",
        default=[],
        help="Filter by language code (e.g. fr). Repeatable or comma-separated.",
    )
    parser.add_argument("--since", default=None, help="Only keep entries with publication_date >= YYYY-MM-DD")
    parser.add_argument("--until", default=None, help="Only keep entries with publication_date <= YYYY-MM-DD")
    parser.add_argument(
        "--include-downloaded",
        action="store_true",
        help="Include entries already marked as downloaded in index.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--state-out",
        default="",
        help="Optional path where updated index state is written after downloads",
    )
    parser.add_argument("--timeout", type=int, default=45, help="HTTP timeout in seconds per file")
    return parser.parse_args(argv)


def _parse_iso_date(value: str | None, flag_name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{flag_name} must be in YYYY-MM-DD format, got: {value}") from exc


def _parse_entry_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _parse_languages(values: list[str]) -> set[str]:
    parsed: set[str] = set()
    for raw in values:
        for token in raw.split(","):
            normalized = token.strip().lower()
            if normalized:
                parsed.add(normalized)
    return parsed


def _is_marked_downloaded(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _derive_filename(entry: dict[str, Any], pdf_url: str) -> str:
    candidate = unquote(Path(urlsplit(pdf_url).path).name)
    if not candidate.lower().endswith(".pdf"):
        issue = str(entry.get("issue_number") or "document").strip()
        language = str(entry.get("language") or "").strip().lower()
        suffix = f"_{language}" if language else ""
        candidate = f"bo_{issue}{suffix}.pdf"

    candidate = candidate.replace(" ", "_")
    safe = _FILENAME_SAFE_RE.sub("_", candidate).strip("._")
    if not safe:
        return "document.pdf"
    if not safe.lower().endswith(".pdf"):
        return f"{safe}.pdf"
    return safe


def _load_index(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    loaded = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(f"Index must be a JSON array, got: {type(loaded).__name__}")
    return [item for item in loaded if isinstance(item, dict)]


def _download_file(pdf_url: str, destination: Path, timeout: int) -> str:
    request = Request(pdf_url, headers={"User-Agent": "haqqi-core-fetch/0.1"})
    temp_path: Path | None = None
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            status = getattr(response, "status", 200) or 200
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")

            digest = hashlib.sha256()
            with NamedTemporaryFile("wb", delete=False, dir=destination.parent, suffix=".part") as handle:
                temp_path = Path(handle.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)

        if temp_path is None:
            raise RuntimeError("No temporary file was created during download")
        temp_path.replace(destination)
        return digest.hexdigest()
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def _write_sidecar(sidecar_path: Path, payload: dict[str, Any]) -> None:
    sidecar_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_sidecar_payload(entry: dict[str, Any], checksum: str, file_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("issue_number", "publication_date", "language", "pdf_url"):
        value = entry.get(key)
        if value is not None:
            payload[key] = value
    payload["checksum"] = checksum
    payload["file_path"] = str(file_path)
    return payload


def _filter_entries(
    entries: list[dict[str, Any]],
    *,
    language_filters: set[str],
    since_date: date | None,
    until_date: date | None,
    include_downloaded: bool,
) -> list[tuple[dict[str, Any], str, date | None]]:
    filtered: list[tuple[dict[str, Any], str, date | None]] = []
    for entry in entries:
        pdf_url = entry.get("pdf_url")
        if not isinstance(pdf_url, str) or not pdf_url.strip():
            continue
        pdf_url = pdf_url.strip()

        if not include_downloaded and _is_marked_downloaded(entry.get("downloaded")):
            continue

        language = str(entry.get("language") or "").strip().lower()
        if language_filters and language not in language_filters:
            continue

        publication_date = _parse_entry_date(entry.get("publication_date"))
        if since_date is not None and (publication_date is None or publication_date < since_date):
            continue
        if until_date is not None and (publication_date is None or publication_date > until_date):
            continue

        filtered.append((entry, pdf_url, publication_date))

    filtered.sort(
        key=lambda item: (
            item[2] or date.min,
            str(item[0].get("issue_number") or ""),
        ),
        reverse=True,
    )
    return filtered


def _update_index_entry(entry: dict[str, Any], checksum: str, destination: Path) -> None:
    entry["downloaded"] = True
    entry["checksum"] = checksum
    entry["file_path"] = str(destination)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        since_date = _parse_iso_date(args.since, "--since")
        until_date = _parse_iso_date(args.until, "--until")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if since_date and until_date and since_date > until_date:
        print("Error: --since must be earlier than or equal to --until", file=sys.stderr)
        return 2

    language_filters = _parse_languages(args.language)
    index_path = Path(args.index_file).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        entries = _load_index(index_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    selected = _filter_entries(
        entries,
        language_filters=language_filters,
        since_date=since_date,
        until_date=until_date,
        include_downloaded=args.include_downloaded,
    )
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]

    if not selected:
        print("No index entries matched the current filters.")
        return 0

    planned = len(selected)
    downloaded = 0
    already_present = 0
    failures = 0

    for entry, pdf_url, _publication_date in selected:
        filename = _derive_filename(entry, pdf_url)
        destination = source_dir / filename
        sidecar_path = destination.with_suffix(".json")

        if destination.exists():
            checksum = entry.get("checksum") if isinstance(entry.get("checksum"), str) else ""
            if not sidecar_path.exists():
                sidecar_payload = _build_sidecar_payload(entry, checksum, destination)
                _write_sidecar(sidecar_path, sidecar_payload)
            _update_index_entry(entry, checksum, destination)
            already_present += 1
            print(f"[exists] {destination}")
            continue

        if args.dry_run:
            print(f"[dry-run] {pdf_url} -> {destination}")
            continue

        try:
            checksum = _download_file(pdf_url, destination, timeout=max(1, args.timeout))
            sidecar_payload = _build_sidecar_payload(entry, checksum, destination)
            _write_sidecar(sidecar_path, sidecar_payload)
            _update_index_entry(entry, checksum, destination)
            downloaded += 1
            print(f"[downloaded] {destination}")
        except (HTTPError, URLError, TimeoutError) as exc:
            failures += 1
            print(f"[failed] {pdf_url} ({exc})", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[failed] {pdf_url} ({exc})", file=sys.stderr)

    if args.state_out.strip():
        state_out = Path(args.state_out).expanduser().resolve()
        state_out.parent.mkdir(parents=True, exist_ok=True)
        state_out.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[state] wrote {state_out}")

    print(
        "Summary: "
        f"planned={planned} downloaded={downloaded} "
        f"already_present={already_present} failures={failures} dry_run={args.dry_run}"
    )
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
