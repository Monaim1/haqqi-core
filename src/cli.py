from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import uvicorn

from src import hf_export
from src import index_fetch
from src.services.semantic_search import SemanticSearchService
from src.services.semantic_search.ingestion import IngestionProgressEvent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Haqqi Core CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Run ingestion pipeline")
    ingest_parser.add_argument("--limit", type=int, default=None, help="Limit number of source PDFs")
    ingest_parser.add_argument("--no-colbert", action="store_true", help="Skip ColBERT collection ingestion")

    query_parser = subparsers.add_parser("query", help="Run a search query")
    query_parser.add_argument("q", type=str, help="Query text")
    query_parser.add_argument("--top-k", type=int, default=None, help="Override number of hits")

    search_parser = subparsers.add_parser("search", help="Alias for query")
    search_parser.add_argument("q", type=str, help="Query text")
    search_parser.add_argument("--top-k", type=int, default=None, help="Override number of hits")

    api_parser = subparsers.add_parser("api", help="Run FastAPI server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    api_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    api_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")

    fetch_parser = subparsers.add_parser("fetch", help="Download PDFs listed in data/index.json")
    fetch_parser.add_argument("--index-file", default="data/index.json", help="Path to index JSON file")
    fetch_parser.add_argument(
        "--source-dir",
        default="data/raw-bulletin-officiel",
        help="Output PDF directory",
    )
    fetch_parser.add_argument("--limit", type=int, default=None, help="Limit number of entries")
    fetch_parser.add_argument(
        "--language",
        action="append",
        default=[],
        help="Filter by language code (repeatable, e.g. --language fr --language ar)",
    )
    fetch_parser.add_argument("--since", default=None, help="Filter publication_date >= YYYY-MM-DD")
    fetch_parser.add_argument("--until", default=None, help="Filter publication_date <= YYYY-MM-DD")
    fetch_parser.add_argument(
        "--include-downloaded",
        action="store_true",
        help="Include entries already marked downloaded in index file",
    )
    fetch_parser.add_argument("--dry-run", action="store_true", help="Print planned downloads only")
    fetch_parser.add_argument(
        "--state-out",
        default="",
        help="Optional path to write updated index state JSON",
    )
    fetch_parser.add_argument("--timeout", type=int, default=45, help="HTTP timeout per file in seconds")

    export_parser = subparsers.add_parser("export", help="Generate Hugging Face metadata bundle")
    export_parser.add_argument("--source-dir", default="data/raw-bulletin-officiel")
    export_parser.add_argument("--glob", default="*.pdf")
    export_parser.add_argument("--output-dir", default="hf_export")
    export_parser.add_argument("--repo-data-dir", default="data/raw-bulletin-officiel")
    export_parser.add_argument("--repo-id", default="")
    export_parser.add_argument("--split", default="train")
    export_parser.add_argument("--limit", type=int, default=None)
    export_parser.add_argument("--checksum-mode", choices=["sidecar", "compute", "none"], default="sidecar")

    return parser


def _print_progress(event: IngestionProgressEvent) -> None:
    stage = event.stage
    if stage == "document_done" and event.document_stats is not None:
        print(
            f"[done] {event.current_document}/{event.total_documents} "
            f"{event.source_filename} chunks={event.document_stats.chunks_indexed}"
        )
    elif stage == "document_failed":
        print(
            f"[fail] {event.current_document}/{event.total_documents} "
            f"{event.source_filename} error={event.error}"
        )
    elif stage == "start":
        print(f"[start] total_documents={event.total_documents}")
    elif stage == "complete":
        print(f"[complete] total_documents={event.total_documents}")


def _run_ingest(limit: int | None, include_colbert: bool) -> int:
    service = SemanticSearchService()
    try:
        dense = service.ingest_dense(limit=limit, progress_callback=_print_progress)
        if include_colbert:
            colbert = service.ingest_colbert(limit=limit, progress_callback=_print_progress)
        else:
            colbert = None
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Hint: create your source folder and add PDFs, "
            "or run `uv run haqqi fetch --limit 5` first.",
            file=sys.stderr,
        )
        return 2

    payload = {
        "dense": asdict(dense),
        "colbert": asdict(colbert) if colbert is not None else None,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run_query(query: str, top_k: int | None) -> int:
    service = SemanticSearchService()
    response = service.search(query, top_k=top_k)
    print(response.model_dump_json(indent=2, ensure_ascii=False))
    return 0


def _run_api(host: str, port: int, reload_enabled: bool) -> int:
    uvicorn.run("src.main:app", host=host, port=port, reload=reload_enabled)
    return 0


def _run_export(args: argparse.Namespace) -> int:
    export_argv = [
        "--source-dir",
        args.source_dir,
        "--glob",
        args.glob,
        "--output-dir",
        args.output_dir,
        "--repo-data-dir",
        args.repo_data_dir,
        "--repo-id",
        args.repo_id,
        "--split",
        args.split,
        "--checksum-mode",
        args.checksum_mode,
    ]
    if args.limit is not None:
        export_argv.extend(["--limit", str(args.limit)])
    return hf_export.main(export_argv)


def _run_fetch(args: argparse.Namespace) -> int:
    fetch_argv = [
        "--index-file",
        args.index_file,
        "--source-dir",
        args.source_dir,
        "--timeout",
        str(args.timeout),
    ]
    if args.limit is not None:
        fetch_argv.extend(["--limit", str(args.limit)])
    if args.since is not None:
        fetch_argv.extend(["--since", args.since])
    if args.until is not None:
        fetch_argv.extend(["--until", args.until])
    if args.include_downloaded:
        fetch_argv.append("--include-downloaded")
    if args.dry_run:
        fetch_argv.append("--dry-run")
    if args.state_out.strip():
        fetch_argv.extend(["--state-out", args.state_out])
    for language in args.language:
        fetch_argv.extend(["--language", language])

    return index_fetch.main(fetch_argv)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        return _run_ingest(limit=args.limit, include_colbert=not args.no_colbert)
    if args.command in {"query", "search"}:
        return _run_query(query=args.q, top_k=args.top_k)
    if args.command == "api":
        return _run_api(host=args.host, port=args.port, reload_enabled=not args.no_reload)
    if args.command == "fetch":
        return _run_fetch(args)
    if args.command == "export":
        return _run_export(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
