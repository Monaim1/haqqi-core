from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from src.services.semantic_search import SemanticSearchService
from src.services.semantic_search.ingestion import IngestionProgressEvent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Haqqi Core CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Run ingestion pipeline")
    ingest_parser.add_argument("--limit", type=int, default=None, help="Limit number of source PDFs")
    ingest_parser.add_argument(
        "--no-colbert",
        action="store_true",
        help="Skip ColBERT collection ingestion",
    )

    query_parser = subparsers.add_parser("query", help="Run a search query")
    query_parser.add_argument("q", type=str, help="Query text")
    query_parser.add_argument("--top-k", type=int, default=None, help="Override number of hits")

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
    dense = service.ingest_dense(limit=limit, progress_callback=_print_progress)
    colbert = service.ingest_colbert(limit=limit, progress_callback=_print_progress) if include_colbert else None

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


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        return _run_ingest(limit=args.limit, include_colbert=not args.no_colbert)
    if args.command == "query":
        return _run_query(query=args.q, top_k=args.top_k)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
