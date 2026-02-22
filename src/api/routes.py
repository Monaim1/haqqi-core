from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status

from src.schemas import IngestRequest, IngestResponse, IngestionReportSchema, QueryRequest, QueryResponse
from src.services.semantic_search import SemanticSearchService, get_semantic_search_service

router = APIRouter(tags=["pipeline"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    semantic_search_service: SemanticSearchService = Depends(get_semantic_search_service),
) -> QueryResponse:
    try:
        return semantic_search_service.search(request.query, top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    semantic_search_service: SemanticSearchService = Depends(get_semantic_search_service),
) -> IngestResponse:
    try:
        dense_report = semantic_search_service.ingest_dense(limit=request.limit)
        colbert_report = (
            semantic_search_service.ingest_colbert(limit=request.limit) if request.include_colbert else None
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return IngestResponse(
        dense=IngestionReportSchema.model_validate(asdict(dense_report)),
        colbert=(
            IngestionReportSchema.model_validate(asdict(colbert_report)) if colbert_report is not None else None
        ),
    )
