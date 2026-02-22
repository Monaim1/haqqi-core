from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=100)


class QueryHit(BaseModel):
    id: str
    score: float | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    hits: list[QueryHit]


class IngestRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1)
    include_colbert: bool = True


class IngestionDocumentStats(BaseModel):
    source_file: str
    source_filename: str
    pages_total: int
    pages_without_extractable_text: int
    pages_indexed: int
    pages_skipped_toc: int
    pages_skipped_boilerplate: int
    segments_indexed: int
    chunks_indexed: int


class IngestionReportSchema(BaseModel):
    documents_indexed: int
    documents_failed: int
    chunks_indexed: int
    document_stats: list[IngestionDocumentStats] = Field(default_factory=list)


class IngestResponse(BaseModel):
    dense: IngestionReportSchema
    colbert: IngestionReportSchema | None = None
