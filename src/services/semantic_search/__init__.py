from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache

from src.config import get_settings
from src.schemas import QueryResponse
from src.services.semantic_search.ingestion import IngestionProgressEvent, IngestionReport, IngestionService
from src.services.semantic_search.qdrant_service import QdrantService
from src.services.semantic_search.retrieval import RetrieverService

logger = logging.getLogger(__name__)


class SemanticSearchService:
    def __init__(self):
        settings = get_settings()
        self._settings = settings
        self._qdrant_service = QdrantService(settings)
        self._ingestion_service = IngestionService(settings)
        self._retriever = RetrieverService(settings)

    def search(self, query: str, top_k: int | None = None) -> QueryResponse:
        client = self._qdrant_service.client
        collection_name = self._qdrant_service.collection_name

        hits = self._retriever.search(client, collection_name, query, top_k=top_k)
        return QueryResponse(query=query, hits=hits)

    def ingest_dense(
        self,
        limit: int | None = None,
        progress_callback: Callable[[IngestionProgressEvent], None] | None = None,
    ) -> IngestionReport:
        return self._ingestion_service.ingest_collection(
            client=self._qdrant_service.client,
            collection_name=self._qdrant_service.collection_name,
            limit=limit,
            progress_callback=progress_callback,
        )

    def ingest_colbert(
        self,
        limit: int | None = None,
        progress_callback: Callable[[IngestionProgressEvent], None] | None = None,
    ) -> IngestionReport:
        self._qdrant_service.ensure_colbert_collection()
        return self._ingestion_service.ingest_colbert_collection(
            client=self._qdrant_service.client,
            collection_name=self._qdrant_service.colbert_collection_name,
            limit=limit,
            progress_callback=progress_callback,
        )


@lru_cache(maxsize=1)
def get_semantic_search_service() -> SemanticSearchService:
    return SemanticSearchService()


__all__ = ["SemanticSearchService", "get_semantic_search_service"]
