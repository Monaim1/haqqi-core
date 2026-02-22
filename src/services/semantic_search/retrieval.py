from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import Settings
from src.schemas import QueryHit
from src.services.semantic_search.rerankers import Candidate, Reranker, build_reranker

logger = logging.getLogger(__name__)


class RetrieverService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._reranker: Reranker = build_reranker(settings)

    def search(self, client: QdrantClient, collection_name: str, query: str, top_k: int | None = None) -> list[QueryHit]:
        final_k = top_k if isinstance(top_k, int) and top_k > 0 else self._settings.query_top_k
        candidate_k = self._settings.retrieval_candidate_k
        if candidate_k <= 0:
            candidate_k = self._settings.retrieval_dense_k
        stage1_k = max(final_k, candidate_k)

        candidates = self._dense_search(client=client, collection_name=collection_name, query=query, top_k=stage1_k)
        if not candidates:
            return []

        ranked = self._reranker.rerank(
            client=client,
            query=query,
            candidates=candidates,
            final_k=final_k,
        )
        return [
            QueryHit(
                id=item.id,
                text=item.text,
                score=item.score,
                metadata=item.metadata,
            )
            for item in ranked[:final_k]
        ]

    def _dense_search(self, client: QdrantClient, collection_name: str, query: str, top_k: int) -> list[Candidate]:
        dense_vector_name = client.get_vector_field_name()
        if not isinstance(dense_vector_name, str) or not dense_vector_name:
            dense_vector_name = None

        try:
            response = client.query_points(
                collection_name=collection_name,
                query=models.Document(
                    text=query,
                    model=self._settings.embedding_model_name,
                ),
                using=dense_vector_name,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning("Qdrant query failed: %s", exc)
            return []

        candidates: list[Candidate] = []
        for point in response.points:
            item_id = str(point.id)
            text = getattr(point, "document", "")
            metadata = getattr(point, "metadata", getattr(point, "payload", {}))
            if not isinstance(metadata, dict):
                metadata = {}
            if not text and "document" in metadata and isinstance(metadata.get("document"), str):
                text = metadata["document"]

            candidates.append(
                Candidate(
                    id=item_id,
                    text=text,
                    metadata=metadata,
                    dense_score=float(point.score) if hasattr(point, "score") else 0.0,
                )
            )
        return candidates
