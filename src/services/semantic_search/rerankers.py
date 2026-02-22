from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import Settings

logger = logging.getLogger(__name__)
_MAX_LOGIT = 20.0


@dataclass(slots=True)
class Candidate:
    id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float


@dataclass(slots=True)
class RankedCandidate:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class Reranker(Protocol):
    def rerank(
        self,
        *,
        client: QdrantClient,
        query: str,
        candidates: list[Candidate],
        final_k: int,
    ) -> list[RankedCandidate]:
        ...


class NoopReranker:
    def rerank(
        self,
        *,
        client: QdrantClient,
        query: str,
        candidates: list[Candidate],
        final_k: int,
    ) -> list[RankedCandidate]:
        del client, query
        ranked = sorted(candidates, key=lambda item: item.dense_score, reverse=True)[:final_k]
        return [
            RankedCandidate(id=item.id, text=item.text, metadata=item.metadata, score=item.dense_score)
            for item in ranked
        ]



class ColbertLateInteractionReranker:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._dense_fallback = NoopReranker()

    def rerank(
        self,
        *,
        client: QdrantClient,
        query: str,
        candidates: list[Candidate],
        final_k: int,
    ) -> list[RankedCandidate]:
        if not candidates:
            return []


        candidate_ids = [item.id for item in candidates]
        score_by_id = self._colbert_scores(
            client=client,
            query=query,
            candidate_ids=candidate_ids,
            limit=len(candidates),
        )
        if score_by_id is None:
            return self._dense_fallback.rerank(
                client=client,
                query=query,
                candidates=candidates,
                final_k=final_k,
            )

        ranked = sorted(
            candidates,
            key=lambda item: score_by_id.get(item.id, item.dense_score),
            reverse=True,
        )[:final_k]
        return [
            RankedCandidate(
                id=item.id,
                text=item.text,
                metadata=item.metadata,
                score=score_by_id.get(item.id, item.dense_score),
            )
            for item in ranked
        ]

    def _colbert_scores(
        self,
        *,
        client: QdrantClient,
        query: str,
        candidate_ids: list[str],
        limit: int,
    ) -> dict[str, float] | None:
        if not candidate_ids:
            return {}

        query_filter = models.Filter(must=[models.HasIdCondition(has_id=candidate_ids)])
        try:
            response = client.query_points(
                collection_name=self._settings.retrieval_colbert_collection_name,
                query=models.Document(
                    text=query,
                    model=self._settings.retrieval_colbert_model_name,
                ),
                using=self._settings.retrieval_colbert_vector_name,
                query_filter=query_filter,
                limit=max(1, limit),
                with_payload=False,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning(
                "Late-interaction reranking unavailable (collection=%s, model=%s). "
                "Falling back to dense ranking: %s",
                self._settings.retrieval_colbert_collection_name,
                self._settings.retrieval_colbert_model_name,
                exc,
            )
            return None

        return {str(point.id): float(point.score) for point in response.points}


def build_reranker(settings: Settings) -> Reranker:
    mode = settings.retrieval_reranker_mode.strip().lower()
    if mode == "none":
        return NoopReranker()
    if mode == "late_interaction":
        return ColbertLateInteractionReranker(settings)

    logger.warning("Unknown reranker mode '%s'. Falling back to dense ranking.", mode)
    return NoopReranker()
