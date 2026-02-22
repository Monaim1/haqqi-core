from __future__ import annotations

from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import Settings


class QdrantService:
    def __init__(self, settings: Settings):
        self._settings = settings
        qdrant_url = self._settings.qdrant_url.strip()
        if qdrant_url:
            api_key = self._settings.qdrant_api_key.strip() or None
            self._client = QdrantClient(url=qdrant_url, api_key=api_key)
        else:
            Path(self._settings.qdrant_path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=self._settings.qdrant_path)
            
        # Configure Dense and Sparse Models via fastembed
        self._client.set_model(self._settings.embedding_model_name)
        self._client.set_sparse_model("prithivida/Splade_PP_en_v1") ### not good for fr/ar, fallback to Qdrant/bm25

    @property
    def client(self) -> QdrantClient:
        return self._client
        
    @property
    def collection_name(self) -> str:
        return self._settings.qdrant_collection_name

    @property
    def colbert_collection_name(self) -> str:
        return self._settings.retrieval_colbert_collection_name

    def needs_ingestion(self) -> bool:
        if not self._client.collection_exists(self.collection_name):
            return True
        collection_info = self._client.get_collection(self.collection_name)
        return collection_info.points_count == 0

    def needs_colbert_ingestion(self) -> bool:
        if not self._client.collection_exists(self.colbert_collection_name):
            return True
        colbert_count = self._client.get_collection(self.colbert_collection_name).points_count or 0
        if colbert_count == 0:
            return True
        if not self._client.collection_exists(self.collection_name):
            return False
        dense_count = self._client.get_collection(self.collection_name).points_count or 0
        return dense_count > 0 and colbert_count < dense_count

    def ensure_colbert_collection(self) -> None:
        if self._client.collection_exists(self.colbert_collection_name):
            return

        vector_size = self._client.get_embedding_size(model_name=self._settings.retrieval_colbert_model_name)
        vector_params = models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
            on_disk=self._settings.retrieval_colbert_on_disk_vectors,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
        )
        self._client.create_collection(
            collection_name=self.colbert_collection_name,
            vectors_config={self._settings.retrieval_colbert_vector_name: vector_params},
            on_disk_payload=True,
        )
