from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_prefix="HAQQI_CORE_",
        extra="ignore",
    )

    app_name: str = "Haqqi Core"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    source_dir: str = "data/raw-bulletin-officiel"
    source_glob_pattern: str = "**/*.pdf"
    pdf_parser_backend: str = "auto"

    qdrant_path: str = "data/qdrant"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "moroccan_law"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    default_chunk_size: int = 1200
    default_chunk_overlap: int = 200
    ingestion_batch_size: int = 64
    query_top_k: int = 6

    retrieval_dense_k: int = 30
    retrieval_candidate_k: int = 500
    retrieval_reranker_mode: str = "late_interaction"
    retrieval_colbert_collection_name: str = "moroccan_law_colbert"
    retrieval_colbert_model_name: str = "colbert-ir/colbertv2.0"
    retrieval_colbert_vector_name: str = "colbert"
    retrieval_colbert_batch_size: int = 8
    retrieval_colbert_on_disk_vectors: bool = True

    @field_validator("retrieval_reranker_mode", mode="before")
    @classmethod
    def parse_reranker_mode(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        allowed = {"none", "late_interaction"}
        if normalized not in allowed:
            raise ValueError(f"retrieval_reranker_mode must be one of {sorted(allowed)}")
        return normalized

    @field_validator("pdf_parser_backend", mode="before")
    @classmethod
    def parse_pdf_parser_backend(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        allowed = {"auto", "pypdf", "docling"}
        if normalized not in allowed:
            raise ValueError(f"pdf_parser_backend must be one of {sorted(allowed)}")
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
