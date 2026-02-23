from __future__ import annotations

import json
import logging
import re
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Literal

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import Settings
from src.services.semantic_search.pdf_parser import extract_pdf_page_texts

_PUNCTUATION_SPLIT_RE = re.compile(r"[.;:!?]\s")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")


def _normalize_text(text: str) -> str:
    cleaned = text.replace("\x00", " ").replace("\r", "\n")
    cleaned = "".join(ch if (ch == "\n" or ch == "\t" or ord(ch) >= 32) else " " for ch in cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _simplify_for_match(text: str) -> str:
    deaccented = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    normalized_chars: list[str] = []
    for ch in deaccented.lower():
        if ch.isalnum() or ch in {" ", "-"}:
            normalized_chars.append(ch)
        else:
            normalized_chars.append(" ")
    return " ".join("".join(normalized_chars).split())


def _find_split_index(text: str, split_start: int, target_end: int) -> int | None:
    if split_start >= target_end:
        return None

    window = text[split_start:target_end]
    punctuation_matches = list(_PUNCTUATION_SPLIT_RE.finditer(window))
    if punctuation_matches:
        return split_start + punctuation_matches[-1].end()

    whitespace_idx = text.rfind(" ", split_start, target_end)
    if whitespace_idx > split_start:
        return whitespace_idx
    return None


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    chunks: list[str] = []
    cursor = 0
    text_len = len(normalized)

    while cursor < text_len:
        target_end = min(cursor + chunk_size, text_len)
        if target_end < text_len:
            split_start = cursor + int(chunk_size * 0.6)
            split_idx = _find_split_index(normalized, split_start, target_end)
            if split_idx and split_idx > cursor:
                target_end = split_idx

        chunk = normalized[cursor:target_end].strip()
        if chunk:
            chunks.append(chunk)

        if target_end >= text_len:
            break
        cursor = max(target_end - chunk_overlap, cursor + 1)

    return chunks


def _split_page_into_segments(page_text: str) -> list[str]:
    normalized = _normalize_text(page_text)
    if not normalized:
        return []

    paragraphs = [part.strip() for part in _PARAGRAPH_SPLIT_RE.split(normalized) if part.strip()]
    if not paragraphs:
        return [normalized]

    # Merge tiny fragments into neighboring paragraphs to avoid indexing headers/noise as standalone chunks.
    segments: list[str] = []
    for paragraph in paragraphs:
        if segments and len(paragraph) < 80:
            segments[-1] = f"{segments[-1]} {paragraph}".strip()
            continue
        segments.append(paragraph)
    return segments


def _detect_segment_type(text: str) -> str:
    first_token = _simplify_for_match(text).split(" ", 1)[0] if text else ""
    if first_token == "article":
        return "article"
    if first_token == "chapitre":
        return "chapter"
    if first_token == "section":
        return "section"
    if first_token in {"dahir", "loi", "decret", "arrete"}:
        return "legal_act"
    return "paragraph"


def _extract_article_id(text: str) -> str | None:
    tokens = _simplify_for_match(text).split()
    for idx, token in enumerate(tokens[:-1]):
        if token != "article":
            continue
        candidate = tokens[idx + 1]
        if candidate in {"premier"} or candidate.isdigit():
            return candidate
        if all(ch in set("ivxlcdm") for ch in candidate):
            return candidate
    return None


def _extract_act_type(text: str) -> str | None:
    first_token = _simplify_for_match(text).split(" ", 1)[0] if text else ""
    if first_token in {"dahir", "loi", "decret", "arrete"}:
        return first_token
    return None


@dataclass(slots=True)
class ChunkRecord:
    id: str
    text: str
    metadata: dict[str, str | int | float | bool]


@dataclass(slots=True)
class DocumentIngestionStats:
    source_file: str
    source_filename: str
    pages_total: int
    pages_without_extractable_text: int
    pages_indexed: int
    pages_skipped_toc: int
    pages_skipped_boilerplate: int
    segments_indexed: int
    chunks_indexed: int


@dataclass(slots=True)
class IngestionReport:
    documents_indexed: int = 0
    documents_failed: int = 0
    chunks_indexed: int = 0
    document_stats: list[DocumentIngestionStats] = field(default_factory=list)


@dataclass(slots=True)
class IngestionProgressEvent:
    stage: Literal["start", "document_start", "document_done", "document_failed", "complete"]
    total_documents: int
    current_document: int = 0
    source_file: str | None = None
    source_filename: str | None = None
    document_stats: DocumentIngestionStats | None = None
    error: str | None = None


class IngestionService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._logger = logging.getLogger(__name__)

    def ingest_collection(
        self,
        client: QdrantClient,
        collection_name: str,
        limit: int | None = None,
        progress_callback: Callable[[IngestionProgressEvent], None] | None = None,
    ) -> IngestionReport:
        report = IngestionReport()
        source_dir = Path(self._settings.source_dir)
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"source_dir does not exist or is not a directory: {source_dir}")

        pdf_paths = sorted(source_dir.glob(self._settings.source_glob_pattern))
        if limit is not None:
            pdf_paths = pdf_paths[: max(0, limit)]
        if not pdf_paths:
            self._logger.warning("No PDFs found for indexing in %s", source_dir)
            return report
        self._ensure_dense_collection(client, collection_name)

        dense_model = self._settings.embedding_model_name
        sparse_model = getattr(client, "sparse_embedding_model_name", None)
        if not isinstance(sparse_model, str) or not sparse_model.strip():
            sparse_model = "prithivida/Splade_PP_en_v1"

        dense_vector_name = client.get_vector_field_name()
        if not isinstance(dense_vector_name, str) or not dense_vector_name:
            dense_vector_name = "vector"

        sparse_vector_name = client.get_sparse_vector_field_name()
        if not isinstance(sparse_vector_name, str) or not sparse_vector_name:
            sparse_vector_name = None

        total_documents = len(pdf_paths)
        if progress_callback is not None:
            progress_callback(IngestionProgressEvent(stage="start", total_documents=total_documents))

        for current_document, pdf_path in enumerate(pdf_paths, start=1):
            if progress_callback is not None:
                progress_callback(
                    IngestionProgressEvent(
                        stage="document_start",
                        total_documents=total_documents,
                        current_document=current_document,
                        source_file=str(pdf_path),
                        source_filename=pdf_path.name,
                    )
                )
            try:
                records, doc_stats = self._extract_records(pdf_path)
                for idx in range(0, len(records), self._settings.ingestion_batch_size):
                    batch = records[idx : idx + self._settings.ingestion_batch_size]
                    points = [
                        models.PointStruct(
                            id=record.id,
                            vector={
                                dense_vector_name: models.Document(
                                    text=record.text,
                                    model=dense_model,
                                ),
                                **(
                                    {
                                        sparse_vector_name: models.Document(
                                            text=record.text,
                                            model=sparse_model,
                                        )
                                    }
                                    if sparse_vector_name
                                    else {}
                                ),
                            },
                            payload={
                                "document": record.text,
                                **record.metadata,
                            },
                        )
                        for record in batch
                    ]
                    client.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=True,
                    )
                report.documents_indexed += 1
                report.chunks_indexed += len(records)
                report.document_stats.append(doc_stats)
                if progress_callback is not None:
                    progress_callback(
                        IngestionProgressEvent(
                            stage="document_done",
                            total_documents=total_documents,
                            current_document=current_document,
                            source_file=str(pdf_path),
                            source_filename=pdf_path.name,
                            document_stats=doc_stats,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                report.documents_failed += 1
                self._logger.exception("Failed to index %s: %s", pdf_path, exc)
                if progress_callback is not None:
                    progress_callback(
                        IngestionProgressEvent(
                            stage="document_failed",
                            total_documents=total_documents,
                            current_document=current_document,
                            source_file=str(pdf_path),
                            source_filename=pdf_path.name,
                            error=str(exc),
                        )
                    )

        if progress_callback is not None:
            progress_callback(
                IngestionProgressEvent(
                    stage="complete",
                    total_documents=total_documents,
                    current_document=total_documents,
                )
            )
        return report

    @staticmethod
    def _ensure_dense_collection(client: QdrantClient, collection_name: str) -> None:
        if client.collection_exists(collection_name):
            return
        client.create_collection(
            collection_name=collection_name,
            vectors_config=client.get_fastembed_vector_params(),
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
            on_disk_payload=True,
        )

    def ingest_colbert_collection(
        self,
        client: QdrantClient,
        collection_name: str,
        limit: int | None = None,
        progress_callback: Callable[[IngestionProgressEvent], None] | None = None,
    ) -> IngestionReport:
        report = IngestionReport()
        source_dir = Path(self._settings.source_dir)
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"source_dir does not exist or is not a directory: {source_dir}")

        pdf_paths = sorted(source_dir.glob(self._settings.source_glob_pattern))
        if limit is not None:
            pdf_paths = pdf_paths[: max(0, limit)]
        if not pdf_paths:
            self._logger.warning("No PDFs found for ColBERT indexing in %s", source_dir)
            return report

        total_documents = len(pdf_paths)
        if progress_callback is not None:
            progress_callback(IngestionProgressEvent(stage="start", total_documents=total_documents))

        colbert_model = self._settings.retrieval_colbert_model_name
        vector_name = self._settings.retrieval_colbert_vector_name

        for current_document, pdf_path in enumerate(pdf_paths, start=1):
            if progress_callback is not None:
                progress_callback(
                    IngestionProgressEvent(
                        stage="document_start",
                        total_documents=total_documents,
                        current_document=current_document,
                        source_file=str(pdf_path),
                        source_filename=pdf_path.name,
                    )
                )

            try:
                records, doc_stats = self._extract_records(pdf_path)
                for idx in range(0, len(records), self._settings.retrieval_colbert_batch_size):
                    batch = records[idx : idx + self._settings.retrieval_colbert_batch_size]
                    points = [
                        models.PointStruct(
                            id=record.id,
                            vector={
                                vector_name: models.Document(
                                    text=record.text,
                                    model=colbert_model,
                                )
                            },
                            payload={
                                "document": record.text,
                                **record.metadata,
                            },
                        )
                        for record in batch
                    ]
                    client.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=True,
                    )

                report.documents_indexed += 1
                report.chunks_indexed += len(records)
                report.document_stats.append(doc_stats)
                if progress_callback is not None:
                    progress_callback(
                        IngestionProgressEvent(
                            stage="document_done",
                            total_documents=total_documents,
                            current_document=current_document,
                            source_file=str(pdf_path),
                            source_filename=pdf_path.name,
                            document_stats=doc_stats,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                report.documents_failed += 1
                self._logger.exception("Failed to index %s into ColBERT collection: %s", pdf_path, exc)
                if progress_callback is not None:
                    progress_callback(
                        IngestionProgressEvent(
                            stage="document_failed",
                            total_documents=total_documents,
                            current_document=current_document,
                            source_file=str(pdf_path),
                            source_filename=pdf_path.name,
                            error=str(exc),
                        )
                    )

        if progress_callback is not None:
            progress_callback(
                IngestionProgressEvent(
                    stage="complete",
                    total_documents=total_documents,
                    current_document=total_documents,
                )
            )
        return report

    def _extract_records(self, pdf_path: Path) -> tuple[list[ChunkRecord], DocumentIngestionStats]:
        page_texts = extract_pdf_page_texts(pdf_path=pdf_path, logger=self._logger)
        sidecar_metadata = self._load_sidecar_metadata(pdf_path)
        base_metadata = self._sanitize_metadata(
            {
                **sidecar_metadata,
                "file_path": str(pdf_path),
                "source_file": str(pdf_path),
                "source_filename": pdf_path.name,
                "parser_backend": "docling",
                "parser_model": "ibm-granite/granite-docling-258M",
            }
        )
        source_fingerprint = self._build_source_fingerprint(pdf_path, sidecar_metadata)

        records: list[ChunkRecord] = []
        pages_without_extractable_text = 0
        pages_indexed = 0
        pages_skipped_toc = 0
        pages_skipped_boilerplate = 0
        segments_indexed = 0

        for page_idx, raw_text in enumerate(page_texts, start=1):
            segments = _split_page_into_segments(raw_text)
            if not segments:
                pages_without_extractable_text += 1
                continue

            page_had_records = False
            for segment_idx, segment_text in enumerate(segments, start=1):
                chunks = _chunk_text(
                    text=segment_text,
                    chunk_size=self._settings.default_chunk_size,
                    chunk_overlap=self._settings.default_chunk_overlap,
                )
                if not chunks:
                    continue

                segment_type = _detect_segment_type(segment_text)
                article_id = _extract_article_id(segment_text)
                act_type = _extract_act_type(segment_text)

                for chunk_idx, chunk in enumerate(chunks, start=1):
                    id_str = f"{source_fingerprint}:p{page_idx:04d}:s{segment_idx:04d}:c{chunk_idx:04d}"
                    records.append(
                        ChunkRecord(
                            id=str(uuid.uuid5(uuid.NAMESPACE_URL, id_str)),
                            text=chunk,
                            metadata=self._sanitize_metadata(
                                {
                                    **base_metadata,
                                    "page_number": page_idx,
                                    "page_start": page_idx,
                                    "page_end": page_idx,
                                    "segment_index": segment_idx,
                                    "segment_type": segment_type,
                                    "article_id": article_id,
                                    "act_type": act_type,
                                    "chunk_index": chunk_idx,
                                    "chunk_char_count": len(chunk),
                                }
                            ),
                        )
                    )
                    page_had_records = True
                segments_indexed += 1

            if page_had_records:
                pages_indexed += 1

        return records, DocumentIngestionStats(
            source_file=str(pdf_path),
            source_filename=pdf_path.name,
            pages_total=len(page_texts),
            pages_without_extractable_text=pages_without_extractable_text,
            pages_indexed=pages_indexed,
            pages_skipped_toc=pages_skipped_toc,
            pages_skipped_boilerplate=pages_skipped_boilerplate,
            segments_indexed=segments_indexed,
            chunks_indexed=len(records),
        )

    @staticmethod
    def _load_sidecar_metadata(pdf_path: Path) -> dict[str, Any]:
        sidecar_path = pdf_path.with_suffix(".json")
        if not sidecar_path.exists():
            return {}

        with sidecar_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _build_source_fingerprint(pdf_path: Path, sidecar_metadata: dict[str, Any]) -> str:
        checksum = sidecar_metadata.get("checksum")
        if isinstance(checksum, str) and checksum:
            return checksum
        payload = f"{pdf_path}:{pdf_path.stat().st_size}:{pdf_path.stat().st_mtime}"
        return sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        sanitized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized
