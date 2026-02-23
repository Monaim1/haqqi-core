from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Literal
import unicodedata
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import Settings
from src.services.semantic_search.pdf_parser import extract_pdf_page_texts

_PUNCTUATION_SPLIT_RE = re.compile(r"[.;:!?]\s")


def _normalize_text(text: str) -> str:
    cleaned = text.replace("\x00", " ").replace("\r", "\n")
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


@dataclass(slots=True)
class ParsedSegment:
    text: str
    segment_type: str
    article_id: str | None
    act_type: str | None


@dataclass(slots=True)
class PageParseResult:
    segments: list[ParsedSegment]
    skipped_toc: bool = False
    skipped_boilerplate: bool = False


class LegalPageParser:
    _SEGMENT_STARTERS = ("article", "chapitre", "section", "titre", "livre", "dahir", "loi", "decret", "arrete")
    _ACT_TYPES = {"dahir", "loi", "decret", "arrete"}
    _TOC_MARKERS = ("sommaire", "textes particuliers")
    _BOILERPLATE_PREFIXES = (
        "pages pages",
        "bulletin officiel",
        "issn",
        "royaume du maroc",
        "edition",
        "imprimerie officielle",
        "tarifs d abonnement",
        "abonnement",
        "au maroc",
        "a l etranger",
    )

    def parse_page(self, raw_text: str) -> PageParseResult:
        page_text = self.normalize_page_text(raw_text)
        if not page_text:
            return PageParseResult(segments=[])

        lines = [line for line in page_text.splitlines() if line]
        if self.is_toc_page(lines):
            return PageParseResult(segments=[], skipped_toc=True)

        content_lines = self.strip_boilerplate_lines(lines)
        if not content_lines:
            return PageParseResult(segments=[], skipped_boilerplate=True)

        segment_texts = self.split_legal_elements("\n".join(content_lines))
        if not segment_texts:
            segment_texts = [_normalize_text("\n".join(content_lines))]

        segments = [
            ParsedSegment(
                text=segment,
                segment_type=self.detect_segment_type(segment),
                article_id=self.extract_article_id(segment),
                act_type=self.extract_act_type(segment),
            )
            for segment in segment_texts
            if segment
        ]
        return PageParseResult(segments=segments)

    def normalize_page_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\x00", " ").replace("\r", "\n")
        lines = [" ".join(line.split()) for line in cleaned.split("\n")]
        return "\n".join(lines).strip()

    def is_boilerplate_line(self, line: str) -> bool:
        normalized = _simplify_for_match(line)
        if not normalized:
            return False

        parts = normalized.split()
        if len(parts) == 1 and parts[0].isdigit() and len(parts[0]) <= 4:
            return True

        if any(normalized.startswith(prefix) for prefix in self._BOILERPLATE_PREFIXES):
            return True

        if "bulletin officiel" in normalized and parts:
            first = parts[0]
            if first.isdigit() or first in {"n", "no"}:
                return True

        return False

    def _looks_like_toc_entry(self, line: str) -> bool:
        if line.count(".") < 3:
            return False
        tokens = line.rstrip().split()
        if not tokens:
            return False
        page_token = tokens[-1].rstrip(".)")
        return page_token.isdigit() and len(page_token) <= 4

    def is_toc_page(self, lines: list[str]) -> bool:
        content_lines = [line for line in lines if line.strip()]
        if not content_lines:
            return False

        dotted_line_count = sum(1 for line in content_lines if self._looks_like_toc_entry(line))
        has_toc_marker = any(
            marker in _simplify_for_match(line)
            for line in content_lines[:30]
            for marker in self._TOC_MARKERS
        )
        return dotted_line_count >= 5 or (has_toc_marker and dotted_line_count >= 3)

    def strip_boilerplate_lines(self, lines: list[str]) -> list[str]:
        return [line for line in lines if line.strip() and not self.is_boilerplate_line(line)]

    def _starts_new_segment(self, text: str) -> bool:
        normalized = _simplify_for_match(text)
        if not normalized:
            return False
        first_token = normalized.split(" ", 1)[0]
        return first_token in self._SEGMENT_STARTERS

    def split_legal_elements(self, page_text: str) -> list[str]:
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        if not lines:
            return []

        raw_segments: list[str] = []
        buffer: list[str] = []
        for line in lines:
            if self._starts_new_segment(line) and buffer:
                raw_segments.append(" ".join(buffer).strip())
                buffer = [line]
            else:
                buffer.append(line)

        if buffer:
            raw_segments.append(" ".join(buffer).strip())

        merged: list[str] = []
        for segment in raw_segments:
            if merged and len(segment) < 180 and not self._starts_new_segment(segment):
                merged[-1] = f"{merged[-1]} {segment}".strip()
            else:
                merged.append(segment)
        return merged

    def detect_segment_type(self, text: str) -> str:
        normalized = _simplify_for_match(text)
        if not normalized:
            return "paragraph"

        first_token = normalized.split(" ", 1)[0]
        if first_token == "article":
            return "article"
        if first_token == "chapitre":
            return "chapter"
        if first_token == "section":
            return "section"
        if first_token in self._ACT_TYPES:
            return "legal_act"
        return "paragraph"

    @staticmethod
    def _looks_like_article_id(token: str) -> bool:
        if not token:
            return False
        if token == "premier":
            return True
        if token.isdigit():
            return True

        roman_chars = set("ivxlcdm")
        if all(ch in roman_chars for ch in token):
            return True

        if "-" in token:
            parts = [part for part in token.split("-") if part]
            return bool(parts) and all(part.isdigit() for part in parts)

        return False

    def extract_article_id(self, text: str) -> str | None:
        tokens = _simplify_for_match(text).split()
        for idx, token in enumerate(tokens[:-1]):
            if token != "article":
                continue
            candidate = tokens[idx + 1]
            if self._looks_like_article_id(candidate):
                return candidate
        return None

    def extract_act_type(self, text: str) -> str | None:
        normalized = _simplify_for_match(text)
        if not normalized:
            return None
        first_token = normalized.split(" ", 1)[0]
        if first_token in self._ACT_TYPES:
            return first_token
        return None


_DEFAULT_PAGE_PARSER = LegalPageParser()


def _normalize_page_text(text: str) -> str:
    return _DEFAULT_PAGE_PARSER.normalize_page_text(text)


def _is_boilerplate_line(line: str) -> bool:
    return _DEFAULT_PAGE_PARSER.is_boilerplate_line(line)


def _is_toc_page(lines: list[str]) -> bool:
    return _DEFAULT_PAGE_PARSER.is_toc_page(lines)


def _strip_boilerplate_lines(lines: list[str]) -> list[str]:
    return _DEFAULT_PAGE_PARSER.strip_boilerplate_lines(lines)


def _split_legal_elements(page_text: str) -> list[str]:
    return _DEFAULT_PAGE_PARSER.split_legal_elements(page_text)


def _detect_segment_type(text: str) -> str:
    return _DEFAULT_PAGE_PARSER.detect_segment_type(text)


def _extract_article_id(text: str) -> str | None:
    return _DEFAULT_PAGE_PARSER.extract_article_id(text)


def _extract_act_type(text: str) -> str | None:
    return _DEFAULT_PAGE_PARSER.extract_act_type(text)


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
        self._page_parser = LegalPageParser()

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
        page_texts = extract_pdf_page_texts(
            pdf_path=pdf_path,
            backend=self._settings.pdf_parser_backend,
            logger=self._logger,
        )
        sidecar_metadata = self._load_sidecar_metadata(pdf_path)
        base_metadata = self._sanitize_metadata(
            {
                **sidecar_metadata,
                "file_path": str(pdf_path),
                "source_file": str(pdf_path),
                "source_filename": pdf_path.name,
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
            parsed_page = self._page_parser.parse_page(raw_text)
            if not parsed_page.segments and not parsed_page.skipped_toc and not parsed_page.skipped_boilerplate:
                pages_without_extractable_text += 1
                continue

            if parsed_page.skipped_toc:
                pages_skipped_toc += 1
                continue

            if parsed_page.skipped_boilerplate:
                pages_skipped_boilerplate += 1
                continue

            page_had_records = False
            for segment_idx, segment in enumerate(parsed_page.segments, start=1):
                chunks = _chunk_text(
                    text=segment.text,
                    chunk_size=self._settings.default_chunk_size,
                    chunk_overlap=self._settings.default_chunk_overlap,
                )
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
                                    "segment_type": segment.segment_type,
                                    "article_id": segment.article_id,
                                    "act_type": segment.act_type,
                                    "chunk_index": chunk_idx,
                                    "chunk_char_count": len(chunk),
                                }
                            ),
                        )
                    )
                    page_had_records = True
                if chunks:
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
