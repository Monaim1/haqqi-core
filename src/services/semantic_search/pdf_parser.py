from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pypdf import PdfReader

_KNOWN_BACKENDS = {"auto", "pypdf", "docling"}


def extract_pdf_page_texts(pdf_path: Path, backend: str, logger: logging.Logger) -> list[str]:
    backend_name = backend.strip().lower()
    if backend_name not in _KNOWN_BACKENDS:
        raise ValueError(f"Unknown parser backend: {backend}")

    if backend_name == "pypdf":
        return _extract_with_pypdf(pdf_path)

    if backend_name == "docling":
        return _extract_with_docling(pdf_path)

    try:
        return _extract_with_docling(pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Docling parser unavailable for %s. Falling back to pypdf: %s", pdf_path.name, exc)
        return _extract_with_pypdf(pdf_path)


def _extract_with_pypdf(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def _extract_with_docling(pdf_path: Path) -> list[str]:
    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Docling is not installed. Install it and set HAQQI_CORE_PDF_PARSER_BACKEND=docling.") from exc

    converter = DocumentConverter()
    conversion = converter.convert(str(pdf_path))
    document = getattr(conversion, "document", conversion)

    page_texts = _try_docling_pages(document)
    if page_texts:
        return page_texts

    full_text = _try_docling_full_text(document)
    if full_text.strip():
        return [full_text]

    return [""]


def _try_docling_pages(document: Any) -> list[str]:
    pages = getattr(document, "pages", None)
    if pages is None:
        return []

    if isinstance(pages, dict):
        values = [pages[key] for key in sorted(pages.keys())]
    elif isinstance(pages, list):
        values = pages
    else:
        return []

    page_texts: list[str] = []
    for page in values:
        page_text = _extract_page_text(page)
        page_texts.append(page_text)
    return page_texts


def _extract_page_text(page: Any) -> str:
    for attr in ("text", "raw_text", "markdown"):
        value = getattr(page, attr, None)
        if isinstance(value, str):
            return value
    for method_name in ("export_to_text", "to_text", "export_to_markdown"):
        method = getattr(page, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, str):
                return value
    return str(page) if page is not None else ""


def _try_docling_full_text(document: Any) -> str:
    for method_name in ("export_to_text", "to_text", "export_to_markdown"):
        method = getattr(document, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, str):
                return value
    for attr in ("text", "markdown"):
        value = getattr(document, attr, None)
        if isinstance(value, str):
            return value
    return ""
