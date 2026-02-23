from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

def extract_pdf_page_texts(pdf_path: Path, logger: logging.Logger) -> list[str]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Docling is required for PDF parsing. Install project dependencies with `uv sync`."
        ) from exc

    options = PdfPipelineOptions(
        do_table_structure=False,
        do_picture_classification=False,
        do_picture_description=False,
        do_chart_extraction=False,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_ocr=False,
        force_backend_text=True,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
        generate_parsed_pages=False,
    )

    options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.AUTO,
        num_threads=max(1, os.cpu_count() or 1),
    )

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    try:
        result = converter.convert(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "huggingface.co" in message or "snapshot folder" in message or "LocalEntryNotFoundError" in message:
            raise RuntimeError(
                "Docling models are not available locally. "
                "Connect to the internet once to download Docling layout/OCR models, then retry ingestion."
            ) from exc
        raise
    document = getattr(result, "document", result)

    page_texts = _extract_page_texts(document)
    if not page_texts:
        logger.warning("Docling returned no page content for %s", pdf_path.name)
        return [""]
    return page_texts


def _extract_page_texts(document: Any) -> list[str]:
    pages = getattr(document, "pages", None)
    if pages is None:
        full_text = _export_text(document).strip()
        return [full_text] if full_text else []

    if isinstance(pages, dict):
        ordered_pages = [pages[key] for key in sorted(pages.keys())]
    elif isinstance(pages, list):
        ordered_pages = pages
    else:
        full_text = _export_text(document).strip()
        return [full_text] if full_text else []

    texts: list[str] = []
    for page in ordered_pages:
        text = _export_text(page).strip()
        texts.append(text)
    return texts


def _export_text(node: Any) -> str:
    for method_name in ("export_to_text", "to_text", "export_to_markdown"):
        method = getattr(node, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, str):
                return value
    for attr_name in ("text", "markdown"):
        value = getattr(node, attr_name, None)
        if isinstance(value, str):
            return value
    return str(node) if node is not None else ""
