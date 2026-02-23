from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_EXPECTED_MODEL_REPO = "ibm-granite/granite-docling-258M"


def extract_pdf_page_texts(pdf_path: Path, logger: logging.Logger) -> list[str]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Docling is required for PDF parsing. Install project dependencies with `uv sync`."
        ) from exc

    options = VlmPipelineOptions()
    model_spec = getattr(getattr(options, "vlm_options", None), "model_spec", None)
    default_repo_id = getattr(model_spec, "default_repo_id", None)
    if default_repo_id != _EXPECTED_MODEL_REPO:
        raise RuntimeError(
            f"Unexpected Docling model preset: {default_repo_id!r}. Expected {_EXPECTED_MODEL_REPO!r}."
        )

    # This project only extracts text; table/layout-heavy processing is intentionally disabled.
    if hasattr(options, "do_table_structure"):
        setattr(options, "do_table_structure", False)
    if hasattr(options, "do_cell_matching"):
        setattr(options, "do_cell_matching", False)
    if hasattr(options, "do_picture_classification"):
        setattr(options, "do_picture_classification", False)
    if hasattr(options, "do_picture_description"):
        setattr(options, "do_picture_description", False)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=options,
            )
        }
    )
    try:
        result = converter.convert(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "huggingface.co" in message or "snapshot folder" in message or "LocalEntryNotFoundError" in message:
            raise RuntimeError(
                "Docling Granite model is not available locally. "
                "Connect to the internet once to download "
                "'ibm-granite/granite-docling-258M', then retry ingestion."
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
