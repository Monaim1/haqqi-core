# Open Source Pipeline Extraction Plan

This plan outlines the steps to extract the core ingestion, retrieval, and API components of the Haqqi backend into a standalone, open-source-ready project folder. We will strip away any authentication, database session management, or front-end specific logic.

## Goal Description
Create a clean, well-structured Python project containing only the RAG pipeline (FastAPI + Qdrant + FastEmbed + ColBERT reranking) so it can be published as an open-source template or standalone microservice.

## Proposed Changes

the directory is at `haqqi-core` with the following structure:

### Project Configuration
*   **[NEW] `haqqi-core/pyproject.toml`**: A standard modern Python package configuration using `uv` or `pip`, declaring dependencies like `fastapi`, `qdrant-client`, `fastembed`, `pydantic`, `pydantic-settings`, and `uvicorn`.
*   **[NEW] `haqqi-core/README.md`**: Guide for users on how to run ingestion and start the API.
*   **[NEW] `haqqi-core/.env.example`**: Example environment variables for Qdrant and fastembed configurations.

### Application Logic (in `src/`)
*   **[NEW] `src/config.py`**: Extracted from [backend/src/app/config.py](file:///Users/mounselam/Developer/Haqqi/backend/src/app/config.py), keeping only the retrieval and Qdrant settings. (Removing Clerk, SQLite, CORS origins meant for specific frontends).
*   **[NEW] `src/schemas.py`**: Extracted from [backend/src/app/schemas.py](file:///Users/mounselam/Developer/Haqqi/backend/src/app/schemas.py), keeping only [QueryRequest](file:///Users/mounselam/Developer/Haqqi/backend/src/app/schemas.py#8-11), [QueryHit](file:///Users/mounselam/Developer/Haqqi/backend/src/app/schemas.py#13-18), and [QueryResponse](file:///Users/mounselam/Developer/Haqqi/backend/src/app/schemas.py#20-23).
*   **[NEW] `src/main.py`**: A clean FastAPI application initialization that mounts the `api/routes.py`.

### API & Services
*   **[NEW] `src/api/routes.py`**: Extracted from [routers/query.py](file:///Users/mounselam/Developer/Haqqi/backend/src/app/routers/query.py), providing the `/query` endpoint, and adding a simple `/ingest` endpoint or keeping it as a CLI command.
*   **[NEW] `src/services/`**: We will copy the entire [semantic_search](file:///Users/mounselam/Developer/Haqqi/backend/src/app/services/semantic_search/__init__.py#31-34) directory from `backend/src/app/services/semantic_search/`. This includes:
    *   `__init__.py`
    *   `qdrant_service.py`
    *   `ingestion.py`
    *   `retrieval.py`
    *   `rerankers.py`

*(Note: If `backend/src/app/services/semantic_search` references `app.config` or `app.schemas`, these imports will be updated to point to the new `src.config` structure.)*

## Verification Plan

1.  **Structure Verification**: Ensure all files are copied and stripped of Haqqi-specific proprietary models (like User schemas, DB setup).
2.  **Linting / Type Checking**: Run `uv run ruff check` or `mypy` on the new folder to ensure no broken imports.
3.  **Run Server**: Start the new open-source pipeline using `uvicorn src.main:app --reload` to verify the FastAPI server boots successfully without auth or database dependency errors.
4.  **CLI / Ingestion Test**: Verify the ingestion script can be invoked cleanly from the new structure.
