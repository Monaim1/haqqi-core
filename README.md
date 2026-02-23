# Haqqi Core

Standalone open-source pipeline for legal-document retrieval:
- FastAPI API
- Qdrant vector store
- FastEmbed dense + sparse retrieval
- Optional ColBERT late-interaction reranking

## Layout

- `src/config.py`: settings
- `src/schemas.py`: API schemas
- `src/main.py`: FastAPI app entrypoint
- `src/api/routes.py`: `/query` and `/ingest` routes
- `src/services/semantic_search/*`: ingestion + retrieval services
- `src/cli.py`: ingestion CLI
- `src/hf_export.py`: Hugging Face metadata/export helper

## Setup

```bash
cd haqqi-core
uv sync
cp .env.example .env
```

## Run API

```bash
cd haqqi-core
uv run uvicorn src.main:app --reload
```

API endpoints:
- `POST /api/v1/query`
- `POST /api/v1/ingest`
- `GET /api/v1/health`

## Ingestion CLI

```bash
cd haqqi-core
uv run python -m src.cli ingest --limit 5
```

## Hugging Face Metadata Export

Generate upload metadata for PDF + sidecar files:

```bash
cd haqqi-core
uv run python -m src.hf_export \
  --source-dir data/raw-bulletin-officiel \
  --output-dir ./hf_export \
  --repo-data-dir data/raw-bulletin-officiel \
  --repo-id your-org/your-dataset
```

This writes:
- `hf_export/metadata.csv`
- `hf_export/metadata.jsonl`
- `hf_export/upload_manifest.json`
- `hf_export/README.md`

## Local Dataset Dry Run

By default, ingestion reads from `data/raw-bulletin-officiel`:

```bash
cd haqqi-core
uv run python -m src.cli ingest --limit 1
```
To override source data at runtime, set `HAQQI_CORE_SOURCE_DIR=/path/to/your/folder`.
