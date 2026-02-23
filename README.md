# Haqqi Core

Standalone open-source pipeline for legal-document retrieval:
- FastAPI API
- Qdrant vector store
- FastEmbed dense + sparse retrieval
- Optional ColBERT late-interaction reranking
- Docling PDF parsing with Granite Docling 258M (single parser path)

## Quickstart (5 minutes)

### 1) Install and configure

```bash
git clone <your-repo-url>
cd haqqi-core
uv sync
cp .env.example .env
```

### 2) Put your source documents in the data folder

Default source path:

`data/raw-bulletin-officiel`

Expected files:
- `*.pdf`
- optional sidecars `*.json` with metadata (`issue_number`, `publication_date`, `language`, etc.)

### 3) Run ingestion

Dense + ColBERT:

```bash
uv run haqqi ingest --limit 5
```

Dense only (faster smoke test):

```bash
uv run haqqi ingest --limit 1 --no-colbert
```

### 4) Run a query

```bash
uv run haqqi query "loi sur la gouvernance des établissements publics" --top-k 5
```

For better results, prefer sentence-style queries over one-word queries.

### 5) Run the API

```bash
uv run haqqi api
```

Health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Ingest via API:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"limit": 1, "include_colbert": true}'
```

Query via API:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"loi sur les établissements publics","top_k":5}'
```

## Configuration

Main env vars:
- `HAQQI_CORE_SOURCE_DIR` default: `data/raw-bulletin-officiel`
- `HAQQI_CORE_SOURCE_GLOB_PATTERN` default: `**/*.pdf`
- `HAQQI_CORE_QDRANT_PATH` default: `data/qdrant`
- `HAQQI_CORE_QDRANT_COLLECTION_NAME` default: `moroccan_law`
- `HAQQI_CORE_RETRIEVAL_RERANKER_MODE` default: `late_interaction` (`none` or `late_interaction`)

Override source directory at runtime:

```bash
HAQQI_CORE_SOURCE_DIR=/path/to/your/pdfs uv run haqqi ingest --limit 10
```

PDF parsing is fixed to Docling + `ibm-granite/granite-docling-258M`.
No table extraction pipeline is enabled in this project.
The first ingestion run downloads Granite model artifacts from Hugging Face.

## Common Pitfalls

- Query returns short chunks (`"* * *"`, headers, very short fragments):
  - Cause: OCR/segmentation noise + very broad query.
  - Fix: re-ingest after parser/chunking improvements and use richer queries (4-10 words).
- Query has low relevance with one-word input like `"loi"`:
  - Cause: too broad semantically; many legal headings match.
  - Fix: include intent and scope, e.g. `"loi sur la fiscalité des entreprises publiques"`.
- ColBERT fallback warning:
  - Cause: ColBERT collection not ingested yet.
  - Fix: run ingestion without `--no-colbert`.
- Docling Granite download error:
  - Cause: model artifacts were not downloaded yet and the machine is offline.
  - Fix: run ingestion once with internet access to cache `ibm-granite/granite-docling-258M`.

## Layout

- `src/config.py`: settings
- `src/schemas.py`: API schemas
- `src/main.py`: FastAPI app entrypoint
- `src/api/routes.py`: `/query` and `/ingest` routes
- `src/services/semantic_search/*`: ingestion + retrieval services
- `src/cli.py`: CLI (`haqqi ingest|query|api|export`)
- `src/hf_export.py`: Hugging Face metadata/export helper

API endpoints:
- `POST /api/v1/query`
- `POST /api/v1/ingest`
- `GET /api/v1/health`

## Hugging Face Metadata Export

Generate upload metadata for PDF + sidecar files:

```bash
uv run haqqi export \
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
