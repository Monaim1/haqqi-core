# Haqqi Core

Retrieval backend for a legal AI assistant:
- FastAPI API
- Qdrant vector store (local path or remote URL)
- FastEmbed dense retrieval + sparse vectors
- Optional ColBERT late-interaction reranking
- Docling PDF parsing

This repo contains the retrieval pipeline (ingest + search) and a small LangGraph ReAct agent that uses retrieval tools for grounded answers.

## Quickstart (10 minutes)

### 1) Install

Requirements:
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

```bash
git clone <your-repo-url>
cd haqqi-core
uv sync
cp .env.example .env
```

If you plan to use the agent, set your Gemini key in `.env`:

```bash
GOOGLE_API_KEY=your_key_here
```

### 2) Fetch PDFs from `data/index.json` (recommended)

`data/index.json` contains source metadata and `pdf_url` links.  
Use the built-in fetcher to download files and generate sidecar metadata:

```bash
# Preview what would be downloaded
uv run haqqi fetch --limit 3 --language fr --dry-run

# Download recent FR files (example)
uv run haqqi fetch --limit 20 --language fr --since 2024-01-01
```

Files are written to `data/raw-bulletin-officiel` by default as:
- `*.pdf`
- matching `*.json` sidecars with:
  `issue_number`, `publication_date`, `language`, `pdf_url`, `checksum`, `file_path`

Optional: write an updated state file after downloads:

```bash
uv run haqqi fetch --limit 20 --state-out data/index.state.json
```

If you already have PDFs, copy them into `data/raw-bulletin-officiel` (or set `HAQQI_CORE_SOURCE_DIR`).

### 3) Build the vector database

Dense + ColBERT:

```bash
uv run haqqi ingest --limit 20
```

Dense only (faster smoke test):

```bash
uv run haqqi ingest --limit 3 --no-colbert
```

### 4) Retrieve

```bash
uv run haqqi query "loi sur la gouvernance des établissements publics" --top-k 5
```

For better relevance, use sentence-like queries (not one-word queries like `loi`).

### 5) Run the API

```bash
uv run haqqi api
```

Health:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Ingest:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"limit": 1, "include_colbert": true}'
```

Query:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"loi sur les établissements publics","top_k":5}'
```

## CLI Commands

```bash
uv run haqqi --help
```

Main commands:
- `haqqi fetch`: Download PDFs from `index.json`
- `haqqi ingest`: Parse PDFs and index vectors in Qdrant
- `haqqi query` / `haqqi search`: Run retrieval
- `haqqi agent`: Run LangGraph ReAct agent over retrieval tools
- `haqqi api`: Start FastAPI server
- `haqqi export`: Generate Hugging Face metadata bundle

## Configuration

Main env vars:
- `HAQQI_CORE_SOURCE_DIR` default: `data/raw-bulletin-officiel`
- `HAQQI_CORE_SOURCE_GLOB_PATTERN` default: `**/*.pdf`
- `HAQQI_CORE_QDRANT_PATH` default: `data/qdrant`
- `HAQQI_CORE_QDRANT_URL` default: empty (when set, remote Qdrant is used)
- `HAQQI_CORE_QDRANT_COLLECTION_NAME` default: `moroccan_law`
- `HAQQI_CORE_RETRIEVAL_RERANKER_MODE` default: `late_interaction` (`none` or `late_interaction`)
- `GOOGLE_API_KEY` required for `haqqi agent`
- `HAQQI_CORE_AGENT_MODEL_NAME` default: `gemini-2.0-flash`
- `HAQQI_CORE_AGENT_RECURSION_LIMIT` default: `25`

Runtime override example:

```bash
HAQQI_CORE_SOURCE_DIR=/path/to/your/pdfs uv run haqqi ingest --limit 10
```

Notes:
- Docling models are downloaded on first use.
- If the machine is offline and models are not cached yet, ingestion will fail.

## Common Pitfalls

- `source_dir does not exist` on ingest:
  - Run `uv run haqqi fetch --limit 5` or create/populate `data/raw-bulletin-officiel`.
- Query returns empty hits:
  - Ingestion has not run yet or collection is empty.
- ColBERT fallback warning:
  - ColBERT collection has not been ingested yet; run ingest without `--no-colbert`.
- Broad queries are noisy:
  - Prefer specific queries like `loi sur la fiscalité des entreprises publiques`.

## Open-Source Smoke Test

Before publishing, these commands should all work:

```bash
uv run haqqi --help
uv run haqqi fetch --limit 2 --dry-run
uv run haqqi ingest --limit 1 --no-colbert
uv run haqqi query "loi sur la fiscalité" --top-k 3
# requires GOOGLE_API_KEY
uv run haqqi agent --question "Résume le sujet du document" --thread-id smoke --recursion-limit 10
```

## LangGraph Agent (Grounded ReAct)

The repository now includes a simple **LangGraph ReAct agent** that can:
- retrieve grounded snippets from the indexed legal corpus,
- trigger ingestion when needed,
- keep thread-level state across turns.
- recurse through tool calls up to a configurable recursion limit.

Set your Gemini API key:

```bash
export GOOGLE_API_KEY="your_key"
```

Single-turn:

```bash
uv run haqqi agent --question "Quelle est la règle sur ... ?" --thread-id demo --recursion-limit 25
```

Interactive:

```bash
uv run haqqi agent --thread-id demo
```

Notes:
- `--thread-id` keeps memory/state for that session.
- `--recursion-limit` maps to LangGraph recursion limit (tool-call loop bound).
- Agent tools available: `retrieve_context`, `run_ingestion`, `get_session_state`, `clear_session_state`.
- System prompt is separated in `src/agent/prompts.py`.
- A placeholder HITL middleware is wired in `src/agent/middleware.py` (currently allow-all + tool-call counting).
- The agent is instructed to cite retrieved evidence (`[1]`, `[2]`, ...).

## Project Layout

- `src/config.py`: settings
- `src/schemas.py`: API schemas
- `src/main.py`: FastAPI app
- `src/api/routes.py`: `/query`, `/ingest`, `/health`
- `src/services/semantic_search/*`: ingestion + retrieval services
- `src/agent/react_agent.py`: LangGraph ReAct agent + stateful tools
- `src/agent/prompts.py`: system prompt definition
- `src/agent/middleware.py`: placeholder HITL middleware
- `src/cli.py`: CLI entrypoint
- `src/index_fetch.py`: `index.json` downloader
- `src/hf_export.py`: Hugging Face export helper

## Hugging Face Metadata Export

```bash
uv run haqqi export \
  --source-dir data/raw-bulletin-officiel \
  --output-dir ./hf_export \
  --repo-data-dir data/raw-bulletin-officiel \
  --repo-id your-org/your-dataset
```

Writes:
- `hf_export/metadata.csv`
- `hf_export/metadata.jsonl`
- `hf_export/upload_manifest.json`
- `hf_export/README.md`
