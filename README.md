# ragdoll - Kev-Based RAG Server

**ragdoll** is a high-performance RAG (Retrieval-Augmented Generation) server combining **FastAPI**, **SQLite** with the **`sqlite-vec`** extension, **`embeddinggemma2`** for dense vector embeddings, and a **Kev 4B** decision model for intelligent boundary detection and semantic re-ranking.

---

## Features

- **Strict Dependencies & Fail-Fast Policy**:
  - Verifies connectivity to Kev 4B and EmbeddingGemma2 during startup.
  - Runtime errors from dependencies immediately abort processing and fail fast (no silent degradation).
- **Two-Stage Multi-Scale Indexing**:
  - **Immediate Ingestion**: Saves files and returns `202 Accepted` immediately.
  - **Stage 1 (Coarse Indexing)**: Splits at structural boundaries (~2,000–4,000 chars), embeds with `embeddinggemma2`, and marks asset as `coarse_indexed` (searchable immediately).
  - **Stage 2 (Parallel Kev Fine Indexing)**: Shards across coarse chunks concurrently (`asyncio.TaskGroup` + `asyncio.Semaphore`), using Kev 4B (`/v1/systemone`) to detect semantic topic transitions (~300–600 chars), embeds fine chunks in batches, and indexes with parent-child linkage.
- **Unified SQLite Vector Store**:
  - Relational metadata and cosine embeddings unified in SQLite via `sqlite-vec`.
- **Search & Semantic Re-ranking**:
  - Hybrid vector search with parent coarse chunk context retrieval.
  - Optional semantic re-ranking powered by Kev 4B relevance scoring.
- **Health & Readiness Check**:
  - Active connectivity probe via `GET /health`.

---

## Installation & Setup

1. **Clone repository and setup virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

2. **Configure environment variables** (optional, defaults provided):
   ```bash
   cp .env.example .env
   ```

3. **Start the server**:
   ```bash
   # CLI command
   ragdoll

   # Or with uvicorn
   uvicorn ragdoll.main:app --host 0.0.0.0 --port 8000
   ```

---

## API Reference

### 1. Upload Asset
- **Endpoint**: `POST /store` (alias: `POST /api/v1/assets`)
- **Content-Type**: `multipart/form-data`
- **Response**: `202 Accepted`
  ```json
  {
    "asset_id": "8b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "filename": "annual_report.pdf",
    "size_bytes": 1048576,
    "status": "queued",
    "created_at": "2026-10-14T16:00:00Z"
  }
  ```

### 2. Check Indexing Status
- **Endpoint**: `GET /assets/{asset_id}/status`
- **Response**: `200 OK`
  ```json
  {
    "asset_id": "8b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "status": "fine_indexed",
    "coarse_chunks": 12,
    "fine_chunks": 48,
    "error": null
  }
  ```

### 3. Search Chunks
- **Endpoint**: `POST /search`
- **Payload**:
  ```json
  {
    "query": "operating profit breakdown Q3",
    "top_k": 5,
    "stage": "fine",
    "rerank": true,
    "include_parent_context": true
  }
  ```
- **Response**: `200 OK`
  ```json
  {
    "results": [
      {
        "chunk_id": 142,
        "asset_id": "8b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        "filename": "annual_report.pdf",
        "score": 1.826,
        "text": "Operating profit in Q3 rose by 14%...",
        "parent_context": "Full section text from coarse chunk...",
        "created_at": "2026-10-14T16:00:00Z"
      }
    ]
  }
  ```

### 4. Download Raw Asset
- **Endpoint**: `GET /get/{asset_id}` (alias: `GET /assets/{asset_id}`)
- **Response**: Binary file stream with `Content-Disposition`.

### 5. Health Check
- **Endpoint**: `GET /health`
- **Response**: `200 OK` (or `503 Service Unavailable` if external dependencies are down)
  ```json
  {
    "status": "healthy",
    "embeddinggemma2": "healthy",
    "kev_4b": "healthy"
  }
  ```

---

## Running Tests

Run the full automated test suite:
```bash
pytest -v
```
