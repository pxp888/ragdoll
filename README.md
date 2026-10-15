# ragdoll - RAG Server with Multi-Scale Indexing & Dual Retrieval

**ragdoll** is a high-performance RAG (Retrieval-Augmented Generation) server combining **FastAPI**, **SQLite** with the **`sqlite-vec`** extension, **`embeddinggemma2`** for dense vector embeddings, a **Kev 4B** decision model for intelligent semantic boundary detection and LLM-based re-ranking, and a high-speed **standard cross-encoder re-ranker** (FlashRank).

The embedding server and decision models are both external servers, so they can be swapped out for better alternatives as they develop.  

The idea behind this is that a decision model can make better decisions regarding where to separate chunks for embedding than other naive strategies.  This is a trade off because ingesting data takes longer with a decision model in the loop, but the quality of separation is better, therefore the search results are better.  

There are also three possible retrieval methods here.  One uses no re-ranking, don't do this.  One uses standard re-ranking which is fast, and one uses the decision model which gives better results but is rather slow.  


[initial test results](tests.md)

---

## Features

- **Strict Dependencies & Fail-Fast Policy**:
  - Verifies connectivity to Kev 4B and EmbeddingGemma2 during startup.
  - Runtime errors from dependencies immediately abort processing and fail fast (no silent degradation).
- **Two-Stage Multi-Scale Indexing**:
  - **Immediate Ingestion**: Saves files and returns `202 Accepted` immediately.
  - **Stage 1 (Coarse Indexing)**: Splits at structural boundaries (~2,000–4,000 chars), embeds with `embeddinggemma2`, and marks asset as `coarse_indexed` (searchable immediately).
  - **Stage 2 (Parallel Kev Fine Indexing)**: Shards across coarse chunks concurrently (`asyncio.TaskGroup` + `asyncio.Semaphore`), using Kev 4B (`/v1/systemone`) to detect semantic topic transitions (~300–1,500 chars), embeds fine chunks in batches, and indexes with parent-child linkage.
- **Unified SQLite Vector Store**:
  - Relational metadata and cosine embeddings unified in SQLite via `sqlite-vec`.
- **Dual Retrieval & Re-ranking Endpoints**:
  - **Kev-Based Retrieval (`POST /retrieve/kev`)**: Semantic relevance re-ranking powered by Kev 4B System One decisions.
  - **Standard Re-ranking Retrieval (`POST /retrieve/standard`)**: Ultra-fast (~5–15ms) neural cross-encoder re-ranking (FlashRank ONNX) running locally without network roundtrips or GPU requirements.
  - **Unified Search (`POST /search`)**: Vector similarity search with configurable optional re-ranking (`"kev"` or `"standard"`).
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

### 3. Kev-Based Retrieval
- **Endpoint**: `POST /retrieve/kev` (alias: `POST /search/kev`)
- **Description**: Retrieves candidate chunks via vector search and scores them using Kev 4B System One questions.
- **Payload**:
  ```json
  {
    "query": "operating profit breakdown Q3",
    "top_k": 5,
    "stage": "fine",
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
        "score": 1.95,
        "text": "Operating profit in Q3 rose by 14%...",
        "parent_context": "Full section text from coarse chunk...",
        "created_at": "2026-10-14T16:00:00Z"
      }
    ],
    "rerank_method": "kev"
  }
  ```

### 4. Standard Re-ranking Retrieval (Without Kev)
- **Endpoint**: `POST /retrieve/standard` (alias: `POST /search/standard`)
- **Description**: Retrieves candidate chunks via vector search and re-ranks them using a lightweight local neural cross-encoder (FlashRank ONNX), executing in ~5–15ms.
- **Payload**:
  ```json
  {
    "query": "operating profit breakdown Q3",
    "top_k": 5,
    "stage": "fine",
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
        "score": 0.9821,
        "text": "Operating profit in Q3 rose by 14%...",
        "parent_context": "Full section text from coarse chunk...",
        "created_at": "2026-10-14T16:00:00Z"
      }
    ],
    "rerank_method": "standard"
  }
  ```

### 5. Unified Search
- **Endpoint**: `POST /search`
- **Description**: Vector search with optional re-ranking (`rerank: true/false`, `rerank_method: "kev" | "standard"`).
- **Payload**:
  ```json
  {
    "query": "operating profit breakdown Q3",
    "top_k": 5,
    "stage": "fine",
    "rerank": false,
    "include_parent_context": true
  }
  ```

### 6. Download Raw Asset
- **Endpoint**: `GET /get/{asset_id}` (alias: `GET /assets/{asset_id}`)
- **Response**: Binary file stream with `Content-Disposition`.

### 7. Delete a Specific Asset
- **Endpoint**: `DELETE /assets/{asset_id}` (alias: `DELETE /delete/{asset_id}`)
- **Query param** (optional): `?target_apikey=<owner>` — defaults to the caller's key; pass another key to delete its assets.
- **Description**: Deletes an asset, coarse/fine chunks, vector embeddings, and disk file.
- **Response**: `200 OK`
  ```json
  {
    "deleted": true,
    "asset_id": "8b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "message": "Asset 8b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d deleted successfully"
  }
  ```

### 8. Delete All Assets Under an API Key
- **Endpoint**: `DELETE /assets/by-key/{apikey}` (alias: `DELETE /assets?apikey={apikey}`)
- **Description**: Deletes all assets, chunks, embeddings, and disk files belonging to a specified API key.
- **Response**: `200 OK`
  ```json
  {
    "deleted_count": 2,
    "apikey": "test-user-key",
    "asset_ids": [
      "8b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "7a1deb4d-2a7d-4bad-9bdd-2b0d7b3dcb6c"
    ],
    "message": "Deleted 2 asset(s) under apikey 'test-user-key'"
  }
  ```

### 9. Health Check
- **Endpoint**: `GET /health`
- **Response**: `200 OK` (or `503 Service Unavailable` if external dependencies are down)
  ```json
  {
    "status": "healthy",
    "embeddinggemma2": "healthy",
    "kev_4b": "healthy",
    "standard_reranker": "healthy"
  }
  ```

---

## Running Tests

Run the full automated test suite:
```bash
pytest -v
```
