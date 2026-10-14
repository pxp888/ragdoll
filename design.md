# ragdoll - Kev-Based RAG Server

A standalone, high-performance RAG (Retrieval-Augmented Generation) server combining **FastAPI**, **SQLite** with **`sqlite-vec`**, **`embeddinggemma2`** for dense embeddings, and a **Kev 4B** decision model for intelligent boundary detection and semantic re-ranking.

---

## 1. Architecture & Core Philosophy

### Strict Dependencies & Fail-Fast Policy
Both the embedding model and the Kev decision model are hard, non-negotiable requirements for this service:
- **No silent fallbacks**: If Kev or the embedding server is unavailable, ragdoll must **fail fast** rather than degrading silently into naive chunking.
- **Startup probe**: The server verifies connectivity to both external services on startup. If either dependency is unreachable, startup aborts immediately.
- **Runtime failures**: If an external dependency fails during an operation, the operation fails with a clear, descriptive HTTP 503 or sets the asset status to `failed`.

### Two-Stage Ingestion Pipeline
1. **Immediate Ingestion**: Assets are saved and assigned a unique ID. The upload request returns `202 Accepted` immediately so callers are not blocked.
2. **Stage 1 — Coarse Indexing (Fast Path)**:
   - The file is split into large structural chunks (~2,000–4,000 characters) snapped to major breaks (headings, paragraph breaks).
   - Embedded with `embeddinggemma2` and indexed in `sqlite-vec`.
   - Asset status transitions to `coarse_indexed`. The asset is now immediately searchable.
3. **Stage 2 — Parallel Fine Indexing (Kev Semantic Boundaries)**:
   - To avoid the $O(N)$ latency bottleneck of serial boundary scanning, fine chunking runs **concurrently across the document**.
   - Stage 1 coarse chunks serve as natural parallel sharding anchors. A worker pool processes multiple coarse chunks simultaneously using Kev 4B (`POST /v1/systemone`) to discover optimal internal semantic breaks.
   - Fine chunks are embedded in batches with `embeddinggemma2` and indexed in `sqlite-vec`.
   - Asset status transitions to `fine_indexed` (ready).
   - *Multi-scale index*: Both coarse and fine chunks are retained in a parent-child relationship. Fine chunks provide high-precision vector matches; parent coarse chunks provide broader surrounding context when retrieving context for downstream LLMs.

---

## 2. Infrastructure Dependencies

| Service | Address | Protocol / Port | Auth | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Kev 4B** | `http://192.168.0.12:8008` | HTTP (`/v1/systemone`) | Bearer `hellothere` | Boundary decisions & query re-ranking |
| **EmbeddingGemma2** | `http://192.168.0.12:11434` | HTTP (`/api/embed`) | None | 768-dim vector embeddings |
| **ragdoll Server** | Local / Target Host | HTTP (`:8000`) | Configurable | FastAPI, SQLite (`sqlite-vec`), File Storage |

---

## 3. Database & Storage Schema

SQLite is used with the `sqlite-vec` extension for unified relational metadata and vector storage.

```
assets
  ├── id (UUID, PK)
  ├── filename (TEXT)
  ├── content_type (TEXT)
  ├── size_bytes (INTEGER)
  ├── sha256 (TEXT, UNIQUE)
  ├── status (TEXT)  -- "queued", "coarse_indexed", "fine_indexed", "failed"
  ├── error_message (TEXT, NULLABLE)
  ├── storage_path (TEXT)
  ├── created_at (TIMESTAMP)
  └── updated_at (TIMESTAMP)

chunks
  ├── id (INTEGER, PK)
  ├── asset_id (UUID, FK -> assets.id ON DELETE CASCADE)
  ├── stage (TEXT)  -- "coarse" | "fine"
  ├── parent_chunk_id (INTEGER, NULLABLE, FK -> chunks.id)
  ├── chunk_index (INTEGER)
  ├── text (TEXT)
  ├── byte_offset_start (INTEGER)
  └── byte_offset_end (INTEGER)

vec_chunks (sqlite-vec virtual table)
  ├── chunk_id (INTEGER PRIMARY KEY)
  └── embedding (FLOAT[768] distance_metric=cosine)
```

---

## 4. Kev Protocol Usage & Parallel Execution

### A. Semantic Boundary Detection Protocol
When evaluating a candidate boundary between adjacent text blocks $A$ and $B$, ragdoll queries Kev at `POST /v1/systemone`:

* **Payload**:
  ```json
  {
    "state": "Preceding text:\n<block_a>\n\nCandidate continuation:\n<block_b>",
    "questions": {
      "boundary": {
        "type": "choice",
        "instructions": "Does the subject matter shift significantly enough between these blocks to warrant a separate semantic chunk?",
        "criteria": {
          "continue": "The text continues the same topic or contextual narrative.",
          "split": "The text begins a distinct topic, section, or logical transition."
        }
      }
    }
  }
  ```
* If `boundary.choice == "split"` with high probability, a new fine-chunk boundary is created.

### B. Parallel Fine-Chunking Pipeline (Anchor Sharding)
To maximize GPU throughput on the Kev 4B server and eliminate serial network roundtrip latency:

1. **Coarse Chunks as Natural Anchors**:
   - Stage 1 has already partitioned the asset at strong structural boundaries (e.g. double newlines, headers, section breaks).
   - Each coarse chunk acts as an independent shard root.
2. **Concurrent Worker Pool**:
   - Fine chunking is parallelized across coarse chunks using `asyncio.gather` governed by an `asyncio.Semaphore` (concurrency limit = 4–8 workers).
   - Worker $i$ scans coarse chunk $i$ sequentially, dividing it into fine chunks (~300–600 characters) via Kev boundary decisions.
   - All fine chunks produced by Worker $i$ retain `parent_chunk_id = coarse_chunk_i.id`.
3. **Fail-Fast Error Propagation**:
   - If any worker encounters an HTTP or timeout error communicating with Kev, the entire gathering task cancels immediately, aborts remaining workers, and marks the asset as `failed` with the error logged in `assets.error_message`.
4. **Batch Embedding**:
   - Once all fine chunks are derived, their texts are batched (e.g. 16–32 chunks per call) and submitted to `embeddinggemma2` at `/api/embed`, minimizing HTTP request overhead.

### C. Optional Semantic Re-ranking (Search Stage)
For high-precision queries, candidate vector hits are re-evaluated against the query:

* **Payload**:
  ```json
  {
    "state": "Query: <user_query>\nPassage: <chunk_text>",
    "questions": {
      "relevance": {
        "type": "score",
        "instructions": "How relevant is this passage to the user query?",
        "criteria": [
          "irrelevant",
          "tangentially related",
          "directly answers or addresses query"
        ]
      }
    }
  }
  ```

---

## 5. API Specification

### `POST /store` (or `/api/v1/assets`)
Upload an asset. Saves file to disk, enqueues background processing, and returns immediately.

* **Request**: `multipart/form-data` with `file=@<filename>`
* **Response** (`202 Accepted`):
  ```json
  {
    "asset_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "filename": "annual_report.pdf",
    "size_bytes": 1048576,
    "status": "queued",
    "created_at": "2026-10-14T16:00:00Z"
  }
  ```

### `GET /assets/{asset_id}/status`
Query the indexing status of an asset.

* **Response** (`200 OK`):
  ```json
  {
    "asset_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "status": "fine_indexed",
    "coarse_chunks": 12,
    "fine_chunks": 48,
    "error": null
  }
  ```

### `POST /search`
Perform hybrid / vector search across indexed chunks.

* **Request**:
  ```json
  {
    "query": "operating profit breakdown Q3",
    "top_k": 5,
    "stage": "fine",
    "rerank": false,
    "include_parent_context": true
  }
  ```
* **Response** (`200 OK`):
  ```json
  {
    "results": [
      {
        "chunk_id": 142,
        "asset_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        "filename": "annual_report.pdf",
        "score": 0.892,
        "text": "Operating profit in Q3 rose by 14%...",
        "parent_context": "Full section text from coarse chunk...",
        "created_at": "2026-10-14T16:00:00Z"
      }
    ]
  }
  ```

### `GET /get/{asset_id}` (or `/assets/{asset_id}`)
Retrieve the original raw stored asset file.

* **Response**: Binary stream with appropriate `Content-Type` and `Content-Disposition`.

### `DELETE /assets/{asset_id}` (or `/delete/{asset_id}`)
Delete a specific asset by ID, along with its coarse/fine chunks, sqlite-vec embeddings, and file on disk. Scoped to the caller's authenticated API key by default.

* **Response** (`200 OK`):
  ```json
  {
    "deleted": true,
    "asset_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "message": "Asset 9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d deleted successfully"
  }
  ```

### `DELETE /assets/by-key/{apikey}` (or `DELETE /assets?apikey={apikey}`)
Delete all assets, chunks, vector embeddings, and storage files associated with a specified API key.

* **Response** (`200 OK`):
  ```json
  {
    "deleted_count": 2,
    "apikey": "user_api_key",
    "asset_ids": [
      "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "8c2deb4d-2a7d-4bad-9bdd-2b0d7b3dcb6e"
    ],
    "message": "Deleted 2 asset(s) under apikey 'user_api_key'"
  }
  ```

### `GET /health`
Liveness and readiness check. Actively tests connectivity to both `embeddinggemma2` and `kev 4B`. Returns `503 Service Unavailable` if either dependency is down.

---

## 6. Execution Flow & Parallel Worker Diagram

```
Client Upload
     │
     ▼
[ POST /store ] ──────────► Save file to disk
     │                     Insert asset (status='queued')
     │                     Return 202 Accepted
     ▼
[ Background Worker ]
     │
     ├─► Check Embedding Server (Fail fast -> status='failed' if down)
     │
     ├─► Stage 1: Coarse Split (2k-4k chars, structural anchors)
     │   └─► Generate embeddings via embeddinggemma2
     │   └─► Insert into vec_chunks (stage='coarse')
     │   └─► Update asset status='coarse_indexed'
     │
     ├─► Check Kev Server (Fail fast -> status='failed' if down)
     │
     └─► Stage 2: Parallel Kev Fine-Chunking
         │
         ├── Coarse Chunk 1 ──► [Worker 1 (Kev)] ──┐
         ├── Coarse Chunk 2 ──► [Worker 2 (Kev)] ──┤ Fan-out via asyncio.gather
         ├── Coarse Chunk 3 ──► [Worker 3 (Kev)] ──┤ (Semaphore concurrency: 4-8)
         └── Coarse Chunk N ──► [Worker N (Kev)] ──┘
                                      │
                                      ▼ Fan-in
         └─► Batch Embed fine chunks via embeddinggemma2
         └─► Insert fine chunks & vectors (linked to parent coarse chunks)
         └─► Update asset status='fine_indexed'
```
