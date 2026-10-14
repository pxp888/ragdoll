---
name: ragdoll
description: How to talk to a running RAGDOLL server (FastAPI document ingestion + retrieval). Use when a project needs to upload documents, poll indexing status, or run vector search with reranking against ragdoll.
---

# Ragdoll client skill

Ragdoll is a FastAPI RAG server: upload files → background pipeline (extract text → chunk at two scales → embed via EmbeddingGemma2) → query with sqlite-vec + optional reranking (Kev LLM judge or FlashRank cross-encoder). Tenancy is per API key.

## Connection & auth

- Base URL defaults to `http://localhost:8000` (server binds 0.0.0.0:8000; confirm the actual host with the user).
- **Every endpoint requires an API key**, including `/health`. Send it as the `X-API-Key` header. Alternatives: `Authorization: Bearer <key>` or `?api_key=<key>` query param.
- The key is any non-empty string and doubles as the tenant id — all assets are scoped to it. A "wrong" key doesn't error; it just sees no assets, so cross-key access shows up as 404, not 403.

## Endpoints (canonical paths)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/store` | Upload a file (multipart field `file`). Returns **202** with `{asset_id, filename, size_bytes, status:"queued", created_at}`. Duplicates by sha256 within the same key return the existing asset without re-processing. |
| GET | `/assets/{asset_id}/status` | Poll indexing progress: `{asset_id, status, coarse_chunks, fine_chunks, error}`. Statuses: `queued` → `coarse_indexed` (searchable at stage=coarse only) → `fine_indexed` (fully ready), or `failed`. |
| POST | `/search` | Hybrid search with optional reranking. |
| POST | `/search/kev` | Always rerank with Kev LLM judge. |
| POST | `/search/standard` | Always rerank with FlashRank cross-encoder (no Kev/GPU involvement — cheaper, faster). |
| GET | `/get/{asset_id}` | Download the original stored file. |
| DELETE | `/assets/{asset_id}?target_apikey=` | Delete one asset + chunks + embeddings + disk file. Defaults to caller's key; `target_apikey` overrides owner. |
| DELETE | `/assets/by-key/{apikey}` | Delete all assets under a given key. (Also: `DELETE /assets?apikey=<k>`.) |
| GET | `/health` | Liveness of embedding service, Kev 4B, and standard reranker. Returns **503** if embedding or Kev is down. Requires the API key. |

Unlisted aliases exist (`/retrieve`, `/retrieve/kev`, `/retrieve/standard`, `/api/v1/assets`) but are not in the OpenAPI schema — use the canonical paths above.

## Search request / response

```json
// POST /search  (and /search/{kev,standard}, which omit rerank fields)
{
  "query": "...",
  "top_k": 5,                    // 1..100
  "stage": "fine",               // "fine" = small precise chunks (default); "coarse" = large context chunks
  "rerank": false,               // /search only: true reranks via Kev by default
  "rerank_method": null,         // /search only: "kev" | "standard" | "none"
  "include_parent_context": true,// include each chunk's surrounding text
  "candidate_k": null            // vector candidates before rerank; defaults to max(top_k*3, 10) when reranking
}

// response (all three search endpoints)
{
  "rerank_method": "kev",        // or "standard" / "none"
  "results": [{
    "chunk_id": 123, "asset_id": "...", "filename": "...",
    "score": 0.87,               // rerank score when reranked, vector similarity otherwise
    "text": "the chunk text",
    "parent_context": "surrounding context or null",
    "created_at": "..."
  }]
}
```

## Typical flow

1. `POST /store` the file → get `asset_id`.
2. Poll `GET /assets/{id}/status` until `fine_indexed` (or bail on `failed`). Use `coarse` stage searches while at `coarse_indexed`.
3. `POST /search` with your query; pass `rerank: true, rerank_method: "standard"` for cheap/fast results or `"kev"` for LLM-judged relevance.

## Gotchas

- A search needs the embedding service up (503 if down); Kev-down only breaks `kev` reranking, standard is independent and vice versa.
- First-ever upload of a document type can take a while — poll status rather than retrying `/store`.
- The server fail-fasts on startup unless both external deps are reachable; operators can bypass with `RAGDOLL_SKIP_STARTUP_PROBE=true`. All settings use the `RAGDOLL_` env prefix (`RAGDOLL_PORT`, `RAGDOLL_DB_PATH`, `RAGDOLL_EMBED_URL`, `RAGDOLL_KEV_URL`, ...).

## Examples

```bash
curl -s -H "X-API-Key: myproject" \
  -F file=@report.pdf http://localhost:8000/store
# → {"asset_id":"...","status":"queued",...}

curl -s -H "X-API-Key: myproject" \
  http://localhost:8000/assets/<asset_id>/status

curl -s -H "X-API-Key: myproject" -H 'Content-Type: application/json' \
  -d '{"query":"deployment steps","top_k":5,"rerank":true,"rerank_method":"standard"}' \
  http://localhost:8000/search
```

Python (httpx):

```python
import time
import httpx

BASE, KEY = "http://localhost:8000", {"X-API-Key": "myproject"}

def store(path):
    with open(path, "rb") as f:
        r = httpx.post(f"{BASE}/store", headers=KEY, files={"file": (path.split("/")[-1], f)}, timeout=300)
    return r.json()["asset_id"]  # 202

def wait_ready(asset_id, interval=5.0):
    while True:
        s = httpx.get(f"{BASE}/assets/{asset_id}/status", headers=KEY).json()
        if s["status"] in ("fine_indexed", "coarse_indexed"): return s
        if s["status"] == "failed": raise RuntimeError(s["error"])
        time.sleep(interval)

def search(query, top_k=5, stage="fine", rerank_method=None):
    body = {"query": query, "top_k": top_k, "stage": stage}
    if rerank_method:
        body |= {"rerank": True, "rerank_method": rerank_method}
    return httpx.post(f"{BASE}/search", headers=KEY, json=body).json()["results"]
```
