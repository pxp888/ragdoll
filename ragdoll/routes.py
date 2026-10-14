import asyncio
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse

from ragdoll.config import settings
from ragdoll.database import (
    get_connection,
    insert_asset,
    get_asset,
    get_asset_by_sha256,
    get_asset_chunk_counts,
    search_vec_chunks,
)
from ragdoll.models import (
    StoreAssetResponse,
    AssetStatusResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    HealthResponse,
)
from ragdoll.clients.embedding import EmbeddingClient
from ragdoll.clients.kev import KevClient
from ragdoll.pipeline.worker import process_asset_pipeline

logger = logging.getLogger("ragdoll.api")
router = APIRouter()


def get_embedding_client(request: Request) -> EmbeddingClient:
    return getattr(request.app.state, "embedding_client", EmbeddingClient())


def get_kev_client(request: Request) -> KevClient:
    return getattr(request.app.state, "kev_client", KevClient())


@router.post(
    "/store",
    response_model=StoreAssetResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@router.post(
    "/api/v1/assets",
    response_model=StoreAssetResponse,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
async def store_asset(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    kev_client: KevClient = Depends(get_kev_client),
):
    """
    Upload an asset. Saves file to disk, enqueues background processing, and returns immediately.
    """
    asset_id = str(uuid.uuid4())
    filename = file.filename or f"upload_{asset_id}"
    content_type = file.content_type or "application/octet-stream"

    # Save file and compute sha256
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    destination = settings.storage_path / f"{asset_id}_{filename}"

    hasher = hashlib.sha256()
    size_bytes = 0

    with open(destination, "wb") as out_f:
        while chunk := await file.read(1024 * 1024):  # 1MB buffer
            hasher.update(chunk)
            size_bytes += len(chunk)
            out_f.write(chunk)

    sha256 = hasher.hexdigest()

    conn = get_connection(settings.db_path)
    try:
        # Check if identical sha256 was already ingested
        existing = get_asset_by_sha256(conn, sha256)
        if existing:
            # Clean up the newly written duplicate file
            try:
                destination.unlink(missing_ok=True)
            except Exception:
                pass
            return StoreAssetResponse(
                asset_id=existing["id"],
                filename=existing["filename"],
                size_bytes=existing["size_bytes"],
                status=existing["status"],
                created_at=existing["created_at"],
            )

        asset_record = insert_asset(
            conn=conn,
            asset_id=asset_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            status="queued",
            storage_path=str(destination),
        )
    finally:
        conn.close()

    # Enqueue background ingestion pipeline
    background_tasks.add_task(
        process_asset_pipeline,
        asset_id=asset_id,
        db_path=settings.db_path,
        embedding_client=embedding_client,
        kev_client=kev_client,
    )

    return StoreAssetResponse(
        asset_id=asset_record["id"],
        filename=asset_record["filename"],
        size_bytes=asset_record["size_bytes"],
        status=asset_record["status"],
        created_at=asset_record["created_at"],
    )


@router.get("/assets/{asset_id}/status", response_model=AssetStatusResponse)
async def get_asset_status(asset_id: str):
    """
    Query the indexing status of an asset.
    """
    conn = get_connection(settings.db_path)
    try:
        asset = get_asset(conn, asset_id)
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {asset_id} not found",
            )
        coarse_chunks, fine_chunks = get_asset_chunk_counts(conn, asset_id)
        return AssetStatusResponse(
            asset_id=asset["id"],
            status=asset["status"],
            coarse_chunks=coarse_chunks,
            fine_chunks=fine_chunks,
            error=asset["error_message"],
        )
    finally:
        conn.close()


@router.post("/search", response_model=SearchResponse)
async def search_chunks(
    req: SearchRequest,
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    kev_client: KevClient = Depends(get_kev_client),
):
    """
    Perform hybrid / vector search across indexed chunks, with optional semantic re-ranking.
    """
    # 1. Embed query
    try:
        query_embedding = await embedding_client.embed_single(req.query)
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {e}",
        )

    # 2. Vector search in sqlite-vec
    conn = get_connection(settings.db_path)
    try:
        # If re-ranking, fetch more candidate hits
        candidate_k = max(req.top_k * 3, 10) if req.rerank else req.top_k
        hits = search_vec_chunks(
            conn=conn,
            query_embedding=query_embedding,
            top_k=req.top_k if not req.rerank else candidate_k,
            stage=req.stage,
            candidate_k=candidate_k,
        )
    finally:
        conn.close()

    # 3. Optional Semantic Re-ranking with Kev
    if req.rerank and hits:
        try:
            # Score each candidate concurrently
            async def score_hit(hit: dict):
                score = await kev_client.score_relevance(
                    query=req.query, passage=hit["text"]
                )
                hit["rerank_score"] = score

            await asyncio.gather(*(score_hit(h) for h in hits))

            # Sort descending by re-rank score
            hits.sort(key=lambda h: h.get("rerank_score", 0.0), reverse=True)
            hits = hits[: req.top_k]
            # Replace score with rerank score normalized
            for h in hits:
                h["score"] = round(h["rerank_score"], 4)
        except Exception as e:
            logger.error(f"Re-ranking failed via Kev: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Kev re-ranking service unavailable: {e}",
            )

    results = []
    for h in hits:
        parent_ctx = h["parent_context"] if req.include_parent_context else None
        results.append(
            SearchResultItem(
                chunk_id=h["chunk_id"],
                asset_id=h["asset_id"],
                filename=h["filename"],
                score=h["score"],
                text=h["text"],
                parent_context=parent_ctx,
                created_at=h["created_at"],
            )
        )

    return SearchResponse(results=results)


@router.get("/get/{asset_id}")
@router.get("/assets/{asset_id}", include_in_schema=False)
async def get_raw_asset(asset_id: str):
    """
    Retrieve the original raw stored asset file.
    """
    conn = get_connection(settings.db_path)
    try:
        asset = get_asset(conn, asset_id)
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {asset_id} not found",
            )
        file_path = Path(asset["storage_path"])
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Asset storage file not found on disk",
            )
        return FileResponse(
            path=str(file_path),
            media_type=asset["content_type"],
            filename=asset["filename"],
        )
    finally:
        conn.close()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    kev_client: KevClient = Depends(get_kev_client),
):
    """
    Liveness and readiness check. Actively tests connectivity to both
    embeddinggemma2 and kev 4B. Returns 503 Service Unavailable if either dependency is down.
    """
    embed_ok = False
    embed_err = None
    try:
        embed_ok = await embedding_client.check_health()
    except Exception as e:
        embed_err = str(e)

    kev_ok = False
    kev_err = None
    try:
        kev_ok = await kev_client.check_health()
    except Exception as e:
        kev_err = str(e)

    all_healthy = embed_ok and kev_ok
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    content = {
        "status": "healthy" if all_healthy else "unhealthy",
        "embeddinggemma2": "healthy" if embed_ok else "unhealthy",
        "kev_4b": "healthy" if kev_ok else "unhealthy",
        "details": {
            "embeddinggemma2_error": embed_err,
            "kev_4b_error": kev_err,
        },
    }

    return JSONResponse(status_code=status_code, content=content)
