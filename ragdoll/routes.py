import asyncio
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    Query,
    status,
)
from fastapi.responses import FileResponse, JSONResponse

from ragdoll.auth import get_api_key
from ragdoll.config import settings
from ragdoll.database import (
    get_connection,
    insert_asset,
    get_asset,
    get_asset_by_sha256,
    get_asset_chunk_counts,
    search_vec_chunks,
    delete_asset,
    delete_assets_by_apikey,
)
from ragdoll.models import (
    StoreAssetResponse,
    AssetStatusResponse,
    SearchRequest,
    RetrievalRequest,
    SearchResponse,
    SearchResultItem,
    HealthResponse,
    DeleteAssetResponse,
    DeleteAllAssetsResponse,
)
from ragdoll.clients.embedding import EmbeddingClient
from ragdoll.clients.kev import KevClient
from ragdoll.clients.standard_reranker import StandardReranker
from ragdoll.pipeline.worker import process_asset_pipeline
from ragdoll.pipeline.manager import BackgroundTaskManager

logger = logging.getLogger("ragdoll.api")
router = APIRouter(dependencies=[Depends(get_api_key)])


def get_embedding_client(request: Request) -> EmbeddingClient:
    return getattr(request.app.state, "embedding_client", EmbeddingClient())


def get_kev_client(request: Request) -> KevClient:
    return getattr(request.app.state, "kev_client", KevClient())


def get_standard_reranker(request: Request) -> StandardReranker:
    reranker = getattr(request.app.state, "standard_reranker", None)
    if reranker is None:
        reranker = StandardReranker(
            model_name=settings.reranker_model,
            cache_dir=settings.reranker_cache_dir,
        )
        request.app.state.standard_reranker = reranker
    return reranker


def get_task_manager(request: Request) -> BackgroundTaskManager:
    manager = getattr(request.app.state, "task_manager", None)
    if manager is None:
        manager = BackgroundTaskManager(
            max_concurrent_tasks=settings.max_active_tasks,
        )
        request.app.state.task_manager = manager
    return manager


async def _retrieve_candidates(
    query: str,
    stage: str,
    top_k: int,
    candidate_k: Optional[int],
    embedding_client: EmbeddingClient,
    apikey: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # 1. Embed query
    try:
        query_embedding = await embedding_client.embed_single(query)
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {e}",
        )

    # 2. Vector search in sqlite-vec
    conn = get_connection(settings.db_path)
    try:
        hits = search_vec_chunks(
            conn=conn,
            query_embedding=query_embedding,
            top_k=candidate_k if candidate_k is not None else top_k,
            stage=stage,
            candidate_k=candidate_k,
            apikey=apikey,
        )
    finally:
        conn.close()
    return hits


def _build_search_response(
    hits: List[Dict[str, Any]],
    include_parent_context: bool,
    rerank_method: Optional[str] = None,
) -> SearchResponse:
    results = []
    for h in hits:
        parent_ctx = h["parent_context"] if include_parent_context else None
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
    return SearchResponse(results=results, rerank_method=rerank_method)


async def _rerank_with_kev(
    hits: List[Dict[str, Any]],
    query: str,
    top_k: int,
    kev_client: KevClient,
) -> List[Dict[str, Any]]:
    if not hits:
        return []
    try:
        async def score_hit(hit: dict):
            score = await kev_client.score_relevance(
                query=query, passage=hit["text"]
            )
            hit["rerank_score"] = score

        await asyncio.gather(*(score_hit(h) for h in hits))
        hits.sort(key=lambda h: h.get("rerank_score", 0.0), reverse=True)
        hits = hits[:top_k]
        for h in hits:
            h["score"] = round(h["rerank_score"], 4)
        return hits
    except Exception as e:
        logger.error(f"Re-ranking failed via Kev: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Kev re-ranking service unavailable: {e}",
        )


async def _rerank_with_standard(
    hits: List[Dict[str, Any]],
    query: str,
    top_k: int,
    standard_reranker: StandardReranker,
) -> List[Dict[str, Any]]:
    if not hits:
        return []
    try:
        passages = [{"id": h["chunk_id"], "text": h["text"], "_hit": h} for h in hits]
        ranked_passages = await standard_reranker.rerank(query=query, passages=passages)
        final_hits = []
        for item in ranked_passages[:top_k]:
            h = item["_hit"]
            h["score"] = round(float(item["score"]), 4)
            final_hits.append(h)
        return final_hits
    except Exception as e:
        logger.error(f"Standard re-ranking failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Standard re-ranking failed: {e}",
        )


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
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    kev_client: KevClient = Depends(get_kev_client),
    task_manager: BackgroundTaskManager = Depends(get_task_manager),
):
    """
    Upload an asset. Saves file to disk, enqueues background processing, and returns immediately.
    Scoped to the authenticated API key.
    """
    asset_id = str(uuid.uuid4())
    filename = file.filename or f"upload_{asset_id}"
    content_type = file.content_type or "application/octet-stream"

    # Save file and compute sha256
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    destination = settings.storage_path / f"{asset_id}_{filename}"

    hasher = hashlib.sha256()
    size_bytes = 0

    with destination.open("wb") as buffer:
        while chunk := await file.read(1024 * 64):
            buffer.write(chunk)
            hasher.update(chunk)
            size_bytes += len(chunk)

    sha256 = hasher.hexdigest()

    # Deduplication check scoped to caller's apikey
    conn = get_connection(settings.db_path)
    try:
        existing = get_asset_by_sha256(conn, sha256=sha256, apikey=api_key)
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
            apikey=api_key,
        )
    finally:
        conn.close()

    # Enqueue background ingestion pipeline via task_manager
    background_tasks.add_task(
        task_manager.process_asset,
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
async def get_asset_status(
    asset_id: str,
    api_key: str = Depends(get_api_key),
):
    """
    Query the indexing status of an asset owned by the authenticated API key.
    """
    conn = get_connection(settings.db_path)
    try:
        asset = get_asset(conn, asset_id, apikey=api_key)
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {asset_id} not found",
            )
        coarse_chunks, fine_chunks = get_asset_chunk_counts(conn, asset_id, apikey=api_key)
        return AssetStatusResponse(
            asset_id=asset["id"],
            status=asset["status"],
            coarse_chunks=coarse_chunks,
            fine_chunks=fine_chunks,
            error=asset["error_message"],
        )
    finally:
        conn.close()


@router.post("/retrieve/kev", response_model=SearchResponse)
@router.post("/search/kev", response_model=SearchResponse)
async def retrieve_with_kev(
    req: RetrievalRequest,
    api_key: str = Depends(get_api_key),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    kev_client: KevClient = Depends(get_kev_client),
):
    """
    Kev-based retrieval endpoint.
    Performs vector candidate search across assets owned by the caller followed by Kev 4B re-ranking.
    """
    candidate_k = req.candidate_k or max(req.top_k * 3, 10)
    hits = await _retrieve_candidates(
        query=req.query,
        stage=req.stage,
        top_k=req.top_k,
        candidate_k=candidate_k,
        embedding_client=embedding_client,
        apikey=api_key,
    )
    reranked = await _rerank_with_kev(hits, req.query, req.top_k, kev_client)
    return _build_search_response(reranked, req.include_parent_context, rerank_method="kev")


@router.post("/retrieve/standard", response_model=SearchResponse)
@router.post("/search/standard", response_model=SearchResponse)
async def retrieve_standard(
    req: RetrievalRequest,
    api_key: str = Depends(get_api_key),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    standard_reranker: StandardReranker = Depends(get_standard_reranker),
):
    """
    Standard re-ranking retrieval endpoint (does not involve Kev).
    Performs vector candidate search across assets owned by the caller followed by FlashRank cross-encoder re-ranking.
    """
    candidate_k = req.candidate_k or max(req.top_k * 3, 10)
    hits = await _retrieve_candidates(
        query=req.query,
        stage=req.stage,
        top_k=req.top_k,
        candidate_k=candidate_k,
        embedding_client=embedding_client,
        apikey=api_key,
    )
    reranked = await _rerank_with_standard(hits, req.query, req.top_k, standard_reranker)
    return _build_search_response(reranked, req.include_parent_context, rerank_method="standard")


@router.post("/search", response_model=SearchResponse)
@router.post("/retrieve", response_model=SearchResponse, include_in_schema=False)
async def search_chunks(
    req: SearchRequest,
    api_key: str = Depends(get_api_key),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    kev_client: KevClient = Depends(get_kev_client),
    standard_reranker: StandardReranker = Depends(get_standard_reranker),
):
    """
    Perform hybrid / vector search across indexed chunks owned by the caller, with optional semantic re-ranking.
    Can re-rank via Kev (default when rerank=True) or standard FlashRank reranker.
    """
    candidate_k = req.candidate_k or (max(req.top_k * 3, 10) if req.rerank else req.top_k)
    hits = await _retrieve_candidates(
        query=req.query,
        stage=req.stage,
        top_k=req.top_k if not req.rerank else candidate_k,
        candidate_k=candidate_k,
        embedding_client=embedding_client,
        apikey=api_key,
    )

    rerank_method = "none"
    if req.rerank and hits:
        if req.rerank_method == "standard":
            hits = await _rerank_with_standard(hits, req.query, req.top_k, standard_reranker)
            rerank_method = "standard"
        else:
            # Default to Kev for backward compatibility
            hits = await _rerank_with_kev(hits, req.query, req.top_k, kev_client)
            rerank_method = "kev"

    return _build_search_response(hits, req.include_parent_context, rerank_method=rerank_method)


@router.get("/get/{asset_id}")
@router.get("/assets/{asset_id}", include_in_schema=False)
async def get_raw_asset(
    asset_id: str,
    api_key: str = Depends(get_api_key),
):
    """
    Retrieve the original raw stored asset file.
    Scoped to the authenticated API key.
    """
    conn = get_connection(settings.db_path)
    try:
        asset = get_asset(conn, asset_id, apikey=api_key)
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
    api_key: str = Depends(get_api_key),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    kev_client: KevClient = Depends(get_kev_client),
    standard_reranker: StandardReranker = Depends(get_standard_reranker),
    task_manager: BackgroundTaskManager = Depends(get_task_manager),
):
    """
    Liveness and readiness check. Requires valid API key. Actively tests connectivity to both
    embeddinggemma2 and kev 4B. Returns 503 Service Unavailable if either dependency is down.
    Also reports the status of the standard reranker.
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

    standard_ok = False
    standard_err = None
    try:
        standard_ok = await standard_reranker.check_health()
    except Exception as e:
        standard_err = str(e)

    all_healthy = embed_ok and kev_ok
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    content = {
        "status": "healthy" if all_healthy else "unhealthy",
        "embeddinggemma2": "healthy" if embed_ok else "unhealthy",
        "kev_4b": "healthy" if kev_ok else "unhealthy",
        "standard_reranker": "healthy" if standard_ok else "unhealthy",
        "details": {
            "embeddinggemma2_error": embed_err,
            "kev_4b_error": kev_err,
            "standard_reranker_error": standard_err,
            "task_manager": task_manager.get_stats(),
        },
    }

    return JSONResponse(status_code=status_code, content=content)


@router.delete("/assets/{asset_id}", response_model=DeleteAssetResponse)
@router.delete("/delete/{asset_id}", response_model=DeleteAssetResponse, include_in_schema=False)
@router.delete("/get/{asset_id}", response_model=DeleteAssetResponse, include_in_schema=False)
async def delete_specific_asset(
    asset_id: str,
    target_apikey: Optional[str] = Query(None, description="Optional target API key owner. If omitted, defaults to authenticated API key."),
    api_key: str = Depends(get_api_key),
):
    """
    Delete a specific asset by ID, including its coarse/fine chunks, sqlite-vec embeddings,
    and stored file on disk. Scoped to the caller's API key by default.
    """
    key_to_use = target_apikey if target_apikey is not None else api_key
    conn = get_connection(settings.db_path)
    try:
        deleted = delete_asset(conn, asset_id, apikey=key_to_use)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset {asset_id} not found",
            )
        return DeleteAssetResponse(
            deleted=True,
            asset_id=asset_id,
            message=f"Asset {asset_id} deleted successfully",
        )
    finally:
        conn.close()


@router.delete("/assets/by-key/{apikey}", response_model=DeleteAllAssetsResponse)
@router.delete("/assets/apikey/{apikey}", response_model=DeleteAllAssetsResponse, include_in_schema=False)
@router.delete("/keys/{apikey}/assets", response_model=DeleteAllAssetsResponse, include_in_schema=False)
async def delete_all_assets_under_key(
    apikey: str,
    api_key: str = Depends(get_api_key),
):
    """
    Delete all assets under a specified API key, including their chunks, vector embeddings,
    and stored files on disk.
    """
    conn = get_connection(settings.db_path)
    try:
        deleted_ids = delete_assets_by_apikey(conn, apikey)
        return DeleteAllAssetsResponse(
            deleted_count=len(deleted_ids),
            apikey=apikey,
            asset_ids=deleted_ids,
            message=f"Deleted {len(deleted_ids)} asset(s) under apikey '{apikey}'",
        )
    finally:
        conn.close()


@router.delete("/assets", response_model=DeleteAllAssetsResponse)
async def delete_all_assets_query(
    target_apikey: Optional[str] = Query(None, alias="target_apikey"),
    apikey_param: Optional[str] = Query(None, alias="apikey"),
    api_key: str = Depends(get_api_key),
):
    """
    Delete all assets under a specified API key.
    The target API key can be specified via query parameter ('apikey' or 'target_apikey').
    If not specified in query, defaults to the caller's authenticated API key.
    """
    key_to_use = target_apikey or apikey_param or api_key
    conn = get_connection(settings.db_path)
    try:
        deleted_ids = delete_assets_by_apikey(conn, key_to_use)
        return DeleteAllAssetsResponse(
            deleted_count=len(deleted_ids),
            apikey=key_to_use,
            asset_ids=deleted_ids,
            message=f"Deleted {len(deleted_ids)} asset(s) under apikey '{key_to_use}'",
        )
    finally:
        conn.close()
