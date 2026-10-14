import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from ragdoll.config import settings
from ragdoll.database import (
    get_connection,
    get_asset,
    update_asset_status,
    insert_chunks_and_embeddings,
)
from ragdoll.clients.embedding import EmbeddingClient
from ragdoll.clients.kev import KevClient
from ragdoll.pipeline.extractor import extract_text
from ragdoll.pipeline.chunker import (
    split_coarse_chunks,
    split_fine_chunks_with_kev,
)

logger = logging.getLogger("ragdoll.worker")


def format_error_message(e: BaseException) -> str:
    if isinstance(e, BaseExceptionGroup):
        parts = [format_error_message(sub) for sub in e.exceptions]
        return "; ".join(parts)
    return str(e)


async def process_asset_pipeline(
    asset_id: str,
    db_path: Path = settings.db_path,
    embedding_client: EmbeddingClient = None,
    kev_client: KevClient = None,
) -> None:
    """
    Executes the two-stage ingestion pipeline for an asset:
    1. Coarse indexing (fast path) -> status='coarse_indexed'
    2. Parallel Kev fine indexing -> status='fine_indexed'
    """
    if embedding_client is None:
        embedding_client = EmbeddingClient()
    if kev_client is None:
        kev_client = KevClient()

    conn = get_connection(db_path)
    try:
        asset = get_asset(conn, asset_id)
        if not asset:
            logger.error(f"Asset {asset_id} not found in database")
            return

        storage_path = Path(asset["storage_path"])
        if not storage_path.exists():
            raise FileNotFoundError(f"Asset file not found at {storage_path}")

        logger.info(f"[{asset_id}] Extracting text from {storage_path}")
        text = extract_text(storage_path, asset["content_type"])
        if not text.strip():
            raise ValueError("Document contains no readable text")

        # Step 1: Health check embedding server
        logger.info(f"[{asset_id}] Checking embedding server connectivity")
        is_embed_ok = await embedding_client.check_health()
        if not is_embed_ok:
            raise ConnectionError("Embedding server health check failed")

        # Stage 1: Coarse Split (2k-4k chars, structural anchors)
        logger.info(f"[{asset_id}] Stage 1: Splitting coarse chunks")
        coarse_chunks = split_coarse_chunks(
            text,
            min_chars=settings.coarse_min_chars,
            max_chars=settings.coarse_max_chars,
        )
        if not coarse_chunks:
            # Fallback if document is smaller than min_chars
            coarse_chunks = [{
                "chunk_index": 0,
                "text": text,
                "byte_offset_start": 0,
                "byte_offset_end": len(text.encode("utf-8")),
            }]

        logger.info(f"[{asset_id}] Stage 1: Generating embeddings for {len(coarse_chunks)} coarse chunks")
        coarse_texts = [c["text"] for c in coarse_chunks]
        coarse_embeddings = await embedding_client.embed_all(coarse_texts)

        logger.info(f"[{asset_id}] Stage 1: Inserting coarse chunks into DB")
        coarse_ids = insert_chunks_and_embeddings(
            conn,
            asset_id=asset_id,
            stage="coarse",
            chunks=coarse_chunks,
            embeddings=coarse_embeddings,
        )

        for c_dict, c_id in zip(coarse_chunks, coarse_ids):
            c_dict["id"] = c_id

        # Update status to coarse_indexed (asset is now immediately searchable)
        update_asset_status(conn, asset_id, status="coarse_indexed")
        logger.info(f"[{asset_id}] Status transitioned to 'coarse_indexed'")

        # Step 2: Health check Kev server
        logger.info(f"[{asset_id}] Checking Kev 4B server connectivity")
        is_kev_ok = await kev_client.check_health()
        if not is_kev_ok:
            raise ConnectionError("Kev 4B server health check failed")

        # Stage 2: Parallel Kev Fine-Chunking (Anchor Sharding)
        logger.info(
            f"[{asset_id}] Stage 2: Starting parallel fine chunking with {settings.concurrency_limit} concurrency limit"
        )
        sem = asyncio.Semaphore(settings.concurrency_limit)
        fine_results: List[List[Dict[str, Any]]] = [[] for _ in coarse_chunks]

        async def worker(index: int, coarse_c: Dict[str, Any]):
            async with sem:
                res = await split_fine_chunks_with_kev(
                    coarse_chunk=coarse_c,
                    kev_client=kev_client,
                    min_chars=settings.fine_min_chars,
                    max_chars=settings.fine_max_chars,
                )
                fine_results[index] = res

        # Fail-fast error propagation: TaskGroup aborts all workers on first exception
        async with asyncio.TaskGroup() as tg:
            for idx, c_chunk in enumerate(coarse_chunks):
                tg.create_task(worker(idx, c_chunk))

        # Flatten fine chunks
        fine_chunks: List[Dict[str, Any]] = []
        for res_list in fine_results:
            fine_chunks.extend(res_list)

        # Re-index fine chunks sequentially
        for f_idx, f_chunk in enumerate(fine_chunks):
            f_chunk["chunk_index"] = f_idx

        logger.info(
            f"[{asset_id}] Stage 2: Produced {len(fine_chunks)} fine chunks; generating embeddings in batches"
        )
        fine_texts = [f["text"] for f in fine_chunks]
        fine_embeddings = await embedding_client.embed_all(fine_texts)

        logger.info(f"[{asset_id}] Stage 2: Inserting fine chunks into DB")
        insert_chunks_and_embeddings(
            conn,
            asset_id=asset_id,
            stage="fine",
            chunks=fine_chunks,
            embeddings=fine_embeddings,
        )

        update_asset_status(conn, asset_id, status="fine_indexed")
        logger.info(f"[{asset_id}] Status transitioned to 'fine_indexed' (complete)")

    except BaseException as e:
        err_msg = format_error_message(e)
        logger.exception(f"[{asset_id}] Ingestion pipeline failed: {err_msg}")
        try:
            update_asset_status(conn, asset_id, status="failed", error_message=err_msg)
        except Exception as db_err:
            logger.error(f"[{asset_id}] Failed to record error status in DB: {db_err}")
    finally:
        conn.close()
