import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from ragdoll.database import init_db, get_connection, insert_asset, get_asset, get_asset_chunk_counts
from ragdoll.pipeline.worker import process_asset_pipeline
from ragdoll.clients.embedding import EmbeddingClient
from ragdoll.clients.kev import KevClient


@pytest.mark.asyncio
async def test_worker_success_pipeline(tmp_path: Path):
    db_file = tmp_path / "worker_test.db"
    init_db(db_file)
    storage_file = tmp_path / "sample.txt"
    sample_text = (
        "## Section 1: Overview\n"
        "Ragdoll is a retrieval engine combining fast coarse-grained chunking with precise fine-grained chunking. " * 30
        + "\n\n## Section 2: Implementation\n"
        "The architecture relies on SQLite with sqlite-vec and Kev for boundary splitting. " * 30
    )
    storage_file.write_text(sample_text)

    conn = get_connection(db_file)
    insert_asset(
        conn=conn,
        asset_id="worker-asset-1",
        filename="sample.txt",
        content_type="text/plain",
        size_bytes=len(sample_text.encode("utf-8")),
        sha256="fake_sha",
        status="queued",
        storage_path=str(storage_file),
    )
    conn.close()

    # Mock clients for fast deterministic pipeline test
    mock_embed = AsyncMock(spec=EmbeddingClient)
    mock_embed.check_health.return_value = True
    # Return 768-dim vector for each text
    mock_embed.embed_all.side_effect = lambda texts: [[0.05] * 768 for _ in texts]

    mock_kev = AsyncMock(spec=KevClient)
    mock_kev.check_health.return_value = True
    mock_kev.decide_boundary.return_value = True

    await process_asset_pipeline(
        asset_id="worker-asset-1",
        db_path=db_file,
        embedding_client=mock_embed,
        kev_client=mock_kev,
    )

    conn = get_connection(db_file)
    asset = get_asset(conn, "worker-asset-1")
    coarse_cnt, fine_cnt = get_asset_chunk_counts(conn, "worker-asset-1")
    conn.close()

    assert asset["status"] == "fine_indexed"
    assert asset["error_message"] is None
    assert coarse_cnt >= 2
    assert fine_cnt >= 4


@pytest.mark.asyncio
async def test_worker_fail_fast_on_kev_error(tmp_path: Path):
    db_file = tmp_path / "fail_test.db"
    init_db(db_file)
    storage_file = tmp_path / "sample_fail.txt"
    sample_text = "Section A content. " * 50 + "\n\nSection B content. " * 50
    storage_file.write_text(sample_text)

    conn = get_connection(db_file)
    insert_asset(
        conn=conn,
        asset_id="fail-asset-1",
        filename="sample_fail.txt",
        content_type="text/plain",
        size_bytes=len(sample_text.encode("utf-8")),
        sha256="fail_sha",
        status="queued",
        storage_path=str(storage_file),
    )
    conn.close()

    mock_embed = AsyncMock(spec=EmbeddingClient)
    mock_embed.check_health.return_value = True
    mock_embed.embed_all.side_effect = lambda texts: [[0.01] * 768 for _ in texts]

    mock_kev = AsyncMock(spec=KevClient)
    mock_kev.check_health.return_value = True
    # Simulate a network/HTTP timeout error during Kev worker execution
    mock_kev.decide_boundary.side_effect = ConnectionResetError("Kev connection reset")

    await process_asset_pipeline(
        asset_id="fail-asset-1",
        db_path=db_file,
        embedding_client=mock_embed,
        kev_client=mock_kev,
    )

    conn = get_connection(db_file)
    asset = get_asset(conn, "fail-asset-1")
    conn.close()

    assert asset["status"] == "failed"
    assert "Kev connection reset" in asset["error_message"]
