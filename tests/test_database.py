import pytest
from pathlib import Path
from ragdoll.database import (
    get_connection,
    init_db,
    insert_asset,
    get_asset,
    get_asset_by_sha256,
    update_asset_status,
    insert_chunks_and_embeddings,
    get_asset_chunk_counts,
    search_vec_chunks,
)


@pytest.fixture
def test_db(tmp_path: Path):
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


def test_asset_lifecycle(test_db: Path):
    conn = get_connection(test_db)
    asset = insert_asset(
        conn=conn,
        asset_id="test-uuid-1",
        filename="test.txt",
        content_type="text/plain",
        size_bytes=100,
        sha256="abc123sha",
        status="queued",
        storage_path="/tmp/test.txt",
    )
    assert asset["id"] == "test-uuid-1"

    fetched = get_asset(conn, "test-uuid-1")
    assert fetched is not None
    assert fetched["status"] == "queued"

    by_sha = get_asset_by_sha256(conn, "abc123sha")
    assert by_sha is not None
    assert by_sha["id"] == "test-uuid-1"

    update_asset_status(conn, "test-uuid-1", "coarse_indexed")
    updated = get_asset(conn, "test-uuid-1")
    assert updated["status"] == "coarse_indexed"
    conn.close()


def test_chunks_and_vector_search(test_db: Path):
    conn = get_connection(test_db)
    insert_asset(
        conn=conn,
        asset_id="a1",
        filename="doc.txt",
        content_type="text/plain",
        size_bytes=500,
        sha256="doc_sha",
        status="fine_indexed",
        storage_path="/tmp/doc.txt",
    )

    # 1. Insert coarse chunk
    coarse_vec = [0.1] * 768
    coarse_chunks = [{
        "chunk_index": 0,
        "text": "Parent coarse chunk text covering the full context.",
        "byte_offset_start": 0,
        "byte_offset_end": 100,
    }]
    coarse_ids = insert_chunks_and_embeddings(
        conn,
        asset_id="a1",
        stage="coarse",
        chunks=coarse_chunks,
        embeddings=[coarse_vec],
    )
    assert len(coarse_ids) == 1
    parent_id = coarse_ids[0]

    # 2. Insert fine chunks linked to parent
    fine_vec1 = [0.2] * 768
    fine_vec2 = [-0.2] * 768
    fine_chunks = [
        {
            "parent_chunk_id": parent_id,
            "chunk_index": 0,
            "text": "Fine chunk 1 discussing details.",
            "byte_offset_start": 0,
            "byte_offset_end": 50,
        },
        {
            "parent_chunk_id": parent_id,
            "chunk_index": 1,
            "text": "Fine chunk 2 discussing conclusions.",
            "byte_offset_start": 50,
            "byte_offset_end": 100,
        },
    ]
    fine_ids = insert_chunks_and_embeddings(
        conn,
        asset_id="a1",
        stage="fine",
        chunks=fine_chunks,
        embeddings=[fine_vec1, fine_vec2],
    )
    assert len(fine_ids) == 2

    # Verify chunk counts
    c_count, f_count = get_asset_chunk_counts(conn, "a1")
    assert c_count == 1
    assert f_count == 2

    # 3. Vector search fine chunks
    query_vec = [0.2] * 768
    results = search_vec_chunks(
        conn,
        query_embedding=query_vec,
        top_k=2,
        stage="fine",
    )
    assert len(results) == 2
    assert results[0]["chunk_id"] == fine_ids[0]
    assert results[0]["score"] > 0.9  # Identical direction
    assert results[0]["parent_context"] == "Parent coarse chunk text covering the full context."

    conn.close()
