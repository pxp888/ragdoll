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
    delete_asset,
    delete_assets_by_apikey,
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
        apikey="user_test",
        filename="test.txt",
        content_type="text/plain",
        size_bytes=100,
        sha256="abc123sha",
        status="queued",
        storage_path="/tmp/test.txt",
    )
    assert asset["id"] == "test-uuid-1"
    assert asset["apikey"] == "user_test"

    fetched = get_asset(conn, "test-uuid-1", apikey="user_test")
    assert fetched is not None
    assert fetched["status"] == "queued"
    assert fetched["apikey"] == "user_test"

    by_sha = get_asset_by_sha256(conn, "abc123sha", apikey="user_test")
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
        apikey="user_test",
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
    c_count, f_count = get_asset_chunk_counts(conn, "a1", apikey="user_test")
    assert c_count == 1
    assert f_count == 2

    # 3. Vector search fine chunks
    query_vec = [0.2] * 768
    results = search_vec_chunks(
        conn,
        query_embedding=query_vec,
        top_k=2,
        stage="fine",
        apikey="user_test",
    )
    assert len(results) == 2
    assert results[0]["chunk_id"] == fine_ids[0]
    assert results[0]["score"] > 0.9  # Identical direction
    assert results[0]["parent_context"] == "Parent coarse chunk text covering the full context."

    conn.close()


def test_multi_user_isolation_in_database(test_db: Path):
    conn = get_connection(test_db)
    # Alice asset & chunks
    insert_asset(
        conn=conn,
        asset_id="alice_asset",
        apikey="alice_key",
        filename="alice.txt",
        content_type="text/plain",
        size_bytes=100,
        sha256="same_sha",
        status="fine_indexed",
        storage_path="/tmp/alice.txt",
    )
    # Bob asset & chunks (same sha256 allowed across different users)
    insert_asset(
        conn=conn,
        asset_id="bob_asset",
        apikey="bob_key",
        filename="bob.txt",
        content_type="text/plain",
        size_bytes=100,
        sha256="same_sha",
        status="fine_indexed",
        storage_path="/tmp/bob.txt",
    )

    vec = [0.5] * 768
    alice_chunks = [{
        "chunk_index": 0,
        "text": "Alice secret information.",
        "byte_offset_start": 0,
        "byte_offset_end": 25,
    }]
    bob_chunks = [{
        "chunk_index": 0,
        "text": "Bob confidential information.",
        "byte_offset_start": 0,
        "byte_offset_end": 28,
    }]

    insert_chunks_and_embeddings(conn, "alice_asset", "fine", alice_chunks, [vec])
    insert_chunks_and_embeddings(conn, "bob_asset", "fine", bob_chunks, [vec])

    # 1. Asset ownership check
    assert get_asset(conn, "alice_asset", apikey="alice_key") is not None
    assert get_asset(conn, "alice_asset", apikey="bob_key") is None
    assert get_asset(conn, "bob_asset", apikey="bob_key") is not None
    assert get_asset(conn, "bob_asset", apikey="alice_key") is None

    # 2. Scoped sha256 lookup check
    assert get_asset_by_sha256(conn, "same_sha", apikey="alice_key")["id"] == "alice_asset"
    assert get_asset_by_sha256(conn, "same_sha", apikey="bob_key")["id"] == "bob_asset"
    assert get_asset_by_sha256(conn, "same_sha", apikey="eve_key") is None

    # 3. Vector search isolation check
    alice_results = search_vec_chunks(conn, vec, top_k=10, stage="fine", apikey="alice_key")
    assert len(alice_results) == 1
    assert alice_results[0]["asset_id"] == "alice_asset"
    assert "Alice" in alice_results[0]["text"]

    bob_results = search_vec_chunks(conn, vec, top_k=10, stage="fine", apikey="bob_key")
    assert len(bob_results) == 1
    assert bob_results[0]["asset_id"] == "bob_asset"
    assert "Bob" in bob_results[0]["text"]

    eve_results = search_vec_chunks(conn, vec, top_k=10, stage="fine", apikey="eve_key")
    assert len(eve_results) == 0

    conn.close()


def test_delete_asset(test_db: Path, tmp_path: Path):
    conn = get_connection(test_db)
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("some data")

    insert_asset(
        conn=conn,
        asset_id="del_asset_1",
        apikey="alice_key",
        filename="test_file.txt",
        content_type="text/plain",
        size_bytes=9,
        sha256="sha_del_1",
        status="fine_indexed",
        storage_path=str(file_path),
    )
    chunk_ids = insert_chunks_and_embeddings(
        conn=conn,
        asset_id="del_asset_1",
        stage="fine",
        chunks=[{"chunk_index": 0, "text": "hello", "byte_offset_start": 0, "byte_offset_end": 5}],
        embeddings=[[0.1] * 768],
    )
    assert len(chunk_ids) == 1
    assert file_path.exists()

    # Bob cannot delete Alice's asset
    res_bob = delete_asset(conn, "del_asset_1", apikey="bob_key")
    assert res_bob is False
    assert file_path.exists()
    assert get_asset(conn, "del_asset_1") is not None

    # Alice can delete her asset
    res_alice = delete_asset(conn, "del_asset_1", apikey="alice_key")
    assert res_alice is True
    assert not file_path.exists()
    assert get_asset(conn, "del_asset_1") is None
    # Verify chunks and vec_chunks are deleted
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE asset_id = 'del_asset_1'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM vec_chunks WHERE chunk_id = ?", (chunk_ids[0],)).fetchone()[0] == 0

    # Repeating returns False
    assert delete_asset(conn, "del_asset_1", apikey="alice_key") is False
    conn.close()


def test_delete_assets_by_apikey(test_db: Path, tmp_path: Path):
    conn = get_connection(test_db)
    f1 = tmp_path / "alice1.txt"
    f1.write_text("alice 1")
    f2 = tmp_path / "alice2.txt"
    f2.write_text("alice 2")
    fb = tmp_path / "bob.txt"
    fb.write_text("bob")

    insert_asset(conn, asset_id="a1", apikey="alice_key", filename="alice1.txt", content_type="text/plain", size_bytes=7, sha256="sha_a1", status="fine_indexed", storage_path=str(f1))
    insert_asset(conn, asset_id="a2", apikey="alice_key", filename="alice2.txt", content_type="text/plain", size_bytes=7, sha256="sha_a2", status="fine_indexed", storage_path=str(f2))
    insert_asset(conn, asset_id="b1", apikey="bob_key", filename="bob.txt", content_type="text/plain", size_bytes=3, sha256="sha_b1", status="fine_indexed", storage_path=str(fb))

    c1 = insert_chunks_and_embeddings(conn, "a1", "fine", [{"chunk_index": 0, "text": "c1", "byte_offset_start": 0, "byte_offset_end": 2}], [[0.1]*768])
    c2 = insert_chunks_and_embeddings(conn, "a2", "fine", [{"chunk_index": 0, "text": "c2", "byte_offset_start": 0, "byte_offset_end": 2}], [[0.2]*768])
    cb = insert_chunks_and_embeddings(conn, "b1", "fine", [{"chunk_index": 0, "text": "cb", "byte_offset_start": 0, "byte_offset_end": 2}], [[0.3]*768])

    # Delete all Alice assets
    deleted = delete_assets_by_apikey(conn, "alice_key")
    assert set(deleted) == {"a1", "a2"}

    # Alice files and records removed
    assert not f1.exists()
    assert not f2.exists()
    assert get_asset(conn, "a1") is None
    assert get_asset(conn, "a2") is None
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE asset_id IN ('a1', 'a2')").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM vec_chunks WHERE chunk_id IN (?, ?)", (c1[0], c2[0])).fetchone()[0] == 0

    # Bob files and records untouched
    assert fb.exists()
    assert get_asset(conn, "b1") is not None
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE asset_id = 'b1'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM vec_chunks WHERE chunk_id = ?", (cb[0],)).fetchone()[0] == 1

    # Calling again for empty apikey returns empty list
    assert delete_assets_by_apikey(conn, "alice_key") == []
    assert delete_assets_by_apikey(conn, "nonexistent_key") == []
    conn.close()
