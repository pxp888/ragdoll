import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import sqlite_vec

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    storage_path TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    parent_chunk_id INTEGER REFERENCES chunks(id),
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    byte_offset_start INTEGER NOT NULL,
    byte_offset_end INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_asset_stage ON chunks(asset_id, stage);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[768] distance_metric=cosine
);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db(db_path: Path) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def insert_asset(
    conn: sqlite3.Connection,
    asset_id: str,
    filename: str,
    content_type: str,
    size_bytes: int,
    sha256: str,
    status: str,
    storage_path: str,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO assets (
                id, filename, content_type, size_bytes, sha256,
                status, error_message, storage_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                filename,
                content_type,
                size_bytes,
                sha256,
                status,
                error_message,
                storage_path,
                now,
                now,
            ),
        )
    return {
        "id": asset_id,
        "filename": filename,
        "content_type": content_type,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "status": status,
        "error_message": error_message,
        "storage_path": storage_path,
        "created_at": now,
        "updated_at": now,
    }


def get_asset(conn: sqlite3.Connection, asset_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM assets WHERE id = ?", (asset_id,)
    ).fetchone()
    if not row:
        return None
    return dict(row)


def get_asset_by_sha256(conn: sqlite3.Connection, sha256: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM assets WHERE sha256 = ?", (sha256,)
    ).fetchone()
    if not row:
        return None
    return dict(row)


def update_asset_status(
    conn: sqlite3.Connection,
    asset_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            UPDATE assets
            SET status = ?, error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, error_message, now, asset_id),
        )


def insert_chunks_and_embeddings(
    conn: sqlite3.Connection,
    asset_id: str,
    stage: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> List[int]:
    """
    Inserts chunks into `chunks` table and their embeddings into `vec_chunks`.
    Returns the list of generated chunk IDs.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks count and embeddings count must match")

    chunk_ids: List[int] = []
    with conn:
        for chunk, emb in zip(chunks, embeddings):
            cursor = conn.execute(
                """
                INSERT INTO chunks (
                    asset_id, stage, parent_chunk_id, chunk_index,
                    text, byte_offset_start, byte_offset_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    stage,
                    chunk.get("parent_chunk_id"),
                    chunk["chunk_index"],
                    chunk["text"],
                    chunk["byte_offset_start"],
                    chunk["byte_offset_end"],
                ),
            )
            c_id = cursor.lastrowid
            chunk_ids.append(c_id)

            conn.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (c_id, json.dumps(emb)),
            )

    return chunk_ids


def get_asset_chunk_counts(conn: sqlite3.Connection, asset_id: str) -> Tuple[int, int]:
    coarse_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE asset_id = ? AND stage = 'coarse'",
        (asset_id,),
    ).fetchone()[0]
    fine_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE asset_id = ? AND stage = 'fine'",
        (asset_id,),
    ).fetchone()[0]
    return coarse_count, fine_count


def search_vec_chunks(
    conn: sqlite3.Connection,
    query_embedding: List[float],
    top_k: int,
    stage: str = "fine",
    candidate_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Performs cosine vector search against vec_chunks for the specified stage.
    """
    # Over-fetch candidate KNNs so stage filtering does not starve the results
    if candidate_k is None:
        search_k = max(top_k * 5, 50)
    else:
        search_k = max(candidate_k, top_k * 5, 50)

    query = """
    SELECT 
        c.id AS chunk_id,
        c.asset_id,
        a.filename,
        c.stage,
        c.text,
        c.parent_chunk_id,
        p.text AS parent_text,
        a.created_at,
        v.distance
    FROM vec_chunks v
    JOIN chunks c ON c.id = v.chunk_id
    JOIN assets a ON a.id = c.asset_id
    LEFT JOIN chunks p ON p.id = c.parent_chunk_id
    WHERE v.embedding MATCH ? AND v.k = ? AND c.stage = ?
    ORDER BY v.distance ASC
    LIMIT ?
    """

    rows = conn.execute(
        query,
        (json.dumps(query_embedding), search_k, stage, top_k),
    ).fetchall()

    results: List[Dict[str, Any]] = []
    for r in rows:
        distance = r["distance"]
        # Convert cosine distance to score in [0.0, 1.0]
        score = max(0.0, min(1.0, 1.0 - distance))
        results.append({
            "chunk_id": r["chunk_id"],
            "asset_id": r["asset_id"],
            "filename": r["filename"],
            "stage": r["stage"],
            "score": round(score, 4),
            "text": r["text"],
            "parent_context": r["parent_text"],
            "created_at": r["created_at"],
        })

    return results
