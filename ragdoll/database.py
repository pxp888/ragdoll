import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import sqlite_vec

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    apikey TEXT NOT NULL,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    storage_path TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    UNIQUE(apikey, sha256)
);

CREATE INDEX IF NOT EXISTS idx_assets_apikey ON assets(apikey);

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
        # Handle migration for existing databases missing the apikey column
        cursor = conn.execute("PRAGMA table_info(assets);")
        columns = [row[1] for row in cursor.fetchall()]
        if columns and "apikey" not in columns:
            conn.execute("ALTER TABLE assets ADD COLUMN apikey TEXT NOT NULL DEFAULT 'default';")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_apikey ON assets(apikey);")
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
    apikey: str = "default",
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO assets (
                id, apikey, filename, content_type, size_bytes, sha256,
                status, error_message, storage_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                apikey,
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
        "apikey": apikey,
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


def get_asset(
    conn: sqlite3.Connection,
    asset_id: str,
    apikey: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if apikey is not None:
        row = conn.execute(
            "SELECT * FROM assets WHERE id = ? AND apikey = ?", (asset_id, apikey)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM assets WHERE id = ?", (asset_id,)
        ).fetchone()
    if not row:
        return None
    return dict(row)


def get_asset_by_sha256(
    conn: sqlite3.Connection,
    sha256: str,
    apikey: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if apikey is not None:
        row = conn.execute(
            "SELECT * FROM assets WHERE sha256 = ? AND apikey = ?", (sha256, apikey)
        ).fetchone()
    else:
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


def get_asset_chunk_counts(
    conn: sqlite3.Connection,
    asset_id: str,
    apikey: Optional[str] = None,
) -> Tuple[int, int]:
    if apikey is not None:
        coarse_count = conn.execute(
            """
            SELECT COUNT(*) FROM chunks c
            JOIN assets a ON a.id = c.asset_id
            WHERE c.asset_id = ? AND c.stage = 'coarse' AND a.apikey = ?
            """,
            (asset_id, apikey),
        ).fetchone()[0]
        fine_count = conn.execute(
            """
            SELECT COUNT(*) FROM chunks c
            JOIN assets a ON a.id = c.asset_id
            WHERE c.asset_id = ? AND c.stage = 'fine' AND a.apikey = ?
            """,
            (asset_id, apikey),
        ).fetchone()[0]
    else:
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
    apikey: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Performs cosine vector search against vec_chunks for the specified stage and apikey.
    """
    if candidate_k is None:
        search_k = max(top_k * 10, 100)
    else:
        search_k = max(candidate_k, top_k * 10, 100)

    if apikey is not None:
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
        WHERE v.embedding MATCH ? AND v.k = ? AND c.stage = ? AND a.apikey = ?
        ORDER BY v.distance ASC
        LIMIT ?
        """
        params = (json.dumps(query_embedding), search_k, stage, apikey, top_k)
    else:
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
        params = (json.dumps(query_embedding), search_k, stage, top_k)

    rows = conn.execute(query, params).fetchall()

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



def delete_asset(
    conn: sqlite3.Connection,
    asset_id: str,
    apikey: Optional[str] = None,
) -> bool:
    """
    Deletes an asset, its chunks, associated sqlite-vec embeddings, and file on disk.
    If apikey is provided, ensures the asset belongs to that apikey.
    Returns True if found and deleted, False otherwise.
    """
    if apikey is not None:
        row = conn.execute(
            "SELECT id, storage_path FROM assets WHERE id = ? AND apikey = ?",
            (asset_id, apikey),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, storage_path FROM assets WHERE id = ?",
            (asset_id,),
        ).fetchone()

    if not row:
        return False

    storage_path = row["storage_path"]

    with conn:
        # 1. Clean up vec_chunks (virtual table without foreign key cascade)
        conn.execute(
            "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE asset_id = ?)",
            (asset_id,),
        )
        # 2. Clean up chunks
        conn.execute(
            "DELETE FROM chunks WHERE asset_id = ?",
            (asset_id,),
        )
        # 3. Clean up asset
        conn.execute(
            "DELETE FROM assets WHERE id = ?",
            (asset_id,),
        )

    # 4. Remove physical file on disk
    if storage_path:
        try:
            file_path = Path(storage_path)
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass

    return True


def delete_assets_by_apikey(
    conn: sqlite3.Connection,
    apikey: str,
) -> List[str]:
    """
    Deletes all assets, chunks, sqlite-vec embeddings, and files on disk
    associated with the given apikey.
    Returns the list of deleted asset IDs.
    """
    rows = conn.execute(
        "SELECT id, storage_path FROM assets WHERE apikey = ?",
        (apikey,),
    ).fetchall()

    if not rows:
        return []

    asset_ids = [r["id"] for r in rows]
    storage_paths = [r["storage_path"] for r in rows]

    with conn:
        # 1. Clean up vec_chunks for all chunks belonging to this apikey's assets
        conn.execute(
            """
            DELETE FROM vec_chunks WHERE chunk_id IN (
                SELECT c.id FROM chunks c
                JOIN assets a ON a.id = c.asset_id
                WHERE a.apikey = ?
            )
            """,
            (apikey,),
        )
        # 2. Clean up chunks for all assets belonging to this apikey
        conn.execute(
            """
            DELETE FROM chunks WHERE asset_id IN (
                SELECT id FROM assets WHERE apikey = ?
            )
            """,
            (apikey,),
        )
        # 3. Clean up assets
        conn.execute(
            "DELETE FROM assets WHERE apikey = ?",
            (apikey,),
        )

    # 4. Remove physical files on disk
    for path_str in storage_paths:
        if path_str:
            try:
                p = Path(path_str)
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    return asset_ids
