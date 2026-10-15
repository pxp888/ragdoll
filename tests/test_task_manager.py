import asyncio
import io
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
import httpx

from ragdoll.config import settings
from ragdoll.database import init_db, get_connection, insert_asset, get_asset
from ragdoll.pipeline.manager import BackgroundTaskManager
from ragdoll.main import create_app
from ragdoll.clients.embedding import EmbeddingClient
from ragdoll.clients.kev import KevClient
from ragdoll.clients.standard_reranker import StandardReranker

TEST_HEADERS = {"X-API-Key": "test-key-task-mgr"}


@pytest.mark.asyncio
async def test_concurrency_limit_enforced(tmp_path: Path):
    db_file = tmp_path / "concurrency_test.db"
    init_db(db_file)

    max_active = 2
    manager = BackgroundTaskManager(max_concurrent_tasks=max_active)

    active_counts = []
    current_active = 0

    async def fake_pipeline(asset_id, db_path, embedding_client, kev_client):
        nonlocal current_active
        current_active += 1
        active_counts.append(current_active)
        await asyncio.sleep(0.05)
        current_active -= 1

    with patch("ragdoll.pipeline.manager.process_asset_pipeline", side_effect=fake_pipeline):
        tasks = [
            asyncio.create_task(
                manager.process_asset(
                    asset_id=f"asset-{i}",
                    db_path=db_file,
                    embedding_client=AsyncMock(),
                    kev_client=AsyncMock(),
                )
            )
            for i in range(6)
        ]
        await asyncio.gather(*tasks)

    # Concurrency limit should never have exceeded 2
    assert max(active_counts) <= max_active
    # All tasks should be cleared from active and queued
    stats = manager.get_stats()
    assert stats["active_tasks_count"] == 0
    assert stats["queued_tasks_count"] == 0


@pytest.mark.asyncio
async def test_startup_recovery(tmp_path: Path):
    db_file = tmp_path / "recovery_test.db"
    init_db(db_file)

    # Pre-populate database with stranded queued assets
    conn = get_connection(db_file)
    for i in range(3):
        insert_asset(
            conn=conn,
            asset_id=f"stranded-{i}",
            filename=f"file_{i}.txt",
            content_type="text/plain",
            size_bytes=100,
            sha256=f"sha_{i}",
            status="queued",
            storage_path=str(tmp_path / f"file_{i}.txt"),
        )
    conn.close()

    manager = BackgroundTaskManager(max_concurrent_tasks=2)
    processed = []

    async def fake_pipeline(asset_id, db_path, embedding_client, kev_client):
        processed.append(asset_id)

    with patch("ragdoll.pipeline.manager.process_asset_pipeline", side_effect=fake_pipeline):
        recovered = await manager.recover_pending_assets(
            db_path=db_file,
            embedding_client=AsyncMock(),
            kev_client=AsyncMock(),
        )
        assert recovered == 3

        # Wait for the background tasks created by recovery to finish
        await manager.shutdown(timeout=2.0)

    assert set(processed) == {"stranded-0", "stranded-1", "stranded-2"}


@pytest.mark.asyncio
async def test_graceful_shutdown_timeout(tmp_path: Path):
    db_file = tmp_path / "shutdown_test.db"
    init_db(db_file)

    manager = BackgroundTaskManager(max_concurrent_tasks=2)

    async def slow_pipeline(asset_id, db_path, embedding_client, kev_client):
        await asyncio.sleep(10.0)

    with patch("ragdoll.pipeline.manager.process_asset_pipeline", side_effect=slow_pipeline):
        task = asyncio.create_task(
            manager.process_asset(
                asset_id="slow-asset",
                db_path=db_file,
                embedding_client=AsyncMock(),
                kev_client=AsyncMock(),
            )
        )
        # Give task time to start
        await asyncio.sleep(0.02)
        assert manager.get_stats()["active_tasks_count"] == 1

        # Shutdown with short timeout to trigger cancellation
        start_t = time.time()
        await manager.shutdown(timeout=0.1)
        duration = time.time() - start_t

        assert duration < 1.0
        assert manager.get_stats()["shutting_down"] is True


def test_health_check_reports_task_manager_stats(tmp_path: Path):
    db_file = tmp_path / "health_mgr_test.db"
    init_db(db_file)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    settings.db_path = db_file
    settings.storage_path = uploads_dir
    settings.skip_startup_probe = True

    app = create_app()

    mock_embed = AsyncMock(spec=EmbeddingClient)
    mock_embed.check_health.return_value = True
    mock_kev = AsyncMock(spec=KevClient)
    mock_kev.check_health.return_value = True
    mock_standard = AsyncMock(spec=StandardReranker)
    mock_standard.check_health.return_value = True

    app.state.embedding_client = mock_embed
    app.state.kev_client = mock_kev
    app.state.standard_reranker = mock_standard

    with TestClient(app) as client:
        resp = client.get("/health", headers=TEST_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "task_manager" in data["details"]
        tm_stats = data["details"]["task_manager"]
        assert "max_concurrent_tasks" in tm_stats
        assert "active_tasks_count" in tm_stats
        assert "queued_tasks_count" in tm_stats
        assert tm_stats["max_concurrent_tasks"] == settings.max_active_tasks


@pytest.mark.asyncio
async def test_concurrent_uploads_concurrency_limiting(tmp_path: Path):
    db_file = tmp_path / "upload_flood.db"
    init_db(db_file)
    uploads_dir = tmp_path / "uploads_flood"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    settings.db_path = db_file
    settings.storage_path = uploads_dir
    settings.skip_startup_probe = True
    settings.max_active_tasks = 2

    app = create_app()

    mock_embed = AsyncMock(spec=EmbeddingClient)
    mock_embed.check_health.return_value = True
    mock_embed.embed_all.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

    mock_kev = AsyncMock(spec=KevClient)
    mock_kev.check_health.return_value = True
    mock_kev.decide_boundary.return_value = False

    mock_standard = AsyncMock(spec=StandardReranker)
    mock_standard.check_health.return_value = True

    app.state.embedding_client = mock_embed
    app.state.kev_client = mock_kev
    app.state.standard_reranker = mock_standard

    # Track maximum concurrent pipeline executions
    active_now = 0
    max_active_observed = 0

    from ragdoll.pipeline import worker
    original_pipeline = worker.process_asset_pipeline

    async def instrumented_pipeline(*args, **kwargs):
        nonlocal active_now, max_active_observed
        active_now += 1
        max_active_observed = max(max_active_observed, active_now)
        try:
            await original_pipeline(*args, **kwargs)
        finally:
            active_now -= 1

    with patch("ragdoll.pipeline.manager.process_asset_pipeline", side_effect=instrumented_pipeline):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # Upload 6 files concurrently
            async def upload_one(i: int):
                content = f"Document #{i} test content for concurrent flood testing.\n\nSection 2.".encode()
                files = {"file": (f"doc_{i}.txt", io.BytesIO(content), "text/plain")}
                r = await client.post("/store", files=files, headers=TEST_HEADERS)
                assert r.status_code == 202
                return r.json()["asset_id"]

            asset_ids = await asyncio.gather(*[upload_one(i) for i in range(6)])
            assert len(asset_ids) == 6

    # Verify peak concurrency respected the max_active_tasks limit
    assert max_active_observed <= 2
    assert max_active_observed > 0
