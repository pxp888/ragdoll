import io
import pytest
from pathlib import Path
from unittest.mock import AsyncMock
import httpx
from fastapi.testclient import TestClient

from ragdoll.config import settings
from ragdoll.database import init_db
from ragdoll.main import create_app
from ragdoll.clients.embedding import EmbeddingClient
from ragdoll.clients.kev import KevClient


@pytest.fixture
def test_app(tmp_path: Path):
    db_file = tmp_path / "api_test.db"
    init_db(db_file)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    settings.db_path = db_file
    settings.storage_path = uploads_dir
    settings.skip_startup_probe = True

    app = create_app()

    mock_embed = AsyncMock(spec=EmbeddingClient)
    mock_embed.check_health.return_value = True
    mock_embed.embed_single.return_value = [0.1] * 768
    mock_embed.embed_batch.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    mock_embed.embed_all.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

    mock_kev = AsyncMock(spec=KevClient)
    mock_kev.check_health.return_value = True
    mock_kev.decide_boundary.return_value = False
    mock_kev.score_relevance.return_value = 1.95

    app.state.embedding_client = mock_embed
    app.state.kev_client = mock_kev

    return app, mock_embed, mock_kev


def test_health_check_ok(test_app):
    app, _, _ = test_app
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["embeddinggemma2"] == "healthy"
        assert data["kev_4b"] == "healthy"


def test_health_check_service_down(test_app):
    app, mock_embed, _ = test_app
    mock_embed.check_health.return_value = False

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["embeddinggemma2"] == "unhealthy"


def test_store_and_status_and_download(test_app):
    app, _, _ = test_app
    with TestClient(app) as client:
        content = b"Ragdoll test document content.\n\nSection 2 text for chunking."
        file_obj = io.BytesIO(content)

        resp = client.post(
            "/store",
            files={"file": ("report.txt", file_obj, "text/plain")},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "asset_id" in data
        assert data["filename"] == "report.txt"
        assert data["size_bytes"] == len(content)
        assert data["status"] in ["queued", "coarse_indexed", "fine_indexed"]

        asset_id = data["asset_id"]

        # Deduplication test: re-upload identical file
        file_obj.seek(0)
        resp2 = client.post(
            "/api/v1/assets",
            files={"file": ("report.txt", file_obj, "text/plain")},
        )
        assert resp2.status_code == 202
        assert resp2.json()["asset_id"] == asset_id

        # Query status
        status_resp = client.get(f"/assets/{asset_id}/status")
        assert status_resp.status_code == 200
        s_data = status_resp.json()
        assert s_data["asset_id"] == asset_id
        assert "status" in s_data
        assert "coarse_chunks" in s_data
        assert "fine_chunks" in s_data

        # Download original file
        get_resp = client.get(f"/get/{asset_id}")
        assert get_resp.status_code == 200
        assert get_resp.content == content


def test_search_endpoint(test_app):
    app, _, _ = test_app
    with TestClient(app) as client:
        # Upload a file to populate chunks
        content = b"Financial earnings in Q3 were outstanding.\n\nOperating expenses decreased."
        client.post(
            "/store",
            files={"file": ("finances.txt", io.BytesIO(content), "text/plain")},
        )

        # Search fine chunks without rerank
        search_req = {
            "query": "financial earnings Q3",
            "top_k": 3,
            "stage": "fine",
            "rerank": False,
            "include_parent_context": True,
        }
        resp = client.post("/search", json=search_req)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) >= 1
        first = data["results"][0]
        assert "chunk_id" in first
        assert "score" in first
        assert "text" in first

        # Search with rerank enabled
        search_rerank = {
            "query": "financial earnings Q3",
            "top_k": 3,
            "stage": "fine",
            "rerank": True,
            "include_parent_context": True,
        }
        resp_rerank = client.post("/search", json=search_rerank)
        assert resp_rerank.status_code == 200
        assert len(resp_rerank.json()["results"]) >= 1
