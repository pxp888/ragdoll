import pytest
import io
import time
from pathlib import Path
from fastapi.testclient import TestClient

from ragdoll.config import settings
from ragdoll.database import init_db
from ragdoll.main import create_app
from ragdoll.clients.embedding import EmbeddingClient
from ragdoll.clients.kev import KevClient

LIVE_HEADERS = {"X-API-Key": "live-test-user"}


@pytest.mark.asyncio
async def test_live_end_to_end(tmp_path: Path):
    """
    Full end-to-end integration test with live EmbeddingGemma2 and Kev 4B servers.
    """
    db_file = tmp_path / "live_e2e.db"
    init_db(db_file)
    uploads_dir = tmp_path / "live_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    settings.db_path = db_file
    settings.storage_path = uploads_dir
    settings.skip_startup_probe = False

    app = create_app()

    with TestClient(app) as client:
        # 1. Health check against live servers
        h_resp = client.get("/health", headers=LIVE_HEADERS)
        assert h_resp.status_code == 200
        assert h_resp.json()["status"] == "healthy"

        # 2. Upload sample markdown text
        document_text = (
            "# Artificial Intelligence Infrastructure\n\n"
            "High performance computing clusters require specialized interconnects and parallel file systems "
            "to feed large GPU nodes with sufficient bandwidth. NVLink and InfiniBand are commonly used. " * 15
            + "\n\n# Operating Financials\n\n"
            "In the third quarter, operating margins increased by 4.2% driven by supply chain optimizations "
            "and higher software license renewals across North American enterprise clients. " * 15
        )

        resp = client.post(
            "/store",
            files={"file": ("tech_report.md", io.BytesIO(document_text.encode("utf-8")), "text/markdown")},
            headers=LIVE_HEADERS,
        )
        assert resp.status_code == 202
        asset_id = resp.json()["asset_id"]

        # 3. Wait for background processing (coarse and fine stages) to complete
        max_wait = 30
        status_data = None
        start = time.time()
        while time.time() - start < max_wait:
            s_resp = client.get(f"/assets/{asset_id}/status", headers=LIVE_HEADERS)
            assert s_resp.status_code == 200
            status_data = s_resp.json()
            if status_data["status"] in ("fine_indexed", "failed"):
                break
            time.sleep(1.0)

        assert status_data is not None
        assert status_data["status"] == "fine_indexed"
        assert status_data["coarse_chunks"] >= 2
        assert status_data["fine_chunks"] >= 4
        assert status_data["error"] is None

        # 4. Search with vector match only
        search_resp = client.post(
            "/search",
            json={
                "query": "GPU interconnect and InfiniBand bandwidth",
                "top_k": 3,
                "stage": "fine",
                "rerank": False,
                "include_parent_context": True,
            },
            headers=LIVE_HEADERS,
        )
        assert search_resp.status_code == 200
        results = search_resp.json()["results"]
        assert len(results) >= 1
        assert "interconnect" in results[0]["text"].lower() or "gpu" in results[0]["text"].lower()
        assert results[0]["parent_context"] is not None

        # 5. Search with Kev semantic re-ranking enabled
        rerank_resp = client.post(
            "/search",
            json={
                "query": "third quarter operating margins",
                "top_k": 3,
                "stage": "fine",
                "rerank": True,
                "include_parent_context": True,
            },
            headers=LIVE_HEADERS,
        )
        assert rerank_resp.status_code == 200
        rerank_results = rerank_resp.json()["results"]
        assert len(rerank_results) >= 1
        assert "quarter" in rerank_results[0]["text"].lower() or "operating" in rerank_results[0]["text"].lower()
